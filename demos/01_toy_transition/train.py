"""
Demo 01: Toy transition model on Pendulum-v1.

Learn p(s', r | s, a) with a simple MLP. The minimum viable world model.

Pendulum:
    state   = (cos θ, sin θ, θ_dot)              [3-dim]
    action  = torque ∈ [-2, 2]                   [1-dim]
    reward  = -(θ² + 0.1·θ_dot² + 0.001·u²)

We learn:
    Δs = f_θ(s, a)        → predict s_{t+1} = s_t + Δs
    r̂ = g_φ(s, a)         → MSE against true reward

Predicting the delta (not s' directly) keeps the dynamic range small and
gives a much cleaner training signal.

Usage:
    python demos/01_toy_transition/train.py
    python demos/01_toy_transition/train.py --epochs 50 --rollout-steps 50000

Outputs:
    checkpoints/01_toy_transition/model.pt
    runs/01_toy_transition/loss.png
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ENV_ID = "Pendulum-v1"
DEMO_NAME = "01_toy_transition"


def collect_rollouts(env_id, total_steps, seed):
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)

    states, actions, next_states, rewards = [], [], [], []
    for _ in range(total_steps):
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, _ = env.step(a)
        states.append(obs)
        actions.append(a)
        next_states.append(next_obs)
        rewards.append(r)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    env.close()
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
    )


class TransitionModel(nn.Module):
    """f(s, a) -> (Δs, r̂). Single trunk, two heads."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, s, a):
        h = self.trunk(torch.cat([s, a], dim=-1))
        return self.delta_head(h), self.reward_head(h).squeeze(-1)


def evaluate_kstep_rollout(model, env_id, horizon, episodes, device):
    """Compare imagined rollout against the real env.

    Imagine = recursive 1-step prediction with the same random actions
    that the real env receives. State MSE per step shows how the error
    compounds the further we roll into the future.
    """
    env = gym.make(env_id)
    errors = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        env.action_space.seed(20_000 + ep)
        s_imag = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        for t in range(horizon):
            a_np = env.action_space.sample()
            a_t = torch.from_numpy(a_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                ds, _ = model(s_imag, a_t)
            s_imag = s_imag + ds
            obs, _, term, trunc, _ = env.step(a_np)
            errors[t] += float(np.mean((s_imag.cpu().numpy()[0] - obs) ** 2))
            counts[t] += 1
            if term or trunc:
                break
    env.close()
    return errors / np.maximum(counts, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-steps", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/4] Collecting {args.rollout_steps} random transitions on {ENV_ID}...")
    s, a, sn, r = collect_rollouts(ENV_ID, args.rollout_steps, args.seed)
    delta = sn - s
    state_dim, action_dim = s.shape[1], a.shape[1]
    print(f"  state_dim={state_dim}, action_dim={action_dim}, samples={len(s)}")

    n_val = max(1024, len(s) // 10)
    perm = np.random.permutation(len(s))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(s[idx]),
            torch.from_numpy(a[idx]),
            torch.from_numpy(delta[idx]),
            torch.from_numpy(r[idx]),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    print(f"[2/4] Training on {args.device} for {args.epochs} epochs...")
    model = TransitionModel(state_dim, action_dim, hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_state": [], "train_reward": [], "val_state": [], "val_reward": []}
    for epoch in range(args.epochs):
        model.train()
        ts = tr = n = 0.0
        for sb, ab, db, rb in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            sb, ab = sb.to(args.device), ab.to(args.device)
            db, rb = db.to(args.device), rb.to(args.device)
            ds_pred, r_pred = model(sb, ab)
            loss_s = F.mse_loss(ds_pred, db)
            loss_r = F.mse_loss(r_pred, rb)
            loss = loss_s + loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = sb.size(0)
            ts += loss_s.item() * bs
            tr += loss_r.item() * bs
            n += bs

        model.eval()
        vs = vr = nv = 0.0
        with torch.no_grad():
            for sb, ab, db, rb in val_loader:
                sb, ab = sb.to(args.device), ab.to(args.device)
                db, rb = db.to(args.device), rb.to(args.device)
                ds_pred, r_pred = model(sb, ab)
                bs = sb.size(0)
                vs += F.mse_loss(ds_pred, db).item() * bs
                vr += F.mse_loss(r_pred, rb).item() * bs
                nv += bs

        history["train_state"].append(ts / n)
        history["train_reward"].append(tr / n)
        history["val_state"].append(vs / nv)
        history["val_reward"].append(vr / nv)
        print(
            f"  epoch {epoch+1:2d}: "
            f"train Δs={ts/n:.5f} r={tr/n:.5f} | "
            f"val Δs={vs/nv:.5f} r={vr/nv:.5f}"
        )

    print(f"[3/4] Evaluating {args.horizon}-step imagined rollouts ({args.eval_episodes} episodes)...")
    kstep_err = evaluate_kstep_rollout(model, ENV_ID, args.horizon, args.eval_episodes, args.device)
    for k in sorted({1, 5, 10, 20, args.horizon}):
        if 1 <= k <= args.horizon:
            print(f"  step {k:3d}: state MSE = {kstep_err[k-1]:.5f}")

    print("[4/4] Saving artifacts...")
    ckpt_dir = Path("checkpoints") / DEMO_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "history": history,
        "kstep_err": kstep_err.tolist(),
    }, ckpt_path)
    print(f"  checkpoint: {ckpt_path}")

    try:
        import matplotlib.pyplot as plt
        runs_dir = Path("runs") / DEMO_NAME
        runs_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history["train_state"], label="train Δs")
        axes[0].plot(history["val_state"], label="val Δs")
        axes[0].plot(history["train_reward"], label="train r")
        axes[0].plot(history["val_reward"], label="val r")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE"); axes[0].set_yscale("log")
        axes[0].set_title("training loss"); axes[0].legend()
        axes[1].plot(range(1, args.horizon + 1), kstep_err, marker="o", markersize=3)
        axes[1].set_xlabel("rollout step"); axes[1].set_ylabel("state MSE")
        axes[1].set_title(f"k-step imagined rollout error")
        fig.tight_layout()
        plot_path = runs_dir / "loss.png"
        fig.savefig(plot_path, dpi=120)
        print(f"  plot:       {plot_path}")
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")


if __name__ == "__main__":
    main()
