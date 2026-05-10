# Demo 03: Dreamer / RSSM

複刻 Dreamer 系列的核心：**RSSM (Recurrent State-Space Model)** + 在 imagination 中訓練 actor-critic。

📄 **Papers**:
- [PlaNet (Hafner et al. 2019)](https://arxiv.org/abs/1811.04551) — RSSM + CEM planning
- [Dreamer (Hafner et al. 2019)](https://arxiv.org/abs/1912.01603) — RSSM + actor-critic in imagination
- [DreamerV2 (Hafner et al. 2020)](https://arxiv.org/abs/2010.02193) — discrete latents, Atari SOTA
- [DreamerV3 (Hafner et al. 2023)](https://arxiv.org/abs/2301.04104) — single-config across 150+ tasks

## RSSM 架構

每個時間步的 state 同時有 **deterministic** 跟 **stochastic** 兩部分：

```
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])     # deterministic recurrent state
prior:    p(z_t | h_t)                     # 從 h 預測下一步先驗
posterior: q(z_t | h_t, o_t)                # 看到觀測後的後驗
```

訓練時學三件事：

1. **Reconstruction**：`p(o_t | h_t, z_t)` — 從 state 重建觀測
2. **Reward / continue**：`p(r_t, c_t | h_t, z_t)` — 預測 reward 跟 episode 是否繼續
3. **Dynamics**：`KL(q(z_t | h_t, o_t) || p(z_t | h_t))` — 讓 prior 學會預測 posterior

## Imagination 訓練 actor-critic

跟典型 RL 的差別：**不用真環境 rollout 就能訓練 policy**。

```
1. 從 replay buffer 抽 batch，跑 RSSM encode 得到一串 (h_t, z_t)
2. 從每個 (h_t, z_t) 出發，用 RSSM prior 跟 actor 做 imagination rollout（H=15 步）
3. 用 critic 估 value，用 λ-return 計算 actor / critic loss
4. backprop 全程穿過可微的 RSSM
```

## 實作 checklist

- [ ] `rssm.py` — `RSSM(z_dim=32, h_dim=200)` with prior/posterior/recurrent components
- [ ] `world_model.py` — RSSM + obs encoder + decoder + reward/continue heads
- [ ] `actor_critic.py` — `Actor(stoch=tanh-Normal)` + `Critic(MLP)`
- [ ] `replay.py` — sequence replay buffer (sample length-L chunks)
- [ ] `train.py` — main loop：collect → train world model → imagination → train actor-critic
- [ ] `eval.py` — deterministic eval episode

## 從哪個版本開始

建議從 **Dreamer (v1)** 起步：
- 連續 latent（高斯），不用 straight-through gumbel
- DM Control suite 的 Cheetah/Walker/Hopper 是經典 benchmark
- 環境 dim 低，tabletop GPU 兩三天能跑出像樣結果

V2 的 discrete latent 跟 V3 的 symlog/twohot 等技巧之後再補。

## 預期看到的訊號

- World model loss（recon + KL + reward）穩定下降
- Imagination 解碼出的 frame sequence 看起來像真實 trajectory（前 5–10 步）
- Actor return 隨訓練上升；典型 cheetah-run 能到 800+

## 還沒寫

`train.py` 是個 placeholder。RSSM 是這個 demo 最值得手寫的部分——建議先讀 PlaNet paper Section 3 再動工。
