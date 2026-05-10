# Demo 02: VAE + MDN-RNN + Controller

複刻 Ha & Schmidhuber 2018《World Models》的經典三件組架構。

📄 **Paper**: [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122) · [互動版](https://worldmodels.github.io/)

## 架構

```
        observation o_t  (e.g. 64×64×3 image)
                │
                ▼
        ┌──────────────┐
        │   V (VAE)    │  encoder + decoder, latent z ∈ ℝ³²
        └──────┬───────┘
               │ z_t
               ▼
        ┌──────────────┐
        │ M (MDN-RNN)  │  predict p(z_{t+1} | z_t, a_t, h_t)
        │              │  output = mixture of Gaussians
        └──────┬───────┘
               │ h_t (LSTM hidden)
               ▼
        ┌──────────────┐
        │ C (Controller)│  a_t = W · [z_t; h_t]    (linear, ~1k params)
        └──────────────┘  trained with CMA-ES (no backprop!)
```

**核心洞見**：V 跟 M 用大量 random rollout 做無監督訓練，C 只有 ~1000 個參數所以可以用演化算法直接搜，不需要 backprop 穿過 dynamics。

## 訓練流程（三階段）

1. **Collect**：random policy 跑 N=10000 個 rollout，存原始 frame + action
2. **Train V**：把所有 frame 當圖片 dataset 訓練 VAE（reconstruction + KL）
3. **Encode**：用 V 把 rollout 變成 latent sequence `(z_t, a_t)`
4. **Train M**：MDN-RNN 學 `p(z_{t+1} | z_t, a_t, h_t)`
5. **Train C**：CMA-ES 在環境中 evaluate `(W, b)`，最大化 episode return

## 實作 checklist

- [ ] `collect.py` — random rollouts on CarRacing-v2 / VizDoom，存 `data/02_vae_mdnrnn/rollouts.npz`
- [ ] `vae.py` — Conv encoder（4 層 stride-2）+ deconv decoder，latent 32 維
- [ ] `train_vae.py` — 從 rollouts 取 frame 當 dataset
- [ ] `mdn_rnn.py` — LSTM(256) → MDN with K=5 mixture components
- [ ] `train_mdnrnn.py` — teacher forcing on latent sequences
- [ ] `controller.py` — `nn.Linear(z_dim + h_dim, action_dim)`
- [ ] `train_controller.py` — CMA-ES via `cma` package；可平行 evaluate
- [ ] `dream.py` — 純 latent imagination：給定 z₀，跑 M，解碼回像素看「夢境」

## 預期看到的訊號

- VAE 重建：模糊但能認得車道/賽道
- MDN-RNN：teacher-forced log-likelihood 下降；自由 rollout 解碼後一開始合理，幾十步後開始 hallucinate
- Controller：CarRacing-v2 random ~ -50，目標 ~700+

## 簡化建議（給學習）

- 用 Pendulum-v1 + 隨機產生的 64×64 觀測（renderer）替代 CarRacing，避免 Box2D 依賴
- 用 GMM K=1（即單高斯）跑通 pipeline 後再加 mixture
- Controller 先用 random search 跑通，再換 CMA-ES

## 還沒寫

`train.py` 是個 placeholder。請依 checklist 逐步補完。
