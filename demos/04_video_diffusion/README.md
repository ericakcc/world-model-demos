# Demo 04: Video Diffusion World Model

把 world model 推到像素級、長時序。Action-conditioned video diffusion 是 Sora / GAIA-1 / Genie 這類「general world model」的核心。

📄 **Papers**:
- [Video Diffusion Models (Ho et al. 2022)](https://arxiv.org/abs/2204.03458) — 基本 video DDPM
- [GAIA-1 (Wayve 2023)](https://arxiv.org/abs/2309.17080) — autoregressive driving world model
- [Genie (Bruce et al. 2024)](https://arxiv.org/abs/2402.15391) — playable video, learned actions
- [Sora (OpenAI 2024)](https://openai.com/research/video-generation-models-as-world-simulators) — DiT-based, scaling

## 跟前面 demo 的關係

| Demo | latent | dynamics | output |
|------|--------|----------|--------|
| 01 | raw state | MLP | next state |
| 02 | VAE z (32) | MDN-RNN | next z |
| 03 | RSSM (h, z) | GRU + MLP | next state |
| **04** | VAE / VQ-VAE z (HxW grid) | **Diffusion** denoiser | **video clip** |

關鍵差別：04 不是學 single-step transition，而是**一次生成整段未來 video clip**（conditioned on past frames + actions）。Diffusion 自然處理多模態未來。

## 最簡架構

```
past frames + actions ──► spatiotemporal UNet ──► denoise N steps ──► future frames
                          (3D conv or DiT)
```

訓練：DDPM 的 ε-prediction loss，conditioned on past frames + action sequence。

## 實作 checklist

- [ ] `data.py` — load 影片 dataset (BAIR robot pushing? CarRacing rollouts? Atari?)
- [ ] `unet.py` — 小型 3D UNet（spatiotemporal），conditioned on past + action
- [ ] `diffusion.py` — DDPM forward / sample（cosine schedule, ε-prediction）
- [ ] `train.py` — 標準 diffusion training loop
- [ ] `sample.py` — 給定 past frames，sample 未來 K 幀；存 mp4

## 從哪個 dataset 起步（建議）

- **最簡**：用 demo 02 收集的 random rollout 影片（Pendulum/CarRacing），32×32 解析度，clip 長 8 幀
- **中等**：BAIR robot pushing dataset（64×64，公開）
- **挑戰**：Atari Pong/Breakout 的 frame skip 4 序列

## 預期看到的訊號

- Loss（noise MSE）下降到 ~0.02 量級
- 生成 sample：前 2–3 幀清楚，後面開始 blur 或 mode collapse 是正常的
- 跟前面 demo 的差別：解碼出的影像直接是像素，不需要另外 train decoder

## 還沒寫

`train.py` 是個 placeholder。建議先寫個 toy 1D diffusion（生成 sin 波）跑通流程，再上 video。
