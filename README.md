# World Model Demos

學習世界模型（World Models）概念的小型實驗 codebase。每個 demo 自成一檔（cleanrl 風格），方便邊讀論文邊跟著程式碼跑。

## 概念一句話

> World model = 學一個 `p(s' | s, a)`（與 reward），讓 agent 可以**在腦中想像**未來，不必每次都跟真實環境互動。

## 四個 demo

| # | 主題 | 論文 / 來源 | 狀態 |
|---|------|------------|------|
| 01 | Toy transition model | — | ✅ 可跑 |
| 02 | VAE + MDN-RNN + Controller | Ha & Schmidhuber 2018 | 🚧 scaffold |
| 03 | Dreamer / RSSM | Hafner 2019/2020/2023 | 🚧 scaffold |
| 04 | Video diffusion world model | Sora / GAIA-1 等 | 🚧 scaffold |

建議學習路徑就是 01 → 02 → 03 → 04，難度遞增。

## 安裝

```bash
uv sync
```

## 跑 demo 01

```bash
uv run python demos/01_toy_transition/train.py
```

訓練完會把 checkpoint 存到 `checkpoints/`，並印出 1-step / k-step rollout 的預測誤差。

## 結構

```
├── demos/
│   ├── 01_toy_transition/   # MLP 學 Pendulum 的 (s,a) -> (s', r)
│   ├── 02_vae_mdnrnn/       # 經典 World Models 三件組
│   ├── 03_dreamer_rssm/     # latent-space planning + actor-critic
│   └── 04_video_diffusion/  # 像素級 video world model
├── notebooks/               # 探索與視覺化
├── skills/world-models/     # Claude 用的概念整理
└── .claude/commands/        # /world-model 斜線命令
```

## 參考

各 demo 目錄下的 `README.md` 有對應論文連結與實作要點。`skills/world-models/SKILL.md` 是整體概念地圖。
