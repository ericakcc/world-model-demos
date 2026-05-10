# World Model Demos

學習 World Model 概念的小型實驗 repo。每個 demo 自成一檔（cleanrl 風格）。

## 結構

- `demos/01_toy_transition/` — MLP 學 `(s,a) -> (s', r)` on Pendulum-v1（**完整可跑**）
- `demos/02_vae_mdnrnn/` — Ha & Schmidhuber 2018（scaffold，README 有實作 checklist）
- `demos/03_dreamer_rssm/` — Dreamer 系列（scaffold）
- `demos/04_video_diffusion/` — video diffusion world model（scaffold）
- `skills/world-models/SKILL.md` — 概念地圖與論文索引
- `.claude/commands/world-model.md` — `/world-model` 斜線命令

## 如何跑

```bash
uv sync                                              # 安裝依賴
uv run python demos/01_toy_transition/train.py       # 跑 demo 01
```

## 實作慣例

- 單檔 cleanrl 風格：每個 demo 的訓練腳本盡量自成一檔，少抽象、少跨目錄 import
- checkpoint 存到 `checkpoints/<demo_name>/`
- rollout 資料存到 `data/<demo_name>/`
- 訓練 log 存到 `runs/<demo_name>/`
- 上述三個目錄都在 `.gitignore` 中

## 補完 scaffold demo 時

每個 scaffold demo 的 `README.md` 列了：對應論文、要實作的元件、預期的成功訊號。動工前先讀那份 README。
