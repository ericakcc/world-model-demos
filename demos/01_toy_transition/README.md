# Demo 01: Toy Transition Model

World model 的最小可行版：用一個 MLP 在 Pendulum-v1 上學 `(s, a) → (Δs, r)`。

## 為什麼從這裡開始

- **狀態低維**（3 維），不用先處理視覺輸入
- **動作連續但 1 維**，不用糾結 categorical action
- **環境決定性**（給定 s, a 後 s' 完全確定），不用先處理隨機性
- **跑很快**：CPU 也能在幾分鐘內訓練完並看到 loss 下降

這個 demo 講清楚了 world model 的核心：**用監督學習擬合一個轉移函數**。後面的 demo 02/03 是在這個骨架上加 representation learning（VAE）和 sequence model（RNN/RSSM）。

## 跑

```bash
uv run python demos/01_toy_transition/train.py
```

可調的旗標：

| flag | 預設 | 說明 |
|------|------|------|
| `--rollout-steps` | 20000 | 用 random policy 收集多少 transition |
| `--epochs` | 20 | 訓練 epoch 數 |
| `--batch-size` | 256 | |
| `--lr` | 1e-3 | |
| `--hidden` | 256 | MLP 隱藏層維度 |
| `--horizon` | 50 | 評估 imagined rollout 的長度 |
| `--eval-episodes` | 20 | rollout 評估跑幾個 episode |

## 預期看到的訊號

1. **訓練 loss 跨 epoch 下降**：`train Δs` 從 ~0.3 降到 ~0.001 量級
2. **val loss 跟 train loss 接近**：沒有 overfit
3. **k-step rollout error 隨 k 上升**：1-step 很準（~0.001），50-step 顯著漏（~0.1+）— 這就是 **compounding error**，世界模型的核心難題

## 設計選擇

- **預測 Δs 而不是 s'**：successive states 高度相關，預測差值能讓網路把容易的部分（identity）省下來，專心學動態變化
- **同一個 trunk 出 (Δs, r)**：兩個 head 共享 representation，reward 訊號也能幫助學動態
- **single-file**：刻意 cleanrl 風格，整支腳本可以一口氣讀完

## 接下來

- 把 random policy 換成 model-based control（例如 random shooting / CEM 用 model 評分）
- 加入隨機環境（例如 noisy action）→ deterministic MLP 會 underperform → 動機去學**機率**模型（demo 02 的 MDN）
- 換成 pixel observation（例如 CarRacing）→ 動機去學 **encoder**（demo 02 的 VAE）
