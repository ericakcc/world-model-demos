# World Models — 概念地圖

## 一句話定義

World model 就是學一個能模擬環境動態的神經網路：

```
p(s_{t+1}, r_{t+1} | s_t, a_t)
```

有了它，agent 可以**在腦中想像**未來，不必每一步都跟真實環境互動。對 sample-efficient RL、long-horizon planning、video prediction 都是核心。

## 為什麼重要

1. **Sample efficiency** — 真實環境互動昂貴（機器人、遊戲、模擬器），world model 訓練好之後可以在 imagination 中跑數百萬步
2. **Planning** — 顯式的轉移模型可以做 MPC、tree search
3. **Representation learning** — 為了預測未來，model 必須學到緊湊有意義的 state representation
4. **Scaling** — video diffusion 把 world model 推到像素級、長時序

## 關鍵 paper 與 demo 對應

| Paper | 年份 | 關鍵概念 | 對應 demo |
|-------|------|---------|----------|
| (隱式) MLP transition model | — | 監督學習 `(s,a) → (s', r)`，最小可行 | `demos/01_toy_transition/` |
| **World Models** (Ha & Schmidhuber) | 2018 | VAE 壓縮觀測 + MDN-RNN 預測 latent + 演化算法控制器 | `demos/02_vae_mdnrnn/` |
| **PlaNet** (Hafner et al.) | 2019 | RSSM = deterministic GRU + stochastic latent，CEM planning | `demos/03_dreamer_rssm/` |
| **Dreamer / DreamerV2 / V3** | 2019–2023 | RSSM + actor-critic 在 imagination 中訓練 | `demos/03_dreamer_rssm/` |
| **GAIA-1, Sora** | 2023+ | latent video diffusion 當作 driving / general world model | `demos/04_video_diffusion/` |

## 共通元件

幾乎所有 world model 都長這個樣子：

```
       observation o_t                        action a_t
            │                                     │
            ▼                                     │
       ┌─────────┐                                │
       │ encoder │                                │
       └────┬────┘                                │
            │ z_t (latent / belief)               │
            ▼                                     │
        ┌────────────────────────────────────────┘
        ▼
   ┌──────────────┐    z_{t+1}
   │ transition / │ ─────────────► (decoder → ô_{t+1})
   │   dynamics   │ ─────────────► (reward head → r̂_{t+1})
   └──────────────┘ ─────────────► (continue head → donê_{t+1})
```

不同方法的差異主要在：

- **encoder**：MLP（low-dim state）/ CNN / VAE / VQ-VAE / diffusion encoder
- **transition**：MLP / GRU / LSTM / MDN-RNN / RSSM / Transformer / diffusion
- **怎麼用**：planning（CEM、MCTS）/ 在 imagination 中訓練 actor-critic / 直接生成 video

## 訓練 loss 的常見組合

- 重建 loss：`||decode(z) - o||²` 或 cross-entropy（discrete latent）
- KL：`KL(q(z|o) || p(z))` 或 prior matching（VAE / RSSM）
- reward / done prediction loss
- 在 latent 空間的 dynamics consistency loss

## 評估訊號

學 world model 時，這幾個訊號比 final reward 更早告訴你模型有沒有學到東西：

1. **1-step prediction error** 隨訓練下降 → encoder + dynamics 至少有學到
2. **k-step rollout error**（k=10/50）隨訓練下降 → dynamics 有 generalization
3. **重建 loss** 收斂 → encoder/decoder 有壓縮能力
4. **在 imagination 中跑出的 trajectory 看起來合理**（解碼成像素或低維 plot）

如果這些訊號都沒動，policy 部分通常也救不回來。

## 給 Claude 的提示

當使用者問 world model 相關概念，先確認他指的是哪個層級：
1. 純概念（這份文件就夠用）
2. 某個特定 paper（指向對應 demo 的 README）
3. 怎麼從零訓練（指向 `demos/01_toy_transition/train.py` 當入門範本）

不要把 world model 跟 LLM 的 "world knowledge" 混為一談——這裡專指 RL / 序列決策意義下的環境動態模型。
