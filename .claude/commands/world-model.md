# /world-model

幫使用者在這個 repo 中學習 / 操作 world model demos。

## 輸入解析

`$ARGUMENTS` 可能是：

1. **空** — 互動模式：列出四個 demo 的狀態，問使用者想做什麼
2. **概念名稱**（例如 `RSSM`、`MDN-RNN`、`VAE`、`diffusion`）— 解釋該概念，並指向對應 demo
3. **demo 編號或名稱**（例如 `01`、`toy`、`02`、`vae_mdnrnn`）— 帶使用者跑該 demo 或解釋它
4. **`run 01`** 之類的明確指令 — 直接執行

## 步驟

### 1. 載入背景

讀取 `skills/world-models/SKILL.md` 取得概念地圖。需要時讀對應 demo 的 `README.md`。

### 2. 判斷意圖

如果 `$ARGUMENTS` 為空，用 `AskUserQuestion` 問：

> 想做什麼？
> - 解釋一個 world model 概念
> - 走讀某個 demo 的程式碼
> - 直接跑 demo（目前只有 01 完整可跑）
> - 補完一個 scaffold demo

### 3. 執行

- **解釋概念**：先給一段直覺，再對應到 demo 的具體程式碼行（用 `file_path:line_number` 格式）
- **走讀程式碼**：照論文流程逐塊解釋，標註對應的 paper section
- **跑 demo**：執行 `uv run python demos/<name>/train.py`，解讀輸出
- **補完 scaffold**：讀該 demo 的 README checklist，逐項實作

## 輸出風格

- 教學優先：每段程式碼配一句「這對應論文裡哪個元件」
- 不要直接貼大段論文摘要——使用者要的是把 paper 跟 code 接起來
- 跑 demo 時，先解釋預期看到什麼訊號（loss 該往哪走），再跑
