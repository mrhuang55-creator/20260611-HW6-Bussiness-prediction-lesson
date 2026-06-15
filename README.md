# Kaggle 50 Startups 新創公司利潤預測與多元線性迴歸分析 (CRISP-DM)

本專案是一個完整遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 流程的數據分析與預測專案。我們針對 Kaggle 上的 50 Startups 資料集進行了探索性數據分析 (EDA)、特徵工程、多模型建模與評估、特徵逐步載入實驗、迴歸診斷，並最終打造了一個精美的網頁端互動式儀表板與即時預測計算器。

---

## 🔍 索引 / 目錄 (Index)

*   [🎨 快速體驗：開啟互動式網頁儀表板 (index.html)](#-快速體驗開啟互動式網頁儀表板-indexhtml)
*   [📂 專案目錄結構](#-專案目錄結構)
*   [📈 專案核心成果摘要](#-專案核心成果摘要)
    *   [1. 多模型評估比較](#1-多模型評估比較-modeling--evaluation)
    *   [2. 特徵選取與逐步載入實驗](#2-特徵選取與逐步載入實驗-feature-selection)
    *   [3. 多元線性迴歸診斷與決定因子分析](#3-多元線性迴歸診斷與決定因子分析-regression-diagnostics)
*   [🎨 互動式網頁儀表板 (Dashboard) 詳細說明](#-互動式網頁儀表板-dashboard)
*   [🛠️ 環境配置與運行指南](#-環境配置與運行指南)
    *   [1. 安裝環境套件](#1-安裝環境套件)
    *   [2. 下載數據與預處理](#2-下載數據與預處理)
    *   [3. 模型訓練與評估](#3-模型訓練與評估)
    *   [4. 特徵選取與診斷實驗](#4-特徵選取與診斷實驗)
    *   [5. 產出中文圖表與報告](#5-產出中文圖表與報告)
    *   [6. 命令列即時預測](#6-命令列即時預測)
    *   [7. 開啟網頁互動式儀表板](#7-開啟網頁互動式儀表板)
*   [📝 專案變更與開發日誌](#-專案變更與開發日誌)

---

## 🎨 快速體驗：開啟互動式網頁儀表板 (index.html)

您可以直接點擊或在瀏覽器中開啟 [index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/index.html) 來體驗我們為您設計的學術診斷平台與黑板即時預測計算器！

---

## 📂 專案目錄結構

```text
├── 0612-更動/          # 2026-06-12 實驗更動封存檔（含圖表、報告與網頁備份）
├── data/               # 專案數據目錄（含原始 50_Startups.csv 與劃分後的 train/test 數據）
├── models/             # 訓練完成並導出的模型檔 (.joblib)
├── plots/              # 專案生成之分析與評估圖表 (英文版)
├── src/                # Python 原始碼目錄
│   ├── data_preprocessing.py          # 數據載入、劃分與 One-Hot 編碼預處理
│   ├── data_understanding.py           # 探索性數據分析 (EDA) 與關聯性圖表繪製
│   ├── train.py                        # 訓練 5 種模型 (OLS, Ridge, Lasso, RF, DT) 並儲存模型
│   ├── feature_selection_experiment.py # 1 至 5 個自變數逐步加入的特徵選取實驗
│   ├── regression_analysis.py          # 針對全特徵的多元線性迴歸 OLS 診斷分析
│   ├── plot_summary.py                 # 生成模型效能對比圖與預測對比圖
│   └── generate_plots_zh.py            # 繪製繁體中文版本的對比與診斷圖表
├── 繁中圖表/           # 繁體中文版本的分析、效能對比與實驗診斷圖表
├── index.html          # 整合性互動式總結網頁（含數據看板、學術討論與迴歸預測計算器）
├── classroom_bg.png    # 網頁背景圖 (溫馨教室風格)
├── walkthrough.md      # 專案完整 CRISP-DM 成果報告與學術原理討論
├── log.md              # 專案開發工作日誌 (Work Log & Changelog)
├── requirements.txt    # 專案環境依賴套件清單
└── predict.py          # 預測命令行腳本 (讀取導出模型進行即時預測)
```

---

## 📈 專案核心成果摘要

### 1. 多模型評估比較 (Modeling & Evaluation)
我們建立了 5 種不同的迴歸模型並在測試集上進行了嚴謹的評估：
*   **隨機森林迴歸 (Random Forest)** 表現最優，測試集 $R^2 \approx 0.9147$，MAE 僅約 **$6,132**。
*   **多元線性迴歸 (Multiple Linear Regression)** 與正則化模型（Ridge, Lasso）表現非常穩定，測試集 $R^2 \approx 0.8987$。
*   詳細的數據對比與圖表，請參考 [walkthrough.md](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/walkthrough.md)。

### 2. 特徵選取與逐步載入實驗 (Feature Selection)
我們設計了從 1 個自變數到 5 個自變數逐步載入的線性迴歸實驗：
*   **最佳特徵組合**：2個特徵 `['R&D Spend', 'Marketing Spend']`，測試集 $R^2$ 達到最高 (**94.76%**)，RMSE 降至最低 (**$5,367.34**)。
*   **過擬合崩盤**：當特徵數量增加到 5 個（加入不顯著的 `Administration`）時，由於樣本量小 ($N=40$)，殘差自由度流失，導致測試集 $R^2$ 崩塌至 **0.8443**，RMSE 懸崖式暴增至 **$9,252.64**。

### 3. 多元線性迴歸診斷與決定因子分析 (Regression Diagnostics)
利用 `statsmodels` 進行 OLS 迴歸估計，得出以下統計診斷結果：
*   **研發支出 (R&D Spend)**：標準化 Beta 達 **0.9183**，P 值為 **0.0000**，為絕對主導之核心特徵。
*   **行銷支出 (Marketing Spend)**：雖然與利潤有高度 Pearson 相關，但在多元迴歸中因多重共線性被稀釋，獨立邊際貢獻不顯著 ($P = 0.1871 > 0.05$)。
*   **行政支出 (Administration) 與地區 (State)**：對利潤無顯著影響。使用 One-Hot Encoding 時，我們剔除了第一類別 `California` 作為 Baseline，以避免**虛擬變數陷阱 (Dummy Variable Trap)**。

---

## 🎨 互動式網頁儀表板 (Dashboard)

專案包含了一個高視覺質感的 `index.html` 網頁：
*   **主題視覺**：採用高畫質教室背景、木紋書夾頂部、玻璃擬態（Glassmorphism）紙張卡片與橫線筆記本紙張紋理。
*   **黑板預測器**：底部的即時預測計算器採用**深綠色黑板質感、實木板框、發光粉筆字體與手繪圖示**。
*   **即時互動**：使用者可以自由拖曳 R&D 研發、Marketing 行銷、Administration 行政的滑桿，並切換所屬地區，網頁將即時調用 OLS 迴歸公式計算並渲染預測的 Profit。

---

## 🛠️ 環境配置與運行指南

### 1. 安裝環境套件
請使用 Python 3.8+ 並安裝依賴套件：
```bash
pip install -r requirements.txt
```

### 2. 下載數據與預處理
```bash
# 下載 Kaggle 50 Startups 原始資料
python download_data.py

# 執行數據理解與探索性分析 (EDA)
python src/data_understanding.py

# 執行數據預處理與劃分
python src/data_preprocessing.py
```

### 3. 模型訓練與評估
```bash
# 訓練 5 種模型並導出至 models/ 目錄
python src/train.py

# 生成模型效能與預測對比圖表
python src/plot_summary.py
```

### 4. 特徵選取與診斷實驗
```bash
# 執行特徵逐步載入實驗並生成折線圖
python src/feature_selection_experiment.py

# 進行多元線性迴歸 OLS 診斷分析並生成指標圖
python src/regression_analysis.py
```

### 5. 產出中文圖表與報告
```bash
# 產生所有圖表的繁體中文翻譯版本（存於 繁中圖表/）
python src/generate_plots_zh.py
```

### 6. 命令列即時預測
可以使用 `predict.py` 腳本在命令列進行預測：
```bash
python predict.py
```
執行後會提示輸入各項支出與地區，並同時輸出多元線性迴歸與隨機森林模型的預測結果。

### 7. 開啟網頁互動式儀表板
直接點擊開啟 [index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/index.html)（或在專案根目錄下直接在瀏覽器中開啟 `index.html`），即可在溫馨的教室與黑板風格 UI 中拉動滑桿，進行實時的利潤預測與學術討論研讀。

---

## 📝 專案變更與開發日誌
專案的詳細開發時序與優化歷程記錄於 [log.md](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/log.md)。
