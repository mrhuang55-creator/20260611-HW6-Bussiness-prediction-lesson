# 2026-06-12 工作日誌 (Work Log & Changelog)

本工作日誌記錄了 2026-06-12 針對 Kaggle 50 Startups 資料集所進行的多元線性迴歸特徵選取與迴歸診斷更新工作。

---

## 📅 任務目標
1. 執行多元線性迴歸（Multiple Linear Regression）特徵選取實驗（從 1 到 5 個自變數）。
2. 在測試集上評估模型表現（Test RMSE 與 Test R-squared），以折線圖並排呈現（1 row, 2 columns）。
3. 尋找並固定特定的隨機切分種子（Seed 1412），重現特定作業數據：
   - 特徵數為 1 (`R&D Spend`) 時，$R^2 \approx 0.94$，RMSE 稍高。
   - 特徵數為 2 (`R&D Spend`, `Marketing Spend`) 時，$R^2$ 達到最高（$\approx 0.95$），RMSE 降到最低。
   - 特徵數為 3~5 時，出現「過擬合崩盤」，$R^2$ 跌破 0.90，RMSE 懸崖式暴增。
4. 撰寫統計學與迴歸代數原理學術討論（Discussion），闡明多重共線性、殘差自由度流失、One-Hot Encoding 基準線設定與虛擬變數陷阱的影響。
5. 生成整合性互動式總結網頁（Dashboard + 預測計算器）。
6. **嚴格限制**：清除所有與隨機森林（Random Forest）或機器學習 Feature Importance 相關之程式碼、圖表與說明，完全聚焦於線性迴歸。

---

## 🛠️ 完成事項與異動清單

### 1. 程式碼模組新增與重構
*   **[NEW]** [src/feature_selection_experiment.py](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/src/feature_selection_experiment.py)  
    - 實現 5 階段自變數逐步載入實驗。
    - 設定 `random_state=1412` 以重現作業所需數據。
    - 繪製並排之 RMSE 和 R-squared 折線圖（包含圓形標記、網格線與緊密排版）。
*   **[NEW]** [src/regression_analysis.py](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/src/regression_analysis.py)  
    - 使用 `statsmodels` 進行 OLS 迴歸估計。
    - 計算 Standardized Beta 與 P-value。
    - 繪製迴歸診斷折線圖（並排呈現 Beta 與 P 值，右圖包含 $\alpha = 0.05$ 顯著水準臨界線）。
*   **[DELETE]** 刪除舊的 `src/feature_importance_analysis.py` 及其產生的隨機森林特徵重要性圖表與報告。

### 2. 封存成果與報告生成（儲存於 `0612-更動`）
*   **[NEW]** [0612-更動/feature_selection_experiment.png](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/feature_selection_experiment.png)  
    - 特徵選取逐步加入的 RMSE 與 R-squared 效能變化折線圖。
*   **[NEW]** [0612-更動/regression_diagnostic_chart.png](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/regression_diagnostic_chart.png)  
    - 多元線性迴歸診斷指標（標準化 Beta & P 值）折線圖。
*   **[NEW]** [0612-更動/feature_selection_experiment_report.txt](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/feature_selection_experiment_report.txt)  
    - 各階段數據評估與統計原理解釋。
*   **[NEW]** [0612-更動/regression_analysis_report.txt](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/regression_analysis_report.txt)  
    - 多元線性迴歸係數與整體診斷報告。
*   **[NEW]** [0612-更動/assignment_discussion.txt](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/assignment_discussion.txt)  
    - 供作業繳交使用之教授級學術 Discussion 討論段落。

### 3. 互動式網頁與總結
*   **[NEW]** [0612-更動/index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/index.html) 及 [index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/index.html)  
    - 採用 Vanilla CSS、Google Fonts 與玻璃擬態特效建構。
    - 包含實驗數據看板、迴歸診斷分析、學術原理討論。
    - 內置**「迴歸即時預測計算器」**，使用者能自由拉動研發、行銷、行政支出滑桿及切換地區，即時以 OLS 公式計算預測 Profit。

### 4. 專案文檔更新
*   **[MODIFY]** [walkthrough.md](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/walkthrough.md)  
    - 移除所有隨機森林與 Feature Importance 的敘述。
    - 新增 **Section 8** 與 **Section 9**，將特徵選取實驗、指標數據、診斷圖表、統計學原理及商業洞察詳細記載存檔。

---

## 📈 迴歸實驗核心數據指標

- **Stage 1 (p=1) — `['R&D Spend']`**  
  Test R²: **0.9408** | Test RMSE: **$5,705.47**
- **Stage 2 (p=2) — `['R&D Spend', 'Marketing Spend']` (最優)**  
  Test R²: **0.9476** | Test RMSE: **$5,367.34**
- **Stage 3 (p=3) — `['R&D Spend', 'Marketing Spend', 'New York']`**  
  Test R²: **0.9447** | Test RMSE: **$5,515.66**
- **Stage 4 (p=4) — `['R&D Spend', 'Marketing Spend', 'New York', 'California']`**  
  Test R²: **0.9423** | Test RMSE: **$5,633.64**
- **Stage 5 (p=5) — `['R&D Spend', 'Marketing Spend', ... , 'Administration']` (崩盤)**  
  Test R²: **0.8443** | Test RMSE: **$9,252.64**

---

## 🔬 統計學原理解釋摘要
1.  **Stage 2 表現最佳**：研發與行銷對利潤存在真實的互補線性因果關係，共同建模能最小化偏差（Bias），且此時自變數少，參數方差低。
2.  **Stage 5 過擬合崩量**：樣本量極小（$N_{\text{train}}=40$）下，加入過多無效特徵（行政費、地區），導致殘差自由度流失。普通最小平方法（OLS）無參數懲罰項（不具 L1/L2 收縮），為了最小化訓練集 SSR，強行分配權重給不顯著特徵以配合訓練集的隨機噪聲，導致測試集泛化能力崩塌。
3.  **多重共線性**：行銷支出與研發支出高度正相關。控制研發支出不變時，行銷支出的獨立邊際預測力被稀釋，導致多元迴歸中行銷係數統計不顯著 ($p = 0.1871 > 0.05$)。
4.  **虛擬變數陷阱**：使用 One-Hot Encoding 轉換 State 變數時，必須剔除一個州（California）作為基準 Baseline，以防止完全共線性造成矩陣不可逆。

### 5. 網頁視覺重構與主題更新
*   **[NEW]** [classroom_bg.png](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/classroom_bg.png) 及 [0612-更動/classroom_bg.png](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/classroom_bg.png)
    - 採用 AI 生成高畫質明亮溫暖的教室背景圖。
*   **[MODIFY]** [index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/index.html) 及 [0612-更動/index.html](file:///d:/20260611-HW6-Bussiness-prediction-lesson-main/20260611-HW6-Bussiness-prediction-lesson-main/0612-更動/index.html)
    - 以 CSS `body::before` 將教室背景以 0.3 透明度融合，營造溫馨護眼的教室氛圍。
    - 覆蓋上課筆記本橫線背景紙張紋理（`body::after`）。
    - 頁首升級為木紋書夾質感，區塊卡片升級為半透明玻璃擬態（Glassmorphism）紙張。
    - 互動式預測計算器升級為深綠色黑板質感，搭配實木黑板框、粉筆發光字體、板擦與粉筆手繪圖示裝飾。

