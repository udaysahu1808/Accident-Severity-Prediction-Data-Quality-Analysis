# 🚦 Accident Severity Prediction & Data Quality Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-6A1B9A)
![Data Visualization](https://img.shields.io/badge/Data%20Visualization-Matplotlib-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Classification](https://img.shields.io/badge/Task-Classification-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A comprehensive end-to-end data science project that analyzes road traffic accident (RTA) data to identify key risk factors, predict accident severity using machine learning, and deliver actionable insights for road safety improvement — paired with an interactive machine learning evaluation of data quality techniques.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analysis Phases](#analysis-phases)
- [Key Insights](#key-insights)
- [Strategic Recommendations](#strategic-recommendations)
- [Tools & Technologies](#tools--technologies)
- [How to Run](#how-to-run)

---

## Project Introduction

Road Traffic Accident (RTA) datasets are widely used for accident severity prediction, traffic safety analysis, and policy-making. However, the effectiveness of any analysis or machine learning model largely depends on the quality of the underlying data. Real-world accident datasets often contain missing values, duplicate records, inconsistent entries, and noisy information, which can negatively impact analytical outcomes and predictive performance.

This project focuses on conducting a comprehensive data quality assessment of the RTA dataset by examining the impact of missing values and duplicate records on data integrity. The project investigates how duplicate counts change under different missing value handling strategies and evaluates multiple imputation techniques such as filling missing values with "Unknown", Median/Mode Imputation, and K-Nearest Neighbors (KNN) Imputation.

Additionally, the project explores whether duplicate records are true duplicates or only appear as duplicates due to the presence of missing values. By comparing various imputation methods and their influence on duplicate detection and model performance, this project provides valuable insights into data preprocessing strategies and their significance in real-world data science projects.

This study demonstrates the importance of data quality engineering as a critical step before performing exploratory data analysis (EDA) and machine learning.

## 📖 Problem Statement

Real-world Road Traffic Accident datasets frequently suffer from missing values and duplicate records, which can distort statistical analysis, bias machine learning models, and reduce the reliability of insights derived from the data.

A common challenge faced by data analysts is determining:
- Whether duplicate records are genuine duplicates or artifacts created due to missing values.
- How different missing value treatment methods affect duplicate detection.
- Which imputation strategy preserves data quality while improving model performance.

Therefore, there is a need to systematically investigate the relationship between missing values, duplicate records, and imputation techniques to identify the most effective data preprocessing approach for Road Traffic Accident datasets.

## 🎯 Objectives of the Study

**Primary Objective:**
- To perform a detailed data quality assessment of the RTA dataset by identifying missing values and duplicate records.
- To analyze the extent and distribution of missing values across different variables.
- To investigate whether duplicate records are exact duplicates or are influenced by missing values present in the dataset.
- To visualize missing values using graphical techniques such as Missing Value Matrix, Missing Value Heatmap, and Bar Plot of Missing Values.
- To compare different missing value handling strategies: filling missing values with "Unknown", Median and Mode Imputation, and K-Nearest Neighbors (KNN) Imputation.
- To examine how each imputation strategy affects duplicate record count, dataset consistency, data distribution, and model performance.
- To identify the most effective imputation technique for improving data quality and analytical accuracy.

## 📚 Scope of the Study

The scope of this project is limited to the analysis of data quality issues present in the Road Traffic Accident (RTA) dataset and does not primarily focus on accident prediction models. The study covers:

- **Data Quality Assessment** — identification of missing values, detection of duplicate records, analysis of duplicate behavior before and after imputation.
- **Missing Value Analysis** — missing value matrix, missing value heatmap, percentage of missing values, column-wise missing value comparison.
- **Imputation Techniques** — Unknown Value Imputation, Median Imputation, Mode Imputation, KNN Imputation.
- **Duplicate Analysis** — exact duplicate detection, duplicate count comparison, duplicate analysis after imputation.
- **Comparative Analysis** — data quality before vs. after imputation, duplicate count variations, impact on machine learning model accuracy.

## 📚 Significance of the Study

This study is important because it:
- Highlights the impact of poor data quality on analysis and machine learning.
- Demonstrates practical approaches to handling missing values in real-world datasets.
- Provides a comparative understanding of different imputation techniques.
- Helps data analysts select appropriate preprocessing methods before building predictive models.
- Emphasizes the importance of data quality engineering in the data science workflow.

---
## 📊 Dataset

**File:** `RTA.csv`

| Column | Description |
|--------|-------------|
| `Time` | Time of accident occurrence |
| `Day_of_week` | Day the accident occurred |
| `Age_band_of_driver` | Driver age group (Under 18 / 18–30 / 31–50 / Over 51) |
| `Sex_of_driver` | Gender of the driver |
| `Driving_experience` | Years of driving experience |
| `Type_of_vehicle` | Category of vehicle involved |
| `Owner_of_vehicle` | Ownership type of the vehicle |
| `Area_accident_occured` | Urban or rural area |
| `Lanes_or_Medians` | Road lane configuration |
| `Road_allignment` | Road alignment type |
| `Types_of_Junction` | Junction type at accident location |
| `Road_surface_type` | Type of road surface |
| `Road_surface_conditions` | Condition of road surface |
| `Light_conditions` | Lighting at time of accident |
| `Weather_conditions` | Weather at time of accident |
| `Type_of_collision` | Nature of the collision |
| `Number_of_vehicles_involved` | Total vehicles involved |
| `Number_of_casualties` | Total casualties resulting |
| `Vehicle_movement` | Movement of vehicle at time of accident |
| `Cause_of_accident` | Primary cause of the accident |
| `Accident_severity` | Target variable — Slight / Serious / Fatal |

> **Dataset Size:** 12,316 records × 32 original features (21 retained after preprocessing)

---

## 📁 Project Structure

```
accident-severity-prediction/
│
├── Accident Severity Prediction & Data Quality Analysis.ipynb            # Jupyter Notebook — EDA & ML
├── Accident Severity Prediction & Data Quality Analysis Dashboard.pbix   # Jupyter Notebook Dashboard
├── RTA.csv                                                        # Dataset
├── images                                                         # Images
├── License                                                        # LICENSE
└── README.md                                                      # Project documentation
```

---

## 🔍 Analysis Phases


## Phase 1: Data Collection & Understanding

**Libraries used:** pandas, numpy, matplotlib, seaborn, missingno, scikit-learn (KNNImputer, LabelEncoder, train_test_split, RandomForestClassifier, accuracy_score).

**Dataset:** `RTA.csv` — Road Traffic Accident data.

**Shape:** 12,316 rows × 32 columns

**Columns (32):**
`Time`, `Day_of_week`, `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Vehicle_driver_relation`, `Driving_experience`, `Type_of_vehicle`, `Owner_of_vehicle`, `Service_year_of_vehicle`, `Defect_of_vehicle`, `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`, `Road_surface_type`, `Road_surface_conditions`, `Light_conditions`, `Weather_conditions`, `Type_of_collision`, `Number_of_vehicles_involved`, `Number_of_casualties`, `Vehicle_movement`, `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Casualty_severity`, `Work_of_casuality`, `Fitness_of_casuality`, `Pedestrian_movement`, `Cause_of_accident`, `Accident_severity`

**Statistical Summary (numeric columns):**

| Statistic | Number_of_vehicles_involved | Number_of_casualties |
|---|---|---|
| count | 12316 | 12316 |
| mean | 2.0407 | 1.5481 |
| std | 0.6888 | 1.0072 |
| min | 1 | 1 |
| 25% | 2 | 1 |
| 50% | 2 | 1 |
| 75% | 2 | 2 |
| max | 7 | 8 |

---

## Phase 2: Missing Value Analysis

**Missing values per column (sorted descending):**

| Column | Missing Count |
|---|---|
| Defect_of_vehicle | 4,427 |
| Service_year_of_vehicle | 3,928 |
| Work_of_casuality | 3,198 |
| Fitness_of_casuality | 2,635 |
| Type_of_vehicle | 950 |
| Types_of_Junction | 887 |
| Driving_experience | 829 |
| Educational_level | 741 |
| Vehicle_driver_relation | 579 |
| Owner_of_vehicle | 482 |
| Lanes_or_Medians | 385 |
| Vehicle_movement | 308 |
| Area_accident_occured | 239 |
| Road_surface_type | 172 |
| Type_of_collision | 155 |
| Road_allignment | 142 |

**Visualizations produced:** Missing Value Bar Plot, Missing Value Heatmap (correlation of missingness across features), Missing Value Matrix (with sparkline for row completeness).

**Key insights:**
- Most columns have very few or no missing observations, indicating good overall data quality.
- The distribution of missing values is not uniform across variables.
- The heatmap showed little strong correlation among missing values, suggesting missingness is largely random rather than systematic.

---

## Phase 3: Duplicate Investigation

- **Exact duplicate rows:** 0 (out of 12,316 rows, 32 columns)
- **Near-duplicate check** using key columns (`Age_band_of_driver`, `Sex_of_driver`, `Driving_experience`, `Vehicle_driver_relation`, `Accident_severity`) revealed near-duplicate records caused primarily by missing values, since common imputed categories (e.g., "Unknown" or mode) can make rows appear identical after preprocessing.

**Insight:** The dataset contains no exact duplicates, but near-duplicates can be created during imputation, so a post-imputation duplicate check is recommended.

---

## Phase 4: Imputation Strategy 1 — Unknown Category

- Approach: `df.fillna("Unknown")`
- **Duplicate rows after imputation: 0**
- **Conclusion:** The dataset maintains 100% row uniqueness even after replacing missing values with "Unknown," reflecting strong data quality and a reliable preprocessing workflow.

## Phase 4 (cont.): Imputation Strategy 2 — Median + Mode

- Approach: numeric columns filled with column median; categorical columns filled with column mode.
- **Duplicate rows after imputation: 0**
- **Conclusion:** Median (numerical) + Mode (categorical) successfully handled missing values without creating duplicate records, preserving overall dataset quality and integrity.

## Phase 5: Imputation Strategy 3 — KNN Imputation

- Approach: Label-encode all categorical columns, then apply `KNNImputer(n_neighbors=5)` across the full encoded dataframe.
- **Duplicate rows after imputation: 0**
- **Conclusion:** KNN Imputation successfully filled missing values without generating any duplicate records, preserving data integrity and supporting accurate downstream analysis.

---

## Phase 6: Duplicate Comparison

| Method | Duplicates |
|---|---|
| Unknown | 0 |
| Median/Mode | 0 |
| KNN | 0 |

**Insight:** All three imputation techniques resulted in zero duplicate rows, demonstrating that the dataset has a robust structure where filling missing values does not adversely affect data integrity, regardless of the method chosen.

---

## Phase 7: Model Comparison (Prediction Accuracy)

A Random Forest Classifier was used to compare prediction accuracy of `Accident_severity` across the three imputation strategies.

| Method | Accuracy |
|---|---|
| Unknown | 74% |
| Median/Mode | 77% |
| KNN | 80% |

**Insight:** KNN Imputation achieved the highest accuracy, indicating it more effectively captures underlying data patterns than the simpler Unknown-category or Median/Mode approaches.

---

## Phase 8: Key Findings

| Method | Duplicates | Accuracy |
|---|---|---|
| Unknown | 0 | 74% |
| Median/Mode | 0 | 77% |
| KNN | 0 | 80% |

- KNN Imputation delivered the best overall performance — 80% accuracy with 0 duplicate rows.
- Median/Mode Imputation achieved 77% accuracy, also with 0 duplicates.
- The Unknown Category method produced the lowest accuracy (74%), though it too maintained 0 duplicates.
- All three imputation techniques resulted in zero duplicate records, so none compromised data uniqueness.
- Model accuracy varied by method even though duplicate prevention remained consistent.
- KNN Imputation improved accuracy by 6 points over Unknown and 3 points over Median/Mode.
- **Conclusion:** KNN Imputation is the most suitable missing-value handling technique for this dataset.

---

## Phase 9: Business Insights

1. **Data Quality Directly Impacts Prediction Accuracy** — high-quality, well-preprocessed data significantly improves model performance and reliability.
2. **Missing Value Treatment Influences Model Performance** — different imputation techniques produce varying accuracies.
3. **KNN Imputation Provides Superior Predictive Results** — preserves underlying data patterns better than traditional methods.
4. **Maintaining Data Uniqueness Enhances Model Reliability** — no duplicates means unbiased learning and better generalization.
5. **Early Data Quality Assessment Reduces Analytical Risks** — catching missing values/duplicates early leads to more dependable outcomes.
6. **Reliable Predictions Support Better Decision-Making** — assists authorities in emergency response and resource allocation.
7. **Data Preprocessing is a Critical Business Process** — essential for robust ML models and maximizing data value.
8. **Predictive Analytics Enables Proactive Safety Measures** — identifies high-risk scenarios in advance.
9. **Improved Prediction Accuracy Reduces Operational Costs** — enables efficient deployment of emergency/healthcare resources.
10. **Combining Data Quality Analysis with ML Creates Greater Business Value** — enhances model effectiveness and road-safety insights.

---

## Phase 10: Strategic Recommendations

1. **Implement Robust Data Quality Management** — continuously monitor missing values, inconsistencies, and duplicates before predictive analysis.
2. **Adopt Advanced Missing Value Imputation Techniques** — prefer KNN where appropriate to improve accuracy while preserving integrity.
3. **Standardize Data Collection Processes** — reduce inconsistencies and improve completeness of accident records.
4. **Integrate Predictive Models into Decision-Making Systems** — connect with traffic management and emergency response systems.
5. **Conduct Periodic Model Evaluation** — regularly retrain/update models with new data.
6. **Strengthen Data Governance Practices** — establish validation, storage, security, and QA policies.
7. **Promote Data-Driven Road Safety Strategies** — use predictive insights to design targeted safety measures.

---

## Phase 11: Executive Summary, Future Scope & Conclusion

### Executive Summary
This project focuses on Accident Severity Prediction and Data Quality Analysis using exploratory data analysis and machine learning techniques. The study emphasizes the importance of data preprocessing — particularly missing value handling, duplicate detection, and data quality assessment — for improving predictive performance. Three imputation techniques (Unknown Category, Median/Mode, and KNN) were compared. All three maintained data uniqueness with zero duplicate records, and KNN Imputation achieved the highest prediction accuracy, confirming that data quality and preprocessing strategy are crucial to building reliable accident severity prediction models.

### 💡 Future Scope
1. **Incorporate Additional Machine Learning Algorithms** — compare XGBoost, LightGBM, and Deep Learning models.
2. **Develop Real-Time Prediction Systems** — integrate models into real-time traffic monitoring.
3. **Expand the Dataset** — use larger, more diverse accident datasets to improve generalization.
4. **Utilize Geospatial and IoT Data** — incorporate GPS, sensor, and geospatial information.
5. **Build Interactive Decision Support Systems** — integrate with dashboards for policymakers and emergency response.
6. **Explore Ensemble Learning Techniques** — Random Forest, Gradient Boosting, Stacking.
7. **Incorporate Explainable AI (XAI)** — SHAP/LIME for interpretability and stakeholder trust.
8. **Develop Mobile and Web-Based Applications** — deploy the predictive model for real-time access by authorities.

### 📌 Conclusion
The study successfully analyzed the impact of data quality on accident severity prediction and demonstrated the importance of effective preprocessing techniques. Missing values were handled using multiple imputation strategies, and all methods preserved data uniqueness without generating duplicate records. KNN Imputation achieved the highest model accuracy, indicating its effectiveness in capturing underlying data relationships. Overall, the project demonstrates that combining data quality analysis with predictive analytics can generate meaningful insights and support better decision-making in road safety management.

---

## 🛠️ Technologies
| Tool | Purpose |
|------|---------|
| Python | Programming |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Seaborn | Statistical Graphics |
| Missingno | Missing-value Visualization |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Development |


## Summary Table

| Imputation Method | Duplicates After Imputation | Model Accuracy |
|---|---|---|
| Unknown Category | 0 | 74% |
| Median (numeric) + Mode (categorical) | 0 | 77% |
| KNN (k=5) | 0 | **80% (best)** |


## 📄 License

Copyright (c) 2026 Uday Sahu

All Rights Reserved.

This project and all associated files are the exclusive property of Uday Sahu.

No part of this project may be reproduced, copied, modified, distributed, published, sublicensed, or sold in any form without prior written permission from the copyright holder.

This project is provided for viewing and portfolio demonstration purposes only.

Unauthorized use of this project is strictly prohibited.

---

## 🙋 Author

**Uday Sahu.**
*Data Science Project | Road Safety & Accident Analytics Domain*

## ▶️ Run
```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn
jupyter notebook






	
