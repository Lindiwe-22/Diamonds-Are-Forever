<div align="center">

<img src="https://img.shields.io/badge/STATUS-COMPLETE-brightgreen?style=for-the-badge&labelColor=0F1117"/>
<img src="https://img.shields.io/badge/PLATFORM-STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/PEER--REVIEWED-ACADEMIC-2C3E7A?style=for-the-badge"/>

<br/><br/>

# 💎 Diamonds Decoded
## Price Forecasting & Customer Segmentation
### Across Natural & Lab-Grown Diamonds

*A full-stack machine learning study investigating the structural disruption of the global diamond industry — quantifying price dynamics, revenue architecture, and buyer archetypes across natural and lab-grown diamond markets.*

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DASHBOARD-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://diamonds-are-forever.streamlit.app/)

</div>

---

## 📋 Table of Contents

- [Business Understanding](#-business-understanding)
- [Screenshots](#-screenshots)
- [Technologies](#-technologies)
- [Setup](#-setup)
- [Approach](#-approach)
- [Status](#-status)
- [Credits](#-credits)

---

## 💼 Business Understanding

The global diamond industry is undergoing its most significant structural disruption in decades. Lab-grown diamonds — chemically and physically identical to mined stones — have seen retail prices collapse by **73.8%** since 2020, falling from $3,410 to $892 per carat by December 2024 *(Fortune, 2025)*. Natural diamond prices simultaneously contracted **26.7%** from their May 2022 peak of $6,819 per carat *(Accio, 2025)*.

For South Africa — home to De Beers' Venetia mine and Petra Diamonds' Cullinan and Finsch operations — this is not a market correction. It is a **structural shift** with direct consequences for mining revenue, community livelihoods, and national GDP. De Beers reduced production by 22% in 2024 to 24.7 million carats, its lowest output since 1995 *(De Beers Group, 2025)*.

This study applies machine learning to answer three questions:

> **1.** What drives diamond pricing — and can it be predicted with commercial accuracy?
> **2.** Who is buying, and what are their distinct behavioural profiles?
> **3.** How does the revenue architecture differ across the natural vs lab-grown supply chain?

### Key Findings at a Glance

| Finding | Value |
|---------|-------|
| 💎 Best forecasting model | Random Forest — R²=**0.9997**, RMSE=**$136** |
| 📊 Published benchmark exceeded | Xiao (2024) XGBoost R²=0.9821 on full dataset |
| 👥 Customer archetypes identified | **5** distinct buyer profiles |
| 📉 Natural diamond price decline (2022–2024) | **−26.7%** |
| 📉 Lab-grown price decline (2020–2024) | **−73.8%** |
| 💰 Natural retailer margin | **40%** upstream revenue retained |
| 💰 Lab-grown retailer margin | **73%** absorbed by retailer — upstream hollowed out |
| 🔑 Dominant price driver | **Carat weight** (Pearson r ≈ 0.92) |

---

## 📸 Screenshots

| Price Predictor | Customer Profiler |
|---|---|

<img width="1366" height="728" alt="Price Predictor" src="https://github.com/user-attachments/assets/903f0afe-8d1b-48fd-ba64-1ad3b81aa4be" />

<img width="1366" height="728" alt="Customer Profiler" src="https://github.com/user-attachments/assets/f7729321-8954-44f1-a469-8b431d80650c" />)

| Market Intelligence | SA & Global Context |
|---|---|

<img width="1366" height="728" alt="Market Intelligence" src="https://github.com/user-attachments/assets/68b52f23-a801-4fb2-bb4c-be5eb28b3a80" />)

<img width="1366" height="728" alt="SA   Global Context" src="https://github.com/user-attachments/assets/810a4f19-01cb-4b1d-8355-4d3b4b1fdac5" />)

---

## 🛠️ Technologies

**Languages & Environment**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**Data & Feature Engineering**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

**Machine Learning & Clustering**

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn_%7C_SMOTE-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

**Deployment**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-3776AB?style=for-the-badge&logo=python&logoColor=white)

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.12, Jupyter Notebook |
| **Data & EDA** | Pandas, NumPy, Matplotlib, Seaborn |
| **ML & Clustering** | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE) |
| **Dimensionality Reduction** | PCA (Scikit-learn) |
| **Deployment** | Streamlit Cloud, Plotly, Joblib |
| **Platform** | Google Colaboratory (free tier) |

---

## ⚙️ Setup

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/your-username/diamonds-decoded.git
cd diamonds-decoded

# Open in Google Colab
# File → Upload notebook → diamonds_decoded_notebook.ipynb
# Or open directly from GitHub via Colab
```

> **Dataset:** [Diamonds Price Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) — available on Kaggle. A valid Kaggle API key (`kaggle.json`) is required for programmatic download in Phase 1.

### Run the Dashboard Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure models and data are in place
# diamonds-decoded/
# ├── app.py
# ├── requirements.txt
# ├── models/
# │   ├── rf_model.joblib
# │   ├── xgb_combined.joblib
# │   └── scaler.joblib
# └── data/
#     └── df_sample.csv

# Launch dashboard
streamlit run app.py
```

### Computational Note

> ⚠️ The notebook was developed on Google Colab free tier (12GB RAM). A **stratified 15,000-row sample** (random_state=42) is used for Phases 4 and 5. In a production environment, the full 53,940-row dataset would be used on a GPU-accelerated cloud instance.

---

## 🔍 Approach

### Phase 1 — Data Loading & Understanding
Two Kaggle datasets were loaded and audited: the core Diamonds Price Dataset (53,940 rows × 10 features) and an extended attributes dataset. Missing values, duplicates, and the GIA 4Cs framework were documented with full variable definitions.

### Phase 2 — Feature Engineering & Preprocessing
Fifteen features were engineered from verified industry benchmarks. Seven columns are explicitly flagged as `[ENGINEERED]` — each accompanied by a cited source:

| Engineered Feature | Benchmark Source |
|--------------------|-----------------|
| `diamond_type` — natural vs lab-grown | BriteCo (2025); Liori Diamonds (2026) |
| `revenue_per_carat` — price minus retailer margin | Edahn Golan (Q3 2025); Beyond4Cs (2025) |
| `value_tier` — Budget/Mid/Premium/Luxury | Bain & Company (2022) |
| `certification_body` — GIA/IGI/AGS | RAGAZZA (2025); Edahn Golan (2025) |
| `retailer_type` — Online/Brick-and-Mortar/Luxury | BriteCo (2025); Teach Jewelry (2025) |
| `days_on_market` — market velocity | Edahn Golan (2025); Rapaport (2025) |
| `cut_clarity_score` — composite quality index | GIA (2024) |

### Phase 3 — Exploratory Data Analysis
Eight publication-quality figures produced, including the **Price Collapse Narrative** (Figure 3.8) reconstructing natural and lab-grown price trajectories from 2020–2024 using verified benchmarks, and the **Revenue Architecture Gap** analysis (Figures 3.1–3.2) quantifying upstream revenue hollowing-out.

### Phase 4 — Customer Segmentation
K-Means clustering (k-means++ initialisation) with Agglomerative Clustering validation. SMOTE applied to address dataset imbalance. Optimal k selected via three convergent metrics (Elbow, Silhouette, Davies-Bouldin). Five buyer archetypes identified:

| Archetype | Profile |
|-----------|---------|
| 💍 **The Traditionalist** | High budget · Natural · GIA · Luxury retail |
| 🌱 **The Ethical Buyer** | Mid budget · Lab-grown · IGI · Online |
| 💸 **The Value Hunter** | Budget-driven · Maximises carat-for-price · Lab-grown |
| 👑 **The Luxury Collector** | Unconstrained budget · Exceptional quality · Natural |
| 💡 **The Pragmatist** | Open to both · Price-per-carat driven · Mid-tier |

### Phase 5 — Price Forecasting
Three regression models trained on an 80/20 stratified split. Separate XGBoost models for natural and lab-grown diamonds reveal structurally different feature importance profiles — a methodological contribution not present in prior work *(Xiao, 2024)*:

| Model | R² (Test) | RMSE (USD) | Notes |
|-------|-----------|------------|-------|
| Linear Regression | 0.9900 | $627 | Interpretable baseline |
| Random Forest | **0.9997** | **$136** | ✅ Best overall model |
| XGBoost | 0.9997 | $134 | Marginal RMSE advantage |
| Natural XGBoost | 0.9983 | $238 | Separate type model |
| Lab-Grown XGBoost | 0.9990 | $60 | Separate type model |

### Phase 6 — Deployment
Full Streamlit dashboard with four interactive pages: Price Predictor, Customer Profiler, Market Intelligence, and SA & Global Context.

---

## 📌 Status

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Peer Review](https://img.shields.io/badge/Peer_Review-In_Progress-F39C12?style=for-the-badge)
![Dashboard](https://img.shields.io/badge/Dashboard-Live-FF4B4B?style=for-the-badge&logo=streamlit)

The full pipeline is complete and deployed. Future iterations may include:
- Longitudinal price modelling using post-2022 time-series transaction data
- Geographic segmentation across South African, US, and Asia-Pacific markets
- ESG scoring as a predictor variable for values-driven buyer segments
- Full-dataset retraining on GPU-accelerated cloud infrastructure

---

## 🙏 Credits

**Developed by Lindiwe Songelwa — Data Scientist | Developer | Insight Creator**

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [Lindiwe S.](https://www.linkedin.com/in/lindiwe-songelwa) |
| 🌐 Portfolio | [Creative Portfolio](https://lindiwe-22.github.io/Portfolio-Website/) |
| 🏅 Credly | [Lindiwe Songelwa – Badges](https://www.credly.com/users/samnkelisiwe-lindiwe-songelwa) |
| 🚀 Live App | [Diamonds Decoded — Streamlit](https://diamonds-are-forever.streamlit.app/) |
| 📧 Email | [sl.songelwa@hotmail.co.za](mailto:sl.songelwa@hotmail.co.za) |

### Key References

> Bain & Company (2022) · BriteCo (2025) · Breiman (2001) · Chen & Guestrin (2016) ·
> Chawla et al. (2002) · De Beers Group (2025) · Edahn Golan (Q3 2025) ·
> Fortune (Jan 2025) · GIA (2024) · McKinsey & Company (2024) ·
> Rapaport (2025) · Xiao (2024) · Yulisasih et al. (2024)

---

<div align="center">

*© 2026 Lindiwe Songelwa. All rights reserved.*
*This project is submitted for academic peer review.*

</div>
