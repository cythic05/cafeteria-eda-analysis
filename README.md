# 🍽️ Cafeteria Order Data: Quick Analysis & Forecast Challenge
### **Kanishka Software Private Limited — Evaluation Assignment**
**Candidate:** Aryan Fursule  
**Submission Contact:** `hr@ksoftpl.com`  

---

## 📌 Project Overview
This repository contains the complete analytical pipeline, interactive Jupyter Notebook, and executive technical report for the 24-hour evaluation challenge assigned by Kanishka Software.

The project analyzes cafeteria transaction data across **18 branches, 72 counters, and 107 categories** in FY 2024–2025:
- **5,961,005 Orders** (~6 Million transactions)
- **7,426,279 Order Detail Items** (~7.4 Million items sold)
- **₹369.6 Million (~₹37 Crores INR)** in total paid transactions
- **54,080 Registered User Profiles**

---

## 📁 Repository Deliverables

| File / Folder | Description |
| :--- | :--- |
| **`Cafeteria_Order_Analysis_and_Forecasting.ipynb`** | **Primary Deliverable:** Fully executed Jupyter Notebook with end-to-end data processing, EDA visualizations, 5 benchmarked time series models, and the 7-day ahead forecast. |
| **`REPORT.md`** | **Executive Report:** Formal technical report with KPI tables, branch diagnostics, menu matrix, forecast evaluation scorecard, and data-driven recommendations. |
| **`notebooks/Cafeteria_Order_Analysis_and_Forecasting.html`** | Standalone rendered HTML version of the notebook for instant browser viewing without needing Jupyter installed. |
| **`scripts/prepare_data.py`** | High-performance streaming data ingestion pipeline that parses the raw 11 GB SQL dump into columnar Parquet and analytical CSVs in 42 seconds. |
| **`data/orders_clean.parquet`** | Cleaned 5.96M order records stored in Snappy-compressed Apache Parquet (88 MB vs. original 11 GB SQL dump). |
| **`data/*.csv`** | Clean analytical summaries (`daily_branch_orders.csv`, `top_dishes.csv`, `hourly_traffic.csv`, `branches.csv`, `counters.csv`, etc.). |
| **`.venv/`** | Configured virtual environment containing all required scientific libraries. |

---

## 🚀 How to Run Locally

### 1. Activate the Virtual Environment
The virtual environment has already been created and populated in `.venv/`:
```bash
source .venv/bin/activate
```

### 2. Verify / Re-run Data Ingestion (Optional)
If you wish to re-generate the clean Parquet and CSV files from the raw SQL dump:
```bash
python scripts/prepare_data.py
```
*(Processes 5.96M orders + 7.42M items in ~42 seconds!)*

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook Cafeteria_Order_Analysis_and_Forecasting.ipynb
# or with jupyter lab:
jupyter lab Cafeteria_Order_Analysis_and_Forecasting.ipynb
```
Select the **`Python (.venv - Kanishka)`** kernel if prompted.

### 4. View the HTML Report in Browser
You can open `notebooks/Cafeteria_Order_Analysis_and_Forecasting.html` directly in any web browser (Chrome, Safari, Edge) to inspect all rendered interactive tables and high-resolution charts without starting a server.

---

## 📈 Key Findings & Insights Summary

1. **Volume & Revenue Champions:**
   - **Embassy Tech Village (Branch 2):** Top volume branch with **2.48M orders** and **₹14.89 Crores** revenue.
   - **Nirlon Knowledge Park (Branch 1):** Top revenue branch with **₹15.34 Crores** across **2.34M orders** (Higher AOV of ₹72.14).
2. **Channel & Payment Dominance:**
   - **82.7% Digital Ordering:** 60.1% Mobile App, 22.6% Self-Ordering Kiosks (SOK), 17.2% Cashier POS.
   - **92.5% Cashless Economy:** Paytm (40.0%), UPI (21.4%), Razorpay (9.4%), CCAvenue (8.0%), Cash (7.5%), Cards (3.9%).
3. **The "Dual Rush" Kitchen Traffic Curve:**
   - **Lunch Rush Peak (1:00 PM):** 722,134 orders.
   - **Evening Chai/Snack Peak (5:00 PM):** 719,940 orders (nearly identical in volume!).
4. **Captive Workforce Retention:**
   - **95.4% Repeat Customer Rate**; over 22,800 employees placed $\ge 50$ orders during the year.
5. **Top Menu Items:**
   - **Ginger Tea** is the undisputed volume leader (**753,840 cups sold**, generating ₹1.15 Crores).
   - **Tandoor Veg Combo** is the top revenue earner (**₹83.3 Lakhs**).

---

## 🎯 7-Day Ahead Forecast (Branch 1: Nirlon Knowledge Park)

Benchmarking across 5 model architectures on a 14-day holdout set:
- **Gradient Boosted Regressor (HistGBM):** **Champion Model** (MAE: **915.54 orders**, MAPE: **24.58%**).
- **SARIMAX (1,1,1)(1,1,1,7):** MAE: **1,039.59 orders**, MAPE: **37.02%**.
- **Meta Prophet:** MAE: **1,087.30 orders**, MAPE: **25.01%**.

### **Production 7-Day Order Forecast (April 1 – April 7, 2025):**
| Date | Day of Week | Predicted Orders | 95% Confidence Interval | Projected Revenue (INR) |
| :---: | :---: | :---: | :---: | :---: |
| **2025-04-01** | Tuesday | **8,359** | [5,616 – 11,280] | **₹603,018.26** |
| **2025-04-02** | Wednesday | **8,284** | [5,321 – 11,180] | **₹597,607.76** |
| **2025-04-03** | Thursday | **8,197** | [5,349 – 11,048] | **₹591,331.58** |
| **2025-04-04** | Friday | **7,032** | [4,213 – 9,926] | **₹507,288.48** |
| **2025-04-05** | Saturday | **855** | [0 – 3,747] | **₹61,679.70** |
| **2025-04-06** | Sunday | **145** | [0 – 2,900] | **₹10,460.30** |
| **2025-04-07** | Monday | **7,358** | [4,498 – 10,230] | **₹530,806.12** |
| **Total 7-Day**| — | **40,230** | — | **₹2,902,192.20** |

---

## ✉️ Submission Details
- **Email:** `hr@ksoftpl.com`
- **Candidate:** Aryan Fursule
- **Attached Deliverables:** GitHub Repo / Google Drive Link, [REPORT.md](REPORT.md), [Cafeteria_Order_Analysis_and_Forecasting.ipynb](Cafeteria_Order_Analysis_and_Forecasting.ipynb), Resume.
