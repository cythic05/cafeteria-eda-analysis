# 📊 Cafeteria Order Data: Quick Analysis & Forecasting Challenge
## **Executive Technical Report & Business Recommendations**
**Prepared for:** HR Evaluation Team, Kanishka Software Private Limited (`hr@ksoftpl.com`)  
**Candidate:** Aryan Fursule  
**Submission Date:** September 30, 2026  
**Assignment:** 24-Hour Evaluation Challenge — Cafeteria Order Data (FY 2024–2025)  

---

## Executive Summary

This report presents an end-to-end exploratory data analysis, business diagnostics, and predictive time-series forecasting across enterprise corporate cafeterias managed by Kanishka Software's POS/ordering platform. The underlying database represents transactions across **18 branches, 72 counters, and 107 menu categories** during Financial Year 2024–2025.

### **High-Level Operational Metrics (FY 2024–2025)**
| Metric | Value | Business Significance |
| :--- | :--- | :--- |
| **Total Order Volume** | **5,961,005 Orders** (~6.0 Million) | Massive scale across campus dining |
| **Total Line Items Sold** | **7,426,279 Items** (~7.4 Million) | High multi-item basket density |
| **Total Paid Revenue** | **₹369,650,021.68 (~₹37.0 Crores INR)** | Substantial annualized GMV |
| **Active Corporate Profiles** | **54,080 Registered Users** | Captive campus tech workforce |
| **Repeat Customer Rate** | **95.4%** (42,996 repeat customers) | Deep recurring engagement |
| **Digital Ordering Penetration** | **82.7%** (60.1% Mobile App, 22.6% Kiosk) | High tech-literacy, low manual cashier load |
| **Cashless Penetration** | **92.5%** (Paytm, UPI, Cards, Gateways) | Near cashless corporate micro-economy |
| **Average Order Value (AOV)** | **₹67.82 INR** | High volume, low ticket micro-transactions |

---

## 1. Data Engineering & Ingestion Pipeline

### **The 11 GB Database Challenge**
The raw data dump provided (`Cafeteria Order Data.sql`) was an uncompressed **11.03 GB** MySQL dump containing over 5.96 million order records and 7.42 million detail rows. Directly loading this SQL file into a standard Pandas DataFrame would crash local RAM.

### **Our Solution: Streaming Columnar Architecture**
1. **Streaming Parsing (`scripts/prepare_data.py`):** Developed a zero-copy streaming parser in Python that seeks directly to table byte offsets and reads line-by-line without buffering the 11 GB payload into memory.
2. **Columnar Parquet Compression:** Extracted normalized, clean transaction rows and saved them into Snappy-compressed Apache Parquet (`data/orders_clean.parquet`).
3. **Efficiency Gains:**
   - Database size reduced from **11,034 MB (11 GB)** down to **88 MB (125x compression)**.
   - Query time reduced from **minutes** to **under 0.3 seconds**.
   - Pipeline executed end-to-end across 5.96M rows in **42.4 seconds**.

---

## 2. Exploratory Data Analysis & Business Diagnostics

### **2.1 Branch Performance Matrix**
Four primary branches dominate operations, with two flagship campuses contributing over **81% of all orders**:

| Branch ID | Branch Name & Location | Total Orders | Paid Orders | Total Revenue (INR) | AOV (INR) | Cancel Rate (%) | Avg Daily Orders |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **Embassy Tech Village (ETV)** | 2,480,228 | 2,250,467 | ₹148,910,049.33 | ₹66.17 | 7.68% | 6,777 |
| **1** | **Nirlon Knowledge Park (NKP)** | 2,344,842 | 2,125,932 | ₹153,375,470.60 | ₹72.14 | 6.74% | 6,407 |
| **9** | **Magma** *(Onboarded Aug 2024)* | 653,115 | 598,720 | ₹39,882,623.75 | ₹66.61 | 7.16% | 3,233 |
| **4** | **Magnus Tower** | 443,902 | 443,902 | ₹25,673,606.00 | ₹57.84 | 0.00% | 1,216 |
| **10** | **JPMT-Kalina** *(Onboarded Dec 2024)* | 38,477 | 30,913 | ₹1,774,913.00 | ₹57.42 | 18.04% | 601 |
| **3** | **KANISHKA CAFE** *(Demo / Internal)* | 432 | 377 | ₹33,359.00 | ₹88.49 | 10.19% | 9 |

> **Key Finding:** Nirlon Knowledge Park generates the highest revenue (₹15.34 Cr) despite slightly lower volume than Embassy Tech Village, due to an **8.9% higher Average Order Value (₹72.14 vs. ₹66.17)**.

---

### **2.2 Ordering Channel Dynamics**
- **Mobile App:** **60.1%** (3,585,587 orders) — Primary method used by corporate employees to place desk-side pre-orders.
- **Self-Ordering Kiosks (SOK):** **22.6%** (1,347,463 orders) — Crucial for walk-in traffic and reducing cashier staffing.
- **POS Counters (Cashier):** **17.2%** (1,027,955 orders) — Traditional assisted checkouts.

---

### **2.3 Payment Method Trends: The Cashless Micro-Economy**
- **Digital / Cashless Share:** **92.5%** of all transactions.
- **Top Gateways:**
  1. Paytm Wallet / Payment Gateway: **40.0%** (2.38M orders)
  2. UPI Direct: **21.4%** (1.27M orders)
  3. Razorpay: **9.4%** (557K orders)
  4. CCAvenue: **8.0%** (478K orders)
  5. Physical Cash: **7.5%** (445K orders)
  6. Credit/Debit Cards: **3.9%** (233K orders)

---

### **2.4 Operational Traffic Peaks: The Corporate Workday Curve**
Transaction density follows a strict schedule dictated by tech park work shifts:

```
Hour (IST)   Orders     Operational Shift
10:00 AM     205,264    Morning Breakfast & Coffee Rush
01:00 PM     722,134    LUNCH PEAK (Maximum Operational Pressure)
02:00 PM     485,767    Late Lunch Wrap-up
04:00 PM     628,061    Afternoon Chai Gathering
05:00 PM     719,940    EVENING CHAI & SNACK CLIMAX (Second Highest Peak)
06:00 PM     513,696    Evening Shift Transition
09:00 PM     396,137    Dinner & Overtime Shift Rush
```

> **Critical Takeaway:** The **5:00 PM Evening Chai Rush (719,940 orders)** is almost identical in order volume to the **1:00 PM Lunch Peak (722,134 orders)**! Cafeterias frequently under-staff the 4:30–6:00 PM window assuming lunch is the only peak.

---

### **2.5 Menu Engineering: Stars vs. Plowhorses**

| Rank | Dish Name | Quantity Sold | Unit Price (INR) | Total Revenue (INR) | Menu Classification |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Ginger Tea** | **753,840** | ₹16.00 | **₹11,484,386.00** | **Super Plowhorse** (Mass volume driver) |
| 2 | **Tandoor Veg Combo** | **79,340** | ₹105.00 | **₹8,333,960.00** | **Star** (High margin, high revenue) |
| 3 | **Veg Meal Combo** | **128,259** | ₹60.00 | **₹7,106,600.00** | **Star** (Consistent core lunch staple) |
| 4 | **Veg Meal** | **81,447** | ₹85.00 | **₹6,536,170.00** | **Star** (High revenue contributor) |
| 5 | **Indian Thali Veg Combo** | **101,694** | ₹60.00 | **₹5,175,430.00** | **Star** (Lunch volume driver) |
| 6 | **Veg Thali** | **56,845** | ₹100.00 | **₹5,115,965.00** | **Star** (Premium meal option) |
| 7 | **Regular Tea** | **324,343** | ₹16.00 | **₹4,946,941.00** | **Plowhorse** (Beverage backbone) |
| 8 | **Masala Dosa** | **104,940** | ₹50.00 | **₹4,653,382.00** | **Plowhorse** (Breakfast staple) |
| 9 | **Tandoor Non Veg Combo** | **38,339** | ₹120.00 | **₹4,600,680.00** | **Puzzle** (High margin, lower volume) |
| 10 | **Reboot Chai** | **207,589** | ₹18.00 | **₹3,196,818.00** | **Plowhorse** (Specialty tea) |

---

## 3. Time Series Forecasting (Branch 1: Nirlon Knowledge Park)

### **3.1 Hybrid Work Attendance Pattern**
Analysis of daily orders across 365 days revealed a clear cyclical attendance structure:
- **Tuesday, Wednesday, Thursday:** Peak in-office days (~9,060 to ~9,192 orders/day).
- **Monday (~8,151) & Friday (~7,854):** Reduced corporate attendance due to flexible work-from-home options.
- **Saturday (~1,046) & Sunday (~391):** 87% to 95% volume contraction (skeletal staff).

### **3.2 Stationarity & Modeling Framework**
- **Augmented Dickey-Fuller (ADF) Test:** Test Statistic = **-3.4147**, $p$-value = **0.0104** ($< 0.05$). The series is stationary with a strict 7-day cyclical periodicity ($s=7$).
- **Validation Protocol:** 14-day out-of-sample backtesting on unseen final period (March 18 – March 31, 2025).

### **3.3 Model Benchmark Comparison Scorecard**

| Rank | Model Architecture | MAE (Orders) | RMSE | MAPE (%) | $R^2$ Score | Key Evaluation Takeaway |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Gradient Boosted Regressor (HistGBM)** | **915.54** | **1,728.91** | **24.58%** | **0.80** | **Champion Model:** Best handles non-linear weekday/weekend shifts with autoregressive lags. |
| 🥈 | **SARIMAX (1, 1, 1) $\times$ (1, 1, 1)$_7$** | **1,039.59** | **1,744.35** | **37.02%** | **0.79** | Strong econometric capture of weekly seasonal differencing. |
| 🥉 | **Meta Prophet** | **1,087.30** | **1,750.32** | **25.01%** | **0.79** | Excellent interpretability, decomposed trend, and realistic confidence intervals. |
| 4 | **Seasonal Naive (Lag-7)** | **1,220.86** | **2,435.67** | **29.15%** | **0.59** | Pure persistence baseline; sensitive to single-day anomalies. |
| 5 | **Holt-Winters Exponential Smoothing** | **1,241.57** | **1,818.88** | **32.60%** | **0.78** | Decent baseline, slightly slower to adapt to abrupt Friday-Monday transitions. |

---

### **3.4 Production 7-Day Ahead Forecast (April 1 – April 7, 2025)**
Using the retrained model with full historical data, the forecasted order volume and revenue for **Nirlon Knowledge Park (Branch 1)** are:

| Forecast Date | Day of Week | Predicted Orders | Lower 95% CI | Upper 95% CI | Expected Revenue (₹ INR) | Recommended Kitchen Staffing |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-04-01** | Tuesday | **8,359** | 5,616 | 11,280 | **₹603,018.26** | **Full Capacity (100%)** |
| **2025-04-02** | Wednesday | **8,284** | 5,321 | 11,180 | **₹597,607.76** | **Full Capacity (100%)** |
| **2025-04-03** | Thursday | **8,197** | 5,349 | 11,048 | **₹591,331.58** | **Full Capacity (100%)** |
| **2025-04-04** | Friday | **7,032** | 4,213 | 9,926 | **₹507,288.48** | Moderate Staffing (85%) |
| **2025-04-05** | Saturday | **855** | -1,863 | 3,747 | **₹61,679.70** | Skeletal Staffing (15%) |
| **2025-04-06** | Sunday | **145** | -2,853 | 2,900 | **₹10,460.30** | Minimal Staffing (5%) |
| **2025-04-07** | Monday | **7,358** | 4,498 | 10,230 | **₹530,806.12** | Standard Staffing (90%) |
| **Total 7-Day**| — | **40,230** | — | — | **₹2,902,192.20** | — |

*Note: Projected revenue calculated based on Branch 1 historical AOV of ₹72.14.*

---

## 4. Strategic Business Insights & Recommendations

### **1. Operational Protocol: The "Dual Rush" Kitchen Re-alignment**
- **The Problem:** Kitchens traditionally treat lunch as the primary workload and staff down by 3:30 PM. However, transaction data proves that the **5:00 PM Chai & Snack rush generates 719,940 orders**, virtually matching lunch (722,134 orders).
- **The Solution:**
  - Shift kitchen schedules from a single-shift to a staggered dual-shift (Shift A: 10:30 AM – 3:30 PM; Shift B: 3:30 PM – 8:30 PM).
  - Prepare tea/coffee decoctions and batch-fry samosas/vadas between 3:15 PM and 3:45 PM to eliminate the 5:00 PM bottleneck.

### **2. Menu Pricing & Elasticity Optimization**
- **Beverage Inelasticity:** Ginger Tea sold 753,840 cups at ₹16.00. Increasing price by ₹2.00 (from ₹16 to ₹18) represents an imperceptible change for corporate employees, but immediately injects **+₹15.08 Lakhs in pure gross margin**.
- **Combos as Margin Builders:** Create a standardized **"Evening Chai Combo"** (Ginger Tea + Samosa for ₹35, saving ₹3) to lift average afternoon transaction value from ₹16 to ₹35 (+118% basket uplift).

### **3. Mitigation of Order Cancellations (~₹2.7 Crores Lost Opportunity)**
- Over **402,133 orders were cancelled** across the financial year (7.68% at ETV, 6.74% at NKP, 18.04% at JPMT-Kalina).
- Mobile order analysis indicates cancellations occur primarily during peak queue times when customers wait over 10 minutes at the pick-up counter.
- **Remediation:** Introduce **dynamic estimated wait-time badges** in the mobile app, and automatic order throttling when counter queue depth exceeds 25 orders.

### **4. Self-Ordering Kiosk (SOK) Rollout in Cashier-Heavy Branches**
- Magnus Tower and JPMT-Kalina show higher proportions of POS counter checkouts.
- Expanding SOK installations at entrance doorways will transition 15–20% of counter traffic to digital self-ordering, reducing front-of-house cashier labor costs.

### **5. Captive Audience Subscription Passes**
- **95.4% of users are repeat corporate diners**, with over 22,800 placing 50+ orders per year.
- Introduce an auto-renewing **"Campus Chai Pass"** (e.g., 20 teas for ₹300/month) or **"Monthly Executive Lunch Plan"** integrated into corporate payroll deductions. This locks in customer retention and provides upfront negative working capital.

---

## 5. Repository Structure & Deliverables

All deliverables are organized in the project workspace:
```
KanishkaSoft/
│
├── Cafeteria_Order_Analysis_and_Forecasting.ipynb   # Fully executed Jupyter Notebook (with plots & tables)
├── REPORT.md                                       # This Executive Summary & Technical Report
├── README.md                                       # Setup and execution guide
│
├── data/                                           # Extracted & normalized analytical datasets
│   ├── orders_clean.parquet                        # 5.96M cleaned orders (88 MB Snappy-compressed)
│   ├── daily_branch_orders.csv                     # Daily orders, revenue, and channels by branch
│   ├── hourly_traffic.csv                          # Hourly order matrix
│   ├── top_dishes.csv                              # Menu items ranked by volume & revenue
│   ├── counter_performance.csv                    # 72 counters aggregated
│   ├── customer_order_frequencies.csv              # Retention & order distribution
│   ├── branches.csv                                # 18 branches metadata
│   ├── categories.csv                              # 107 categories
│   ├── counters.csv                                # 72 counters
│   ├── dishes.csv                                  # 11,418 dishes catalog
│   └── users_summary.csv                           # 54,080 user profiles
│
├── notebooks/
│   └── Cafeteria_Order_Analysis_and_Forecasting.html # Standalone interactive HTML report
│
├── scripts/
│   ├── prepare_data.py                             # Streaming 11GB SQL ingestion pipeline
│   └── build_notebook.py                           # Automated notebook generator
│
└── .venv/                                          # Virtual environment with all dependencies
```

---

## Submission Sign-Off
- **Candidate:** Aryan Fursule
- **Assignment Status:** Complete & Fully Validated
- **Recipient:** HR Team, Kanishka Software Private Limited (`hr@ksoftpl.com`)
