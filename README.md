# 📺 Streaming MMM — Media Mix Model for a Brazilian Streaming Platform

> **End-to-end Marketing Mix Modeling pipeline** with channel contribution decomposition, saturation curves, carryover effects, and budget optimization — built on a synthetic dataset inspired by a real Brazilian streaming platform with 1.2M+ MAU.

---

## Overview

This project demonstrates a production-grade MMM workflow applied to a subscription streaming business. The model quantifies the marginal contribution of each media channel to new subscriber acquisition, controls for organic and seasonal baselines, and outputs an optimized budget allocation that maximizes conversions at flat spend.

**Why this matters for streaming businesses:** Unlike e-commerce, streaming subscription growth is driven by delayed conversion cycles, high seasonality (sports calendars, content releases), and heavy dependence on performance channels with diminishing returns — all of which standard linear attribution fails to capture.

---

## Key Features

- **Channel Contribution Decomposition** — Isolates the incremental effect of each paid channel from organic baseline and seasonality
- **Hill Saturation Curves** — Models diminishing returns per channel using the Hill function (α, γ parameters)
- **Geometric Adstock / Carryover** — Captures lagged media effects with per-channel decay rates
- **Budget Optimizer** — Constrained optimization (SciPy) that maximizes predicted conversions subject to total spend cap
- **Full-Funnel Dashboard** — Channel-level ROAS, contribution %, and budget reallocation recommendations

---

## Model Architecture

```
Raw Spend Data
      │
      ▼
Adstock Transformation (geometric decay per channel)
      │
      ▼
Hill Saturation (diminishing returns)
      │
      ▼
OLS / Ridge Regression (contribution coefficients)
      │
      ├──► Decomposition: Paid | Organic | Seasonality | Trend
      │
      └──► Budget Optimizer (SciPy minimize / SLSQP)
                │
                └──► Recommended Allocation + Expected Lift
```

---

## Dataset

The dataset is **fully synthetic**, generated to reflect realistic behavioral patterns of a Brazilian SVOD platform offering tiered subscription plans:

| Plan | Price | Churn Rate | Primary Channel |
|---|---|---|---|
| Família | R$ 80/mo | ~4% | TV + Meta |
| Premiere (Football) | R$ 60/mo | ~7% | Meta + YouTube |
| Telecine (Movies) | R$ 38/mo | ~9% | Google + TikTok |
| HBO Max bundle | R$ 35/mo | ~11% | TikTok + Influencer |

Channels modeled: **Meta Ads, Google Ads, YouTube, TikTok, TV/OOH, Email, Organic/SEO**

Synthetic data generated with calibrated CAC, LTV, and conversion rates per plan/channel combination, preserving realistic correlations and seasonality patterns.

---

## Results Snapshot

| Channel | Baseline Spend Share | Optimized Share | Expected Lift |
|---|---|---|---|
| TikTok | 8% | 19% | +5.2% conversions |
| Meta | 35% | 31% | — |
| Google | 28% | 18% | — |
| YouTube | 14% | 17% | +1.1% conversions |
| TV/OOH | 15% | 15% | — |

> **Finding:** TikTok was the most under-invested channel relative to its marginal return. Reallocating ~11% of budget from Google to TikTok and YouTube is projected to deliver ~+5% incremental conversions at flat total spend.

---

## Project Structure

```
streaming-mmm/
│
├── data/
│   ├── generate_synthetic_data.py     # Synthetic dataset generator
│   └── streaming_mmm_dataset.csv      # Generated dataset
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_adstock_saturation.ipynb    # Transformation pipeline
│   ├── 03_mmm_model.ipynb             # Model fitting & decomposition
│   └── 04_budget_optimizer.ipynb      # Optimization & recommendations
│
├── src/
│   ├── transformations.py             # Adstock & Hill saturation functions
│   ├── model.py                       # MMM regression wrapper
│   └── optimizer.py                   # Budget optimization engine
│
├── dashboard/
│   └── mmm_dashboard.pbix             # Power BI / Looker Studio dashboard
│
├── outputs/
│   ├── contribution_decomposition.png
│   ├── saturation_curves.png
│   └── budget_recommendation.png
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Generation | Python, NumPy, pandas |
| Modeling | scikit-learn, statsmodels, SciPy |
| Visualization | matplotlib, seaborn, Plotly |
| Optimization | SciPy (`minimize`, SLSQP) |
| Dashboard | Power BI / Looker Studio |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/streaming-mmm.git
cd streaming-mmm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset
python data/generate_synthetic_data.py

# 4. Run notebooks in order
jupyter notebook notebooks/
```

---

## Methodology Notes

**Adstock (Geometric Decay)**
```
adstock_t = spend_t + λ × adstock_(t-1)
```
Decay rates (λ) are set per channel based on expected carryover windows: TV ~0.7, Digital ~0.3–0.5.

**Hill Saturation Function**
```
saturation(x) = x^α / (x^α + γ^α)
```
Parameters estimated via grid search / bounded optimization per channel.

**Identified Limitations**
- No Bayesian priors (Robyn / PyMC-Marketing); priors could improve stability with limited data
- Geo-level or holdout-based incrementality not yet implemented (planned)
- TV and OOH are modeled as a single channel due to data constraints

---

## Roadmap

- [ ] Bayesian MMM with PyMC-Marketing (prior regularization)
- [ ] Geo-experiment / holdout layer for incrementality validation
- [ ] Multi-period budget optimizer (rolling quarterly planning)
- [ ] Automated sensitivity analysis on saturation parameters

---

## Context

This project is part of a **Marketing Science portfolio** built to demonstrate end-to-end MMM capability in a subscription business context. The streaming platform structure (tiered plans, seasonality, multi-channel mix) was chosen to reflect realistic complexity encountered in Growth and Marketing Science roles.

---

## Author

**Bruno** — Technical Architect & Decision Science Lead  
Focused on Growth Analytics, Marketing Mix Modeling, and causal inference for subscription businesses.

[LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)

---

*Synthetic data. All figures are illustrative and do not represent any real company's performance.*

