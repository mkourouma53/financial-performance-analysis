# Cross-Sector Financial Performance Analysis
27 Companies | 9 Sectors | 2006–2024 | Created by Moustapha Kourouma

---

## Overview

This project analyzes 18 years of financial data across 27 publicly traded companies to understand why certain business models consistently generate returns regardless of the macro environment, and what that means for thinking about which companies are best positioned to survive the next economic downturn.

The pipeline pulls raw annual filings directly from SEC EDGAR, derives all financial ratios from scratch in SQL, and produces 11 analytical charts covering growth, margins, leverage, and crisis performance across six distinct macro regimes from the 2008 Financial Crisis through 2024.

---

## Core Finding

Asset-light, high-margin, recurring-revenue business models outperform and survive downturns better than any other model type across every macro regime in this dataset.

Software and Healthcare companies maintain the highest and most stable margins across 18 years (Chart 2). They show the smallest drawdowns in both 2008 and COVID (Charts 6 and 10). They recover fastest when they do fall (Chart 6). Their sector-average margins are the least volatile across every macro regime (Chart 5). Their 20-year operating margin trends show expansion or stability while other sectors compress (Chart 8). The pattern holds across both credit shocks and demand shocks, which is what makes it structural rather than cyclical.

---

## What This Means for the Next Downturn

The cross-crisis analysis (Chart 10) compares 2008 versus COVID across 14 companies with pre-crisis data. The findings suggest resilience is business-model-specific, not cycle-specific.

In 2008, a credit and solvency shock, banking collapsed. Citigroup lost 58 percentage points of net margin in a single year. Healthcare barely moved. Software recovered in one year. The vulnerability was leverage and funding model risk.

In COVID, a demand shock, energy collapsed. XOM and CVX went deeply negative as oil demand evaporated. Healthcare was again resilient. Software accelerated. The vulnerability was revenue concentration in physical activity.

The companies resilient to both, AAPL, AMZN, MSFT, and GOOG, share a common structural profile: gross margins above 40%, low debt relative to assets, high free cash flow conversion, and revenue that is either recurring or demand-inelastic. These are not coincidental. They are the defining characteristics of asset-light, high-margin business models that generate cash in every environment.

If the next downturn is a credit shock (rising rates, tightening credit), the 2008 playbook applies. Avoid high-leverage companies and favor cash-generative businesses with low debt/equity. If it is a demand shock (recession, consumption decline), the COVID playbook applies. Favor companies with inelastic demand and digital delivery. Software and Cloud companies score well on both screens, which is what makes them structurally different from every other sector in this dataset.

---

## Project Structure

```
financial_analysis/
├── 01_ingest.py          # Pulls raw 10-K data from SEC EDGAR XBRL API
├── 02_sql_normalize.py   # Loads into SQLite, derives all financial ratios
├── 03_analyze.py         # Runs 10 FP&A analyses, outputs CSVs
├── 04_visualize.py       # Generates 11 charts from analysis outputs
├── requirements.txt      # Python dependencies — run pip3 install -r requirements.txt before starting
├── data/
│   ├── raw/              # One CSV per company and combined file
│   └── processed/        # SQLite database and analysis CSVs
└── visualizations/       # Chart outputs (.png)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | Python, requests |
| Storage | SQLite, sqlite3 |
| Normalization | SQL, pandas, numpy |
| Analysis | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## Company Universe

| Sector | Tickers |
|---|---|
| Software / Cloud | MSFT, GOOG, CRM |
| Semiconductors | NVDA, INTC, AMD |
| Consumer Hardware | AAPL, HPQ, DELL |
| Banking | JPM, BAC, C |
| Healthcare | JNJ, PFE, UNH |
| Energy | XOM, CVX, SLB |
| Consumer Retail | WMT, COST, TGT |
| Ecommerce / Logistics | AMZN, FDX, UPS |
| Automotive | TSLA, F, GM |

Three companies per sector separates sector dynamics from company-specific execution. One company per sector makes it impossible to distinguish whether JPM recovered faster than BAC because of market tailwinds or management decisions.

---

## Macro Regimes

| Period | Regime |
|---|---|
| 2006–2007 | Pre-crisis leverage buildup |
| 2008–2009 | Financial Crisis, credit and solvency shock |
| 2010–2013 | Post-crisis recovery, divergence by sector |
| 2014–2019 | Low rate bull market |
| 2020–2021 | COVID, acute demand shock and rebound |
| 2022–2024 | Inflation and rate hikes, persistent margin compression |

The three stress regimes cover different dimensions of business model risk. 2008 tests funding model and leverage risk. COVID tests demand elasticity and revenue durability. 2022 tests pricing power and cost structure under sustained input cost pressure. A business model that holds margins across all three is structurally sound. The charts show that very few do.

---

## Data Source

All data is pulled directly from the SEC EDGAR XBRL API at data.sec.gov/api/xbrl/companyfacts. This source was chosen because it allows pulls going back to 2006 for most large-cap companies, covering the full arc from the 2008 Financial Crisis through 2024. All data comes directly from official 10-K annual filings with no third-party aggregation and no API key required.

Coverage spans 2006 to 2024 for most companies, with some variation by company depending on when they began structured XBRL filing.

---

## Data Limitations

**Structural gaps by sector.** Banks do not report gross profit or capex in the standard format. Their income model is interest rate spread, not product margin. Energy companies do not isolate a gross profit line. Logistics companies report costs directly without a COGS line. These are not data errors. They reflect genuine reporting differences. Operating margin is used as the primary profitability metric for these sectors.

**Pre-2009 coverage.** Structured XBRL filing was voluntary before 2009. Most companies are missing one or two fields for 2006 to 2008. These gaps are treated as missing values and cannot be filled from any free public source.

**UNH 2007–2017.** UnitedHealth's XBRL filings for this period report pharmacy benefits segment revenue only, not consolidated company revenue. This produces mathematically impossible net margins above 100%. These years are excluded from all ratio calculations. Clean consolidated data begins in 2018.

**Crisis recovery scope.** The crisis recovery and recovery quality analyses cover 14 companies with valid 2007 baseline data. Ford is excluded because its 2007 net margin was negative, making a recovery threshold undefined. Bank of America and Schlumberger did not return to 90% of their 2007 margin level within the observation window. This reflects the severity of their respective exposures, not a permanent failure.

---

## Chart Insights

**Chart 1: Revenue CAGR**
TSLA (42.3%) and NVDA (30.9%) lead 10-year growth but both are expansion-phase companies growing from small bases. The more analytically meaningful observation is MSFT at 10.9%, consistent compounding from an already-large revenue base. HPQ at -3.1%, XOM at -1.2%, and CVX at -0.9% show structural revenue decline reflecting hardware commoditization and energy sector headwinds over this period.

**Chart 2: Margin Stability**
CRM leads gross margin at 109%. SaaS gross margins can exceed 100% because cost of revenue is near zero for software delivery. The more important signal is stability across 18 years. JNJ and PFE maintain high gross margins with low volatility, indicating genuine pricing power. TSLA's average operating margin of -30% reflects years of pre-profitability investment. The company only turned sustainably profitable in 2020.

**Chart 3: Operating Leverage**
No company in this dataset exceeds 1.0 on average operating leverage across the full period, meaning no company consistently expands margins faster than revenue on a cycle-averaged basis. This is expected because crises compress margins faster than revenue, pulling the average down. INTC at 0.89 and NVDA at 0.45 are the highest, reflecting the semiconductor cost structure where fixed R&D and fabrication costs create significant scale benefits when revenue grows.

**Chart 4: Capital Efficiency**
The gap between ROE and ROA measures exactly how much of a company's return is leverage-manufactured versus operationally earned. The relationship is ROE − ROA = ROA × (Debt/Equity). A wide gap means debt is doing the work. NVDA, MSFT, and GOOG show both high ROE and high ROA, meaning their returns are genuinely earned. DELL shows high ROE with low ROA, meaning leverage is inflating the number. AAPL's 125% ROE reflects aggressive share buybacks shrinking book equity to near zero — their 22.7% ROA is the honest figure. HPQ's -210% ROE reflects negative book equity from the same dynamic taken further. The gap chart on the right panel makes this comparison direct across all 27 companies.

**Chart 5: Sector Timeline 2006–2024**
Automotive is the most volatile sector across the full period. Banking collapsed in 2008 to 2009 and recovered slowly. Healthcare showed the most consistent margins throughout, never going negative across any macro regime in this dataset. Energy went deeply negative in 2020 as oil demand collapsed, then spiked positive in 2021 to 2022 as commodity prices surged, making it the most cyclical sector in the analysis.

**Chart 6: Crisis Recovery**
Citigroup suffered the largest drawdown in the dataset at 58 percentage points from 2007 to trough. JPM also fell sharply but recovered in 2 years. Its more conservative balance sheet pre-crisis gave it a faster recovery path than BAC. AAPL, AMZN, and HPQ showed drawdowns under 2 percentage points. Healthcare showed the strongest defensive characteristics with both JNJ and PFE falling less than 1 percentage point and recovering within one year.

**Chart 7: Macro Stress**
AMD gained 20 percentage points of net margin during COVID, a demand surge as remote work drove PC and data center chip demand. TSLA gained significantly in both COVID and inflation periods, reflecting its transition from loss-making to profitable during this window. XOM lost the most during COVID and gained the most during the inflation period, pure commodity price exposure working in both directions. CRM and F were most compressed by inflation.

**Chart 8: 20-Year Operating Margin Trends**
The Software/Cloud panel shows CRM's operating margin starting near 0% in 2007 and expanding to 14% by 2024 as it scaled, making SaaS operating leverage visible over time. The Banking panel shows the 2008 collapse and slow recovery. The Semiconductor panel shows NVDA's dramatic margin expansion from 2020 onward as AI chip demand drove revenue far faster than cost growth, which is the clearest example of operating leverage in the dataset.

**Chart 9: FP&A Benchmark**
NVDA's 2024 snapshot is the standout: 72.7% gross margin, 54.1% operating margin, 48.8% net margin, 69.2% ROE, 45.3% ROA. INTC's 2024 numbers are the opposite: -22% operating margin, -35% net margin, -18.9% ROE, reflecting the cost of its foundry transition. The contrast between NVDA and INTC in the same sector and same year illustrates how execution within the same business model type produces completely different outcomes.

**Chart 10: Cross-Crisis Comparison**
The key finding is that 2008 and COVID hit different companies. C, BAC, and JPM were most exposed to 2008 due to their credit model. PFE and SLB were most exposed to COVID due to demand and commodity exposure. AAPL, AMZN, and MSFT showed minimal drawdown in both, the clearest evidence of structural resilience across fundamentally different crisis types. Companies showing 7 on the recovery axis did not return to pre-crisis margins within the observation window.

**Chart 11: Recovery Quality**
JPM is the only revenue-led recovery in the dataset, meaning JPMorgan grew its way back through revenue expansion rather than cost cutting. Most companies show balanced recovery with both revenue and margin contributing. BAC and SLB show no recovery within the observation window. The net margin trajectory panel shows PFE's unusual 2013 spike to 42%, driven by patent cliff management and asset sales rather than sustainable operating improvement.

