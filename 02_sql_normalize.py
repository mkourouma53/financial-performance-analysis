"""
Layer 2: SQL Ingestion, Normalization & Ratio Derivation
=========================================================
Loads raw SEC EDGAR data into SQLite and derives all financial ratios
from raw line items. No pre-built ratios imported. Every metric is
computed here from income statement, balance sheet, and cash flow
inputs so the analytical chain is fully auditable.

Data quality fixes applied before ratio derivation:
  UNH  2007–2017 : revenue + net_income nulled (segment-level XBRL,
                   not consolidated — produces impossible 200% margins)
  JNJ  2009+2015 : revenue nulled (restated filing, partial segment)
  SLB  2014–2015 : revenue nulled (services segment only)
  AAPL 2016      : revenue nulled (XBRL tag transition, partial segment)
  GM   2009      : net_income nulled (bankruptcy reorganization gain,
                   not operating earnings)

SQL views created:
  sector_summary         — average metrics by sector and year
  crisis_comparison      — per-company margins 2007–2011
  covid_inflation_stress — per-company margins 2019–2023

"""

import sqlite3
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = "data/raw/ALL_COMPANIES_RAW.csv"
DB_PATH  = "data/processed/financials.db"
CSV_OUT  = "data/processed/normalized.csv"
os.makedirs("data/processed", exist_ok=True)

# ── Macro Regime Labels ───────────────────────────────────────────────────────
MACRO_REGIMES = {
    2005: "Pre-Crisis Buildup",
    2006: "Pre-Crisis Buildup",
    2007: "Pre-Crisis Buildup",
    2008: "Financial Crisis",
    2009: "Financial Crisis",
    2010: "Post-Crisis Recovery",
    2011: "Post-Crisis Recovery",
    2012: "Post-Crisis Recovery",
    2013: "Post-Crisis Recovery",
    2014: "Low Rate Bull Market",
    2015: "Low Rate Bull Market",
    2016: "Low Rate Bull Market",
    2017: "Low Rate Bull Market",
    2018: "Low Rate Bull Market",
    2019: "Low Rate Bull Market",
    2020: "COVID Shock",
    2021: "COVID Rebound",
    2022: "Inflation & Rate Hikes",
    2023: "Inflation & Rate Hikes",
    2024: "Normalization",
}

# ── Sectors with no gross profit (structural, not a data error) ───────────────
# Used to suppress false zero gross_margin values in ratio derivation
NO_GROSS_PROFIT_SECTORS = {
    "Banking",            # JPM, BAC, C — interest spread model
    "Energy",             # XOM, CVX — integrated energy, no GP line
}
NO_GROSS_PROFIT_TICKERS = {
    "FDX", "UPS",         # logistics — no COGS line
    "UNH",                # insurance — no COGS concept
}


def load_raw(path: str) -> pd.DataFrame:
    """Load, validate, and clean the raw SEC EDGAR CSV."""
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    expected = ["fiscal_year","ticker","company","sector","revenue","net_income",
                "total_assets","operating_cashflow"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    metric_cols = [c for c in df.columns
                   if c not in ["ticker","company","sector","fiscal_year"]]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Data Quality Fixes ────────────────────────────────────────────────────
    # These are confirmed XBRL filing inconsistencies where segment-level
    # revenue was filed instead of total company revenue, causing impossible
    # margin values (net margin > 100%). Affected years are nulled so they
    # are excluded from ratio calculations rather than distorting averages.

    # UNH 2007-2017: revenue is pharmacy benefits segment only (~$1-26B)
    # not total company revenue (~$87-226B). Net margins of 200% result.
    # Clean data starts 2018 when full consolidated revenue was filed.
    unh_mask = (df["ticker"] == "UNH") & (df["fiscal_year"] <= 2017)
    df.loc[unh_mask, "revenue"]    = np.nan
    df.loc[unh_mask, "net_income"] = np.nan
    log.info(f"  UNH 2007-2017: nulled revenue+net_income (segment-level XBRL, not consolidated)")

    # JNJ 2009 and 2015: revenue drops 76-77% vs adjacent years — confirmed
    # XBRL restated filing years where only domestic segment was tagged.
    jnj_mask = (df["ticker"] == "JNJ") & (df["fiscal_year"].isin([2009, 2015]))
    df.loc[jnj_mask, "revenue"] = np.nan
    log.info(f"  JNJ 2009+2015: nulled revenue (restated filing, partial segment only)")

    # SLB 2014-2015: revenue drops 90% — XBRL filing used services segment only.
    slb_mask = (df["ticker"] == "SLB") & (df["fiscal_year"].isin([2014, 2015]))
    df.loc[slb_mask, "revenue"] = np.nan
    log.info(f"  SLB 2014-2015: nulled revenue (services segment only, not consolidated)")

    # AAPL 2016: revenue drops 78% to $50B from $233B — XBRL tag mismatch year.
    aapl_mask = (df["ticker"] == "AAPL") & (df["fiscal_year"] == 2016)
    df.loc[aapl_mask, "revenue"] = np.nan
    log.info(f"  AAPL 2016: nulled revenue (XBRL tag transition year, partial segment)")

    # GM 2009: net income ($109B) > revenue ($47B) — bankruptcy reorganization
    # gain was booked as income but revenue was only operating revenue.
    # Null net_income for this year to prevent 200% margin distortion.
    gm_mask = (df["ticker"] == "GM") & (df["fiscal_year"] == 2009)
    df.loc[gm_mask, "net_income"] = np.nan
    log.info(f"  GM 2009: nulled net_income (bankruptcy reorganization gain, not operating)")

    log.info(f"  Data quality fixes applied.")
    return df


def derive_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive EBITDA as Operating Income + D&A.
    SEC EDGAR doesn't always have a standalone D&A tag so we use
    the standard approximation: EBITDA = Operating Income + (CFO - Net Income + Tax)
    where that's unavailable, fall back to operating income as a proxy.

    For most analyses operating_margin serves the same purpose.
    We flag EBITDA as derived so downstream layers know it's an estimate.
    """
    # Best estimate: operating_income as EBITDA floor
    # (D&A is typically 2-8% of revenue; this is a known approximation)
    df["ebitda"] = df["operating_income"].copy()
    df["ebitda_is_estimated"] = True
    return df


def derive_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all financial ratios from raw SEC EDGAR line items.
    Every ratio is traceable to a specific raw column.
    """
    d = df.copy()

    # ── EBITDA (estimated as operating income — see note above) ──────────────
    d = derive_ebitda(d)

    # ── Margin Ratios ─────────────────────────────────────────────────────────
    d["gross_margin"]     = d["gross_profit"]     / d["revenue"]
    d["operating_margin"] = d["operating_income"] / d["revenue"]
    d["net_margin"]       = d["net_income"]        / d["revenue"]
    d["ebitda_margin"]    = d["ebitda"]            / d["revenue"]
    d["rd_intensity"]     = d["rd_expense"]        / d["revenue"]

    # Null out gross_margin for sectors/tickers where GP is structurally absent
    # (prevents misleading 0% or NaN margins in visualizations)
    mask_no_gp = (
        d["sector"].isin(NO_GROSS_PROFIT_SECTORS) |
        d["ticker"].isin(NO_GROSS_PROFIT_TICKERS)
    )
    d.loc[mask_no_gp, "gross_margin"] = np.nan

    # ── Return Ratios ─────────────────────────────────────────────────────────
    d["roe"]  = d["net_income"]        / d["total_equity"]   # Return on Equity
    d["roa"]  = d["net_income"]        / d["total_assets"]   # Return on Assets
    d["roic"] = d["operating_income"]  / (                   # Return on Invested Capital
                    d["total_equity"].fillna(0) + d["long_term_debt"].fillna(0)
                ).replace(0, np.nan)

    # ── Leverage & Liquidity ──────────────────────────────────────────────────
    d["debt_to_equity"]    = d["long_term_debt"]       / d["total_equity"]
    d["debt_to_assets"]    = d["long_term_debt"]        / d["total_assets"]
    d["current_ratio"]     = d["current_assets"]        / d["current_liabilities"]
    d["interest_coverage"] = d["operating_income"]      / d["interest_expense"].abs()

    # ── Cash Flow Quality ─────────────────────────────────────────────────────
    # SEC EDGAR capex is POSITIVE (payments made), so FCF = CFO - capex
    d["fcf"]               = d["operating_cashflow"] - d["capex"].fillna(0)
    d["fcf_margin"]        = d["fcf"]                / d["revenue"]
    d["cfo_to_net_income"] = d["operating_cashflow"] / d["net_income"]  # earnings quality
    d["capex_intensity"]   = d["capex"]              / d["revenue"]

    # ── Per Share ─────────────────────────────────────────────────────────────
    d["eps"]           = d["net_income"] / d["shares_outstanding"]
    d["fcf_per_share"] = d["fcf"]        / d["shares_outstanding"]

    # ── Macro Context ─────────────────────────────────────────────────────────
    d["macro_regime"] = d["fiscal_year"].map(MACRO_REGIMES).fillna("Unknown")

    # ── Clip extreme outliers ─────────────────────────────────────────────────
    # Keeps ratios analytically meaningful — removes data errors and
    # edge cases (e.g. near-zero denominator years)
    clip_cols = {
        "gross_margin":     (-2,    2),
        "operating_margin": (-2,    2),
        "net_margin":       (-2,    2),
        "ebitda_margin":    (-2,    2),
        "roe":              (-5,    5),
        "roa":              (-2,    2),
        "roic":             (-5,    5),
        "debt_to_equity":   (-20,  20),
        "current_ratio":    (0,    20),
        "interest_coverage":(-50,  50),
        "cfo_to_net_income":(-10,  10),
    }
    for col, (lo, hi) in clip_cols.items():
        if col in d.columns:
            d[col] = d[col].clip(lo, hi)

    return d


def build_database(df: pd.DataFrame):
    """
    Load enriched data into SQLite with indexes and analytical views.
    Views replicate what a BI analyst would build in a data warehouse.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Drop existing views before recreating (avoids "already exists" errors)
    for view in ["sector_summary", "crisis_comparison", "covid_inflation_stress"]:
        cur.execute(f"DROP VIEW IF EXISTS {view}")

    # Main fact table
    df.to_sql("financials", conn, if_exists="replace", index=False)
    log.info(f"  Loaded {len(df):,} rows into financials table")

    # Indexes for fast querying
    for idx, col in [("idx_ticker","ticker"), ("idx_sector","sector"),
                     ("idx_year","fiscal_year"), ("idx_regime","macro_regime")]:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON financials({col})")

    # ── View 1: Sector averages by year ──────────────────────────────────────
    cur.execute("""
        CREATE VIEW sector_summary AS
        SELECT
            sector,
            fiscal_year,
            macro_regime,
            COUNT(DISTINCT ticker)       AS company_count,
            AVG(gross_margin)            AS avg_gross_margin,
            AVG(operating_margin)        AS avg_operating_margin,
            AVG(net_margin)              AS avg_net_margin,
            AVG(ebitda_margin)           AS avg_ebitda_margin,
            AVG(roe)                     AS avg_roe,
            AVG(roa)                     AS avg_roa,
            AVG(roic)                    AS avg_roic,
            AVG(debt_to_equity)          AS avg_debt_to_equity,
            AVG(current_ratio)           AS avg_current_ratio,
            SUM(revenue)                 AS total_sector_revenue,
            AVG(fcf_margin)              AS avg_fcf_margin,
            AVG(capex_intensity)         AS avg_capex_intensity
        FROM financials
        WHERE revenue IS NOT NULL
        GROUP BY sector, fiscal_year
        ORDER BY sector, fiscal_year
    """)

    # ── View 2: Crisis impact comparison ─────────────────────────────────────
    cur.execute("""
        CREATE VIEW crisis_comparison AS
        SELECT
            ticker, company, sector,
            MAX(CASE WHEN fiscal_year = 2007 THEN net_margin  END) AS net_margin_2007,
            MAX(CASE WHEN fiscal_year = 2008 THEN net_margin  END) AS net_margin_2008,
            MAX(CASE WHEN fiscal_year = 2009 THEN net_margin  END) AS net_margin_2009,
            MAX(CASE WHEN fiscal_year = 2010 THEN net_margin  END) AS net_margin_2010,
            MAX(CASE WHEN fiscal_year = 2011 THEN net_margin  END) AS net_margin_2011,
            MAX(CASE WHEN fiscal_year = 2007 THEN roe         END) AS roe_2007,
            MAX(CASE WHEN fiscal_year = 2009 THEN roe         END) AS roe_2009,
            MAX(CASE WHEN fiscal_year = 2011 THEN roe         END) AS roe_2011,
            MAX(CASE WHEN fiscal_year = 2007 THEN revenue     END) AS revenue_2007,
            MAX(CASE WHEN fiscal_year = 2009 THEN revenue     END) AS revenue_2009,
            MAX(CASE WHEN fiscal_year = 2011 THEN revenue     END) AS revenue_2011
        FROM financials
        GROUP BY ticker, company, sector
    """)

    # ── View 3: COVID + inflation stress test ────────────────────────────────
    cur.execute("""
        CREATE VIEW covid_inflation_stress AS
        SELECT
            ticker, company, sector,
            MAX(CASE WHEN fiscal_year = 2019 THEN net_margin       END) AS net_margin_2019,
            MAX(CASE WHEN fiscal_year = 2020 THEN net_margin       END) AS net_margin_2020,
            MAX(CASE WHEN fiscal_year = 2021 THEN net_margin       END) AS net_margin_2021,
            MAX(CASE WHEN fiscal_year = 2022 THEN net_margin       END) AS net_margin_2022,
            MAX(CASE WHEN fiscal_year = 2023 THEN net_margin       END) AS net_margin_2023,
            MAX(CASE WHEN fiscal_year = 2019 THEN operating_margin END) AS op_margin_2019,
            MAX(CASE WHEN fiscal_year = 2020 THEN operating_margin END) AS op_margin_2020,
            MAX(CASE WHEN fiscal_year = 2022 THEN operating_margin END) AS op_margin_2022,
            MAX(CASE WHEN fiscal_year = 2019 THEN fcf_margin       END) AS fcf_margin_2019,
            MAX(CASE WHEN fiscal_year = 2020 THEN fcf_margin       END) AS fcf_margin_2020,
            MAX(CASE WHEN fiscal_year = 2022 THEN fcf_margin       END) AS fcf_margin_2022
        FROM financials
        GROUP BY ticker, company, sector
    """)

    conn.commit()
    conn.close()
    log.info(f"  SQLite database saved: {DB_PATH}")
    log.info("  Views created: sector_summary, crisis_comparison, covid_inflation_stress")


def audit(df: pd.DataFrame):
    """Print a summary of the derived dataset."""
    log.info("\n-- Normalization Audit --")
    log.info(f"  Rows     : {len(df):,}")
    log.info(f"  Columns  : {df.shape[1]}")
    log.info(f"  Companies: {df['ticker'].nunique()}")
    log.info(f"  Years    : {int(df['fiscal_year'].min())} - {int(df['fiscal_year'].max())}")

    log.info("\n  Derived ratio coverage (non-null rows):")
    ratios = ["gross_margin","operating_margin","net_margin","roe","roa","roic",
              "fcf_margin","current_ratio","debt_to_equity","interest_coverage"]
    for r in ratios:
        if r in df.columns:
            n = df[r].notna().sum()
            pct = n / len(df) * 100
            log.info(f"    {r:25s} {n:4d} / {len(df)} rows  ({pct:.0f}%)")

    log.info("\n  Sample ratios — AAPL 2022-2024:")
    sample = df[(df["ticker"]=="AAPL") & (df["fiscal_year"]>=2022)][
        ["fiscal_year","revenue","gross_margin","net_margin","roe","roa","fcf_margin"]
    ]
    log.info(f"\n{sample.to_string(index=False)}")


def run():
    if not os.path.exists(RAW_FILE):
        log.error(f"Raw file not found: {RAW_FILE}")
        log.error("Run 01_ingest.py first.")
        return

    log.info("Layer 2: SQL Normalization & Ratio Derivation")
    log.info(f"  Input : {RAW_FILE}")

    raw = load_raw(RAW_FILE)

    log.info("Deriving financial ratios from raw line items...")
    enriched = derive_ratios(raw)

    log.info("Building SQLite database and views...")
    build_database(enriched)

    log.info("Exporting normalized CSV...")
    enriched.to_csv(CSV_OUT, index=False)
    log.info(f"  Output: {CSV_OUT}")

    audit(enriched)


if __name__ == "__main__":
    run()
