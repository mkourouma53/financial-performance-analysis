"""
Layer 3: FP&A 
=============================
Pandas analytical layer. Reads from SQLite, outputs 10 analysis CSVs.

Analyses:
  1.  Revenue CAGR          — 5yr (2019–2024), 10yr (2014–2024), full period
  2.  Margin Stability      — mean + coefficient of variation per company;
                              operating_margin fallback for Banking/Energy/Logistics
  3.  Operating Leverage    — margin expansion rate vs revenue growth rate
  4.  Capital Efficiency    — ROE vs ROA divergence; gap = leverage contribution
  5.  Crisis Recovery       — 2007 baseline → 2008/2009 trough → recovery speed;
                              14 companies with valid pre-crisis data only
  6.  Macro Stress Tests    — COVID (2020 vs 2019) and inflation (2022 vs 2021)
  7.  FP&A Benchmark        — latest year snapshot with sector peer rankings
  8.  Sector Regime         — average metrics by macro period (SQL view)
  9.  Cross-Crisis          — 2008 vs COVID direct comparison; drawdown + recovery
                              speed side-by-side for same 14 companies
  10. Recovery Quality      — classifies HOW companies recovered from 2008:
                              revenue-led (demand returned), margin-led (cost
                              cutting), or balanced; excludes Ford (negative 2007
                              baseline) and UNH (unreliable pre-2018 revenue)

"""

import pandas as pd
import numpy as np
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "data/processed/financials.db"
ANA_DIR = "data/processed"
os.makedirs(ANA_DIR, exist_ok=True)

# Sectors/tickers with no gross profit — use operating_margin as fallback
NO_GROSS_PROFIT = {
    "sectors": {"Banking", "Energy"},
    "tickers": {"FDX", "UPS", "UNH"},
}


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM financials ORDER BY ticker, fiscal_year", conn)
    conn.close()
    log.info(f"Loaded {len(df):,} rows from database")
    return df


# ── 1. Revenue CAGR ───────────────────────────────────────────────────────────
def calc_cagr(start_val, end_val, years) -> float:
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0 or end_val <= 0 or years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def revenue_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """
    CAGR for three windows:
      5yr  : 2019-2024 (post-COVID normalization)
      10yr : 2014-2024 (low rate era through normalization)
      full : earliest available year to 2024 (varies by company)
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.set_index("fiscal_year").sort_index()

        def rev(yr):
            return grp.loc[yr, "revenue"] if yr in grp.index else np.nan

        # Full period: use earliest year with valid revenue
        rev_series = grp["revenue"].dropna()
        if not rev_series.empty:
            earliest_yr  = int(rev_series.index.min())
            earliest_rev = rev_series.iloc[0]
            full_years   = 2024 - earliest_yr
        else:
            earliest_yr = np.nan
            earliest_rev = np.nan
            full_years   = np.nan

        results.append({
            "ticker":        ticker,
            "company":       grp["company"].iloc[0],
            "sector":        grp["sector"].iloc[0],
            "cagr_5yr":      calc_cagr(rev(2019), rev(2024), 5),
            "cagr_10yr":     calc_cagr(rev(2014), rev(2024), 10),
            "cagr_full":     calc_cagr(earliest_rev, rev(2024), full_years),
            "full_period":   f"{earliest_yr}-2024" if not pd.isna(earliest_yr) else "",
            "revenue_start": earliest_rev,
            "revenue_2024":  rev(2024),
        })

    out = pd.DataFrame(results).sort_values("cagr_10yr", ascending=False)
    out.to_csv(f"{ANA_DIR}/analysis_revenue_cagr.csv", index=False)
    log.info("OK Revenue CAGR analysis complete")
    return out


# ── 2. Margin Stability ───────────────────────────────────────────────────────
def margin_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stability score = high average margin / (1 + coefficient of variation).
    Higher score = more stable, higher-margin business model.

    For companies with no gross profit (banks, energy, logistics):
    uses operating_margin as the primary margin metric.
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        sector = grp["sector"].iloc[0]

        # Determine which margin to use as primary
        no_gp = (sector in NO_GROSS_PROFIT["sectors"] or
                 ticker in NO_GROSS_PROFIT["tickers"])
        primary_margin = "operating_margin" if no_gp else "gross_margin"

        grp_clean = grp.dropna(subset=[primary_margin, "net_margin", "operating_margin"])
        if len(grp_clean) < 3:
            continue

        def cv(series):
            mean = series.mean()
            if abs(mean) < 0.001:
                return np.nan
            return series.std() / abs(mean)

        avg_primary = grp_clean[primary_margin].mean()
        cv_primary  = cv(grp_clean[primary_margin])

        row = {
            "ticker":               ticker,
            "company":              grp_clean["company"].iloc[0],
            "sector":               sector,
            "primary_margin_type":  primary_margin,
            "avg_primary_margin":   avg_primary,
            "avg_gross_margin":     grp_clean["gross_margin"].mean() if not no_gp else np.nan,
            "avg_operating_margin": grp_clean["operating_margin"].mean(),
            "avg_net_margin":       grp_clean["net_margin"].mean(),
            "avg_ebitda_margin":    grp_clean["ebitda_margin"].mean(),
            "primary_margin_cv":    cv_primary,
            "net_margin_cv":        cv(grp_clean["net_margin"]),
            "years_observed":       len(grp_clean),
            "margin_stability_score": (
                avg_primary / (1 + cv_primary)
                if cv_primary is not None and not np.isnan(cv_primary) else np.nan
            ),
        }
        results.append(row)

    out = pd.DataFrame(results).sort_values("margin_stability_score", ascending=False)
    out.to_csv(f"{ANA_DIR}/analysis_margin_stability.csv", index=False)
    log.info("OK Margin stability analysis complete")
    return out


# ── 3. Operating Leverage ─────────────────────────────────────────────────────
def operating_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Operating leverage = % change in operating margin / % change in revenue.
    > 1: margin expands faster than revenue (scalable model)
    < 1: cost-heavy, not scalable
    Note: uses operating_margin since ebitda_margin = operating_margin in this dataset.
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("fiscal_year").copy()
        grp["rev_growth"]    = grp["revenue"].pct_change()
        grp["margin_change"] = grp["operating_margin"].diff()

        # Filter years with near-zero revenue growth (unstable denominator)
        valid = grp[grp["rev_growth"].abs() > 0.02].copy()
        valid["op_lev"] = (valid["margin_change"] / valid["rev_growth"]).clip(-5, 5)

        if len(valid) < 3:
            continue

        def get_yr(yr):
            row = valid[valid["fiscal_year"] == yr]
            return float(row["op_lev"].values[0]) if len(row) > 0 else np.nan

        results.append({
            "ticker":             ticker,
            "company":            grp["company"].iloc[0],
            "sector":             grp["sector"].iloc[0],
            "avg_op_leverage":    valid["op_lev"].mean(),
            "median_op_leverage": valid["op_lev"].median(),
            "crisis_op_lev_2008": get_yr(2008),
            "covid_op_lev_2020":  get_yr(2020),
            "inflation_op_lev_2022": get_yr(2022),
        })

    out = pd.DataFrame(results).sort_values("avg_op_leverage", ascending=False)
    out.to_csv(f"{ANA_DIR}/analysis_operating_leverage.csv", index=False)
    log.info("OK Operating leverage analysis complete")
    return out


# ── 4. Capital Efficiency ─────────────────────────────────────────────────────
def capital_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROE vs ROA: ROE-ROA gap = leverage contribution to returns.
    High ROE + high ROA = genuinely efficient.
    High ROE + low ROA = leverage-driven (riskier).
    Uses 2018-2024 to reflect current business model.
    """
    recent = df[df["fiscal_year"] >= 2018].copy()
    results = []

    for ticker, grp in recent.groupby("ticker"):
        grp = grp.dropna(subset=["roe", "roa"])
        if grp.empty:
            continue

        avg_roe = grp["roe"].mean()
        avg_roa = grp["roa"].mean()
        gap     = avg_roe - avg_roa

        if avg_roe > 0.15 and avg_roa > 0.08:
            label = "High Quality (High ROE + High ROA)"
        elif avg_roe > 0.15 and gap > 0.15:
            label = "Leverage-Driven (High ROE, Low ROA)"
        elif avg_roa > 0.05:
            label = "Asset Efficient (Moderate)"
        else:
            label = "Capital Intensive / Low Returns"

        results.append({
            "ticker":           ticker,
            "company":          grp["company"].iloc[0],
            "sector":           grp["sector"].iloc[0],
            "avg_roe":          avg_roe,
            "avg_roa":          avg_roa,
            "avg_roic":         grp["roic"].mean(),
            "roe_roa_gap":      gap,
            "avg_debt_equity":  grp["debt_to_equity"].mean(),
            "efficiency_label": label,
        })

    out = pd.DataFrame(results).sort_values("avg_roa", ascending=False)
    out.to_csv(f"{ANA_DIR}/analysis_capital_efficiency.csv", index=False)
    log.info("OK Capital efficiency analysis complete")
    return out


# ── 5. Crisis Impact & Recovery ───────────────────────────────────────────────
def crisis_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net margin drawdown from 2007 baseline to 2008/2009 trough.
    Recovery year = first year net margin returns to >= 90% of 2007 level.

    Only includes companies with valid 2007 net_margin data (15 of 27).
    Excluded: DELL (data starts 2011), GM (2009 — bankruptcy year),
              GOOG (2012), TSLA (2008 — pre-revenue startup).
    Using non-crisis years as a baseline would misrepresent the recovery
    story, so these companies are excluded rather than approximated.
    """
    # Only companies with genuine pre-crisis 2007 data
    has_2007 = (
        df[(df["fiscal_year"] == 2007) & df["net_margin"].notna()]["ticker"]
        .unique()
    )
    log.info(f"  Crisis recovery: {len(has_2007)} companies with 2007 baseline data")
    excluded = sorted(set(df["ticker"].unique()) - set(has_2007))
    log.info(f"  Excluded (no 2007 data): {excluded}")

    crisis_df = df[df["ticker"].isin(has_2007)].copy()
    results = []

    for ticker, grp in crisis_df.groupby("ticker"):
        grp = grp.set_index("fiscal_year").sort_index()

        def nm(yr):
            return grp.loc[yr, "net_margin"] if yr in grp.index else np.nan

        pre_crisis = nm(2007)

        # Crisis trough — worst of 2008 and 2009
        trough_candidates = [v for v in [nm(2008), nm(2009)] if not pd.isna(v)]
        crisis_low = min(trough_candidates) if trough_candidates else np.nan

        # Recovery year — first year back to 90% of 2007 level
        recovery_yr = np.nan
        if not pd.isna(pre_crisis) and pre_crisis > 0:
            for yr in range(2010, 2016):
                if yr in grp.index and not pd.isna(nm(yr)):
                    if nm(yr) >= pre_crisis * 0.9:
                        recovery_yr = yr
                        break

        drawdown = (
            (crisis_low - pre_crisis)
            if not (pd.isna(pre_crisis) or pd.isna(crisis_low))
            else np.nan
        )
        recovery_yrs = (recovery_yr - 2009) if not pd.isna(recovery_yr) else np.nan

        results.append({
            "ticker":              ticker,
            "company":             grp["company"].iloc[0],
            "sector":              grp["sector"].iloc[0],
            "net_margin_2007":     pre_crisis,
            "net_margin_2008":     nm(2008),
            "net_margin_2009":     crisis_low,
            "net_margin_2010":     nm(2010),
            "net_margin_2011":     nm(2011),
            "margin_drawdown":     drawdown,
            "recovery_year":       recovery_yr,
            "years_to_recover":    recovery_yrs,
        })

    out = pd.DataFrame(results).sort_values("margin_drawdown")
    out.to_csv(f"{ANA_DIR}/analysis_crisis_recovery.csv", index=False)
    log.info(f"OK Crisis recovery analysis complete — {len(out)} companies")
    return out


# ── 6. Macro Stress Tests ─────────────────────────────────────────────────────
def macro_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    COVID shock (2020) vs baseline (2019).
    Inflation shock (2022) vs rebound (2021).
    """
    pivot_cols = ["net_margin", "gross_margin", "operating_margin",
                  "ebitda_margin", "fcf_margin"]
    stress_yrs = [2019, 2020, 2021, 2022, 2023]
    stress = df[df["fiscal_year"].isin(stress_yrs)].copy()

    results = []
    for ticker, grp in stress.groupby("ticker"):
        grp = grp.set_index("fiscal_year")
        row = {
            "ticker":  ticker,
            "company": grp["company"].iloc[0],
            "sector":  grp["sector"].iloc[0],
        }
        for yr in stress_yrs:
            for col in pivot_cols:
                row[f"{col}_{yr}"] = (
                    grp.loc[yr, col] if yr in grp.index else np.nan
                )

        def safe_diff(a, b):
            return (a - b) if not (pd.isna(a) or pd.isna(b)) else np.nan

        row["covid_net_margin_impact"]     = safe_diff(
            row.get("net_margin_2020"), row.get("net_margin_2019"))
        row["covid_op_margin_impact"]      = safe_diff(
            row.get("operating_margin_2020"), row.get("operating_margin_2019"))
        row["inflation_net_margin_impact"] = safe_diff(
            row.get("net_margin_2022"), row.get("net_margin_2021"))
        row["inflation_op_margin_impact"]  = safe_diff(
            row.get("operating_margin_2022"), row.get("operating_margin_2021"))
        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(f"{ANA_DIR}/analysis_macro_stress.csv", index=False)
    log.info("OK Macro stress test analysis complete")
    return out


# ── 7. FP&A Benchmark Table ───────────────────────────────────────────────────
def fpa_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Latest year metrics with sector peer ranking.
    The kind of summary a Strategic Finance team presents to the CFO.
    """
    latest_yr = int(df["fiscal_year"].max())
    latest = df[df["fiscal_year"] == latest_yr].copy()

    metrics = ["gross_margin", "operating_margin", "net_margin",
               "ebitda_margin", "roe", "roa", "fcf_margin",
               "current_ratio", "debt_to_equity"]

    summary = latest[["ticker", "company", "sector", "fiscal_year"] + metrics].copy()

    for metric in metrics:
        summary[f"{metric}_sector_rank"] = summary.groupby("sector")[metric].rank(
            ascending=False, method="min", na_option="bottom"
        )

    rank_cols = [f"{m}_sector_rank" for m in metrics]
    summary["overall_sector_rank"] = summary[rank_cols].mean(axis=1)
    summary = summary.sort_values(["sector", "overall_sector_rank"])

    summary.to_csv(f"{ANA_DIR}/analysis_fpa_benchmark.csv", index=False)
    log.info("OK FP&A benchmark table complete")
    return summary


# ── 8. Sector Regime Analysis ─────────────────────────────────────────────────
def sector_regime_analysis() -> pd.DataFrame:
    """Average metrics by sector x macro regime — pulled from SQL view."""
    conn = sqlite3.connect(DB_PATH)
    out  = pd.read_sql(
        "SELECT * FROM sector_summary ORDER BY sector, fiscal_year", conn
    )
    conn.close()
    out.to_csv(f"{ANA_DIR}/analysis_sector_regimes.csv", index=False)
    log.info("OK Sector regime analysis complete")
    return out



# ── 9. 2008 vs COVID Cross-Crisis Comparison ──────────────────────────────────
def cross_crisis_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Direct side-by-side comparison of 2008 Financial Crisis vs COVID shock.
    Only includes companies with data covering both periods.

    For each crisis measures:
      - Peak-to-trough margin drawdown
      - Revenue contraction
      - Recovery speed (years back to 90% of pre-crisis level)

    This answers: which companies and sectors are structurally resilient
    across fundamentally different crisis types?
      2008 = solvency/credit shock (financial system freeze)
      COVID = demand shock (volume disappeared overnight)

    Different business models fail differently under each.
    The companies resilient to BOTH have genuinely durable models.
    """
    # 2008 crisis window: baseline 2007, shock 2008-2009, recovery 2010-2013
    # COVID crisis window: baseline 2019, shock 2020, recovery 2021-2022
    # Only companies with data in both windows
    has_2007 = set(df[(df["fiscal_year"]==2007) & df["net_margin"].notna()]["ticker"])
    has_2019 = set(df[(df["fiscal_year"]==2019) & df["net_margin"].notna()]["ticker"])
    both = sorted(has_2007 & has_2019)

    log.info(f"  Cross-crisis: {len(both)} companies in both windows")
    excluded_2008 = sorted(set(df["ticker"].unique()) - has_2007)
    excluded_covid = sorted(set(df["ticker"].unique()) - has_2019)
    if excluded_2008:
        log.info(f"  No 2007 data: {excluded_2008}")
    if excluded_covid:
        log.info(f"  No 2019 data: {excluded_covid}")

    results = []
    for ticker, grp in df[df["ticker"].isin(both)].groupby("ticker"):
        grp = grp.set_index("fiscal_year").sort_index()

        def get(col, yr):
            return grp.loc[yr, col] if yr in grp.index and not pd.isna(grp.loc[yr, col]) else np.nan

        # ── 2008 crisis metrics ───────────────────────────────────────────────
        nm_2007   = get("net_margin", 2007)
        nm_trough_2008 = min(
            [v for v in [get("net_margin",2008), get("net_margin",2009)]
             if not pd.isna(v)],
            default=np.nan
        )
        rev_2007  = get("revenue", 2007)
        rev_trough_2008 = min(
            [v for v in [get("revenue",2008), get("revenue",2009)]
             if not pd.isna(v)],
            default=np.nan
        )

        # 2008 recovery: first year margin back to 90% of 2007
        rec_2008 = np.nan
        if not pd.isna(nm_2007) and nm_2007 > 0:
            for yr in range(2010, 2016):
                val = get("net_margin", yr)
                if not pd.isna(val) and val >= nm_2007 * 0.9:
                    rec_2008 = yr
                    break

        # ── COVID metrics ─────────────────────────────────────────────────────
        nm_2019  = get("net_margin", 2019)
        nm_2020  = get("net_margin", 2020)
        rev_2019 = get("revenue", 2019)
        rev_2020 = get("revenue", 2020)

        # COVID recovery: first year margin back to 90% of 2019
        rec_covid = np.nan
        if not pd.isna(nm_2019) and nm_2019 > 0:
            for yr in range(2021, 2025):
                val = get("net_margin", yr)
                if not pd.isna(val) and val >= nm_2019 * 0.9:
                    rec_covid = yr
                    break

        # ── Derived comparisons ───────────────────────────────────────────────
        drawdown_2008  = (nm_trough_2008 - nm_2007) if not (pd.isna(nm_trough_2008) or pd.isna(nm_2007)) else np.nan
        drawdown_covid = (nm_2020 - nm_2019)         if not (pd.isna(nm_2020) or pd.isna(nm_2019)) else np.nan
        rev_drop_2008  = (rev_trough_2008 - rev_2007) / rev_2007 if not (pd.isna(rev_trough_2008) or pd.isna(rev_2007) or rev_2007==0) else np.nan
        rev_drop_covid = (rev_2020 - rev_2019) / rev_2019 if not (pd.isna(rev_2020) or pd.isna(rev_2019) or rev_2019==0) else np.nan
        yrs_rec_2008   = (rec_2008 - 2009)   if not pd.isna(rec_2008)   else np.nan
        yrs_rec_covid  = (rec_covid - 2020)  if not pd.isna(rec_covid)  else np.nan

        # Resilience label: resilient if drawdown < 5pp in absolute terms
        def resilience(drawdown):
            if pd.isna(drawdown):
                return "Insufficient data"
            if drawdown > -0.02:
                return "Resilient (< 2pp drawdown)"
            elif drawdown > -0.05:
                return "Mild impact (2-5pp)"
            elif drawdown > -0.10:
                return "Moderate impact (5-10pp)"
            else:
                return "Severe impact (> 10pp)"

        results.append({
            "ticker":              ticker,
            "company":             grp["company"].iloc[0],
            "sector":              grp["sector"].iloc[0],
            # 2008
            "nm_baseline_2007":    nm_2007,
            "nm_trough_2008":      nm_trough_2008,
            "drawdown_2008":       drawdown_2008,
            "rev_contraction_2008":rev_drop_2008,
            "recovery_year_2008":  rec_2008,
            "yrs_to_recover_2008": yrs_rec_2008,
            "resilience_2008":     resilience(drawdown_2008),
            # COVID
            "nm_baseline_2019":    nm_2019,
            "nm_trough_2020":      nm_2020,
            "drawdown_covid":      drawdown_covid,
            "rev_contraction_covid":rev_drop_covid,
            "recovery_year_covid": rec_covid,
            "yrs_to_recover_covid":yrs_rec_covid,
            "resilience_covid":    resilience(drawdown_covid),
            # Cross-crisis comparison
            "worse_crisis":        "2008" if (
                not pd.isna(drawdown_2008) and not pd.isna(drawdown_covid)
                and drawdown_2008 < drawdown_covid
            ) else "COVID" if (
                not pd.isna(drawdown_covid) and not pd.isna(drawdown_2008)
                and drawdown_covid < drawdown_2008
            ) else "Comparable",
        })

    out = pd.DataFrame(results).sort_values("drawdown_2008")
    out.to_csv(f"{ANA_DIR}/analysis_cross_crisis.csv", index=False)
    log.info(f"OK Cross-crisis comparison complete — {len(out)} companies")
    return out


# ── 10. Recovery Quality ──────────────────────────────────────────────────────
def recovery_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Examines HOW companies recovered from the 2008 crisis — not just when.

    Three recovery drivers:
      Revenue-led:  revenue grew back faster than margins recovered
                    → demand returned, volume-driven recovery
      Margin-led:   margins recovered faster than revenue
                    → cost-cutting or pricing power drove recovery
      Balanced:     both recovered at similar rates

    This separates structurally sound recoveries from cost-cutting
    recoveries that may not last — a key FP&A judgment call.
    """
    has_2007 = set(df[(df["fiscal_year"]==2007) & df["net_margin"].notna()]["ticker"])
    crisis_df = df[df["ticker"].isin(has_2007)].copy()

    results = []
    for ticker, grp in crisis_df.groupby("ticker"):
        grp = grp.set_index("fiscal_year").sort_index()

        def get(col, yr):
            return grp.loc[yr, col] if yr in grp.index and not pd.isna(grp.loc[yr, col]) else np.nan

        rev_2007 = get("revenue", 2007)
        nm_2007  = get("net_margin", 2007)

        # Find recovery year for both revenue and margin independently
        rev_rec_yr = np.nan
        nm_rec_yr  = np.nan

        for yr in range(2009, 2016):
            rev = get("revenue", yr)
            nm  = get("net_margin", yr)
            if pd.isna(rev_rec_yr) and not pd.isna(rev) and not pd.isna(rev_2007):
                if rev >= rev_2007 * 0.9:
                    rev_rec_yr = yr
            if pd.isna(nm_rec_yr) and not pd.isna(nm) and not pd.isna(nm_2007) and nm_2007 > 0:
                if nm >= nm_2007 * 0.9:
                    nm_rec_yr = yr

        # Revenue CAGR during recovery (2009-recovery year)
        rev_2009 = get("revenue", 2009)
        rev_at_rec = get("revenue", int(rev_rec_yr)) if not pd.isna(rev_rec_yr) else np.nan
        rev_cagr_recovery = np.nan
        if not any(pd.isna(x) for x in [rev_2009, rev_at_rec, rev_rec_yr]) and rev_2009 > 0:
            yrs = rev_rec_yr - 2009
            if yrs > 0:
                rev_cagr_recovery = (rev_at_rec / rev_2009) ** (1/yrs) - 1

        # Recovery driver classification
        if pd.isna(rev_rec_yr) and pd.isna(nm_rec_yr):
            driver = "Did not recover by 2015"
        elif pd.isna(rev_rec_yr):
            driver = "Margin-led (cost cutting)"
        elif pd.isna(nm_rec_yr):
            driver = "Revenue-led (volume driven)"
        elif nm_rec_yr < rev_rec_yr - 1:
            driver = "Margin-led (cost cutting)"
        elif rev_rec_yr < nm_rec_yr - 1:
            driver = "Revenue-led (volume driven)"
        else:
            driver = "Balanced recovery"

        results.append({
            "ticker":              ticker,
            "company":             grp["company"].iloc[0],
            "sector":              grp["sector"].iloc[0],
            "revenue_rec_year":    rev_rec_yr,
            "margin_rec_year":     nm_rec_yr,
            "rev_cagr_recovery":   rev_cagr_recovery,
            "recovery_driver":     driver,
            "nm_2007":             nm_2007,
            "nm_2010":             get("net_margin", 2010),
            "nm_2011":             get("net_margin", 2011),
            "nm_2013":             get("net_margin", 2013),
            "rev_2007":            rev_2007,
            "rev_2010":            get("revenue", 2010),
            "rev_2013":            get("revenue", 2013),
        })

    out = pd.DataFrame(results).sort_values("recovery_driver")
    out.to_csv(f"{ANA_DIR}/analysis_recovery_quality.csv", index=False)
    log.info(f"OK Recovery quality analysis complete — {len(out)} companies")
    return out


def run():
    df = load_data()

    log.info("\n-- Running FP&A Analyses --")
    cagr      = revenue_cagr(df)
    stability = margin_stability(df)
    op_lev    = operating_leverage(df)
    cap_eff   = capital_efficiency(df)
    crisis    = crisis_recovery(df)
    stress    = macro_stress(df)
    benchmark = fpa_benchmark_table(df)
    regimes   = sector_regime_analysis()
    cross     = cross_crisis_comparison(df)
    rec_qual  = recovery_quality(df)

    log.info("\n-- Key Findings Preview --")

    log.info("\nTop 5 Revenue CAGR (10yr):")
    log.info(cagr[["ticker","company","sector","cagr_10yr"]]
             .head().to_string(index=False))

    log.info("\nTop 5 Margin Stability:")
    log.info(stability[["ticker","company","sector",
                         "primary_margin_type","avg_primary_margin",
                         "margin_stability_score"]]
             .head().to_string(index=False))

    log.info("\nTop 5 Operating Leverage:")
    log.info(op_lev[["ticker","company","sector","avg_op_leverage"]]
             .head().to_string(index=False))

    log.info("\nCapital Efficiency Labels:")
    log.info(cap_eff[["ticker","company","efficiency_label"]]
             .to_string(index=False))

    log.info("\nCrisis Recovery — Worst Drawdowns:")
    log.info(crisis[["ticker","company","sector",
                      "margin_drawdown","years_to_recover"]]
             .head(5).to_string(index=False))

    log.info("\n2008 vs COVID — Cross-Crisis Resilience:")
    log.info(cross[["ticker","company","sector",
                    "drawdown_2008","drawdown_covid","worse_crisis"]]
             .head(8).to_string(index=False))

    log.info("\nRecovery Drivers (2008):")
    log.info(rec_qual[["ticker","company","sector","recovery_driver",
                        "rev_cagr_recovery"]]
             .to_string(index=False))

    log.info("\n-- All analyses complete --")
    log.info(f"  Outputs: {ANA_DIR}/analysis_*.csv (10 files)")


if __name__ == "__main__":
    run()
