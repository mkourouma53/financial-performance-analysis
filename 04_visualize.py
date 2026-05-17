"""
Layer 4: Visualization 
===================================
11 charts using Matplotlib + Seaborn. Each answers a specific analytical question from the FP&A analyses in Layer 3.

Charts:
  1. Revenue CAGR          — 10yr ranking by company, colored by sector
  2. Margin Stability      — average margin profile heatmap, all companies
  3. Operating Leverage    — scalability scatter, avg vs median leverage
  4. Capital Efficiency    — ROE vs ROA scatter + ROE-ROA gap bar chart
  5. Sector Timeline       — net margin by sector, full period 2006–2024
  6. Crisis Recovery       — drawdown magnitude + recovery speed, 14 companies
  7. Macro Stress          — COVID vs inflation net margin impact, 27 companies
  8. 20-Year Trends        — operating margin by sector, 9 panels with macro bands
  9. FP&A Benchmark        — 2024 key metrics heatmap, all companies
  10. Cross-Crisis         — 2008 vs COVID drawdown + recovery side-by-side
  11. Recovery Quality     — recovery speed + net margin trajectory 2007–2013


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

VIZ_DIR = "visualizations"
ANA_DIR = "data/processed"
DB_PATH = "data/processed/financials.db"
os.makedirs(VIZ_DIR, exist_ok=True)

SECTOR_COLORS = {
    "Software_Cloud":      "#2563EB",
    "Semiconductors":      "#7C3AED",
    "Consumer_Hardware":   "#059669",
    "Banking":             "#DC2626",
    "Healthcare":          "#0891B2",
    "Energy":              "#D97706",
    "Consumer_Retail":     "#DB2777",
    "Ecommerce_Logistics": "#65A30D",
    "Automotive":          "#9333EA",
}

# Only highlight acute shocks (crisis + COVID) — inflation is a slow squeeze
# shown through the sector lines themselves, not a separate shaded region
MACRO_BANDS = [
    (2008, 2009, "#FEE2E2", "Financial\nCrisis"),
    (2020, 2020, "#FEF3C7", "COVID"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(fname):
    p = f"{ANA_DIR}/{fname}"
    if not os.path.exists(p):
        log.warning(f"Missing: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def add_macro_bands(ax):
    ylim = ax.get_ylim()
    for start, end, color, label in MACRO_BANDS:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.2, color=color, zorder=0)
        ax.text((start + end) / 2, ylim[1] * 0.99, label,
                ha="center", va="top", fontsize=6.5,
                color="#6B7280", style="italic")


def sector_legend(ax, loc="lower right"):
    handles = [mpatches.Patch(color=c, label=s.replace("_", " "))
               for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, title="Sector",
              title_fontsize=8, loc=loc, framealpha=0.7)


def pct(x, _): return f"{x*100:.0f}%"


# ── 01: Revenue CAGR ──────────────────────────────────────────────────────────
def chart_revenue_cagr():
    df = load("analysis_revenue_cagr.csv")
    if df.empty: return
    df = df.dropna(subset=["cagr_10yr"]).sort_values("cagr_10yr", ascending=True)
    colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df["sector"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(df["ticker"], df["cagr_10yr"], color=colors, height=0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct))
    ax.set_xlabel("10-Year Revenue CAGR (2014–2024)")
    ax.set_title("Revenue Growth: 10-Year CAGR by Company\nColored by Sector",
                 fontsize=13, fontweight="bold", pad=15)
    sector_legend(ax)
    for bar, val in zip(bars, df["cagr_10yr"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/01_revenue_cagr.png")
    plt.close()
    log.info("OK 01: Revenue CAGR")


# ── 02: Margin Heatmap ────────────────────────────────────────────────────────
def chart_margin_heatmap():
    df = load("analysis_margin_stability.csv")
    if df.empty: return
    df = df.set_index("ticker").sort_index()
    df["display_primary"] = df["avg_gross_margin"].fillna(df["avg_operating_margin"])
    cols = {
        "display_primary":      "Primary\nMargin*",
        "avg_operating_margin": "Operating\nMargin",
        "avg_net_margin":       "Net\nMargin",
        "avg_ebitda_margin":    "EBITDA\nMargin",
    }
    pivot = df[list(cols.keys())].rename(columns=cols) * 100

    fig, ax = plt.subplots(figsize=(11, 13))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Margin %"},
                annot_kws={"size": 8})
    ax.set_title(
        "Average Margin Profile by Company (2006–2024)\n"
        "*Primary = Gross Margin; Operating Margin used for Banking/Energy/Logistics",
        fontsize=11, fontweight="bold", pad=15)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/02_margin_heatmap.png")
    plt.close()
    log.info("OK 02: Margin Heatmap")


# ── 03: Operating Leverage ────────────────────────────────────────────────────
def chart_operating_leverage():
    df = load("analysis_operating_leverage.csv")
    if df.empty: return
    df = df.dropna(subset=["avg_op_leverage", "median_op_leverage"])
    colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df["sector"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(df["median_op_leverage"], df["avg_op_leverage"],
               c=colors, s=110, alpha=0.85, edgecolors="white", linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["ticker"],
                    (row["median_op_leverage"], row["avg_op_leverage"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    for val, col, ls in [(0,"gray","--"), (1,"#10B981",":")]:
        ax.axhline(val, color=col, linewidth=0.8, linestyle=ls)
        ax.axvline(val, color=col, linewidth=0.8, linestyle=ls)

    ylim = ax.get_ylim()
    ax.text(1.05, ylim[1] * 0.95, "Lev=1\n(neutral)", fontsize=7, color="#10B981")
    ax.text(0.5, 0.02,
            "Note: Averaged across full period including crises — values suppressed vs growth-only periods",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#9CA3AF", style="italic")
    ax.set_xlabel("Median Operating Leverage")
    ax.set_ylabel("Average Operating Leverage")
    ax.set_title(
        "Operating Leverage by Company (2006–2024)\n"
        "Values closer to 1: Margins expand closer to revenue growth rate (more scalable)",
        fontsize=12, fontweight="bold", pad=15)
    sector_legend(ax, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/03_operating_leverage.png")
    plt.close()
    log.info("OK 03: Operating Leverage")


# ── 04: Capital Efficiency ────────────────────────────────────────────────────
def chart_capital_efficiency():
    df = load("analysis_capital_efficiency.csv")
    if df.empty: return
    df = df.dropna(subset=["avg_roe", "avg_roa"])
    colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df["sector"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: ROE vs ROA scatter
    ax = axes[0]
    ax.scatter(df["avg_roa"], df["avg_roe"], c=colors, s=120,
               alpha=0.85, edgecolors="white", linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["ticker"], (row["avg_roa"], row["avg_roe"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(0.15, color="#6B7280", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(0.08, color="#6B7280", linewidth=0.7, linestyle="--", alpha=0.6)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.text(0.09, ylim[1]*0.88, "High Quality\n(High ROE + ROA)",
            fontsize=8, color="#059669", fontweight="bold")
    ax.text(xlim[0]+0.002, ylim[1]*0.88, "Leverage-Driven\n(High ROE, Low ROA)",
            fontsize=8, color="#DC2626")
    ax.text(0.5, 0.14,
            "ROE − ROA = ROA × (Debt/Equity)  |  Gap = leverage contribution to returns\n"
            "AAPL: high ROE from buybacks reducing equity base  |  HPQ: negative equity → negative ROE",
            transform=ax.transAxes, ha="center", fontsize=7.5,
            color="#6B7280", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct))
    ax.set_xlabel("Average Return on Assets (ROA)")
    ax.set_ylabel("Average Return on Equity (ROE)")
    ax.set_title("ROE vs ROA (2018–2024)\nGap = leverage contribution to returns",
                 fontsize=11, fontweight="bold", pad=20)

    # Right panel: ROE-ROA gap bar chart (the difference itself)
    df_gap = df.copy().sort_values("roe_roa_gap")
    gap_colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df_gap["sector"]]
    ax2 = axes[1]
    bars = ax2.barh(df_gap["ticker"], df_gap["roe_roa_gap"] * 100,
                    color=gap_colors, height=0.65)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}pp"))
    ax2.set_xlabel("ROE − ROA (percentage points)")
    ax2.set_title("ROE–ROA Gap by Company\nHigher gap = more leverage inflating returns",
                  fontsize=11, fontweight="bold")
    for bar, val in zip(bars, df_gap["roe_roa_gap"]):
        offset = 0.5 if val >= 0 else -0.5
        ha = "left" if val >= 0 else "right"
        ax2.text(val * 100 + offset, bar.get_y() + bar.get_height()/2,
                 f"{val*100:.0f}pp", va="center", ha=ha, fontsize=7)

    sector_legend(ax2)
    fig.suptitle("Capital Efficiency Matrix (2018–2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/04_capital_efficiency.png")
    plt.close()
    log.info("OK 04: Capital Efficiency")


# ── 05: Crisis Timeline — FULL PERIOD 2006-2024 ───────────────────────────────
def chart_crisis_timeline():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT fiscal_year, sector, AVG(net_margin) AS avg_net_margin
        FROM financials
        WHERE fiscal_year BETWEEN 2006 AND 2024
          AND net_margin IS NOT NULL
          AND net_margin BETWEEN -0.5 AND 0.5
        GROUP BY fiscal_year, sector
        ORDER BY fiscal_year
    """, conn)
    conn.close()
    if df.empty: return

    fig, ax = plt.subplots(figsize=(16, 7))
    for sector, grp in df.groupby("sector"):
        ax.plot(grp["fiscal_year"], grp["avg_net_margin"] * 100,
                marker="o", markersize=4, linewidth=2,
                label=sector.replace("_", " "),
                color=SECTOR_COLORS.get(sector, "#6B7280"))

    add_macro_bands(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Net Margin")
    ax.set_title(
        "Sector Net Margins: Full Period 2006–2024\n"
        "Showing Financial Crisis, COVID Shock, and Inflation impact",
        fontsize=12, fontweight="bold", pad=15)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax.set_xticks(range(2006, 2025, 2))
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/05_crisis_timeline.png")
    plt.close()
    log.info("OK 05: Crisis Timeline (full 2006-2024)")


# ── 06: Crisis Recovery ───────────────────────────────────────────────────────
def chart_recovery_speed():
    df = load("analysis_crisis_recovery.csv")
    if df.empty: return

    df_draw = df.dropna(subset=["margin_drawdown"]).sort_values("margin_drawdown")
    df_rec  = df.dropna(subset=["years_to_recover"]).sort_values("years_to_recover")
    df_no_rec = df[df["years_to_recover"].isna()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    bar_colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df_draw["sector"]]
    bars = ax1.barh(df_draw["ticker"], df_draw["margin_drawdown"] * 100,
                    color=bar_colors, height=0.65)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}pp"))
    ax1.set_xlabel("Percentage Point Change in Net Margin")
    ax1.set_title("Net Margin Drawdown\n2007 Baseline → Crisis Trough (2008/2009)",
                  fontsize=11, fontweight="bold")
    for bar, val in zip(bars, df_draw["margin_drawdown"]):
        ax1.text(val - 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val*100:.1f}pp", va="center", ha="right", fontsize=7)

    ax2 = axes[1]
    rec_colors = [SECTOR_COLORS.get(s, "#6B7280") for s in df_rec["sector"]]
    bars2 = ax2.barh(df_rec["ticker"], df_rec["years_to_recover"],
                     color=rec_colors, height=0.65)
    ax2.set_xlabel("Years to Recover")
    ax2.set_title("Recovery Speed\nYears to return to 90% of 2007 net margin",
                  fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, df_rec["years_to_recover"]):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f} yrs", va="center", fontsize=8)
    # BAC, F, SLB did not recover by 2015 — noted in README data notes

    fig.suptitle("2008 Financial Crisis: Impact & Recovery\n"
                 "15 companies with pre-crisis (2007) baseline data",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/06_crisis_recovery.png")
    plt.close()
    log.info("OK 06: Crisis Recovery")


# ── 07: Macro Stress ──────────────────────────────────────────────────────────
def chart_macro_stress():
    df = load("analysis_macro_stress.csv")
    if df.empty: return
    df = df.dropna(subset=["covid_net_margin_impact", "inflation_net_margin_impact"])

    fig, axes = plt.subplots(1, 2, figsize=(17, 10))

    df_c = df.sort_values("covid_net_margin_impact")
    ax = axes[0]
    bc = ["#DC2626" if v < 0 else "#059669" for v in df_c["covid_net_margin_impact"]]
    ax.barh(df_c["ticker"], df_c["covid_net_margin_impact"] * 100, color=bc, height=0.65)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}pp"))
    ax.set_title("COVID Shock (2020)\nNet Margin Change vs. 2019", fontsize=11, fontweight="bold")
    ax.set_xlabel("Percentage Point Change")

    df_i = df.sort_values("inflation_net_margin_impact")
    ax2 = axes[1]
    bi = ["#DC2626" if v < 0 else "#059669" for v in df_i["inflation_net_margin_impact"]]
    ax2.barh(df_i["ticker"], df_i["inflation_net_margin_impact"] * 100, color=bi, height=0.65)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}pp"))
    ax2.set_title("Inflation Shock (2022)\nNet Margin Change vs. 2021", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Percentage Point Change")

    fig.suptitle("Macro Stress Tests: COVID vs. Inflation — All 27 Companies\n"
                 "Which companies and sectors held margins?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/07_macro_stress.png")
    plt.close()
    log.info("OK 07: Macro Stress")


# ── 08: 20-Year Trends ────────────────────────────────────────────────────────
def chart_20yr_trends():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT fiscal_year, sector, ticker, operating_margin
        FROM financials
        WHERE operating_margin IS NOT NULL AND ABS(operating_margin) < 1
        ORDER BY sector, ticker, fiscal_year
    """, conn)
    conn.close()
    if df.empty: return

    sectors = sorted(df["sector"].unique())
    ncols, nrows = 3, -(-len(sectors) // 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4.5), sharex=True)
    axes = axes.flatten()

    for i, sector in enumerate(sectors):
        ax = axes[i]
        for ticker, tdf in df[df["sector"]==sector].groupby("ticker"):
            ax.plot(tdf["fiscal_year"], tdf["operating_margin"]*100,
                    linewidth=1.8, alpha=0.85, label=ticker)
        add_macro_bands(ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
        ax.set_title(sector.replace("_"," "), fontsize=10, fontweight="bold",
                     color=SECTOR_COLORS.get(sector, "#6B7280"))
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(2006, 2024)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("20-Year Operating Margin Trends by Sector (2006–2024)\n"
                 "Shaded: Financial Crisis | COVID | Inflation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/08_20yr_trends.png")
    plt.close()
    log.info("OK 08: 20-Year Trends")


# ── 09: FP&A Benchmark ───────────────────────────────────────────────────────
def chart_fpa_benchmark():
    df = load("analysis_fpa_benchmark.csv")
    if df.empty: return
    df = df.set_index("ticker").sort_values("sector")
    df["primary_margin"] = df["gross_margin"].fillna(df["operating_margin"])

    cols = {
        "primary_margin":   "Primary\nMargin*",
        "operating_margin": "Operating\nMargin",
        "net_margin":       "Net\nMargin",
        "roe":              "ROE",
        "roa":              "ROA",
        "fcf_margin":       "FCF\nMargin",
        "current_ratio":    "Current\nRatio",
        "debt_to_equity":   "Debt/\nEquity",
    }
    pivot = df[list(cols.keys())].rename(columns=cols).copy()
    for c in ["Primary\nMargin*","Operating\nMargin","Net\nMargin","ROE","ROA","FCF\nMargin"]:
        pivot[c] = pivot[c] * 100

    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Value (% or ratio)"},
                annot_kws={"size": 8},
                vmin=-50, vmax=100)  # cap color scale to prevent outliers dominating
    ax.set_title(
        "FP&A Benchmark Summary — 2024\n"
        "*Primary Margin = Gross Margin; Operating Margin for Banking/Energy/Logistics\n"
        "Color scale capped at -50% to 100% to prevent outliers (AAPL ROE=165%, HPQ ROE=-210%) from washing out other values",
        fontsize=10, fontweight="bold", pad=15)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/09_fpa_benchmark.png")
    plt.close()
    log.info("OK 09: FP&A Benchmark")


# ── 10: Cross-Crisis ─────────────────────────────────────────────────────────
def chart_cross_crisis():
    df = load("analysis_cross_crisis.csv")
    if df.empty: return
    df = df.dropna(subset=["drawdown_2008", "drawdown_covid"]).sort_values("drawdown_2008")

    x = np.arange(len(df))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    # Left: drawdown comparison
    ax = axes[0]
    ax.barh(x + w/2, df["drawdown_2008"]*100,  w, color="#DC2626", alpha=0.8, label="2008 Crisis")
    ax.barh(x - w/2, df["drawdown_covid"]*100, w, color="#F97316", alpha=0.8, label="COVID 2020")
    ax.set_yticks(x)
    ax.set_yticklabels(df["ticker"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}pp"))
    ax.set_xlabel("Net Margin Drawdown (pp)")
    ax.set_title("Margin Drawdown by Crisis\n2008 (credit shock) vs COVID (demand shock)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: recovery speed
    ax2 = axes[1]
    rec_2008  = df["yrs_to_recover_2008"].fillna(7)
    rec_covid = df["yrs_to_recover_covid"].fillna(7)
    ax2.barh(x + w/2, rec_2008,  w, color="#DC2626", alpha=0.8, label="2008 Recovery")
    ax2.barh(x - w/2, rec_covid, w, color="#F97316", alpha=0.8, label="COVID Recovery")
    ax2.set_yticks(x)
    ax2.set_yticklabels(df["ticker"], fontsize=9)
    ax2.set_xlabel("Years to Recover (7 = did not recover by end of observation window)")
    ax2.set_title("Recovery Speed by Crisis\nYears to return to 90% of pre-crisis margin",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.text(0.98, 0.03,
             "7 = company did not recover\nwithin the observation window\n"
             "(2015 for 2008 | 2024 for COVID)",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7.5, color="#6B7280",
             bbox=dict(boxstyle="round,pad=0.3", fc="#F3F4F6", alpha=0.8))

    fig.suptitle(
        "2008 Financial Crisis vs COVID: Cross-Crisis Resilience\n"
        "15 companies with pre-2008 data  |  2008 = credit shock  |  COVID = demand shock",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/10_cross_crisis.png")
    plt.close()
    log.info("OK 10: Cross-Crisis")


# ── 11: Recovery Quality ─────────────────────────────────────────────────────
def chart_recovery_quality():
    df = load("analysis_recovery_quality.csv")
    if df.empty: return

    # Exclude UNH (unreliable pre-2018 revenue — segment XBRL, not consolidated)
    # Exclude F (negative 2007 margin — recovery threshold is meaningless)
    df = df[~df["ticker"].isin(["UNH", "F"])].copy()

    driver_colors = {
        "Revenue-led (volume driven)": "#059669",
        "Balanced recovery":           "#2563EB",
        "Margin-led (cost cutting)":   "#DC2626",
        "Did not recover by 2015":     "#9CA3AF",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ── Left: years to margin recovery ───────────────────────────────────────
    df["margin_rec_yrs"] = (df["margin_rec_year"] - 2009).clip(lower=0)

    # Companies where margin never recovered
    never_rec = df["margin_rec_year"].isna()
    df.loc[never_rec, "margin_rec_yrs"] = 6.5
    df.loc[never_rec, "recovery_driver"] = "Did not recover by 2015"

    # Minimum bar of 0.3 so 0-year recoveries (same year as trough) are visible
    df["bar_val"] = df["margin_rec_yrs"].clip(lower=0.3)

    df_sorted = df.sort_values(["recovery_driver", "margin_rec_yrs"], ascending=[True, True])
    bar_colors = [driver_colors.get(d, "#6B7280") for d in df_sorted["recovery_driver"]]

    ax = axes[0]
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted["bar_val"], color=bar_colors, height=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["ticker"], fontsize=9)
    ax.set_xlabel("Years from Crisis Trough (2009) to Margin Recovery")
    ax.set_title("Years to Margin Recovery Post-2008\n"
                 "Short bar = fast | 6.5 = did not recover by 2015",
                 fontsize=11, fontweight="bold")
    ax.axvline(6.5, color="#9CA3AF", linewidth=0.8, linestyle=":", alpha=0.6)

    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        label = row["recovery_driver"].split(" (")[0]
        if label == "Did not recover by 2015":
            label = "No recovery"
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=7, color="#374151")

    handles = [mpatches.Patch(color=c, label=d) for d, c in driver_colors.items()]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.8)

    # ── Right: net margin trajectory 2007-2013 ────────────────────────────────
    ax2 = axes[1]
    margin_years = ["nm_2007", "nm_2010", "nm_2011", "nm_2013"]
    year_labels  = [2007, 2010, 2011, 2013]

    for _, row in df.iterrows():
        margins = [row.get(m) for m in margin_years]
        color = driver_colors.get(row["recovery_driver"], "#6B7280")
        # Only plot sensible values (exclude > 100% margin anomalies)
        # Keep values as decimals — pct formatter handles the % display
        valid = [(yr, m) for yr, m in zip(year_labels, margins)
                 if not pd.isna(m) and abs(m) < 1.0]
        if len(valid) >= 2:
            yrs, mgs = zip(*valid)
            ax2.plot(yrs, list(mgs),
                     marker="o", markersize=5, linewidth=1.8,
                     color=color, alpha=0.85)
            ax2.annotate(row["ticker"], (yrs[-1], mgs[-1]),
                         textcoords="offset points", xytext=(5, 2), fontsize=7.5)

    ax2.axvline(2009, color="#DC2626", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_ylim(-0.05, 0.45)
    ax2.text(2009.1, 0.43, "Crisis trough", fontsize=7, color="#DC2626", style="italic")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct))
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Net Margin")
    ax2.set_title("Net Margin Trajectory 2007-2013\nColored by recovery driver",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(year_labels)

    handles2 = [mpatches.Patch(color=c, label=d) for d, c in driver_colors.items()]
    ax2.legend(handles=handles2, fontsize=7.5, loc="upper left", framealpha=0.8)

    fig.suptitle(
        "Recovery Quality: How Did Companies Recover from 2008?\n"
        "Revenue-led = demand returned  |  Balanced = both  |  Margin-led = cost cutting",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/11_recovery_quality.png")
    plt.close()
    log.info("OK 11: Recovery Quality")


def run():
    log.info("Building visualization suite — 11 charts")
    chart_revenue_cagr()
    chart_margin_heatmap()
    chart_operating_leverage()
    chart_capital_efficiency()
    chart_crisis_timeline()
    chart_recovery_speed()
    chart_macro_stress()
    chart_20yr_trends()
    chart_fpa_benchmark()
    chart_cross_crisis()
    chart_recovery_quality()
    log.info(f"\nAll charts saved to {VIZ_DIR}/")


if __name__ == "__main__":
    run()
