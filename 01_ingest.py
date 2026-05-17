"""
Layer 1: Raw Financial Data Ingestion via SEC EDGAR XBRL API
=============================================================
Pulls raw annual financial data directly from SEC EDGAR public filings.

Data source: https://data.sec.gov/api/xbrl/companyfacts/
Same raw data the SEC uses — pulled from 10-K annual report filings.

XBRL concepts pulled:
  Income Statement : Revenue, COGS, Gross Profit, Operating Income, Net Income, R&D Expense, Interest Expense
  Balance Sheet    : Total Assets, Total Equity, Long Term Debt,Current Assets, Current Liabilities, Cash
  Cash Flow        : Operating Cash Flow, CapEx
  Other            : Shares Outstanding

All ratios derived in Layer 2 (SQL).

"""

import requests
import pandas as pd
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
USER_AGENT = "Financial Analysis Project mkourouma53@gmail.com"
START_YEAR = 2005
END_YEAR   = 2024
RAW_DIR    = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ── Company Universe ──────────────────────────────────────────────────────────
COMPANIES = {
    "MSFT": {"name": "Microsoft",         "sector": "Software_Cloud",      "cik": "0000789019"},
    "GOOG": {"name": "Alphabet",          "sector": "Software_Cloud",      "cik": "0001652044"},
    "CRM":  {"name": "Salesforce",        "sector": "Software_Cloud",      "cik": "0001108524"},
    "NVDA": {"name": "NVIDIA",            "sector": "Semiconductors",      "cik": "0001045810"},
    "INTC": {"name": "Intel",             "sector": "Semiconductors",      "cik": "0000050863"},
    "AMD":  {"name": "AMD",               "sector": "Semiconductors",      "cik": "0000002488"},
    "AAPL": {"name": "Apple",             "sector": "Consumer_Hardware",   "cik": "0000320193"},
    "HPQ":  {"name": "HP Inc",            "sector": "Consumer_Hardware",   "cik": "0000047217"},
    "DELL": {"name": "Dell",              "sector": "Consumer_Hardware",   "cik": "0001571123"},
    "JPM":  {"name": "JPMorgan Chase",    "sector": "Banking",             "cik": "0000019617"},
    "BAC":  {"name": "Bank of America",   "sector": "Banking",             "cik": "0000070858"},
    "C":    {"name": "Citigroup",         "sector": "Banking",             "cik": "0000831001"},
    "JNJ":  {"name": "Johnson & Johnson", "sector": "Healthcare",          "cik": "0000200406"},
    "PFE":  {"name": "Pfizer",            "sector": "Healthcare",          "cik": "0000078003"},
    "UNH":  {"name": "UnitedHealth",      "sector": "Healthcare",          "cik": "0000731766"},
    "XOM":  {"name": "ExxonMobil",        "sector": "Energy",              "cik": "0000034088"},
    "CVX":  {"name": "Chevron",           "sector": "Energy",              "cik": "0000093410"},
    "SLB":  {"name": "Schlumberger",      "sector": "Energy",              "cik": "0000087347"},
    "WMT":  {"name": "Walmart",           "sector": "Consumer_Retail",     "cik": "0000104169"},
    "COST": {"name": "Costco",            "sector": "Consumer_Retail",     "cik": "0000909832"},
    "TGT":  {"name": "Target",            "sector": "Consumer_Retail",     "cik": "0000027419"},
    "AMZN": {"name": "Amazon",            "sector": "Ecommerce_Logistics", "cik": "0001018724"},
    "FDX":  {"name": "FedEx",             "sector": "Ecommerce_Logistics", "cik": "0001048911"},
    "UPS":  {"name": "UPS",               "sector": "Ecommerce_Logistics", "cik": "0001090727"},
    "TSLA": {"name": "Tesla",             "sector": "Automotive",          "cik": "0001318605"},
    "F":    {"name": "Ford",              "sector": "Automotive",          "cik": "0000037996"},
    "GM":   {"name": "General Motors",    "sector": "Automotive",          "cik": "0001467858"},
}

# ── XBRL Concept Mappings ─────────────────────────────────────────────────────
# All aliases merged across years. Earlier alias = higher priority on overlap.
# Covers: tag changes over time, sector-specific tags, company-specific tags,
# and the 2024 tag migration where many companies switched to Nonoperating variants.
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",   # post-2017 standard
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",                                        # pre-2017 standard
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "Revenues",
        "TotalRevenuesAndOtherIncome",
        "InterestAndDividendIncomeOperating",                     # banks
        "InterestIncomeOperating",
        "RevenuesNetOfInterestExpense",
        "NoninterestIncome",
    ],
    "cogs": [
        # Used to derive gross_profit = revenue - cogs where GrossProfit tag missing
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
        "CostOfSales",
        "CostOfRevenueAmortization",
        "CostsAndExpenses",                                       # SLB post-2018
        "OperatingCostsAndExpenses",
        "CostOfPurchasedOilAndGas",                               # CVX/XOM
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        # TGT used these during Canada restructuring 2014-2016
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "OperatingIncomeLossFromContinuingOperations",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "NetIncomeLossAttributableToParent",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        "AutomotiveResearchAndDevelopmentExpense",               # GM uses this
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
        "InterestExpenseOther",                                  # AAPL early years
        "InterestExpenseRelatedParty",
        "InterestExpenseNonoperating",                           # 2024 migration
        "InterestCostsIncurred",
        "FinanceLeaseInterestExpense",                           # 2024 lease standard
        "InterestAndFeeIncomeLoansAndLeases",                    # banks 2024
        "InterestExpenseLongTermDebt",
        "InterestExpenseShortTermBorrowings",
        "InterestPaidNet",
    ],
    "total_assets": [
        "Assets",
        "AssetsNet",                                             # JNJ 2009/2015
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "CommonStockholdersEquity",
        "RetainedEarningsAccumulatedDeficit",                    # fallback
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",      # SLB 2006-2012
        "DebtAndCapitalLeaseObligations",                        # GM
        "DebtLongtermAndShorttermCombinedAmount",                # Ford
        "LongTermNotesPayable",
        "SeniorLongTermNotes",
        "ConvertibleNotesPayable",                               # CRM
        "UnsecuredLongTermDebt",
        "NotesPayable",                                          # GM alternate
        "LongTermDebtFairValue",                                 # JPM 2014+
        "SubordinatedLongTermDebt",                              # JPM alternate
        "JuniorSubordinatedLongTermDebt",
        "FinanceLeaseLiabilityNoncurrent",                       # F 2021+ (lease standard)
        "LongTermDebtExcludingCurrentMaturities",
        "OtherLongTermDebt",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
        "CashAndCashEquivalentsAndRestrictedCash",               # CVX 2019/2024
        "CashAndCashEquivalentsAndShortTermInvestments",         # INTC 2019-2023
        "CashEquivalentsAtCarryingValue",                        # SLB
        "CashAndDueFromBanks",                                   # banks
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsFairValueDisclosure",             # SLB alternate
        "RestrictedCashAndCashEquivalents",
        "CashCashEquivalentsAndFederalFundsSold",                # WMT early years
        "CashAndCashEquivalentsPeriodIncreaseDecrease",          # WMT 2007-2008
    ],
    "operating_cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByOperatingActivities",
        "NetCashFromOperatingActivities",
        "CashGeneratedFromOperations",
        "NetCashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PurchaseOfPropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsForProceedsFromProductiveAssets",               # NVDA 2013-2019
        "PaymentsToAcquireOtherProductiveAssets",
        "CapitalExpenditures",
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets", # NVDA 2017-2018
        "CapitalExpenditureDiscontinuedOperations",
        "PaymentsForProceedsFromOtherInvestingActivities",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
}


def fetch_company_facts(cik: str, ticker: str) -> dict | None:
    """Fetch all XBRL facts for one company from SEC EDGAR."""
    url     = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"  HTTP {resp.status_code} for {ticker} (CIK {cik})")
        return None
    except Exception as e:
        log.error(f"  Request failed for {ticker}: {e}")
        return None


def extract_annual_series(facts: dict, concept_aliases: list) -> dict:
    """
    Extract annual 10-K values for one concept.
    Merges ALL matching aliases by year — handles companies that changed
    XBRL tags over time. Earlier alias = higher priority on overlap.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    merged  = {}

    for priority, alias in enumerate(concept_aliases):
        if alias not in us_gaap:
            continue
        units    = us_gaap[alias].get("units", {})
        usd_data = units.get("USD") or units.get("shares") or []

        for entry in usd_data:
            if entry.get("form") != "10-K":
                continue
            end_date = entry.get("end", "")
            if not end_date:
                continue
            year = int(end_date[:4])
            if year < START_YEAR or year > END_YEAR:
                continue
            filed = entry.get("filed", "")
            val   = entry.get("val")

            if year not in merged:
                merged[year] = {"val": val, "filed": filed, "priority": priority}
            else:
                existing = merged[year]
                if priority < existing["priority"]:
                    merged[year] = {"val": val, "filed": filed, "priority": priority}
                elif priority == existing["priority"] and filed > existing["filed"]:
                    merged[year] = {"val": val, "filed": filed, "priority": priority}

    return {yr: d["val"] for yr, d in merged.items()}


def pull_company(ticker: str, info: dict) -> pd.DataFrame | None:
    """Pull all financial line items for one company."""
    facts = fetch_company_facts(info["cik"], ticker)
    if not facts:
        return None

    rows = {}
    for field, aliases in CONCEPTS.items():
        for year, val in extract_annual_series(facts, aliases).items():
            if year not in rows:
                rows[year] = {"fiscal_year": year}
            rows[year][field] = val

    if not rows:
        log.warning(f"  No data extracted for {ticker}")
        return None

    df = pd.DataFrame(list(rows.values())).sort_values("fiscal_year").copy()
    df["ticker"]  = ticker
    df["company"] = info["name"]
    df["sector"]  = info["sector"]

    # Derive gross_profit = revenue - cogs where GrossProfit tag missing
    # Fixes: AMZN, GOOG, WMT, COST, TGT, DELL, HPQ, SLB, F, GM (partial)
    if "gross_profit" not in df.columns:
        df["gross_profit"] = np.nan
    if "cogs" in df.columns and "revenue" in df.columns:
        mask = (df["gross_profit"].isnull() &
                df["revenue"].notna() &
                df["cogs"].notna())
        df.loc[mask, "gross_profit"] = df.loc[mask, "revenue"] - df.loc[mask, "cogs"]

    # Drop cogs — intermediate derivation field, not needed downstream
    cols = (["fiscal_year", "ticker", "company", "sector"] +
            [c for c in df.columns
             if c not in ["fiscal_year", "ticker", "company", "sector", "cogs"]])
    df = df[cols].reset_index(drop=True)

    log.info(f"  OK {ticker:5s} ({info['name']:25s}) "
             f"{int(df['fiscal_year'].min())}-{int(df['fiscal_year'].max())}  "
             f"({len(df)} years)")
    return df


def print_missing_summary(df: pd.DataFrame):
    """Full missing data report after all fixes applied."""
    log.info("\n-- Missing Data Summary (after all fixes) --")
    metrics = ["revenue", "gross_profit", "operating_income", "net_income",
               "total_assets", "total_equity", "long_term_debt",
               "current_assets", "current_liabilities", "cash",
               "operating_cashflow", "capex", "shares_outstanding",
               "rd_expense", "interest_expense"]
    for m in metrics:
        if m not in df.columns:
            continue
        n = df[m].isnull().sum()
        if n == 0:
            log.info(f"  {m:25s} COMPLETE")
            continue
        log.info(f"  {m:25s} {n:3d} missing")
        for ticker, grp in df.groupby("ticker"):
            yrs = grp[grp[m].isnull()]["fiscal_year"].tolist()
            if yrs:
                log.info(f"    {ticker:5s}: {yrs}")


def audit(df: pd.DataFrame):
    """Coverage report."""
    log.info("\n-- Coverage Report --")
    log.info(f"  Total rows : {len(df):,}")
    log.info(f"  Companies  : {df['ticker'].nunique()}")
    log.info(f"  Year range : {int(df['fiscal_year'].min())} - {int(df['fiscal_year'].max())}")
    log.info("\n  Years per company:")
    summary = (
        df.groupby(["sector", "ticker", "company"])["fiscal_year"]
        .agg(["min", "max", "count"]).reset_index()
        .rename(columns={"min": "from", "max": "to", "count": "years"})
    )
    for _, row in summary.sort_values(["sector", "ticker"]).iterrows():
        flag = "OK     " if row["years"] >= 10 else "PARTIAL"
        log.info(f"    [{flag}] {row['ticker']:5s} {row['company']:25s} "
                 f"{int(row['from'])}-{int(row['to'])}  ({int(row['years'])} yrs)")
    missing_cos = set(COMPANIES.keys()) - set(df["ticker"].unique())
    if missing_cos:
        log.warning(f"\n  Missing companies: {missing_cos}")


def run():
    log.info("SEC EDGAR Financial Data Ingestion")
    log.info(f"  Target range : {START_YEAR}-{END_YEAR}")
    log.info(f"  Companies    : {len(COMPANIES)}")
    log.info(f"  Source       : data.sec.gov/api/xbrl/companyfacts/\n")

    all_frames = []
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        log.info(f"[{i:2d}/{len(COMPANIES)}] Pulling {ticker}...")
        df = pull_company(ticker, info)
        if df is not None:
            df.to_csv(f"{RAW_DIR}/{ticker}.csv", index=False)
            all_frames.append(df)
        time.sleep(0.12)  # SEC rate limit: max 10 req/sec

    if not all_frames:
        log.error("No data pulled. Check USER_AGENT and internet connection.")
        return

    combined = (pd.concat(all_frames, ignore_index=True)
                .sort_values(["sector", "ticker", "fiscal_year"])
                .reset_index(drop=True))
    combined.to_csv(f"{RAW_DIR}/ALL_COMPANIES_RAW.csv", index=False)
    log.info(f"\nSaved: {RAW_DIR}/ALL_COMPANIES_RAW.csv")

    audit(combined)
    print_missing_summary(combined)


if __name__ == "__main__":
    run()
