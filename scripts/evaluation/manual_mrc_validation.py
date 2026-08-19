"""
Manual validation audit for the FinBERT-MRC news extraction stage.

The gold labels below are a manually curated, event-stratified set of real
headlines from data/processed/news_with_events_28stocks.csv. They are intended
to validate the processed extraction artifacts used by the manuscript, not to
retrain the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GoldCase:
    title_contains: str
    stock: str
    gold_event_type: str
    gold_entity_a: str
    gold_entity_b: str = ""
    note: str = ""


GOLD_CASES = [
    GoldCase("Wells Fargo On Oracle Following Earnings", "ORCL", "Analyst Rating", "ORCL"),
    GoldCase("Susquehanna Maintains Positive on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
    GoldCase("TD Securities Downgrades Enbridge Inc to Buy", "ENB", "Analyst Rating", "ENB"),
    GoldCase("3M to Sell Identity Management Business to Gemalto", "MMM", "Mergers and Acquisitions", "MMM"),
    GoldCase("Oracle's Low Expectations Lead To Well-Received Q3 Results", "ORCL", "Earnings Report", "ORCL"),
    GoldCase("BP PLC shares are trading lower after the company reported its Q3 earnings", "BP", "Earnings Report", "BP"),
    GoldCase("Amazon.com Q1 EPS $5.010 Misses", "AMZN", "Earnings Report", "AMZN"),
    GoldCase("Cisco's Q1: What Was Bullish", "CSCO", "Earnings Report", "CSCO"),
    GoldCase("Coca-Cola Q1'16 Earnings Conference Call", "KO", "Earnings Report", "KO"),
    GoldCase("Oracle Jury Says Google Did Not Infringe on Oracle's Patents", "ORCL", "Legal and Regulation", "ORCL"),
    GoldCase("House Judiciary Panel Wants To Know About Amazon", "AMZN", "Legal and Regulation", "AMZN"),
    GoldCase("Oracle Lawyer: No Excuse For Google Copying Of Java", "ORCL", "Legal and Regulation", "ORCL"),
    GoldCase("Amazon Could Relinquish Key Trademark to Apple", "JNJ", "Legal and Regulation", "AMZN", "AAPL"),
    GoldCase("Amazon Urges Court To Halt Progress On Microsoft's JEDI Contract", "ORCL", "Legal and Regulation", "AMZN", "MSFT"),
    GoldCase("FDA Votes In Favor Of Removing Side-Effect Warning On Pfizer's Chantix", "PFE", "Legal and Regulation", "PFE"),
    GoldCase("Oracle Buys InQuira", "ORCL", "Mergers and Acquisitions", "ORCL"),
    GoldCase("Amazon in Talks to Buy $2B Stake", "AMZN", "Mergers and Acquisitions", "AMZN"),
    GoldCase("Pfizer in Talks to Merge Off-patent Drugs Business with Mylan", "PFE", "Mergers and Acquisitions", "PFE"),
    GoldCase("The Coca-Cola Company and Coca-Cola FEMSA to acquire AdeS", "KO", "Mergers and Acquisitions", "KO"),
    GoldCase("Enbridge Partners Acquires 50% Ownership from EnBW", "ENB", "Mergers and Acquisitions", "ENB"),
    GoldCase("Apple's iPhone 9 Launch Could Be Imminent", "AAPL", "Product Launch", "AAPL"),
    GoldCase("Pfizer's ABRILADA Approved by the FDA", "PFE", "Product Launch", "PFE"),
    GoldCase("Pfizer's Hospira Unveils LifeCare PCA", "PFE", "Product Launch", "PFE"),
    GoldCase("Amazon launches its first big-budget game", "AMZN", "Product Launch", "AMZN"),
    GoldCase("Yara and IBM Launch an Open Collaboration", "IBM", "Product Launch", "IBM"),
    GoldCase("Apple shares are trading higher potentially due", "AAPL", "Stock Movement", "AAPL"),
    GoldCase("Oracle shares are trading lower after the company reported worse-than-expected Q2", "ORCL", "Stock Movement", "ORCL"),
    GoldCase("7 Stocks To Watch For October 13, 2017", "BAC", "Stock Movement", "BAC"),
    GoldCase("Cisco shares are trading higher after the company reported better-than-expected Q3", "CSCO", "Stock Movement", "CSCO"),
    GoldCase("Coke CEO Steps Down", "KO", "Stock Movement", "KO"),
    GoldCase("Apple's Top Supplier Foxconn Launched New Recruitment", "AAPL", "Supply Chain", "AAPL"),
    GoldCase("Apple supplier TSMC is reportedly building", "AAPL", "Supply Chain", "AAPL"),
    GoldCase("Amazon Looks To Establish Supply Chain Trust", "AMZN", "Supply Chain", "AMZN"),
    GoldCase("Wal-Mart, Coke And Pepsi Will Supply Flint", "KO", "Supply Chain", "KO"),
    GoldCase("Enbridge's $2.6B Sandpiper Project Secures Anchor Shipper", "ENB", "Supply Chain", "ENB"),
    GoldCase("Verizon CEO Talks 5G, China, Trade", "VZ", "Macroeconomic", "VZ"),
    GoldCase("Home Depot CEO Talks About Recovery", "HD", "Macroeconomic", "HD"),
    GoldCase("What Would A $5.8 Billion NIH Cut Look Like", "MRK", "Macroeconomic", "MRK"),
    GoldCase("China's Global Times Mentions Qualcomm, Cisco, Apple And Boeing", "AAPL", "Macroeconomic", "AAPL"),
    GoldCase("T-Mobile Won The U.S. Growth Battle Last Quarter", "VZ", "Macroeconomic", "VZ"),
]

GOLD_CASES.extend(
    [
        # Analyst Rating
        GoldCase("JMP Securities Maintains Market Outperform on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Morgan Stanley Maintains Overweight on Amazon.com, Raises Price Target to $2800", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Morgan Stanley Maintains Overweight on Amazon.com, Raises Price Target to $2600", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("MKM Partners Maintains Buy on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Drexel Hamilton's Brian White on Cisco", "CSCO", "Analyst Rating", "CSCO"),
        GoldCase("BMO Capital Maintains Outperform on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("RBC Capital Maintains Outperform on Amazon.com, Raises Price Target to $3300", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Credit Suisse Maintains Outperform on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("B of A Securities Reiterates Buy on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Canaccord Genuity Maintains Buy on Amazon.com", "AMZN", "Analyst Rating", "AMZN"),
        GoldCase("Oracle's Cloud Transition Isn't Fast Enough", "ORCL", "Analyst Rating", "ORCL"),
        GoldCase("Deutsche Bank Comments On Oracle's Apps Situation", "ORCL", "Analyst Rating", "ORCL"),
        # Earnings Report
        GoldCase("Oracle's Q3 Earnings Preview", "ORCL", "Earnings Report", "ORCL"),
        GoldCase("Oracle's Earnings: 'Tug Of War Between Old And New Continues'", "ORCL", "Earnings Report", "ORCL"),
        GoldCase("Oracle's Future May Be In The Cloud", "ORCL", "Earnings Report", "ORCL"),
        GoldCase("Cisco's Disappointing Outlook Sure To Overshadow Earnings Beat", "CSCO", "Earnings Report", "CSCO"),
        GoldCase("Pfizer's Q3 Earnings Preview", "PFE", "Earnings Report", "PFE"),
        GoldCase("Pfizer, Inc. Q3 EPS $0.78 Beats", "PFE", "Earnings Report", "PFE"),
        GoldCase("Pfizer Sees FY Rennue", "PFE", "Earnings Report", "PFE"),
        GoldCase("BP shares are trading higher after the company reported better-than-expected Q4", "BP", "Earnings Report", "BP"),
        GoldCase("IBM +5.93% Premarket", "IBM", "Earnings Report", "IBM"),
        GoldCase("Oracle Corporation Reports Q4 EPS", "ORCL", "Earnings Report", "ORCL"),
        # Legal and Regulation
        GoldCase("Oracle to Pay $199.5M to Resolve False Claims Act Lawsuit", "ORCL", "Legal and Regulation", "ORCL"),
        GoldCase("Pfizer's Biologics License Application for Tanezumab Accepted", "PFE", "Legal and Regulation", "PFE"),
        GoldCase("Pfizer's Drug TRAZIMERA Wins FDA Approval", "PFE", "Legal and Regulation", "PFE"),
        GoldCase("Pfizer's BOSULIF Regulatory Submissions Accepted", "PFE", "Legal and Regulation", "PFE"),
        GoldCase("Gilead defied a government HIV patent", "GILD", "Legal and Regulation", "GILD"),
        GoldCase("Oracle Seeking Up to $6.1 Billion in Google Patent Lawsuit", "ORCL", "Legal and Regulation", "ORCL"),
        GoldCase("JNJ's Sterilmed Gets FDA Letter", "JNJ", "Legal and Regulation", "JNJ"),
        GoldCase("New York AG Raises Concerns about Amazon", "AMZN", "Legal and Regulation", "AMZN"),
        GoldCase("Apple Fined", "AAPL", "Legal and Regulation", "AAPL"),
        # Mergers and Acquisitions
        GoldCase("Endeca to be Acquired by Oracle", "ORCL", "Mergers and Acquisitions", "ORCL"),
        GoldCase("Oracle To Acquire Ravello Systems", "ORCL", "Mergers and Acquisitions", "ORCL"),
        GoldCase("Red Hat gave Google and other buyers a chance to bid before IBM", "IBM", "Mergers and Acquisitions", "IBM"),
        GoldCase("Oracle Moves Forward With $9.3 Billion NetSuite Acquisition", "ORCL", "Mergers and Acquisitions", "ORCL"),
        GoldCase("Pfizer To Buy Anacor", "PFE", "Mergers and Acquisitions", "PFE"),
        GoldCase("Cisco To Buy Michigan-Based Duo Security", "CSCO", "Mergers and Acquisitions", "CSCO"),
        GoldCase("Cisco To Purchase Sentryo", "CSCO", "Mergers and Acquisitions", "CSCO"),
        GoldCase("Apple Acquires Dark Sky Weather App", "AAPL", "Mergers and Acquisitions", "AAPL"),
        GoldCase("Pfizer to Acquire Array BioPharma", "PFE", "Mergers and Acquisitions", "PFE"),
        # Product Launch
        GoldCase("Apple's computerized glasses won't be ready", "AAPL", "Product Launch", "AAPL"),
        GoldCase("Apple's iPhone 5 Brings New Customers to Sprint", "VZ", "Product Launch", "AAPL"),
        GoldCase("Apple, 'Transformers' Lead 2015 Product Placement", "KO", "Product Launch", "AAPL"),
        GoldCase("Pfizer, Eli Lilly Late Thurs. Announced Top-Line Results", "PFE", "Product Launch", "PFE"),
        GoldCase("Crucible' Is Out", "AMZN", "Product Launch", "AMZN"),
        GoldCase("ORTHOCON Announces Uninterrupted Availability", "JNJ", "Product Launch", "JNJ"),
        GoldCase("YouTube Makes Its Dedicated Children App Available On Apple TV", "AAPL", "Product Launch", "AAPL"),
        GoldCase("Volume Production of New iPhone Models", "AAPL", "Product Launch", "AAPL"),
        GoldCase("Merck Yesterday Announced KEYTRUDA plus LENVIMA", "MRK", "Product Launch", "MRK"),
        GoldCase("Halozyme Announces J&J's Janssen Receives European Marketing Authorization", "JNJ", "Product Launch", "JNJ"),
        # Stock Movement
        GoldCase("Oracle shares are trading higher after the company reported better-than-expected Q3", "ORCL", "Stock Movement", "ORCL"),
        GoldCase("Oracle shares are trading higher after the company reported better-than-expected Q4", "ORCL", "Stock Movement", "ORCL"),
        GoldCase("Amazon shares are trading higher despite market weakness", "AMZN", "Stock Movement", "AMZN"),
        GoldCase("Cisco shares are trading lower after the company issued Q1 EPS", "CSCO", "Stock Movement", "CSCO"),
        GoldCase("Oracle shares are trading higher after the company reported better-than-expected Q4 EPS", "ORCL", "Stock Movement", "ORCL"),
        GoldCase("Cisco shares are trading lower after the company issued Q2 EPS guidance", "CSCO", "Stock Movement", "CSCO"),
        GoldCase("Gilead shares are trading lower after the company reportedly ended", "GILD", "Stock Movement", "GILD"),
        GoldCase("Cisco shares are trading higher after the company reported better-than-expected Q3 EPS", "CSCO", "Stock Movement", "CSCO"),
        GoldCase("Johnson & Johnson shares are trading higher", "JNJ", "Stock Movement", "JNJ"),
        GoldCase("Shares of several technology companies are trading lower", "IBM", "Stock Movement", "IBM"),
        # Supply Chain
        GoldCase("Boeing Supplier Spirit Aerosystems Announces", "BA", "Supply Chain", "BA"),
        GoldCase("Amazon Announces New Fulfillment Center", "AMZN", "Supply Chain", "AMZN"),
        GoldCase("Corning And Pfizer Announce Supply Agreement", "PFE", "Supply Chain", "PFE"),
        GoldCase("BDR Pharmaceuticals Sought India Approval to Manufacture Remdesivir", "GILD", "Supply Chain", "GILD"),
        GoldCase("LG reportedly supplying 20 million OLED panels", "AAPL", "Supply Chain", "AAPL"),
        GoldCase("Apple Looking To Diversify Its Manufacturing Base", "AAPL", "Supply Chain", "AAPL"),
        GoldCase("Hetero Enters Into a Licensing Agreement With Gilead", "GILD", "Supply Chain", "GILD"),
        GoldCase("Pfizer Begins Human Testing Of Coronavirus Vaccine", "PFE", "Supply Chain", "PFE"),
        GoldCase("3M Partners With Cummins", "MMM", "Supply Chain", "MMM"),
        GoldCase("Stanley Black & Decker To Supply DEWALT", "MMM", "Supply Chain", "MMM"),
        # Macroeconomic
        GoldCase("BP Cuts 10K Jobs", "BP", "Macroeconomic", "BP"),
        GoldCase("US Debt Market Raised $22.5B This Week", "AAPL", "Macroeconomic", "AAPL"),
        GoldCase("Taking A Breath: Jobless Claims Disappoint", "BAC", "Macroeconomic", "BAC"),
        GoldCase("IBM Is Cutting 'Several Thousand' Jobs", "IBM", "Macroeconomic", "IBM"),
        GoldCase("Smartphone Sales Picking up in China", "AAPL", "Macroeconomic", "AAPL"),
        GoldCase("The Oil Crisis Can Harm US Economy", "BP", "Macroeconomic", "BP"),
        GoldCase("Caterpillar 8-K Shows Global Rolling", "CAT", "Macroeconomic", "CAT"),
        GoldCase("Investor Movement Index Summary: April 2020", "BAC", "Macroeconomic", "BAC"),
        GoldCase("Apple CEO Tim Cook: Store Traffic", "AAPL", "Macroeconomic", "AAPL"),
        GoldCase("Pfizer begins coronavirus vaccine testing", "PFE", "Macroeconomic", "PFE"),
    ]
)


def normalize_entity(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def find_case(df: pd.DataFrame, case: GoldCase) -> pd.Series:
    mask = df["title"].str.contains(case.title_contains, case=False, regex=False, na=False)
    mask &= df["stock"].astype(str).str.upper().eq(case.stock)
    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(f"No processed row matched: {case.stock} / {case.title_contains}")
    return matches.iloc[0]


def build_validation(news_path: Path) -> pd.DataFrame:
    df = pd.read_csv(news_path)
    rows = []

    for case in GOLD_CASES:
        row = find_case(df, case)
        pred_entity_a = normalize_entity(row.get("entity_a"))
        pred_entity_b = normalize_entity(row.get("entity_b"))
        gold_entity_a = normalize_entity(case.gold_entity_a)
        gold_entity_b = normalize_entity(case.gold_entity_b)
        pred_event_type = str(row.get("event_type", "")).strip()

        entity_a_correct = pred_entity_a == gold_entity_a
        entity_b_correct = pred_entity_b == gold_entity_b
        event_type_correct = pred_event_type == case.gold_event_type

        rows.append(
            {
                "title": row["title"],
                "date": row.get("date", ""),
                "stock": normalize_entity(row.get("stock")),
                "pred_event_type": pred_event_type,
                "gold_event_type": case.gold_event_type,
                "event_type_correct": event_type_correct,
                "pred_entity_a": pred_entity_a,
                "gold_entity_a": gold_entity_a,
                "entity_a_correct": entity_a_correct,
                "pred_entity_b": pred_entity_b,
                "gold_entity_b": gold_entity_b,
                "entity_b_correct": entity_b_correct,
                "event_confidence": float(row.get("event_confidence", 0.0)),
                "note": case.note,
            }
        )

    return pd.DataFrame(rows)


def summarize(validation: pd.DataFrame) -> pd.DataFrame:
    both_entities = validation["entity_a_correct"] & validation["entity_b_correct"]
    rows = [
        ("n_headlines", len(validation)),
        ("event_type_accuracy", validation["event_type_correct"].mean()),
        ("entity_a_accuracy", validation["entity_a_correct"].mean()),
        ("entity_b_accuracy", validation["entity_b_correct"].mean()),
        ("strict_tuple_accuracy", both_entities.mean()),
        ("mean_confidence_correct_entity_a", validation.loc[validation["entity_a_correct"], "event_confidence"].mean()),
        ("mean_confidence_incorrect_entity_a", validation.loc[~validation["entity_a_correct"], "event_confidence"].mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual MRC validation on processed news output.")
    parser.add_argument("--news-path", type=Path, default=Path("data/processed/news_with_events_28stocks.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/validation"))
    args = parser.parse_args()

    validation = build_validation(args.news_path)
    summary = summarize(validation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.out_dir / "mrc_manual_validation_28stocks.csv"
    summary_path = args.out_dir / "mrc_manual_validation_summary_28stocks.csv"
    validation.to_csv(validation_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved validation rows: {validation_path}")
    print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
