"""
Table 1: Descriptive statistics for the main estimation sample.

The +Governance subset of panel_main.csv is the same 20,318 firm-year
observations of 5,148 listed focal firms used by the baseline regression
in Table 2 column 3, so the descriptives line up with the regressions.
"""
import pandas as pd
from utils import load_panel, ensure_output_dir

GOV_VARS = ["top1_ownership", "board_size", "board_independence", "industry_hhi"]

TABLE_1_VARS = [
    ("resilience_composite",  "Supply chain resilience (composite of resistance and recovery sub-indices)"),
    ("dd_patent_range",       "Supplier-base digital disparity (patent-based range)"),
    ("size",                  "Firm size (log of total assets)"),
    ("leverage",              "Leverage (total liabilities / total assets)"),
    ("roa",                   "Return on assets (net income / total assets)"),
    ("top1_ownership",        "Top-1 ownership"),
    ("board_size",            "Board size"),
    ("board_independence",    "Board independence"),
    ("industry_hhi",          "Industry HHI"),
]


def main():
    panel = load_panel("panel_main.csv")
    # Restrict to the +Governance estimation sample so descriptives match the
    # regressions reported in Table 2 column 3.
    sample = panel.dropna(subset=GOV_VARS)
    print(f"Estimation sample: N={len(sample):,}, "
          f"firms={sample['stkcd'].nunique():,}, "
          f"years={sample['year'].min()}-{sample['year'].max()}\n")

    rows = []
    for col, label in TABLE_1_VARS:
        s = pd.to_numeric(sample[col], errors="coerce").dropna()
        rows.append({
            "Variable": label,
            "N": len(s),
            "Mean": round(s.mean(), 3),
            "SD": round(s.std(), 3),
            "Min": round(s.min(), 3),
            "Max": round(s.max(), 3),
        })
    desc = pd.DataFrame(rows)
    print(desc.to_string(index=False))

    out_dir = ensure_output_dir()
    desc.to_csv(out_dir / "table1_descriptive.csv", index=False)
    print(f"\nSaved: {out_dir / 'table1_descriptive.csv'}")


if __name__ == "__main__":
    main()
