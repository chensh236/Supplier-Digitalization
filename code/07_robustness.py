"""
Robustness checks (selected from Appendix Tables C1 and C5).

Reproduces:

  1. Alternative measures of supplier-base digital disparity
     (recruitment-based range, composite range, coefficient of variation,
     standard deviation). Reported in Appendix Table C1.

  2. Manufacturing and high-technology subsamples. Reported in
     Appendix Table C5.

All rows use the +Governance specification from Table 2 column 3.
"""
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from utils import load_panel, standardise_in_sample, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]
GOV_CONTROLS = ["top1_ownership", "board_size", "board_independence", "industry_hhi"]
ALL_CONTROLS = MAIN_CONTROLS + GOV_CONTROLS


def fit(df: pd.DataFrame, x: str, label: str) -> dict:
    cols = ["resilience_composite", x] + ALL_CONTROLS
    d = standardise_in_sample(df.dropna(subset=cols), cols)
    d = d.set_index(["stkcd", "year"])
    Y = d["resilience_composite"].astype(float)
    X = d[[x] + ALL_CONTROLS].astype(float)
    res = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(
        cov_type="clustered", cluster_entity=True
    )
    coef = res.params[x]
    se = res.std_errors[x]
    p = res.pvalues[x]
    n = int(res.nobs)
    print(f"  {label:42s}  coef={coef:+.4f}  SE={se:.4f}  p={p:.4f}  N={n:,}")
    return {"Spec": label, "Coef": coef, "SE": se, "p": p, "N": n}


def main():
    panel = load_panel("panel_main.csv")
    print(f"Sample: N={len(panel):,}\n")

    out = []

    # ---- Table C1: Alternative disparity measures ----
    print("Table C1: Alternative measures of supplier digital disparity")
    out.append(fit(panel, "dd_patent_range",       "Patent-based range (baseline)"))
    out.append(fit(panel, "dd_recruitment_range",  "Recruitment-based range"))
    out.append(fit(panel, "dd_composite_range",    "Composite range"))
    out.append(fit(panel, "dd_patent_cv",          "Patent-based coefficient of variation"))
    out.append(fit(panel, "dd_patent_std",         "Patent-based standard deviation"))

    # ---- Table C5: Subsample analyses ----
    print("\nTable C5: Subsample analyses")
    manuf = panel[panel["industry_code"].astype(str).str.upper().str.startswith("C")].copy()
    out.append(fit(manuf, "dd_patent_range",       "Manufacturing subsample"))
    if "is_high_tech" in panel.columns:
        ht = panel[panel["is_high_tech"] == 1].copy()
        out.append(fit(ht,    "dd_patent_range",       "High-technology subsample"))

    out_dir = ensure_output_dir()
    pd.DataFrame(out).to_csv(out_dir / "tableC_robustness.csv", index=False)
    print(f"\nSaved: {out_dir / 'tableC_robustness.csv'}")


if __name__ == "__main__":
    main()
