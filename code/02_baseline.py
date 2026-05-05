"""
Table 2: Baseline two-way fixed-effects estimates.

Estimates the effect of supplier-base digital disparity on supply chain
resilience using a panel two-way fixed-effects model:

    Resilience_{it} = beta * DD_{it} + gamma' * X_{it}
                      + mu_i + lambda_t + epsilon_{it}

Three nested specifications are reported, matching the columns of Table 2:

  Column 1   Firm and year fixed effects only.
  Column 2   Adds the main controls: firm size, leverage, return on assets.
  Column 3   Adds governance controls: top-1 ownership, board size,
             board independence, industry HHI.

All three columns are estimated on the +Governance sample (N = 20,318
firm-year observations, 5,148 firms). Continuous variables are
standardised to within-sample z-scores; standard errors are clustered
at the firm level.
"""
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from utils import load_panel, standardise_in_sample, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]
GOV_CONTROLS = ["top1_ownership", "board_size", "board_independence", "industry_hhi"]


def fit(df: pd.DataFrame, controls: list, label: str) -> dict:
    """Fit one specification on the +Governance sample."""
    cols = ["resilience_composite", "dd_patent_range"] + MAIN_CONTROLS + GOV_CONTROLS
    d = standardise_in_sample(df.dropna(subset=cols),
                              ["resilience_composite", "dd_patent_range"] + controls)
    d = d.set_index(["stkcd", "year"])

    y = d["resilience_composite"].astype(float)
    X = d[["dd_patent_range"] + controls].astype(float) if controls \
        else d[["dd_patent_range"]].astype(float)

    res = PanelOLS(y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    coef = res.params["dd_patent_range"]
    se = res.std_errors["dd_patent_range"]
    pv = res.pvalues["dd_patent_range"]
    n = int(res.nobs)
    print(f"  {label:20s}  coef={coef:+.4f}  SE={se:.4f}  p={pv:.4f}  N={n:,}  R2_within={res.rsquared_within:.4f}")
    return {"Spec": label, "Coef": coef, "SE": se, "p": pv, "N": n,
            "R2_within": res.rsquared_within}


def main():
    panel = load_panel("panel_main.csv")
    # Restrict to the +Governance sample so all three columns share an N.
    panel = panel.dropna(subset=GOV_CONTROLS)
    print(f"Sample: N={len(panel):,}, firms={panel['stkcd'].nunique():,}\n")
    print("Two-way fixed-effects estimates (firm + year FE, firm-clustered SEs):")
    rows = [
        fit(panel, [], "(1) FE only"),
        fit(panel, MAIN_CONTROLS, "(2) Main controls"),
        fit(panel, MAIN_CONTROLS + GOV_CONTROLS, "(3) +Governance"),
    ]
    out = pd.DataFrame(rows)

    out_dir = ensure_output_dir()
    out.to_csv(out_dir / "table2_baseline.csv", index=False)
    print(f"\nSaved: {out_dir / 'table2_baseline.csv'}")


if __name__ == "__main__":
    main()
