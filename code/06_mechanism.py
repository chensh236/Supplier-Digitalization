"""
Table 6: Supplier-base changes consistent with the predicted mechanism.

The theory predicts that as supplier-base digital disparity widens, two
related changes appear in the supplier base:

  Hypothesis 2a - Procurement concentrates on digitally advanced suppliers.
                  Tested with two proxies:
                    * procurement-weighted supplier digitalisation,
                    * share of procurement allocated to suppliers above
                      the network-year median digitalisation.

  Hypothesis 2b - The active operational presence of lower-digital
                  suppliers narrows. Tested with the share of suppliers in
                  the bottom decile of the network-year digitalisation
                  distribution.

These regressions are mechanism-consistent diagnostics, not formal
mediation tests. They document whether disparity moves with the predicted
supplier-base changes; they do not estimate a full mediation system.

The procurement-share variables have lower coverage than the main panel,
which is why columns 1 and 2 use a smaller sample (N = 16,993).
"""
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from utils import load_panel, standardise_in_sample, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]


def fit(panel: pd.DataFrame, dv: str, label: str) -> dict:
    cols = [dv, "dd_patent_range"] + MAIN_CONTROLS
    d = standardise_in_sample(panel.dropna(subset=cols), cols)
    d = d.set_index(["stkcd", "year"])
    Y = d[dv].astype(float)
    X = d[["dd_patent_range"] + MAIN_CONTROLS].astype(float)
    res = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(
        cov_type="clustered", cluster_entity=True
    )
    coef = res.params["dd_patent_range"]
    se = res.std_errors["dd_patent_range"]
    p = res.pvalues["dd_patent_range"]
    n = int(res.nobs)
    print(f"  {label:36s}  coef={coef:+.4f}  SE={se:.4f}  p={p:.4f}  N={n:,}")
    return {"Spec": label, "Coef": coef, "SE": se, "p": p, "N": n}


def main():
    panel = load_panel("panel_main.csv")
    print("Table 6: Supplier-base changes")
    rows = []
    for dv, label in [
        ("procurement_weighted_digitalization", "(1) Proc.-weighted digitalisation"),
        ("high_digital_procurement_share",     "(2) High-digital procurement share"),
        ("low_digital_supplier_share",         "(3) Low-digital supplier share"),
    ]:
        rows.append(fit(panel, dv, label))
    out_dir = ensure_output_dir()
    pd.DataFrame(rows).to_csv(out_dir / "table6_mechanism.csv", index=False)
    print(f"\nSaved: {out_dir / 'table6_mechanism.csv'}")


if __name__ == "__main__":
    main()
