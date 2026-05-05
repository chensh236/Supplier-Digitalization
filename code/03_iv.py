"""
Table 3: Instrumental-variable (2SLS) estimates.

Leave-one-out industry-year peer-disparity instrument:

    Z_{it} = mean( DD_{jt} : j != i, j in same industry-year cell )

Estimation steps:

  1. Build the peer instrument on the broader supplier-disparity panel
     (drop NA on Y, X, main controls).
  2. Restrict to industry-year cells with at least ten firms.
  3. Standardise within the IV sample, then absorb firm and year fixed
     effects by sequential demeaning.
  4. Estimate IV2SLS with firm-clustered standard errors. The first-stage
     F is reported as (peer-disparity coefficient / SE)^2 from a
     firm-clustered first-stage OLS.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from utils import load_panel, standardise_in_sample, demean, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]
MIN_PEERS = 10  # minimum firms in an industry-year cell


def main():
    panel = load_panel("panel_main.csv")
    panel = panel.dropna(subset=["resilience_composite", "dd_patent_range",
                                 "size", "leverage", "roa", "industry_code"])
    print(f"IV starting panel: N={len(panel):,}, firms={panel['stkcd'].nunique():,}")

    # Build the leave-one-out industry-year peer instrument
    g = panel.groupby(["industry_code", "year"])
    cell_sum = g["dd_patent_range"].transform("sum")
    cell_n = g["dd_patent_range"].transform("count")
    panel["peer_dd"] = (cell_sum - panel["dd_patent_range"]) / (cell_n - 1)

    # Restrict to industry-year cells with at least MIN_PEERS firms
    iv_sample = panel[cell_n >= MIN_PEERS].copy()
    print(f"After cell-size restriction (>= {MIN_PEERS} firms): "
          f"N={len(iv_sample):,}, firms={iv_sample['stkcd'].nunique():,}")

    # Standardise within the IV sample, then sequentially demean
    cols = ["resilience_composite", "dd_patent_range", "peer_dd"] + MAIN_CONTROLS
    d = standardise_in_sample(iv_sample.dropna(subset=cols), cols)
    d = d[["stkcd", "year"] + cols].copy()
    d = demean(d, cols, by="stkcd")
    d = demean(d, cols, by="year")
    d = d.dropna(subset=cols)

    # First stage: regress disparity on the instrument and exogenous controls
    Y_fs = d["dd_patent_range"].astype(float)
    X_fs = sm.add_constant(d[["peer_dd"] + MAIN_CONTROLS].astype(float))
    fs = sm.OLS(Y_fs, X_fs).fit(cov_type="cluster",
                                 cov_kwds={"groups": d["stkcd"]})
    iv_coef = fs.params["peer_dd"]
    iv_se = fs.bse["peer_dd"]
    f_stat = (iv_coef / iv_se) ** 2
    print("\nFirst stage (DV: dd_patent_range):")
    print(f"  peer_dd coefficient = {iv_coef:+.4f}  (SE={iv_se:.4f})")
    print(f"  First-stage F = {f_stat:.2f}  (Stock-Yogo weak-IV threshold = 10)")
    print(f"  N = {int(fs.nobs):,}")

    # Second stage via 2SLS on the demeaned series
    Y2 = d["resilience_composite"].astype(float)
    Xexog = sm.add_constant(d[MAIN_CONTROLS].astype(float))
    endog = d[["dd_patent_range"]].astype(float)
    instr = d[["peer_dd"]].astype(float)
    iv = IV2SLS(Y2, Xexog, endog, instr).fit(cov_type="clustered",
                                              clusters=d["stkcd"])
    coef = iv.params["dd_patent_range"]
    se = iv.std_errors["dd_patent_range"]
    pv = iv.pvalues["dd_patent_range"]
    print("\nSecond stage (DV: resilience_composite):")
    print(f"  Instrumented disparity coefficient = {coef:+.4f}  "
          f"(SE={se:.4f}, p={pv:.4f})")
    print(f"  N = {int(iv.nobs):,}")

    out = pd.DataFrame([
        {"Stage": "First stage",  "Coef": iv_coef, "SE": iv_se,
         "F_or_p": f"F={f_stat:.2f}", "N": int(fs.nobs)},
        {"Stage": "Second stage", "Coef": coef,    "SE": se,
         "F_or_p": f"p={pv:.4f}",     "N": int(iv.nobs)},
    ])
    out_dir = ensure_output_dir()
    out.to_csv(out_dir / "table3_iv.csv", index=False)
    print(f"\nSaved: {out_dir / 'table3_iv.csv'}")


if __name__ == "__main__":
    main()
