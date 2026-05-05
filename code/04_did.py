"""
Table 4 and Figure 4: Sanctions-based difference-in-differences.

Treatment is the first calendar year a focal firm appears on a major
U.S. trade or investment restriction list (Entity List or NS-CMIC).
The treated cohort comprises twenty-four A-share listed firms designated
between 2019 and 2024.

Sample. Dynamic-balanced sub-panel of the +Governance sample (Tables 1-2):
5,148 firms, 15,165 firm-years.

Specification. Two-way fixed-effects (firm and year) with lag(resilience)
and the main controls (Size, Leverage, ROA), following standard dynamic
panel practice (Holtz-Eakin, Newey, and Rosen 1988; Roth 2022). Standard
errors are clustered at the firm level. The dynamic event-study window
is t = -5 through t = +4, with t = -1 as the base period; treated firms
in the event study are those designated by 2021. Pooled DID retains all
twenty-four treated firms.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from utils import load_panel, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]
GOV_CONTROLS = ["top1_ownership", "board_size", "board_independence", "industry_hhi"]
EVENT_LEADS = (-5, -4, -3, -2)        # leads (with -1 as base)
EVENT_LAGS = (0, 1, 2, 3, 4)          # lags (event window restricted to t<=+4)


def winsorise_then_standardise(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        lo, hi = out[c].quantile(0.01), out[c].quantile(0.99)
        out[c] = out[c].clip(lo, hi)
        sd = out[c].std()
        if sd and sd > 0:
            out[c] = (out[c] - out[c].mean()) / sd
    return out


def restrict_to_event_window(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop firm-years beyond the event-study horizon.

    For treated firms designated by 2021, observations at rel_year >
    max(EVENT_LAGS) are dropped. Firms designated after 2021 keep all
    observations and are still treated in the pooled DID; only the dynamic
    event study restricts the cohort to designated-by-2021.
    """
    p = panel.copy()
    p["rel_year"] = p["year"] - p["first_sanction_year"]
    drop_mask = (p["first_sanction_year"].notna() &
                 (p["first_sanction_year"] <= 2021) &
                 (p["rel_year"] > max(EVENT_LAGS)))
    return p.loc[~drop_mask].copy()


def add_lag_Y(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a one-year lag of resilience_composite computed within firm."""
    p = panel.sort_values(["stkcd", "year"]).copy()
    p["lag_Y"] = p.groupby("stkcd")["resilience_composite"].shift(1)
    return p


def pooled_did(panel: pd.DataFrame, with_controls: bool, label: str) -> dict:
    """Pooled Treat * Post specification on the dynamic-balanced sub-panel."""
    panel = panel.copy()
    panel["post"] = ((panel["first_sanction_year"].notna()) &
                     (panel["year"] >= panel["first_sanction_year"])).astype(int)
    panel["treat_post"] = panel["post"]

    controls = ["lag_Y"] + (MAIN_CONTROLS if with_controls else [])
    cols = ["resilience_composite", "treat_post"] + controls
    d = panel.dropna(subset=cols).copy()
    d = winsorise_then_standardise(d, ["resilience_composite"] + controls)
    if with_controls and d["treat_post"].std() > 0:
        d["treat_post"] = (d["treat_post"] - d["treat_post"].mean()) / d["treat_post"].std()
    d = d.set_index(["stkcd", "year"])

    rhs = ["treat_post"] + controls
    X = d[rhs].astype(float)
    Y = d["resilience_composite"].astype(float)
    res = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    coef = res.params["treat_post"]; se = res.std_errors["treat_post"]
    p = res.pvalues["treat_post"]; n = int(res.nobs)
    print(f"  {label:20s}  Treat*Post = {coef:+.4f}  SE={se:.4f}  p={p:.4f}  N={n:,}")
    return {"Spec": label, "Coef": coef, "SE": se, "p": p, "N": n}


def event_study(panel: pd.DataFrame) -> tuple:
    """Dynamic event-study with leads/lags for t in [-5, +4].

    Restricts the treated cohort to firms designated by 2021 so the +4
    horizon is observed for all treated. Includes lag(resilience) and
    main controls. Returns the coefficient table plus the joint Wald
    test of zero pre-treatment leads.
    """
    panel = panel.copy()
    panel.loc[panel["first_sanction_year"] > 2021, "first_sanction_year"] = np.nan
    panel["rel_year"] = panel["year"] - panel["first_sanction_year"]
    panel["treated"] = panel["first_sanction_year"].notna().astype(int)

    indicators = []
    for k in EVENT_LEADS + EVENT_LAGS:
        v = f"D_{'m' if k < 0 else 'p'}{abs(k)}"
        panel[v] = ((panel["rel_year"] == k) & (panel["treated"] == 1)).astype(int)
        indicators.append(v)

    cols = ["resilience_composite"] + indicators + MAIN_CONTROLS + ["lag_Y"]
    d = panel.dropna(subset=cols).copy()
    d = winsorise_then_standardise(d, ["resilience_composite"] + MAIN_CONTROLS + ["lag_Y"])
    d = d.set_index(["stkcd", "year"])
    X = d[indicators + MAIN_CONTROLS + ["lag_Y"]].astype(float)
    Y = d["resilience_composite"].astype(float)
    res = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(cov_type="clustered", cluster_entity=True)

    pre_inds = [f"D_m{abs(k)}" for k in EVENT_LEADS]
    wald = res.wald_test(formula=" = 0, ".join(pre_inds) + " = 0")
    chi2, p_joint = float(wald.stat), float(wald.pval)

    rec = []
    for k in EVENT_LEADS + EVENT_LAGS:
        v = f"D_{'m' if k < 0 else 'p'}{abs(k)}"
        rec.append({"t": k,
                    "coef": float(res.params[v]),
                    "se": float(res.std_errors[v]),
                    "p": float(res.pvalues[v])})
    rec.append({"t": -1, "coef": 0.0, "se": 0.0, "p": np.nan})
    es = pd.DataFrame(rec).sort_values("t").reset_index(drop=True)
    return es, int(res.nobs), chi2, p_joint


def plot_event_study(df: pd.DataFrame, save_to) -> None:
    """Plot event-study coefficients with 90% CIs over t in [-5, +4]."""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times"],
                         "font.size": 9.5, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axvspan(-0.5, max(EVENT_LAGS) + 0.5, color="#f6f6f6", alpha=0.6)
    ax.axvline(x=-0.5, color="#999", linestyle=":", linewidth=0.7)
    ax.axhline(y=0, color="#000", linewidth=0.5)

    z90 = 1.645
    pre = df[df["t"] < 0]; base = df[df["t"] == -1]; post = df[df["t"] >= 0]
    ax.errorbar(pre["t"], pre["coef"], yerr=z90 * pre["se"], fmt="o",
                markersize=5, color="#555", markerfacecolor="white",
                ecolor="#888", capsize=2.2, label="Pre-treatment")
    ax.plot(base["t"], base["coef"], "o", markersize=6, color="black",
            label="Base (t = -1)")
    ax.errorbar(post["t"], post["coef"], yerr=z90 * post["se"], fmt="o",
                markersize=5, color="black", ecolor="black",
                capsize=2.2, label="Post-treatment")
    ax.plot(df["t"], df["coef"], color="#444", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Years relative to first U.S. sanction (t)")
    ax.set_ylabel("Effect on supply chain resilience")
    ax.set_xticks(range(min(EVENT_LEADS), max(EVENT_LAGS) + 1))
    ax.set_xlim(min(EVENT_LEADS) - 0.5, max(EVENT_LAGS) + 0.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def main():
    # Start from the +Governance sample, drop treated firm-years beyond t=+4,
    # and add lag(resilience) for the dynamic specification.
    panel = load_panel("panel_main.csv").dropna(subset=GOV_CONTROLS)
    panel = restrict_to_event_window(panel)
    panel = add_lag_Y(panel)
    print(f"DID dynamic-balanced sub-panel: N (after lag construction & "
          f"event-window restriction) reported with each regression below.")
    print(f"Base panel before lag dropna: N={len(panel):,}, "
          f"firms={panel['stkcd'].nunique():,}")

    print("\nTable 4: Pooled difference-in-differences")
    rows = [
        pooled_did(panel, with_controls=False, label="(1) Without controls"),
        pooled_did(panel, with_controls=True,  label="(2) With controls"),
    ]
    out_dir = ensure_output_dir()
    pd.DataFrame(rows).to_csv(out_dir / "table4_did.csv", index=False)

    print("\nFigure 4 / Appendix Table B2: Dynamic event-study coefficients "
          "(with 90% CI), event window t = -5 through t = +4")
    es, es_n, chi2, p_joint = event_study(panel)
    print(es.to_string(index=False))
    print(f"  N = {es_n:,}")
    print(f"  Joint Wald test (D_m5..D_m2 = 0): chi2 = {chi2:.3f}, p = {p_joint:.4f}")
    es.to_csv(out_dir / "figure4_event_study_coefs.csv", index=False)
    plot_event_study(es, out_dir / "figure4_event_study.png")

    print(f"\nSaved: {out_dir / 'table4_did.csv'}")
    print(f"Saved: {out_dir / 'figure4_event_study_coefs.csv'}")
    print(f"Saved: {out_dir / 'figure4_event_study.png'}")


if __name__ == "__main__":
    main()
