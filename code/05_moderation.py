"""
Table 5 and Figure 5: Moderating effects.

Two boundary conditions are examined:

  Column 1 - Supplier geographic dispersion. Suppliers located across more
             provinces are less likely to be jointly affected by the same
             localised disruption, so dispersion should weaken the negative
             effect of disparity on resilience.

  Column 2 - SA index of financial constraints (Hadlock and Pierce, 2010).
             Higher SA = tighter constraints. Constrained firms operate
             with thinner relational slack, so widening disparity has less
             remaining redundancy to erode; the marginal effect is weaker.

  Column 3 - Leverage as an alternative financial-constraint proxy.

The interaction between disparity and the moderator is the parameter
of interest. The raw interaction (raw_x * raw_moderator) is formed first,
and every regressor (Y, X, moderator, interaction, controls) is then
z-scored using within-sample means and standard deviations.

Figure 5 plots the marginal effect of disparity at different levels of
each moderator, with 90 percent confidence intervals.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from utils import load_panel, standardise_in_sample, ensure_output_dir

MAIN_CONTROLS = ["size", "leverage", "roa"]


def fit_moderation(panel: pd.DataFrame, moderator: str, label: str) -> tuple:
    """Estimate disparity x moderator with firm and year fixed effects.

    Convention: form raw interaction first, then standardise everything
    together. This matches the moderation results reported in the paper.
    """
    inter = f"dd_x_{moderator}"
    df = panel.copy()
    df[inter] = df["dd_patent_range"] * df[moderator]

    # When the moderator IS leverage, drop leverage from the main controls
    # to avoid putting it in the regression twice.
    controls = [c for c in MAIN_CONTROLS if c != moderator]

    cols = ["resilience_composite", "dd_patent_range", moderator, inter] + controls
    d = standardise_in_sample(df.dropna(subset=cols), cols)
    d = d.set_index(["stkcd", "year"])

    Y = d["resilience_composite"].astype(float)
    X = d[["dd_patent_range", moderator, inter] + controls].astype(float)
    res = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit(
        cov_type="clustered", cluster_entity=True
    )

    print(f"\n  {label}")
    for v in ["dd_patent_range", moderator, inter]:
        print(f"    {v:36s}  coef={res.params[v]:+.4f}  SE={res.std_errors[v]:.4f}  p={res.pvalues[v]:.4f}")
    print(f"    {'N':36s}  {int(res.nobs):,}")

    return res


def plot_marginal_effects(res_geo, res_sa, save_to) -> None:
    """Plot the marginal effect of disparity at different moderator levels.

    The marginal effect is beta_disparity + beta_interaction * z, where z is
    the standardised moderator value. The variance of this combination is
    Var(beta_disp) + 2 * z * Cov(beta_disp, beta_inter) + z^2 * Var(beta_inter).
    """
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "font.size": 9.5, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    z90 = 1.645
    z = np.linspace(-2, 2, 50)

    for ax, res, mod_name, label in [
        (axes[0], res_geo, "dd_x_geo_dispersion_count",  "Geographic dispersion (std.)"),
        (axes[1], res_sa,  "dd_x_sa_index",              "SA index of financial constraint (std.)"),
    ]:
        bx = float(res.params["dd_patent_range"])
        bi = float(res.params[mod_name])
        cov = res.cov.loc[["dd_patent_range", mod_name], ["dd_patent_range", mod_name]].values
        m = bx + bi * z
        v = cov[0, 0] + 2 * z * cov[0, 1] + z**2 * cov[1, 1]
        s = np.sqrt(v)

        ax.fill_between(z, m - z90 * s, m + z90 * s, color="#cccccc", alpha=0.6)
        ax.plot(z, m, color="#1f4e79", linewidth=1.6)
        ax.axhline(0, color="#000", linewidth=0.5)
        ax.axvline(0, color="#888", linewidth=0.5, linestyle=":")
        ax.set_xlabel(label)
        ax.set_ylabel("Marginal effect of disparity")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def main():
    panel = load_panel("panel_main.csv")
    print(f"Sample: N={len(panel):,}")
    print("\nTable 5: Moderating effects")

    res_geo = fit_moderation(panel, "geo_dispersion_count", "(1) Geographic dispersion")
    res_sa  = fit_moderation(panel, "sa_index",            "(2) SA index")
    res_lev = fit_moderation(panel, "leverage",            "(3) Leverage")

    rows = []
    for label, res, mod in [
        ("(1) Geo dispersion",  res_geo, "geo_dispersion_count"),
        ("(2) SA index",        res_sa,  "sa_index"),
        ("(3) Leverage",        res_lev, "leverage"),
    ]:
        rows.append({
            "Spec": label,
            "DD": res.params["dd_patent_range"],
            "DD_SE": res.std_errors["dd_patent_range"],
            "Mod": res.params[mod],
            "Mod_SE": res.std_errors[mod],
            "DDxMod": res.params[f"dd_x_{mod}"],
            "DDxMod_SE": res.std_errors[f"dd_x_{mod}"],
            "DDxMod_p": res.pvalues[f"dd_x_{mod}"],
            "N": int(res.nobs),
        })
    out_dir = ensure_output_dir()
    pd.DataFrame(rows).to_csv(out_dir / "table5_moderation.csv", index=False)

    print("\nFigure 5: Marginal effects of disparity by moderator level")
    plot_marginal_effects(res_geo, res_sa, out_dir / "figure5_marginal_effects.png")

    print(f"\nSaved: {out_dir / 'table5_moderation.csv'}")
    print(f"Saved: {out_dir / 'figure5_marginal_effects.png'}")


if __name__ == "__main__":
    main()
