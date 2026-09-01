# Input : players_cross_section.csv + google_trends_monthly.csv
# Output: analysis_outputs/
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CROSS_FILE = os.path.join(HERE, "players_cross_section.csv")
OUT = os.path.join(HERE, "analysis_outputs"); os.makedirs(OUT, exist_ok=True)

DV = "ig_followers"; DV2 = "google_trends"; PERF = "G_plus_A"
NUMERIC = [PERF, "minutes", "team_league_position", "ucl_stage_ord"]
BRAND = "club_ig_followers"; CATEG = "position_group"; MIN_GROUP = 15
MIN_MINUTES = 900    # 10 full matches
SENS_MINUTES = 450   # sensitivity check



def load():
    print("\n[1] Load & prepare")
    assert os.path.exists(CROSS_FILE), "players_cross_section.csv not found beside this script."
    df = pd.read_csv(CROSS_FILE); print(f"  Loaded {CROSS_FILE}")
    df = df[df[DV].notna() & (df[DV] > 0)].copy()
    df["log_attention"] = np.log(df[DV])
    if BRAND in df.columns: df["log_club_ig"] = np.log(df[BRAND].replace(0, np.nan))
    df[CATEG] = df[CATEG].fillna("Unknown")
    print(f"  {len(df)} players")
    return df


def model_numeric(df):
    cols = [c for c in NUMERIC if c in df.columns]
    if "log_club_ig" in df.columns: cols = cols + ["log_club_ig"]
    return cols




def apply_minutes_threshold(df, min_minutes):
    # minutes filter
    has_min = df["minutes"].notna()
    kept = df[has_min & (df["minutes"] >= min_minutes)].copy()
    excl_low = df.loc[has_min & (df["minutes"] < min_minutes), ["player_name","minutes"]]
    excl_cov = df.loc[~has_min, "player_name"].tolist()
    print(f"  Threshold {min_minutes} min: kept {len(kept)} | excluded low-minutes: "
          f"{', '.join(f'{r.player_name} ({int(r.minutes)})' for r in excl_low.itertuples()) or 'none'}"
          + (f" | no performance coverage: {', '.join(excl_cov)}" if excl_cov else ""))
    pd.concat([excl_low.assign(reason="below threshold"),
               pd.DataFrame({"player_name": excl_cov, "minutes": np.nan, "reason": "no source coverage"})]
              ).to_csv(os.path.join(OUT, f"excluded_players_{min_minutes}min.csv"), index=False)
    return kept


def sensitivity_threshold(df_full, formula):
    # sensitivity check
    print(f"\n[5c] Sensitivity: re-estimation at {SENS_MINUTES} minutes")
    d = apply_minutes_threshold(df_full, SENS_MINUTES)
    m450 = smf.ols(formula, data=d).fit(cov_type="HC3")
    d900 = apply_minutes_threshold(df_full, MIN_MINUTES)
    m900 = smf.ols(formula, data=d900).fit(cov_type="HC3")
    comp = pd.DataFrame({
        "coef_900": m900.params.round(4), "p_900": m900.pvalues.round(4),
        "coef_450": m450.params.round(4), "p_450": m450.pvalues.round(4)})
    comp.loc["Model n"] = [int(m900.nobs), "", int(m450.nobs), ""]
    comp.loc["R2"] = [round(m900.rsquared,3), "", round(m450.rsquared,3), ""]
    comp.loc["Adjusted R2"] = [round(m900.rsquared_adj,3), "", round(m450.rsquared_adj,3), ""]
    comp.to_csv(os.path.join(OUT, "table_sensitivity_450.csv"))
    print(f"  n={int(m450.nobs)}  R2={m450.rsquared:.3f}  (900-minute model R2={m900.rsquared:.3f})")


def descriptives(df, num):
    print("\n[2] Descriptives + attention by position")
    cols = [DV, DV2, "engagement_rate", PERF, "key_passes", "long_balls", "shots_pg",
            "minutes", "team_league_position", "ucl_stage_ord"] + \
           (["log_club_ig"] if "log_club_ig" in df.columns else [])
    cols = [c for c in cols if c in df.columns]
    labels = {DV: "Instagram followers", DV2: "Google Trends score",
              "engagement_rate": "Engagement rate (%)", PERF: "Goals + assists",
              "key_passes": "Key passes (per game)", "long_balls": "Long balls (per game)",
              "shots_pg": "Shots (per game)", "minutes": "Minutes",
              "team_league_position": "League position", "ucl_stage_ord": "UCL stage (ordinal)",
              "log_club_ig": "Log club Instagram"}
    desc = df[cols].describe().T.round(2).rename(index=labels).rename(
        columns={"count":"n","50%":"median"})
    desc.to_csv(os.path.join(OUT, "table_descriptives.csv"))
    by = df.groupby(CATEG).agg(n=(DV,"size"), mean_followers=(DV,"mean"),
        median_followers=(DV,"median"), mean_trends=(DV2,"mean"), median_trends=(DV2,"median"))
    by[["mean_followers","median_followers"]] = by[["mean_followers","median_followers"]].round(0)
    by[["mean_trends","median_trends"]] = by[["mean_trends","median_trends"]].round(2)
    by.to_csv(os.path.join(OUT,"table_attention_by_position.csv")); print(by.to_string())
    order = df.groupby(CATEG)[DV].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.boxplot([np.log(df[df[CATEG]==g][DV]) for g in order], labels=list(order))
    ax.set_ylabel("log(Instagram followers)"); ax.set_title("Attention by position")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_attention_by_position.png"),dpi=150); plt.close(fig)
    return by


def vif_check(df, num):
    print("\n[3] Correlation + VIF")
    df[["log_attention"]+num].corr().round(3).to_csv(os.path.join(OUT,"table_correlation.csv"))
    X = sm.add_constant(df[num].dropna())
    v = pd.DataFrame({"variable":X.columns,
        "VIF":[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]})
    v = v[v.variable!="const"].round(2); v.to_csv(os.path.join(OUT,"table_vif.csv"), index=False)
    print(v.to_string(index=False))


def _coefs(m, fname):
    pd.DataFrame({"coef":m.params,"std_err":m.bse,"p_value":m.pvalues,
        "ci_low":m.conf_int()[0],"ci_high":m.conf_int()[1]}).round(4).to_csv(os.path.join(OUT,fname))


def main_regression(df, num):
    print("\n[4] BASELINE MODEL (pre-specified, all players)")
    formula = "log_attention ~ " + " + ".join(num) + f" + C({CATEG})"
    m = smf.ols(formula, data=df).fit()
    print(f"  R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")
    open(os.path.join(OUT,"model_baseline_summary.txt"),"w").write(str(m.summary()))
    _coefs(m,"table_baseline_coefficients.csv"); return m, formula


def diag_rows(m, label):
    from statsmodels.stats.stattools import durbin_watson
    r = m.resid; jb = jarque_bera(r); bp = het_breuschpagan(r, m.model.exog)
    cooks = OLSInfluence(m).cooks_distance[0]; thr = 4/len(cooks)
    std_r = r/r.std()
    return [{"model":label,"test":"Jarque-Bera","stat":round(jb[0],3),"p_or_note":round(jb[1],4)},
            {"model":label,"test":"Breusch-Pagan","stat":round(bp[0],3),"p_or_note":round(bp[1],4)},
            {"model":label,"test":"Durbin-Watson","stat":round(durbin_watson(r),3),"p_or_note":""},
            {"model":label,"test":"Max |std residual|","stat":round(abs(std_r).max(),2),"p_or_note":""},
            {"model":label,"test":"Cooks>4/n","stat":int((cooks>thr).sum()),"p_or_note":f">{thr:.3f}"}]


def diagnostics(m):
    print("\n[5] Diagnostics + robust SE")
    r = m.resid; jb = jarque_bera(r); bp = het_breuschpagan(r, m.model.exog)
    cooks = OLSInfluence(m).cooks_distance[0]; thr = 4/len(cooks)
    pd.DataFrame(diag_rows(m,"Baseline")).to_csv(
        os.path.join(OUT,"table_diagnostics.csv"), index=False)
    print(f"  normality p={jb[1]:.3f} | heterosced. p={bp[1]:.3f}")
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].scatter(m.fittedvalues,r,alpha=.6); ax[0].axhline(0,color="r"); ax[0].set_title("Residuals vs Fitted")
    sm.qqplot(r,line="45",fit=True,ax=ax[1]); ax[1].set_title("Q-Q")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_diagnostics.png"),dpi=150); plt.close(fig)
    return bp[1]


def refined(df, formula):
    print("\n[5b] Refined model (HC3 robust SE)")
    m = smf.ols(formula, data=df).fit(cov_type="HC3")
    open(os.path.join(OUT,"model_refined_summary.txt"),"w").write(str(m.summary()))
    _coefs(m,"table_refined_coefficients.csv"); return m




def final_model(df):
    # final model, outfield only
    print("\n[4b] FOLLOWER FINAL MODEL (CV-selected, outfield) — headline specification")
    d = df[df[CATEG] != "Goalkeeper"].copy()
    f = ("log_attention ~ team_league_position + ucl_stage_ord + log_club_ig"
         " + key_passes + long_balls + shots_pg + C(" + CATEG + ")")
    m = smf.ols(f, data=d).fit(cov_type="HC3")
    open(os.path.join(OUT, "model_final_summary.txt"), "w").write(str(m.summary()))
    _coefs(m, "table_final_model_coefficients.csv")
    dg = pd.read_csv(os.path.join(OUT,"table_diagnostics.csv"))
    pd.concat([dg, pd.DataFrame(diag_rows(m,"Follower final"))]).to_csv(
        os.path.join(OUT,"table_diagnostics.csv"), index=False)
    print(f"  n={int(m.nobs)}  R2={m.rsquared:.3f}  adjR2={m.rsquared_adj:.3f}  (CV R2 ~ 0.57)")
    jb_p = jarque_bera(m.resid)[1]
    bp_p = het_breuschpagan(m.resid, m.model.exog)[1]
    print(f"  diagnostics: normality p={jb_p:.3f} | heteroscedasticity p={bp_p:.3f} (HC3 reported)")
    ci = m.conf_int(); keep = [i for i in m.params.index if i != "Intercept" and not i.startswith("C(")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(m.params[keep], range(len(keep)),
                xerr=(m.params[keep]-ci.loc[keep,0], ci.loc[keep,1]-m.params[keep]),
                fmt="o", capsize=4)
    ax.axvline(0, ls="--", lw=1)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep)
    ax.set_xlabel("coefficient (log followers), 95% CI"); ax.set_title("Follower final model coefficients")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_final_coefficients.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(m.fittedvalues, m.model.endog, alpha=0.6)
    lo, hi = min(m.fittedvalues.min(), m.model.endog.min()), max(m.fittedvalues.max(), m.model.endog.max())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1)
    ax.set_xlabel("predicted log followers"); ax.set_ylabel("actual log followers")
    ax.set_title(f"Follower final model fit (R2={m.rsquared:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_final_fit.png"), dpi=150); plt.close(fig)
    return m

POSITION_SPECS = {
    "Forward":    ["log_club_ig", "key_passes", "shots_pg", "fouled"],
    "Midfielder": [PERF, "log_club_ig", "key_passes", "fouled"],
    "Defender":   [PERF, "log_club_ig", "long_balls", "crosses"],
}


def position_analysis(df, num):
    print("\n[6] Position analysis: tailored per-position models (HC3) + interaction")
    rows=[]
    for g, sub in df.groupby(CATEG):
        if len(sub) < MIN_GROUP:
            rows.append({"position":g,"n":len(sub),"note":"too small — descriptive only"}); continue
        preds = [c for c in POSITION_SPECS.get(g, [PERF,"team_league_position","log_club_ig"]) if c in sub.columns]
        try:
            mm = smf.ols("log_attention ~ "+" + ".join(preds), data=sub).fit(cov_type="HC3")
            sig = [p for p in mm.pvalues[mm.pvalues<0.05].index if p!="Intercept"]
            rows.append({"position":g,"n":len(sub),"k":len(preds),"SPV":round(len(sub)/len(preds),1),
                         "R2":round(mm.rsquared,3),"adjR2":round(mm.rsquared_adj,3),
                         "significant":", ".join(sig) if sig else "none"})
            open(os.path.join(OUT,f"model_position_{g}.txt"),"w").write(str(mm.summary()))
        except Exception as e:
            rows.append({"position":g,"n":len(sub),"note":str(e)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT,"table_position_models.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    try:
        inter = smf.ols(f"log_attention ~ {PERF}*C({CATEG}) + minutes", data=df).fit(cov_type="HC3")
        open(os.path.join(OUT,"model_interaction_perf_x_position.txt"),"w").write(str(inter.summary()))
        ip = {k:round(v,3) for k,v in inter.pvalues.items() if ":" in k}
        print("  interaction (perf x position) p-values:", ip)
    except Exception as e:
        print("  interaction skipped:", e)


def secondary_trends(df, num):
    print("\n[7] Secondary outcome: Google Trends (log)")
    if df[DV2].notna().sum() < 10: print("  too few Trends values — skipped"); return
    d = df[df[DV2].notna() & (df[DV2] > 0)].copy(); d["log_trends"] = np.log(d[DV2])
    m = smf.ols("log_trends ~ "+" + ".join(num)+f" + C({CATEG})", data=d).fit(cov_type="HC3")
    open(os.path.join(OUT,"model_trends_summary.txt"),"w").write(str(m.summary()))
    print(f"  primary spec:  n={int(m.nobs)}  R2={m.rsquared:.3f}  adjR2={m.rsquared_adj:.3f}")
    do = d[d[CATEG] != "Goalkeeper"]
    f = ("log_trends ~ team_league_position + ucl_stage_ord"
         " + log_club_ig + key_passes + long_balls + shots_pg + C(" + CATEG + ")")
    m2 = smf.ols(f, data=do.dropna(subset=["key_passes","long_balls","shots_pg"])).fit(cov_type="HC3")
    open(os.path.join(OUT,"model_trends_finalspec_summary.txt"),"w").write(str(m2.summary()))
    print(f"  follower final spec: n={int(m2.nobs)}  R2={m2.rsquared:.3f}  adjR2={m2.rsquared_adj:.3f}"
          f"  (key_passes p={m2.pvalues.get('key_passes', float('nan')):.3f}, shots p={m2.pvalues.get('shots_pg', float('nan')):.3f})")
    f3 = ("log_trends ~ team_league_position + ucl_stage_ord"
          " + log_club_ig + shots_pg + long_balls + C(" + CATEG + ")")
    m3 = smf.ols(f3, data=do.dropna(subset=["shots_pg","long_balls"])).fit(cov_type="HC3")
    open(os.path.join(OUT,"model_trends_optimised_summary.txt"),"w").write(str(m3.summary()))
    _coefs(m3, "table_trends_coefficients.csv")
    print(f"  trends final spec (shots+lb): n={int(m3.nobs)}  R2={m3.rsquared:.3f}  adjR2={m3.rsquared_adj:.3f}"
          f"  (shots p={m3.pvalues.get('shots_pg', float('nan')):.3f})")




def engagement_model(df, num):
    # engagement rate
    print("\n[7b] Secondary outcome: engagement rate")
    if "engagement_rate" not in df.columns or df["engagement_rate"].notna().sum() < 20:
        print("  engagement_rate not collected / too few values — skipped"); return
    d = df[df["engagement_rate"] > 0].copy()
    d["log_engage"] = np.log(d["engagement_rate"])
    # best-performing specification from the engagement search (see spec_search.py)
    f = ("log_engage ~ " + PERF + " + team_league_position + ucl_stage_ord"
         " + log_club_ig + passes_pg + C(" + CATEG + ")")
    d = d[d[CATEG] != "Goalkeeper"].dropna(subset=["passes_pg"])
    m = smf.ols(f, data=d).fit(cov_type="HC3")
    open(os.path.join(OUT, "model_engagement_summary.txt"), "w").write(str(m.summary()))
    _coefs(m, "table_engagement_coefficients.csv")
    print(f"  n={int(m.nobs)}  R2={m.rsquared:.3f}")
    both = d[["ig_followers","engagement_rate"]].dropna()
    if len(both) > 10:
        r = np.corrcoef(np.log(both["ig_followers"]), np.log(both["engagement_rate"]))[0,1]
        print(f"  corr(log followers, log engagement) = {r:.3f}  (reach vs resonance)")
        pd.DataFrame([{"n": len(both), "sample": "outfield, engagement model",
                       "correlation_log_followers_log_engagement": round(r, 3)}]
                     ).to_csv(os.path.join(OUT, "table_reach_resonance_correlation.csv"), index=False)


def clustering(df):
    # player segmentation
    print("\n[8] Player segmentation (attention x match rating)")
    d0 = df[df["Whoscored_Rating"].notna()].copy()
    if "log_attention" not in d0.columns or len(d0) < 20:
        print("  skipped"); return None

    def fit(d, perf):
        d = d.copy(); d["perf_z"] = perf
        X = StandardScaler().fit_transform(d[["log_attention", "perf_z"]].fillna(0))
        d["cluster"] = KMeans(4, random_state=42, n_init=10).fit_predict(X)
        a = d["log_attention"].median()
        cent = d.groupby("cluster")[["log_attention", "perf_z"]].mean()
        names = {c: ("Top Stars" if r["log_attention"] >= a and r["perf_z"] >= 0
                     else "Fan Favourites" if r["log_attention"] >= a
                     else "Underrated Performers" if r["perf_z"] >= 0 else "Squad Players")
                 for c, r in cent.iterrows()}
        d["group"] = d["cluster"].map(names)
        return d

    # raw rating
    raw = fit(d0, (d0["Whoscored_Rating"] - d0["Whoscored_Rating"].mean())
                   / d0["Whoscored_Rating"].std())
    mix_raw = raw.groupby(["group", CATEG]).size().unstack(fill_value=0)
    print("  (a) raw rating — comparison only (attacker bias: forwards mean "
          f"{d0.loc[d0[CATEG]=='Forward','Whoscored_Rating'].mean():.2f} vs goalkeepers "
          f"{d0.loc[d0[CATEG]=='Goalkeeper','Whoscored_Rating'].mean():.2f}):")
    print(mix_raw.to_string())
    if "Underrated Performers" not in mix_raw.index:
        print("  -> high-performing defender/goalkeeper group cannot form under raw ratings")

    # within position
    within = fit(d0, d0.groupby(CATEG)["Whoscored_Rating"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() else 1)))
    print("\n  (b) rating standardised WITHIN position — used for reporting:")
    summ = within.groupby("cluster")[["log_attention", "perf_z"]].mean().round(2)
    summ["n"] = within.groupby("cluster").size()
    summ["group"] = summ.index.map(within.groupby("cluster")["group"].first())
    print(summ.to_string())
    summ.to_csv(os.path.join(OUT, "table_segmentation.csv"))
    mix_raw.to_csv(os.path.join(OUT, "table_segmentation_raw_comparison.csv"))

    extra = [c for c in ["club_2024_25", "league_2024_25", "nationality", "Whoscored_Rating"]
             if c in within.columns]
    cols = ["player_name", CATEG] + extra + ["log_attention", "perf_z", "group"]
    within[cols].to_csv(os.path.join(OUT, "table_player_groups.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for g, sub in within.groupby("group"):
        ax.scatter(sub["perf_z"], sub["log_attention"], alpha=0.7, s=38, label=f"{g} (n={len(sub)})")
    ax.axvline(0, ls="--", lw=1, color="grey"); ax.axhline(within["log_attention"].median(), ls="--", lw=1, color="grey")
    ax.set_xlabel("performance relative to position (standardised match rating)")
    ax.set_ylabel("log(Instagram followers)")
    ax.set_title("Player segmentation: attention vs within-position performance")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_segmentation.png"), dpi=150); plt.close(fig)

    if "league_2024_25" in within.columns:
        league_mix = (within.groupby(["group", "league_2024_25"]).size()
                      .unstack(fill_value=0))
        league_mix.to_csv(os.path.join(OUT, "table_segmentation_league_mix.csv"))
        print("\n  League composition by group -> table_segmentation_league_mix.csv")
    return summ


def trends_monthly_analysis():
    # monthly Trends and transfer DiD
    print("\n[9] Attention dynamics: monthly Google Trends (optional)")
    mfile = os.path.join(HERE, "google_trends_monthly.csv")
    if not os.path.exists(mfile):
        print("  no google_trends_monthly.csv — skipped"); return
    m = pd.read_csv(mfile)
    m["month"] = pd.to_datetime(m["month"])
    m["trends_rel"] = pd.to_numeric(m["trends_rel"], errors="coerce")

    avg = m.groupby("month")["trends_rel"].mean()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(avg.index, avg.values, marker="o")
    ax.set_title("Average search interest by month (relative units)")
    ax.set_ylabel("mean Trends (vs reference)"); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_trends_monthly.png"), dpi=150); plt.close(fig)

    cross = pd.read_csv(CROSS_FILE)
    flag = next((c for c in ["mid_season_tranfers","mid_season_transfers","split_season"] if c in cross.columns), None)
    if flag is None:
        print("  no mid-season flag in cross-section — DiD on Trends skipped"); return
    t = m.merge(cross[["player_name", flag]], on="player_name", how="left")
    t["treated"] = pd.to_numeric(t[flag], errors="coerce").fillna(0).astype(int)
    t["post"] = (t["month"] >= pd.Timestamp("2025-01-01")).astype(int)
    t = t.dropna(subset=["trends_rel"])
    if t["treated"].sum() == 0 or t["treated"].nunique() < 2:
        print("  no treated (mid-season movers) rows — DiD on Trends skipped"); return
    t["log_t"] = np.log(t["trends_rel"].clip(lower=0.001))
    dd = smf.ols("log_t ~ treated + post + treated:post", data=t).fit(
        cov_type="cluster", cov_kwds={"groups": t["player_name"]})
    eff = dd.params.get("treated:post", np.nan)
    open(os.path.join(OUT, "model_did_midseason_summary.txt"), "w").write(str(dd.summary()))
    print(f"  DiD A — mid-season movers only (n={int(t['treated'].sum()>0 and t.loc[t.treated==1,'player_name'].nunique())}, "
          f"post=Jan 2025): {eff:.4f} log-points (~{(np.exp(eff)-1)*100:.1f}%), p={dd.pvalues.get('treated:post', np.nan):.3f}")

    # DiD B: all 2025 transfers
    if "transferred" in cross.columns:
        t2 = m.merge(cross[["player_name", "transferred"]], on="player_name", how="left")
        t2["treated"] = pd.to_numeric(t2["transferred"], errors="coerce").fillna(0).astype(int)
        t2["post"] = (t2["month"] >= pd.Timestamp("2025-05-01")).astype(int)
        t2 = t2.dropna(subset=["trends_rel"]); t2 = t2[t2["trends_rel"] > 0]
        if t2["treated"].sum() > 0 and t2["treated"].nunique() > 1:
            t2["log_t"] = np.log(t2["trends_rel"])
            dd2 = smf.ols("log_t ~ treated * post", data=t2).fit(
                cov_type="cluster", cov_kwds={"groups": t2["player_name"]})
            eff2 = dd2.params.get("treated:post", np.nan)
            open(os.path.join(OUT, "model_did_alltransfers_summary.txt"), "w").write(str(dd2.summary()))
            print(f"  DiD B — all 2025 transfers (n={t2.loc[t2.treated==1,'player_name'].nunique()}, "
                  f"post=May-Jul 2025 window): {eff2:.4f} log-points "
                  f"(~{(np.exp(eff2)-1)*100:.1f}%), p={dd2.pvalues.get('treated:post', np.nan):.4f}")
            grp = (t2.groupby(["month", "treated"])["trends_rel"].mean().unstack())
            fig, ax = plt.subplots(figsize=(9, 4.5))
            if 0 in grp.columns: ax.plot(grp.index, grp[0], marker="o", label="Did not transfer")
            if 1 in grp.columns: ax.plot(grp.index, grp[1], marker="o", label="Transferred in 2025")
            ax.axvline(pd.Timestamp("2025-05-01"), ls="--", color="grey", lw=1)
            ax.text(pd.Timestamp("2025-05-01"), ax.get_ylim()[1]*0.97, " transfer window",
                    fontsize=8, color="grey", va="top")
            ax.set_ylabel("mean search interest (relative units)"); ax.set_xlabel("")
            ax.set_title("Search interest: transferred vs non-transferred players")
            ax.legend(fontsize=9)
            fig.autofmt_xdate()
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_did_movers.png"), dpi=150); plt.close(fig)

    # per-player search interest around each transfer window
    cs = pd.read_csv(CROSS_FILE)
    recs = []
    for _, row in cs[(cs.get("transferred", 0) == 1) | (cs.get("mid_season_tranfers", 0) == 1)].iterrows():
        sub = m[m["player_name"] == row["player_name"]].dropna(subset=["trends_rel"])
        if not len(sub): continue
        for lbl, cut, a, b in [("Mid-season", "2025-01-01", "club_season_start", "club_2024_25"),
                               ("Summer", "2025-05-01", "club_2024_25", "club_2025_26")]:
            flag = "mid_season_tranfers" if lbl == "Mid-season" else "transferred"
            if row.get(flag, 0) != 1: continue
            pre = sub[sub.month < cut]["trends_rel"].mean()
            post = sub[sub.month >= cut]["trends_rel"].mean()
            recs.append({"player_name": row["player_name"], "window": lbl,
                         "previous_club": row.get(a), "new_club": row.get(b),
                         "pre_event": round(pre, 3), "post_event": round(post, 3),
                         "ratio": round(post / pre, 2) if pre else None})
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "table_transfer_player_changes.csv"), index=False)



def report(df, by, main, ref, final):
    print("\n[10] Summary report")
    L=["DISSERTATION ANALYSIS — SUMMARY","="*40,f"Players: {len(df)}","",
       f"FOLLOWER FINAL MODEL (CV-selected, outfield): n={int(final.nobs)}  "
       f"R2={final.rsquared:.3f}  adjR2={final.rsquared_adj:.3f}  — headline specification",
       f"Baseline model (pre-specified, all players): R2={main.rsquared:.3f} (adj {main.rsquared_adj:.3f}; HC3 inference)",
       "","Attention by position (median followers):"]
    for g,r in by.iterrows(): L.append(f"  {g}: n={int(r['n'])}, median={int(r['median_followers']):,}")
    open(os.path.join(OUT,"RESULTS_SUMMARY.txt"),"w").write("\n".join(L)); print("\n".join(L))


def main():
    print("="*60+"\nRUNNING DISSERTATION ANALYSIS\n"+"="*60)
    df_full=load(); num=model_numeric(df_full)
    by=descriptives(df_full,num)
    print(f"\n[3b] Applying inclusion threshold ({MIN_MINUTES} minutes)")
    df=apply_minutes_threshold(df_full, MIN_MINUTES)
    vif_check(df,num)
    main_m,formula=main_regression(df,num)
    diagnostics(main_m); ref=refined(df,formula)
    sensitivity_threshold(df_full, formula)
    final_m = final_model(df)
    position_analysis(df,num); secondary_trends(df,num); engagement_model(df,num)
    clustering(df); trends_monthly_analysis(); report(df,by,main_m,ref,final_m)


if __name__ == "__main__":
    main()
