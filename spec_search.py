# Specification search.
# Input : players_cross_section.csv
# Output: analysis_outputs/ comparison tables
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_outputs"); os.makedirs(OUT, exist_ok=True)
MIN_MINUTES = 900

d = pd.read_csv(os.path.join(HERE, "players_cross_section.csv"))
d = d[d["ig_followers"].notna() & (d["ig_followers"] > 0)].copy()
d["log_attention"] = np.log(d["ig_followers"])
d["log_trends"] = np.log(d["google_trends"].where(d["google_trends"] > 0))
d["log_club_ig"] = np.log(d["club_ig_followers"])
d = d[d["minutes"].notna() & (d["minutes"] >= MIN_MINUTES)]
outfield = d[d["position_group"] != "Goalkeeper"]


def cv_r2(dd, formula, dv, k=5):
    dd = dd.reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    y, yh = [], []
    for tr, te in kf.split(dd):
        m = smf.ols(formula, data=dd.iloc[tr]).fit()
        y += list(dd.iloc[te][dv]); yh += list(m.predict(dd.iloc[te]))
    y, yh = np.array(y), np.array(yh); ok = ~np.isnan(yh)
    return 1 - ((y[ok]-yh[ok])**2).sum() / ((y[ok]-y[ok].mean())**2).sum()


def evaluate(sub, preds, dv, label, sample, pos_dummies=True):
    dd = sub.dropna(subset=preds + [dv])
    f = f"{dv} ~ " + " + ".join(preds) + (" + C(position_group)" if pos_dummies else "")
    m = smf.ols(f, data=dd).fit(cov_type="HC3")
    sig = [p for p in m.pvalues[m.pvalues < 0.05].index
           if p != "Intercept" and not p.startswith("C(")]
    return {"sample": sample, "spec": label, "predictors": " + ".join(preds),
            "k": len(preds), "n": int(m.nobs), "SPV": round(m.nobs/len(preds), 1),
            "R2": round(m.rsquared, 3), "adjR2": round(m.rsquared_adj, 3),
            "CV_R2": round(cv_r2(dd, f, dv), 3),
            "significant": ", ".join(sig) if sig else "none"}


# Pooled models (followers)
BASE = ["G_plus_A", "team_league_position", "ucl_stage_ord", "log_club_ig"]
POOLED = {
    "Baseline predictors":            BASE + ["minutes"],
    "Baseline without minutes":       BASE,
    "Base + key passes":              BASE + ["key_passes"],
    "Base + shots":                   BASE + ["shots_pg"],
    "Base + dribbles":                BASE + ["dribbles"],
    "Base + long balls":              BASE + ["long_balls"],
    "Base + tackles and interceptions": BASE + ["tackles", "interceptions"],
    "Base + key passes and shots":    BASE + ["key_passes", "shots_pg"],
    "Base + key passes and long balls": BASE + ["key_passes", "long_balls"],
    "Base + key passes, shots and dribbles": BASE + ["key_passes", "shots_pg", "dribbles"],
    "Key passes, long balls and shots, G+A dropped (FINAL)":
        ["team_league_position", "ucl_stage_ord", "log_club_ig",
         "key_passes", "long_balls", "shots_pg"],
    "Full specification":             BASE + ["key_passes", "shots_pg", "dribbles",
                                              "long_balls", "tackles", "interceptions"],
}

rows = []
for label, preds in POOLED.items():
    rows.append(evaluate(d, preds, "log_attention", label, "all players"))
    rows.append(evaluate(outfield, preds, "log_attention", label, "outfield only"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_pooled_spec_comparison.csv"), index=False)
print(f"pooled followers: {len(rows)} specifications written")

# Position models (followers)
SUBSETS = {g: d[d["position_group"] == g] for g in ["Forward", "Midfielder", "Defender"]}
B3 = ["G_plus_A", "team_league_position", "log_club_ig"]
POSITION = {
 "Forward": {
    "Minutes, shots and dribbles":        B3 + ["minutes", "shots_pg", "dribbles"],
    "Key passes, shots and dribbles":     B3 + ["key_passes", "shots_pg", "dribbles"],
    "Dispossessed, shots and dribbles":   B3 + ["dispossessed", "shots_pg", "dribbles"],
    "Goals per 90, shots and dribbles":   B3 + ["goals_p90", "shots_pg", "dribbles"],
    "Shots and dribbles":                 B3 + ["shots_pg", "dribbles"],
    "Brand, key passes, shots, dribbles and fouled":
        ["log_club_ig", "key_passes", "shots_pg", "dribbles", "fouled"],
    "Brand, key passes, shots and fouled (FINAL)":
        ["log_club_ig", "key_passes", "shots_pg", "fouled"],
 },
 "Midfielder": {
    "Minutes, key passes, passes and long balls": B3 + ["minutes", "key_passes", "passes_pg", "long_balls"],
    "Pass accuracy, key passes, passes and long balls": B3 + ["pass_pct", "key_passes", "passes_pg", "long_balls"],
    "Dribbles, key passes, passes and long balls": B3 + ["dribbles", "key_passes", "passes_pg", "long_balls"],
    "Key passes and pass accuracy":       B3 + ["key_passes", "pass_pct"],
    "Key passes and dribbles":            B3 + ["key_passes", "dribbles"],
    "Key passes only":                    B3 + ["key_passes"],
    "G+A, brand, key passes and fouled (FINAL)":
        ["G_plus_A", "log_club_ig", "key_passes", "fouled"],
 },
 "Defender": {
    "Minutes, tackles and interceptions": B3 + ["minutes", "tackles", "interceptions"],
    "Long balls, tackles and interceptions": B3 + ["long_balls", "tackles", "interceptions"],
    "Clearances, tackles and interceptions": B3 + ["clearances", "tackles", "interceptions"],
    "Long balls and clearances":          B3 + ["long_balls", "clearances"],
    "Tackles and interceptions":          B3 + ["tackles", "interceptions"],
    "Long balls only":                    B3 + ["long_balls"],
    "G+A, brand, long balls and crosses (FINAL)":
        ["G_plus_A", "log_club_ig", "long_balls", "crosses"],
    "Brand, tackles and interceptions (no G+A)":
        ["log_club_ig", "tackles", "interceptions"],
    "Brand, long balls, crosses and blocks (no G+A)":
        ["log_club_ig", "long_balls", "crosses", "blocks"],
 },
}

rows = []
for pos, specs in POSITION.items():
    for label, preds in specs.items():
        rows.append(evaluate(SUBSETS[pos], preds, "log_attention", label, pos,
                             pos_dummies=False))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_position_spec_comparison.csv"), index=False)
print(f"position followers: {len(rows)} specifications written")

# Trends models
TRENDS = {
    "Baseline predictors":            BASE + ["minutes"],
    "Baseline without minutes":       BASE,
    "Follower final predictors":      ["team_league_position", "ucl_stage_ord", "log_club_ig",
                                       "key_passes", "long_balls", "shots_pg"],
    "Shots and long balls, G+A dropped (FINAL)":
                                      ["team_league_position", "ucl_stage_ord",
                                       "log_club_ig", "shots_pg", "long_balls"],
    "Full specification":             BASE + ["key_passes", "shots_pg", "dribbles",
                                              "long_balls", "tackles", "interceptions"],
    "Base + key passes":              BASE + ["key_passes"],
    "Base + shots":                   BASE + ["shots_pg"],
    "Base + dribbles":                BASE + ["dribbles"],
    "Base + long balls":              BASE + ["long_balls"],
    "Base + fouled":                  BASE + ["fouled"],
    "Base + crosses":                 BASE + ["crosses"],
    "Base + tackles":                 BASE + ["tackles"],
    "Base + interceptions":           BASE + ["interceptions"],
    "Base + passes":                  BASE + ["passes_pg"],
    "Base + pass accuracy":           BASE + ["pass_pct"],
    "Base + goals per 90":            BASE + ["goals_p90"],
    "Base + shots and fouled":        BASE + ["shots_pg", "fouled"],
    "Base + shots and crosses":       BASE + ["shots_pg", "crosses"],
    "Base + shots and dribbles":      BASE + ["shots_pg", "dribbles"],
    "Base + long balls and crosses":  BASE + ["long_balls", "crosses"],
    "Base + shots and long balls":    BASE + ["shots_pg", "long_balls"],
    "Base + shots, long balls and fouled":   BASE + ["shots_pg", "long_balls", "fouled"],
    "Base + shots, long balls and crosses":  BASE + ["shots_pg", "long_balls", "crosses"],
    "Base + shots, long balls and dribbles": BASE + ["shots_pg", "long_balls", "dribbles"],
    "Base + shots, long balls and key passes": BASE + ["shots_pg", "long_balls", "key_passes"],
    "Base + shots, long balls, tackles and interceptions":
                                      BASE + ["shots_pg", "long_balls", "tackles", "interceptions"],
}

rows = [evaluate(d, preds, "log_trends", label, "all players") for label, preds in TRENDS.items()]
rows += [evaluate(outfield, preds, "log_trends", label, "outfield only") for label, preds in TRENDS.items()]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_trends_spec_comparison.csv"), index=False)
print(f"trends: {len(rows)} specifications written")

# Engagement rate
eng = d[d["engagement_rate"].notna() & (d["engagement_rate"] > 0)].copy()
eng["log_engage"] = np.log(eng["engagement_rate"])
BASE_E = ["G_plus_A", "team_league_position", "ucl_stage_ord", "log_club_ig"]
ENGAGEMENT = {
    "Baseline predictors":            BASE_E + ["minutes"],
    "Baseline without minutes":       BASE_E,
    "Follower final predictors":      ["team_league_position", "ucl_stage_ord", "log_club_ig",
                                       "key_passes", "long_balls", "shots_pg"],
    "Trends final predictors":        ["team_league_position", "ucl_stage_ord",
                                       "log_club_ig", "shots_pg", "long_balls"],
    "Full specification":             BASE_E + ["key_passes", "shots_pg", "dribbles",
                                                "long_balls", "tackles", "interceptions"],
    "Base + key passes":              BASE_E + ["key_passes"],
    "Base + shots":                   BASE_E + ["shots_pg"],
    "Base + dribbles":                BASE_E + ["dribbles"],
    "Base + long balls":              BASE_E + ["long_balls"],
    "Base + fouled":                  BASE_E + ["fouled"],
    "Base + crosses":                 BASE_E + ["crosses"],
    "Base + tackles":                 BASE_E + ["tackles"],
    "Base + interceptions":           BASE_E + ["interceptions"],
    "Base + passes":                  BASE_E + ["passes_pg"],
    "Base + pass accuracy":           BASE_E + ["pass_pct"],
    "Base + goals per 90":            BASE_E + ["goals_p90"],
    "Base + shots and fouled":        BASE_E + ["shots_pg", "fouled"],
    "Base + shots and crosses":       BASE_E + ["shots_pg", "crosses"],
    "Base + shots and dribbles":      BASE_E + ["shots_pg", "dribbles"],
    "Base + long balls and crosses":  BASE_E + ["long_balls", "crosses"],
    "Base + shots and long balls":    BASE_E + ["shots_pg", "long_balls"],
    "Base + shots, long balls and fouled":   BASE_E + ["shots_pg", "long_balls", "fouled"],
    "Base + shots, long balls and crosses":  BASE_E + ["shots_pg", "long_balls", "crosses"],
    "Base + shots, long balls and dribbles": BASE_E + ["shots_pg", "long_balls", "dribbles"],
    "Base + shots, long balls and key passes": BASE_E + ["shots_pg", "long_balls", "key_passes"],
    "Base + shots, long balls, tackles and interceptions":
                                      BASE_E + ["shots_pg", "long_balls", "tackles", "interceptions"],
}

def evaluate_eng(sub, preds, label, sample):
    dd = sub.dropna(subset=preds + ["log_engage"])
    f = "log_engage ~ " + " + ".join(preds) + " + C(position_group)"
    m = smf.ols(f, data=dd).fit(cov_type="HC3")
    sig = [p for p in m.pvalues[m.pvalues < 0.05].index if p != "Intercept"]
    return {"sample": sample, "spec": label, "predictors": " + ".join(preds),
            "k": len(preds), "n": int(m.nobs), "SPV": round(m.nobs/len(preds), 1),
            "R2": round(m.rsquared, 3), "adjR2": round(m.rsquared_adj, 3),
            "CV_R2": round(cv_r2(dd, f, "log_engage"), 3),
            "significant": ", ".join(sig) if sig else "none"}

eng_out = eng[eng["position_group"] != "Goalkeeper"]
rows = [evaluate_eng(eng, preds, label, "all players") for label, preds in ENGAGEMENT.items()]
rows += [evaluate_eng(eng_out, preds, label, "outfield only") for label, preds in ENGAGEMENT.items()]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_engagement_spec_comparison.csv"), index=False)
best_cv = max(r["CV_R2"] for r in rows)
print(f"engagement: {len(rows)} specifications written (best CV R2 = {best_cv:.3f} — null finding)")
print("Done — comparison tables in analysis_outputs/.")
