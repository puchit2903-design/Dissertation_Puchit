# Standardised coefficients (beta = B * SDx / SDy)
# Input : players_cross_section.csv
# Output: analysis_outputs/table_standardised_coefficients.csv
import os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
OUT = os.path.join(HERE, "analysis_outputs"); os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(HERE, "players_cross_section.csv"))
d = d[d["ig_followers"].notna() & (d["ig_followers"] > 0)].copy()
d["log_attention"] = np.log(d["ig_followers"])
d["log_club_ig"]   = np.log(d["club_ig_followers"])
d["log_trends"]    = np.log(d["google_trends"].where(d["google_trends"] > 0))
d = d[d["minutes"] >= 900]
out = d[d["position_group"] != "Goalkeeper"]
eng = d[d["engagement_rate"] > 0].copy(); eng["log_engage"] = np.log(eng["engagement_rate"])

BASE = "G_plus_A + minutes + team_league_position + ucl_stage_ord + log_club_ig + C(position_group)"
MODELS = {
 "Baseline":            (d,   "log_attention", "log_attention ~ " + BASE),
 "Final":               (out, "log_attention", "log_attention ~ team_league_position + ucl_stage_ord + log_club_ig + key_passes + long_balls + shots_pg + C(position_group)"),
 "Position: forwards":  (d[d.position_group=="Forward"],    "log_attention", "log_attention ~ log_club_ig + key_passes + shots_pg + fouled"),
 "Position: midfielders":(d[d.position_group=="Midfielder"],"log_attention", "log_attention ~ G_plus_A + log_club_ig + key_passes + fouled"),
 "Position: defenders": (d[d.position_group=="Defender"],   "log_attention", "log_attention ~ G_plus_A + log_club_ig + long_balls + crosses"),
 "Trends-optimised":    (out, "log_trends",    "log_trends ~ team_league_position + ucl_stage_ord + log_club_ig + shots_pg + long_balls + C(position_group)"),
 "Engagement rate":     (eng[eng["position_group"]!="Goalkeeper"], "log_engage",    "log_engage ~ G_plus_A + team_league_position + ucl_stage_ord + log_club_ig + passes_pg + C(position_group)"),
}

rows = []
for name, (data, dv, f) in MODELS.items():
    cols = [c for c in data.columns if c in f]
    dd = data.dropna(subset=[dv] + cols)
    m = smf.ols(f, data=dd).fit(cov_type="HC3")
    sy = dd[dv].std()
    for k in m.params.index:
        if k == "Intercept":
            continue
        # dummies are not standardised
        beta = "" if k.startswith("C(") else round(m.params[k] * dd[k].std() / sy, 3)
        rows.append({"model": name, "n": int(m.nobs), "predictor": k,
                     "B": round(m.params[k], 4), "SE": round(m.bse[k], 3),
                     "p": round(m.pvalues[k], 4), "beta": beta,
                     "model_R2": round(m.rsquared, 3),
                     "model_adjR2": round(m.rsquared_adj, 3),
                     "model_F_p": round(m.f_pvalue, 4)})

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "table_standardised_coefficients.csv"), index=False)
for name in MODELS:
    s = t[t.model == name]
    print(f"\n{name}  (n={s.n.iloc[0]}, adjR2={s.model_adjR2.iloc[0]})")
    print(s[["predictor", "B", "p", "beta"]].to_string(index=False))
