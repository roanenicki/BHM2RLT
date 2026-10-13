"""
Modélisation : prédire la rupture portuaire du mois t+1 à partir des informations du mois t.

Protocole :
- Validation chronologique "walk-forward" annuelle : on entraîne sur toutes les années
  passées, on prédit l'année suivante (2022, 2023, 2024, 2025, puis 2026 = test final).
- 4 modèles : règle naïve (persistance), régression logistique, Random Forest, XGBoost.
- Test central de l'hypothèse : chaque modèle est entraîné DEUX fois,
    A = sans les chocs (état du port, Brent, Covid, contexte pays, saison)
    B = A + les chocs conflictuels (UCDP, ACLED, piraterie)
  Si B > A hors échantillon, les chocs apportent une information prédictive.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RNG = 42
P = pd.read_csv("/home/claude/data/panel_final.csv")
P["date"] = pd.to_datetime(P.date)
P = P.sort_values(["portid", "date"]).reset_index(drop=True)

# ------------------------------------------------------------------ variables de chocs
CHOCS = [c for c in P.columns if c.startswith(("ucdp_", "acled_", "pirat_")) and c != "ucdp_provisoire"]
g = P.groupby("portid")
for c in CHOCS:
    P[f"{c}_cum3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).sum())          # intensité sur 3 mois
    moy = g[c].transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
    P[f"{c}_rel"] = (P[c] - moy) / (moy + 1)                                                  # écart au niveau habituel du port
CHOCS_ALL = CHOCS + [f"{c}_cum3" for c in CHOCS] + [f"{c}_rel" for c in CHOCS]

# ------------------------------------------------------------------ variables "sans chocs"
for c in ["ecart_escales", "ecart_volume"]:
    P[f"{c}_lag1"] = g[c].shift(1)
P["rupture_lag1"] = g["rupture"].shift(1)
BASE = ["rupture", "rupture_lag1", "ecart_escales", "ecart_volume", "ecart_escales_lag1", "ecart_volume_lag1",
        "brent_moy", "brent_vol", "brent_var_1m", "covid_restrictions", "covid_frontieres",
        "ocde_risque_pays", "sipri_import_armes", "sipri_depmil_pib",
        "port_part_petroliers", "port_part_import_pays", "mois"]
P[BASE + CHOCS_ALL] = P[BASE + CHOCS_ALL].replace([np.inf, -np.inf], np.nan)


def modeles(pos_rate):
    return {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features="sqrt",
                                                class_weight="balanced_subsample", random_state=RNG, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                                 scale_pos_weight=(1 - pos_rate) / pos_rate, random_state=RNG, verbosity=0),
    }


def walk_forward(cible, feats, annees=(2022, 2023, 2024, 2025, 2026)):
    """Renvoie les prédictions hors échantillon de chaque modèle, année par année."""
    D = P.dropna(subset=[cible]).copy()
    out = []
    for an in annees:
        tr = D[D.date.dt.year < an]
        # on retire le dernier mois d'entraînement : sa cible tombe dans l'année testée
        tr = tr[tr.date < pd.Timestamp(f"{an}-01-01") - pd.offsets.MonthBegin(1)]
        te = D[D.date.dt.year == an]
        if te.empty or tr[cible].nunique() < 2:
            continue
        Xtr, Xte = tr[feats].fillna(tr[feats].median()), te[feats].fillna(tr[feats].median())
        pr = tr[cible].mean()
        res = te[["portid", "portname", "date", cible]].copy()
        res["annee_test"] = an
        res["Naïf (persistance)"] = te["rupture"].fillna(0).values if cible == "rupture" else \
            te[cible.replace("_t+1", "")].fillna(0).values
        for nom, mdl in modeles(pr).items():
            mdl.fit(Xtr, tr[cible])
            res[nom] = mdl.predict_proba(Xte)[:, 1]
            # seuil d'alerte : on signale la même proportion de mois que le taux de rupture d'entraînement
            seuil = np.quantile(mdl.predict_proba(Xtr)[:, 1], 1 - pr)
            res[nom + "__alerte"] = (res[nom] >= seuil).astype(int)
        out.append(res)
    return pd.concat(out)


def scores(res, cible, cols):
    y = res[cible].values
    rows = []
    for c in cols:
        p = res[c].values
        a = res[c + "__alerte"].values if c + "__alerte" in res else (p >= 0.5).astype(int)
        rows.append({"modèle": c,
                     "PR-AUC": average_precision_score(y, p) if len(np.unique(p)) > 2 else np.nan,
                     "ROC-AUC": roc_auc_score(y, p) if len(np.unique(p)) > 2 else np.nan,
                     "Rappel": recall_score(y, a), "Précision": precision_score(y, a, zero_division=0),
                     "F1": f1_score(y, a)})
    return pd.DataFrame(rows)


def boot_diff(res_a, res_b, cible, col, n=1000):
    """Intervalle de confiance bootstrap (par mois) du gain de PR-AUC apporté par les chocs."""
    rng = np.random.default_rng(RNG)
    a = res_a.set_index(["portid", "date"]); b = res_b.set_index(["portid", "date"]).loc[a.index]
    mois = a.index.get_level_values("date").unique()
    diffs = []
    for _ in range(n):
        s = rng.choice(mois, len(mois), replace=True)
        ia = a.loc[a.index.get_level_values("date").isin(s)]
        if ia[cible].nunique() < 2: continue
        ib = b.loc[ia.index]
        diffs.append(average_precision_score(ib[cible], ib[col]) - average_precision_score(ia[cible], ia[col]))
    return np.percentile(diffs, [2.5, 50, 97.5]), np.mean(np.array(diffs) > 0)


if __name__ == "__main__":
    MOD = ["Naïf (persistance)", "Régression logistique", "Random Forest", "XGBoost"]
    toutes = []
    for cible in ["rupture_t+1", "rupture_petrole_t+1", "rupture_conteneurs_t+1"]:
        base_feats = [f for f in BASE if f != "rupture"] + ([cible.replace("_t+1", "")] if cible != "rupture_t+1" else ["rupture"])
        ra = walk_forward(cible, base_feats)
        rb = walk_forward(cible, base_feats + CHOCS_ALL)
        sa = scores(ra, cible, MOD).assign(variables="A : sans chocs")
        sb = scores(rb, cible, MOD[1:]).assign(variables="B : avec chocs")
        s = pd.concat([sa, sb]).assign(cible=cible)
        s["taux_rupture"] = ra[cible].mean(); s["n"] = len(ra)
        toutes.append(s)
        print(f"\n=== {cible}  (n={len(ra)}, taux de rupture={ra[cible].mean():.3f}) ===")
        print(s.drop(columns=["cible"]).round(3).to_string(index=False))
        for m in MOD[1:]:
            ci, ppos = boot_diff(ra, rb, cible, m)
            print(f"  gain PR-AUC des chocs [{m}] : médiane {ci[1]:+.3f}  IC95% [{ci[0]:+.3f} ; {ci[2]:+.3f}]  P(gain>0)={ppos:.2f}")
        rb.to_csv(f"/home/claude/data/predictions_{cible.replace('+','plus')}.csv", index=False)
    pd.concat(toutes).to_csv("/home/claude/data/resultats_modeles.csv", index=False)
