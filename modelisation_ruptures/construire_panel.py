"""
Construction du jeu de données final : 1 ligne = 1 port x 1 mois.
Y  : rupture portuaire (PortWatch)
X  : chocs (UCDP, ACLED, OMI piraterie), contrôles (Brent), contexte (OCDE, SIPRI)
Règle anti-fuite : toutes les variables X décrivent le mois t ou avant ;
les variables annuelles (SIPRI) utilisent l'année précédente.
"""
import glob, re
import numpy as np
import pandas as pd

D = "/home/claude/data/"
OUT = "/home/claude/data/panel_final.csv"


def dist(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- Y : PortWatch
d = pd.read_csv(D + "activite.csv", parse_dates=["date"])
d = d[d.date < "2026-10-01"]
cols = [c for c in d.columns if c.startswith(("portcalls", "import", "export"))]
m = d.groupby(["portid", "portname", "ISO3", pd.Grouper(key="date", freq="MS")])[cols].sum().reset_index()
m["volume"] = m["import"] + m["export"]
keep = m.groupby("portid").portcalls.mean()
m = m[m.portid.isin(keep[keep >= 15].index)].sort_values(["portid", "date"]).copy()


def base(s):  # médiane des 12 mois PRÉCÉDENTS (aucune info future)
    return s.shift(1).rolling(12, min_periods=6).median()


g = m.groupby("portid")
for c in ["portcalls", "volume", "portcalls_tanker", "portcalls_container"]:
    m[f"base_{c}"] = g[c].transform(base)
m["rupture"] = ((m.portcalls < 0.8 * m.base_portcalls) | (m.volume < 0.7 * m.base_volume)).astype(float)
m["rupture_petrole"] = (m.portcalls_tanker < 0.7 * m.base_portcalls_tanker).astype(float)
m["rupture_conteneurs"] = (m.portcalls_container < 0.7 * m.base_portcalls_container).astype(float)
for c in ["rupture", "rupture_petrole", "rupture_conteneurs"]:
    m.loc[m.base_portcalls.isna(), c] = np.nan
# flux trop faibles pour juger une rupture par type
m.loc[m.base_portcalls_tanker < 5, "rupture_petrole"] = np.nan
m.loc[m.base_portcalls_container < 5, "rupture_conteneurs"] = np.nan
m["ecart_escales"] = m.portcalls / m.base_portcalls - 1
m["ecart_volume"] = m.volume / m.base_volume - 1
m = m.drop(columns=[c for c in m.columns if c.startswith("base_")])

ports = pd.read_csv(D + "ports.csv").set_index("portid").loc[m.portid.unique()]
rows = []

# ---------------------------------------------------------------- X1 : UCDP
u = pd.read_csv(D + "ucdp.csv", low_memory=False, parse_dates=["date_start"])
u = u.drop_duplicates(["date_start", "latitude", "longitude", "best", "side_a", "side_b"])
u = u[u.where_prec <= 3]
u["date"] = u.date_start.dt.to_period("M").dt.to_timestamp()
u["candidate"] = (u.fichier_source != "GEDEvent_v26_1.csv").astype(int)
TV = {1: "etat", 2: "nonetat", 3: "unilaterale"}

# ---------------------------------------------------------------- X2 : ACLED (agrégé)
a = pd.concat([pd.read_csv(f) for f in glob.glob(D + "acled/*.csv")]).drop_duplicates()
a["date"] = pd.to_datetime(a.week).dt.to_period("M").dt.to_timestamp()
ET = {"Battles": "combats", "Violence against civilians": "violence_civils",
      "Explosions/Remote violence": "explosions", "Riots": "emeutes",
      "Protests": "manifestations", "Strategic developments": "strategique"}
a["cat"] = a.event_type.map(ET)

# ---------------------------------------------------------------- X3 : piraterie (OMI)
p = pd.read_csv(D + "imo_piraterie.csv", parse_dates=["Date"]).dropna(subset=["lat"])
p["date"] = p.Date.dt.to_period("M").dt.to_timestamp()
p["enlevement"] = p["Consequences for crew etc"].astype(str).str.contains("kidnap|hostage", case=False).astype(int)

for pid, pt in ports.iterrows():
    # UCDP
    du = dist(pt.lat, pt.lon, u.latitude.values, u.longitude.values)
    for r in [50, 100, 250]:
        s = u[du <= r].groupby("date").agg(ev=("id", "size"), morts=("best", "sum"))
        s.columns = [f"ucdp_{c}_{r}km" for c in s.columns]
        s["portid"] = pid; rows.append(s.reset_index())
    s = u[du <= 100].pivot_table(index="date", columns="type_of_violence", values="id", aggfunc="size", fill_value=0)
    s.columns = [f"ucdp_{TV[c]}_100km" for c in s.columns]; s["portid"] = pid; rows.append(s.reset_index())
    # ACLED : régions administratives dont le centre est à moins de 150 km du port
    da = dist(pt.lat, pt.lon, a.centroid_latitude.values, a.centroid_longitude.values)
    s = a[da <= 150].pivot_table(index="date", columns="cat", values="events", aggfunc="sum", fill_value=0)
    s.columns = [f"acled_{c}" for c in s.columns]
    s2 = a[da <= 150].groupby("date").fatalities.sum().rename("acled_morts")
    s = s.join(s2, how="outer"); s["portid"] = pid; rows.append(s.reset_index())
    # Piraterie
    dp = dist(pt.lat, pt.lon, p.lat.values, p.lon.values)
    for r in [100, 250]:
        s = p[dp <= r].groupby("date").agg(n=("lat", "size"), enl=("enlevement", "sum"))
        s.columns = [f"pirat_incidents_{r}km", f"pirat_enlevements_{r}km"]
        s["portid"] = pid; rows.append(s.reset_index())

X = pd.concat(rows).groupby(["portid", "date"]).sum().reset_index()
P = m.merge(X, on=["portid", "date"], how="left")
xcols = [c for c in X.columns if c not in ("portid", "date")]
P[xcols] = P[xcols].fillna(0)

# indicateur : mois couvert par les données UCDP provisoires (Candidate)
P["ucdp_provisoire"] = (P.date >= "2026-01-01").astype(int)
P.loc[P.date > "2026-08-01", [c for c in xcols if c.startswith("ucdp_")]] = np.nan  # UCDP s'arrête fin août 2026

# ---------------------------------------------------------------- Contrôle : Brent
b = pd.read_csv(D + "brent.csv", parse_dates=["observation_date"]).dropna()
b["date"] = b.observation_date.dt.to_period("M").dt.to_timestamp()
bm = b.groupby("date").DCOILBRENTEU.agg(brent_moy="mean", brent_vol="std").reset_index()
bm["brent_var_1m"] = bm.brent_moy.pct_change()
P = P.merge(bm, on="date", how="left")

# ---------------------------------------------------------------- Contexte : OCDE
o = pd.read_csv(D + "ocde_risque_pays.csv", parse_dates=["debut"])
o["note"] = pd.to_numeric(o.note, errors="coerce")
o = o.dropna().sort_values("debut")
P = P.sort_values("date")
res = []
for iso, g2 in P.groupby("ISO3"):
    oo = o[o.iso == iso][["debut", "note"]].rename(columns={"debut": "date", "note": "ocde_risque_pays"})
    res.append(pd.merge_asof(g2.sort_values("date"), oo, on="date", direction="backward"))
P = pd.concat(res)

# ---------------------------------------------------------------- Contexte : SIPRI (année précédente)
tiv = pd.read_csv(D + "sipri_tiv.csv", skiprows=9, nrows=9)
tiv = tiv.melt(id_vars="Recipient", value_vars=[str(y) for y in range(2019, 2026)],
               var_name="annee", value_name="sipri_import_armes")
tiv["sipri_import_armes"] = pd.to_numeric(tiv.sipri_import_armes.astype(str).str.strip(), errors="coerce").fillna(0)
tiv["annee"] = tiv.annee.astype(int) + 1  # valeur de l'année N utilisée en N+1
mx = pd.read_excel(D + "milex.xlsx", "Share of GDP", header=None)
hr = mx.index[mx.iloc[:, 0].astype(str).str.strip() == "Country"][0]
mx = pd.read_excel(D + "milex.xlsx", "Share of GDP", header=hr)
mx["Country"] = mx.Country.astype(str).str.strip()
mx = mx.melt(id_vars="Country", value_vars=[c for c in mx.columns if str(c).isdigit() and 2017 <= int(c) <= 2025],
             var_name="annee", value_name="sipri_depmil_pib")
mx["annee"] = mx.annee.astype(int) + 1
mx["sipri_depmil_pib"] = pd.to_numeric(mx.sipri_depmil_pib, errors="coerce") * 100
NOM = {"BEN": ("Benin", "Benin"), "CMR": ("Cameroon", "Cameroon"), "COG": ("Congo", "Congo, Republic"),
       "CIV": ("Cote d'Ivoire", "Cote d'Ivoire"), "GNQ": ("Equatorial Guinea", "Equatorial Guinea"),
       "GAB": ("Gabon", "Gabon"), "GHA": ("Ghana", "Ghana"), "NGA": ("Nigeria", "Nigeria"), "TGO": ("Togo", "Togo")}
P["annee"] = P.date.dt.year
P["_tiv"] = P.ISO3.map(lambda i: NOM[i][0]); P["_mx"] = P.ISO3.map(lambda i: NOM[i][1])
P = P.merge(tiv, left_on=["_tiv", "annee"], right_on=["Recipient", "annee"], how="left")
P = P.merge(mx, left_on=["_mx", "annee"], right_on=["Country", "annee"], how="left")
P["sipri_import_armes"] = P.sipri_import_armes.fillna(0)
P = P.drop(columns=["_tiv", "_mx", "Recipient", "Country"])

# ---------------------------------------------------------------- Choc sanitaire : Covid (OxCGRT)
cv = pd.read_csv(D + "covid_oxcgrt.csv", parse_dates=["date"])
# Guinée équatoriale absente d'OxCGRT : moyenne de ses voisins CEMAC (Cameroun, Gabon)
gnq = cv[cv.CountryCode.isin(["CMR", "GAB"])].groupby("date")[["covid_restrictions", "covid_frontieres"]].mean().reset_index()
gnq["CountryCode"] = "GNQ"
cv = pd.concat([cv, gnq]).rename(columns={"CountryCode": "ISO3"})
cv["date"] = cv.date.dt.strftime("%Y-%m")
P["_d"] = P.date.dt.strftime("%Y-%m")
P = P.merge(cv.rename(columns={"date": "_d"}), on=["ISO3", "_d"], how="left").drop(columns="_d")
P[["covid_restrictions", "covid_frontieres"]] = P[["covid_restrictions", "covid_frontieres"]].fillna(0)  # hors 2020-2022 : pas de restrictions

# ---------------------------------------------------------------- Caractéristiques fixes du port
pc = pd.read_csv(D + "ports.csv").set_index("portid")
P["port_part_petroliers"] = P.portid.map(pc.vessel_count_tanker / pc.vessel_count_total)
P["port_part_import_pays"] = P.portid.map(pc.share_country_maritime_import)
P["mois"] = P.date.dt.month

# ---------------------------------------------------------------- Cibles futures
P = P.sort_values(["portid", "date"])
for c in ["rupture", "rupture_petrole", "rupture_conteneurs"]:
    P[f"{c}_t+1"] = P.groupby("portid")[c].shift(-1)

P["date"] = P.date.dt.strftime("%Y-%m")
P.to_csv(OUT, index=False)
print("Panel final :", P.shape, "| ports :", P.portid.nunique(), "| mois :", P.date.min(), "->", P.date.max())
print("Ruptures (t+1) :", int(P["rupture_t+1"].sum()), "sur", int(P["rupture_t+1"].notna().sum()))
print("Ruptures pétrole :", int(P["rupture_petrole_t+1"].sum()), "/", int(P["rupture_petrole_t+1"].notna().sum()),
      "| conteneurs :", int(P["rupture_conteneurs_t+1"].sum()), "/", int(P["rupture_conteneurs_t+1"].notna().sum()))
print("Valeurs manquantes par colonne (>0) :")
na = P.isna().sum(); print(na[na > 0].to_string())
