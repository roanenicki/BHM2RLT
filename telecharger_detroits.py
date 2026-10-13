"""
Téléchargement du trafic quotidien des détroits (IMF PortWatch) :
Ormuz, Bab el-Mandeb, Suez, Cap de Bonne-Espérance, Gibraltar, etc.
À exécuter sur votre ordinateur :  python telecharger_detroits.py
Prérequis : pip install requests pandas
Résultat  : detroits_liste.csv et detroits_quotidien.csv
"""
import os
import time
import requests
import pandas as pd

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
PAYS = ["CMR", "NGA", "BEN", "TGO", "GHA", "CIV", "GAB", "COG", "GNQ", "STP"]
SEUIL_NAVIRES = 20      # on ignore les terminaux quasi inactifs (moins de 20 navires)
TAILLE_PAGE = 1000
DOSSIER = "portwatch_detroits"
ESSAIS = 5


def requete(url, params):
    """Requête avec reprises automatiques en cas d'erreur serveur (504, etc.)."""
    for essai in range(1, ESSAIS + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as e:
            attente = 10 * essai
            print(f"    erreur ({e}) - nouvel essai {essai}/{ESSAIS} dans {attente} s")
            time.sleep(attente)
    raise RuntimeError("Échec après plusieurs essais")


def interroger(service, where, champs="*", tri=None):
    """Récupère toutes les lignes d'une couche ArcGIS, page par page."""
    url = f"{BASE}/{service}/FeatureServer/0/query"
    lignes, offset = [], 0
    while True:
        params = {"where": where, "outFields": champs, "resultOffset": offset,
                  "resultRecordCount": TAILLE_PAGE, "f": "json"}
        if tri:
            params["orderByFields"] = tri
        data = requete(url, params)
        lot = [f["attributes"] for f in data.get("features", [])]
        lignes.extend(lot)
        if not data.get("exceededTransferLimit") and len(lot) < TAILLE_PAGE:
            break
        offset += len(lot)
        time.sleep(0.5)
    return pd.DataFrame(lignes)


# 1) Liste des 28 détroits
print("Téléchargement de la liste des détroits...")
det = interroger("PortWatch_chokepoints_database", "1=1")
det.to_csv("detroits_liste.csv", index=False)
print(det[["portid", "portname"]].to_string(index=False))

# 2) Trafic quotidien, détroit par détroit
os.makedirs(DOSSIER, exist_ok=True)
for pid, nom in zip(det["portid"], det["portname"]):
    fichier = os.path.join(DOSSIER, f"{pid}.csv")
    if os.path.exists(fichier):
        print(f"{nom} : déjà téléchargé"); continue
    print(f"{nom} ({pid})...")
    df = interroger("Daily_Chokepoints_Data", f"portid = '{pid}'", tri="date")
    df.to_csv(fichier, index=False); print(f"    {len(df)} jours"); time.sleep(1)

tout = pd.concat([pd.read_csv(os.path.join(DOSSIER, f)) for f in os.listdir(DOSSIER) if f.endswith(".csv")], ignore_index=True)
if "date" in tout.columns and pd.api.types.is_numeric_dtype(tout["date"]):
    tout["date"] = pd.to_datetime(tout["date"], unit="ms", errors="coerce")
tout.to_csv("detroits_quotidien.csv", index=False)
print(f"Terminé : {len(tout)} lignes. Période : {tout['date'].min()} -> {tout['date'].max()}")
print("Envoyez à Claude : detroits_quotidien.csv et detroits_liste.csv")
