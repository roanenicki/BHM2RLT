"""
Téléchargement des données IMF PortWatch pour les ports des cas de fermeture (monde).
À exécuter sur votre ordinateur :  python telecharger_portwatch.py
Prérequis :  pip install requests pandas

Version 2 : téléchargement port par port, avec reprises automatiques.
Si le script s'arrête, relancez-le : les ports déjà téléchargés sont conservés.

Produit :
  - ports_golfe_guinee.csv            : liste des ports (coordonnées, profil des navires)
  - portwatch_ports/<portid>.csv      : activité quotidienne, un fichier par port
  - activite_quotidienne_ports.csv    : tous les ports réunis
"""
import os
import time
import requests
import pandas as pd

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
PAYS = ["SEN", "GIN", "MRT", "UKR", "ISR", "LBY", "HTI", "PAK", "SDN", "DJI", "EGY", "SAU",
        "ARE", "IRN", "QAT", "KWT", "IRQ", "OMN", "BHR", "YEM", "ROU", "TUR"]
SEUIL_NAVIRES = 20      # on ignore les terminaux quasi inactifs (moins de 20 navires)
TAILLE_PAGE = 1000
DOSSIER = "portwatch_ports_monde"
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


# 1) Liste des ports
liste_pays = ",".join(f"'{p}'" for p in PAYS)
print("Téléchargement de la liste des ports...")
ports = interroger("PortWatch_ports_database", f"ISO3 IN ({liste_pays})")
ports.to_csv("ports_monde.csv", index=False)
retenus = ports[ports["vessel_count_total"] >= SEUIL_NAVIRES] \
    .sort_values("vessel_count_total", ascending=False)
print(f"{len(ports)} ports trouvés, {len(retenus)} retenus (>= {SEUIL_NAVIRES} navires)\n")

# 2) Activité quotidienne, port par port
os.makedirs(DOSSIER, exist_ok=True)
for i, (pid, nom) in enumerate(zip(retenus["portid"], retenus["portname"]), 1):
    fichier = os.path.join(DOSSIER, f"{pid}.csv")
    if os.path.exists(fichier):
        print(f"[{i}/{len(retenus)}] {nom} : déjà téléchargé")
        continue
    print(f"[{i}/{len(retenus)}] {nom} ({pid})...")
    df = interroger("Daily_Ports_Data", f"portid = '{pid}'", tri="date")
    df.to_csv(fichier, index=False)
    print(f"    {len(df)} jours")
    time.sleep(1)

# 3) Fusion
print("\nFusion des fichiers...")
tout = pd.concat(
    [pd.read_csv(os.path.join(DOSSIER, f)) for f in os.listdir(DOSSIER) if f.endswith(".csv")],
    ignore_index=True)
if "date" in tout.columns and pd.api.types.is_numeric_dtype(tout["date"]):
    tout["date"] = pd.to_datetime(tout["date"], unit="ms", errors="coerce")
tout.to_csv("activite_ports_monde.csv", index=False)

print(f"Terminé : {len(tout)} lignes, {tout['portid'].nunique()} ports.")
if "date" in tout.columns:
    print(f"Période couverte : {tout['date'].min()} -> {tout['date'].max()}")
print("Colonnes :", list(tout.columns))
print("\nEnvoyez à Claude : ports_monde.csv et activite_ports_monde.csv")
