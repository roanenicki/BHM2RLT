"""
Téléchargement des incidents de piraterie / attaques de navires (NGA ASAM)
pour le golfe de Guinée, 2019 -> aujourd'hui.
À exécuter sur votre ordinateur :  python telecharger_asam.py
Prérequis : pip install requests pandas
Résultat  : asam_golfe_guinee.csv
"""
import requests
import pandas as pd

URL = "https://msi.nga.mil/api/publications/asam"
PARAMS = {"minOccurDate": "2019-01-01", "maxOccurDate": "2026-12-31",
          "sort": "date", "output": "json"}

# Zone : golfe de Guinée élargi (de la Côte d'Ivoire à l'Angola nord)
LAT_MIN, LAT_MAX = -8.0, 8.0
LON_MIN, LON_MAX = -10.0, 14.0

print("Téléchargement des incidents ASAM...")
r = requests.get(URL, params=PARAMS, timeout=120,
                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
print("Code réponse :", r.status_code)
r.raise_for_status()
data = r.json()

# L'API renvoie une liste d'incidents, éventuellement dans une clé "asam"
incidents = data.get("asam", data) if isinstance(data, dict) else data
df = pd.DataFrame(incidents)
print(f"{len(df)} incidents dans le monde depuis 2019")
print("Colonnes :", list(df.columns))

# Repérage des colonnes latitude / longitude quel que soit leur nom exact
lat = next(c for c in df.columns if c.lower() in ("latitude", "lat"))
lon = next(c for c in df.columns if c.lower() in ("longitude", "lon", "lng"))
df[lat] = pd.to_numeric(df[lat], errors="coerce")
df[lon] = pd.to_numeric(df[lon], errors="coerce")

gg = df[df[lat].between(LAT_MIN, LAT_MAX) & df[lon].between(LON_MIN, LON_MAX)]
gg.to_csv("asam_golfe_guinee.csv", index=False)
print(f"\nTerminé : {len(gg)} incidents dans le golfe de Guinée -> asam_golfe_guinee.csv")