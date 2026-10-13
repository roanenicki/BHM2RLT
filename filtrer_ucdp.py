"""
Filtre les fichiers UCDP (GED, Candidate, VPP-GED) sur les 10 pays du golfe de Guinée
et la période 2019+, pour obtenir un fichier léger à envoyer.

Version 2 : cherche automatiquement les fichiers UCDP (CSV ou ZIP) dans
le dossier courant ET dans votre dossier Téléchargements.

Utilisation :  python filtrer_ucdp.py
Résultat    :  ucdp_golfe_guinee.csv (dans le dossier courant)
"""
import os
import glob
import zipfile
import pandas as pd

PAYS = ["Cameroon", "Nigeria", "Benin", "Togo", "Ghana", "Ivory Coast",
        "Côte d'Ivoire", "Cote d'Ivoire", "Gabon", "Congo",
        "Equatorial Guinea", "Sao Tome and Principe", "São Tomé and Príncipe"]

DOSSIERS = [os.getcwd(), os.path.join(os.path.expanduser("~"), "Downloads")]


def lire_ucdp(source, nom):
    """Lit un CSV ; renvoie les lignes filtrées, ou None si ce n'est pas un fichier UCDP."""
    try:
        df = pd.read_csv(source, low_memory=False)
    except Exception as e:
        print(f"  illisible : {nom} ({e})")
        return None
    if "country" not in df.columns or "date_start" not in df.columns:
        return None
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    garde = df[df["country"].isin(PAYS) & (df["date_start"] >= "2019-01-01")].copy()
    garde["fichier_source"] = nom
    print(f"  {nom} : {len(df)} lignes -> {len(garde)} gardées")
    return garde


morceaux, vus = [], set()
for dossier in DOSSIERS:
    print(f"Recherche dans : {dossier}")
    for chemin in sorted(glob.glob(os.path.join(dossier, "*"))):
        nom = os.path.basename(chemin)
        if nom in vus or nom.startswith("ucdp_golfe_guinee"):
            continue
        if nom.lower().endswith(".csv"):
            res = lire_ucdp(chemin, nom)
            if res is not None:
                morceaux.append(res); vus.add(nom)
        elif nom.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(chemin) as z:
                    for interne in z.namelist():
                        if interne.lower().endswith(".csv") and interne not in vus:
                            with z.open(interne) as f:
                                res = lire_ucdp(f, f"{nom}/{interne}")
                            if res is not None:
                                morceaux.append(res); vus.add(interne)
            except zipfile.BadZipFile:
                pass

if not morceaux:
    print("\nAucun fichier UCDP trouvé.")
    print("Vérifiez que les fichiers GED / Candidate (CSV ou ZIP) sont bien dans")
    print("le dossier Téléchargements ou dans le même dossier que ce script.")
else:
    res = pd.concat(morceaux, ignore_index=True)
    res.to_csv("ucdp_golfe_guinee.csv", index=False)
    print(f"\nTerminé : {len(res)} événements -> {os.path.abspath('ucdp_golfe_guinee.csv')}")
    print(res["country"].value_counts().to_string())