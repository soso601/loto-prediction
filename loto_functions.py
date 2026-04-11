"""
Récupération des tirages du Loto français.
- Scraping depuis le site historique
- Sauvegarde/chargement CSV pour fiabilité
- Mise à jour incrémentale
"""

import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from config import *


def scrap_loto_numbers():
    """
    Scrape les tirages depuis le site historique.
    Retourne un DataFrame avec les colonnes : day, month_year, num0-num4, chance
    """
    loto_url = "http://loto.akroweb.fr/loto-historique-tirages/"

    try:
        page = requests.get(loto_url, timeout=15)
        page.raise_for_status()
    except requests.RequestException as e:
        print(f"[Scraping] Erreur de connexion : {e}")
        return None

    soup = BeautifulSoup(page.text, 'html.parser')
    body = soup.find('table')

    if body is None:
        print("[Scraping] Impossible de trouver le tableau des tirages.")
        return None

    tirage_lines = body.find_all('tr')
    my_list = []

    for value in tirage_lines:
        try:
            res = value.text.split('\n')
            my_dict = {
                'day': res[2].strip(),
                'month_year': res[3].strip(),
            }
            for i, val in enumerate(res[5:10]):
                my_dict[f'num{i}'] = int(val.strip())
            my_dict['chance'] = int(res[10].strip())
            my_list.append(my_dict)
        except (ValueError, IndexError) as e:
            continue  # Ignorer les lignes malformées

    if not my_list:
        print("[Scraping] Aucun tirage trouvé.")
        return None

    df = pd.DataFrame(my_list)
    print(f"[Scraping] {len(df)} tirages récupérés avec succès.")
    return df


def save_draws_to_csv(df, path=None):
    """Sauvegarde les tirages dans un fichier CSV."""
    if path is None:
        path = DRAWS_CSV_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[CSV] {len(df)} tirages sauvegardés dans {path}")


def load_draws_from_csv(path=None):
    """Charge les tirages depuis un fichier CSV."""
    if path is None:
        path = DRAWS_CSV_PATH
    if not os.path.exists(path):
        print(f"[CSV] Fichier {path} introuvable.")
        return None
    df = pd.read_csv(path)
    print(f"[CSV] {len(df)} tirages chargés depuis {path}")
    return df


def get_draws(force_scrape=False):
    """
    Récupère les tirages : scraping si demandé ou si pas de CSV, sinon CSV.
    Sauvegarde automatiquement après scraping.
    """
    if not force_scrape:
        df = load_draws_from_csv()
        if df is not None:
            return df

    print("[Data] Scraping des tirages en cours...")
    df = scrap_loto_numbers()
    if df is not None:
        save_draws_to_csv(df)
        return df

    # Fallback sur le CSV si le scraping échoue
    df = load_draws_from_csv()
    if df is not None:
        print("[Data] Utilisation du CSV existant comme fallback.")
        return df

    raise RuntimeError("Impossible de récupérer les tirages (ni scraping, ni CSV).")


def update_draws(new_draws_df):
    """
    Met à jour le fichier CSV avec de nouveaux tirages.
    Évite les doublons en se basant sur la date.
    Retourne le DataFrame complet mis à jour.
    """
    existing = load_draws_from_csv()

    if existing is None:
        save_draws_to_csv(new_draws_df)
        return new_draws_df

    # Concaténation et suppression des doublons
    combined = pd.concat([new_draws_df, existing], ignore_index=True)
    combined = combined.drop_duplicates(subset=['month_year'], keep='first')
    save_draws_to_csv(combined)
    nb_new = len(combined) - len(existing)
    print(f"[Update] {nb_new} nouveaux tirages ajoutés. Total : {len(combined)}")
    return combined


def add_single_draw(day, month_year, nums, chance):
    """
    Ajoute un seul tirage manuellement.
    nums: liste de 5 entiers
    chance: entier
    """
    assert len(nums) == 5, "Il faut exactement 5 numéros"
    assert all(LOTO_MIN_NUM <= n <= LOTO_MAX_NUM for n in nums), f"Numéros entre {LOTO_MIN_NUM} et {LOTO_MAX_NUM}"
    assert LOTO_MIN_CHANCE <= chance <= LOTO_MAX_CHANCE, f"Chance entre {LOTO_MIN_CHANCE} et {LOTO_MAX_CHANCE}"

    new_row = pd.DataFrame([{
        'day': day,
        'month_year': month_year,
        'num0': nums[0], 'num1': nums[1], 'num2': nums[2],
        'num3': nums[3], 'num4': nums[4],
        'chance': chance,
    }])

    return update_draws(new_row)
