"""
Fonctions utilitaires et feature engineering pour le projet Loto.
Inclut les features originales + nouvelles features enrichies.
"""

import numpy as np
import pandas as pd
from config import *

# ──────────────────────────────────────────────
# Listes de référence
# ──────────────────────────────────────────────
PAIRS = list(range(2, 51, 2))
IMPAIRS = list(range(1, 50, 2))


# ══════════════════════════════════════════════
# FEATURES ORIGINALES (corrigées)
# ══════════════════════════════════════════════

def is_under(data, number):
    """Nombre de numéros (sur les 5) inférieurs ou égaux à `number`."""
    return sum((data[c] <= number).astype(int) for c in NUM_COLS)

def is_pair(data):
    """Nombre de numéros pairs parmi les 5."""
    return sum((data[c].isin(PAIRS)).astype(int) for c in NUM_COLS)

def is_impair(data):
    """Nombre de numéros impairs parmi les 5."""
    return sum((data[c].isin(IMPAIRS)).astype(int) for c in NUM_COLS)

def is_pair_etoile(data):
    """Le numéro chance est-il pair ? (0/1)"""
    return (data[CHANCE_COL].isin(PAIRS)).astype(int)

def is_impair_etoile(data):
    """Le numéro chance est-il impair ? (0/1)"""
    return (data[CHANCE_COL].isin(IMPAIRS)).astype(int)

def sum_diff(data):
    """Somme des différences au carré entre numéros consécutifs."""
    return sum(
        (data[NUM_COLS[i+1]] - data[NUM_COLS[i]])**2
        for i in range(len(NUM_COLS)-1)
    )

def freq_val(data, column):
    """Fréquence cumulative d'apparition de chaque valeur."""
    tab = data[column].values.tolist()
    freqs = []
    for pos, e in enumerate(tab, 1):
        freqs.append(tab[:pos].count(e))
    return freqs


# ══════════════════════════════════════════════
# NOUVELLES FEATURES ENRICHIES
# ══════════════════════════════════════════════

def somme_tirage(data):
    """Somme des 5 numéros du tirage (ex: 1+13+24+35+46 = 119)."""
    return sum(data[c] for c in NUM_COLS)

def somme_chance_incluse(data):
    """Somme des 5 numéros + le numéro chance."""
    return sum(data[c] for c in ALL_DRAW_COLS)

def reduction_numerologique(n):
    """Réduit un nombre à un seul chiffre (1-9). Ex: 24 → 2+4=6, 19 → 1+9=10 → 1+0=1."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n

def numerologie_cols(data):
    """Calcule la réduction numérologique de chaque numéro."""
    result = pd.DataFrame(index=data.index)
    for c in NUM_COLS:
        result[f'numer_{c}'] = data[c].apply(reduction_numerologique)
    result['numer_chance'] = data[CHANCE_COL].apply(reduction_numerologique)
    return result

def somme_numerologique(data):
    """Somme des réductions numérologiques des 5 numéros."""
    return sum(data[c].apply(reduction_numerologique) for c in NUM_COLS)

def reduction_somme_tirage(data):
    """Réduction numérologique de la somme totale du tirage."""
    return somme_tirage(data).apply(reduction_numerologique)

def ecart_entre_tirages(data):
    """
    Écart (différence absolue) entre chaque numéro et sa valeur au tirage précédent.
    Retourne un DataFrame avec les colonnes ecart_num0..ecart_num4, ecart_chance.
    """
    result = pd.DataFrame(index=data.index)
    for c in ALL_DRAW_COLS:
        result[f'ecart_{c}'] = data[c].diff().abs().fillna(0)
    return result

def ecart_moyen_tirage(data):
    """Écart moyen entre numéros consécutifs dans un même tirage."""
    ecarts = [data[NUM_COLS[i+1]] - data[NUM_COLS[i]] for i in range(len(NUM_COLS)-1)]
    return sum(ecarts) / len(ecarts)

def moyennes_mobiles(data, windows=None):
    """
    Moyennes mobiles de chaque numéro sur différentes fenêtres.
    Pas de look-ahead : on utilise .shift(1) pour ne prendre que les tirages passés.
    """
    if windows is None:
        windows = MOVING_AVG_WINDOWS
    result = pd.DataFrame(index=data.index)
    for c in ALL_DRAW_COLS:
        shifted = data[c].shift(1)  # Décalage pour éviter le look-ahead
        for w in windows:
            result[f'ma_{c}_{w}'] = shifted.rolling(window=w, min_periods=1).mean()
    return result

def entropie_tirage(data):
    """
    Entropie de Shannon du tirage (mesure de "désordre" dans la répartition).
    Calculée sur la distribution des 5 numéros normalisés par 49.
    """
    def calc_entropy(row):
        nums = [row[c] / LOTO_MAX_NUM for c in NUM_COLS]
        total = sum(nums)
        if total == 0:
            return 0
        probs = [n / total for n in nums]
        return -sum(p * np.log2(p + 1e-10) for p in probs)
    return data.apply(calc_entropy, axis=1)

def entropie_glissante(data, window=10):
    """Entropie moyenne sur une fenêtre glissante (tendance du désordre)."""
    ent = entropie_tirage(data)
    return ent.shift(1).rolling(window=window, min_periods=1).mean()

def amplitude_tirage(data):
    """Différence entre le plus grand et le plus petit des 5 numéros."""
    return data[NUM_COLS].max(axis=1) - data[NUM_COLS].min(axis=1)

def decade_distribution(data):
    """
    Nombre de numéros dans chaque dizaine (1-10, 11-20, 21-30, 31-40, 41-49).
    """
    result = pd.DataFrame(index=data.index)
    decades = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 49)]
    for low, high in decades:
        col_name = f'decade_{low}_{high}'
        result[col_name] = sum(
            ((data[c] >= low) & (data[c] <= high)).astype(int)
            for c in NUM_COLS
        )
    return result

def retard_numeros(data):
    """
    Pour chaque tirage, calcule le retard moyen des 5 numéros
    (nombre de tirages depuis leur dernière apparition).
    """
    retards = []
    historique = {n: -1 for n in range(1, LOTO_MAX_NUM + 1)}

    for idx, (_, row) in enumerate(data.iterrows()):
        nums = [int(row[c]) for c in NUM_COLS]
        retard_list = []
        for n in nums:
            if historique[n] == -1:
                retard_list.append(idx)  # Jamais apparu avant
            else:
                retard_list.append(idx - historique[n])
        retards.append(np.mean(retard_list))
        for n in nums:
            historique[n] = idx

    return pd.Series(retards, index=data.index)

def consecutifs(data):
    """Nombre de paires de numéros consécutifs dans le tirage."""
    def count_consecutive(row):
        nums = sorted([int(row[c]) for c in NUM_COLS])
        return sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    return data.apply(count_consecutive, axis=1)


# ══════════════════════════════════════════════
# NOUVELLES FEATURES V2 (jour, mois, terminaisons, etc.)
# ══════════════════════════════════════════════

def terminaisons(data):
    """Chiffre des unités de chaque numéro."""
    result = pd.DataFrame(index=data.index)
    for c in NUM_COLS:
        result[f'term_{c}'] = data[c] % 10
    result['term_chance'] = data[CHANCE_COL] % 10
    return result

def nb_terminaisons_identiques(data):
    """Nombre de numéros qui partagent la même terminaison."""
    def count_same_term(row):
        terms = [int(row[c]) % 10 for c in NUM_COLS]
        from collections import Counter
        c = Counter(terms)
        return max(c.values())  # Le max de doublons
    return data.apply(count_same_term, axis=1)

def tendance_recente(data, window=20):
    """
    Pour chaque numéro, fréquence glissante sur les N derniers tirages
    vs fréquence globale. Ratio > 1 = en hausse, < 1 = en baisse.
    Sans look-ahead (shift).
    """
    result = pd.DataFrame(index=data.index)
    for c in NUM_COLS:
        # Fréquence glissante (sur les window derniers)
        shifted = data[c].shift(1)
        for n in [1, 10, 25]:  # mini, moyen, historique
            col_name = f'hot_{c}_{n}'
            # Combien de fois la valeur actuelle est apparue dans les N derniers tirages
            vals = data[c].values
            hot_scores = []
            for idx in range(len(vals)):
                start = max(0, idx - n)
                recent = vals[start:idx]  # Pas d'inclusion du courant
                hot_scores.append(np.sum(recent == vals[idx]) if len(recent) > 0 else 0)
            result[col_name] = hot_scores
    return result

def retard_par_numero(data):
    """
    Pour chaque tirage, le retard de chaque numéro tiré
    (combien de tirages depuis sa dernière apparition).
    """
    result = pd.DataFrame(index=data.index)
    for c in NUM_COLS:
        retards = []
        last_seen = {}
        for idx, val in enumerate(data[c].values):
            if val in last_seen:
                retards.append(idx - last_seen[val])
            else:
                retards.append(idx)  # Jamais vu avant
            last_seen[val] = idx
        result[f'retard_{c}'] = retards
    return result

def repetitions_avec_precedent(data):
    """Nombre de numéros en commun avec le tirage précédent."""
    reps = [0]  # Premier tirage = 0
    for i in range(1, len(data)):
        curr = set(int(data.iloc[i][c]) for c in NUM_COLS)
        prev = set(int(data.iloc[i-1][c]) for c in NUM_COLS)
        reps.append(len(curr & prev))
    return pd.Series(reps, index=data.index)

def encode_jour(df_full):
    """
    Encode le jour de la semaine en numérique.
    Lundi=1, Mercredi=3, Samedi=6, etc.
    """
    jour_map = {'Lundi': 1, 'Mardi': 2, 'Mercredi': 3, 'Jeudi': 4,
                'Vendredi': 5, 'Samedi': 6, 'Dimanche': 7}
    if 'day' in df_full.columns:
        return df_full['day'].map(jour_map).fillna(0).astype(int)
    return pd.Series(0, index=df_full.index)

def encode_mois(df_full):
    """Encode le mois en numérique (1-12)."""
    mois_map = {
        'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
        'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
        'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
    }
    if 'month_year' in df_full.columns:
        def get_month(ds):
            parts = str(ds).strip().split()
            if len(parts) >= 2:
                return mois_map.get(parts[1].lower(), 0)
            return 0
        return df_full['month_year'].apply(get_month)
    return pd.Series(0, index=df_full.index)

def ratio_position(data):
    """
    Pour chaque numéro, sa position relative dans l'intervalle [1,49].
    num/49 donne une idée de si c'est un petit ou grand numéro.
    """
    result = pd.DataFrame(index=data.index)
    for c in NUM_COLS:
        result[f'pos_ratio_{c}'] = data[c] / LOTO_MAX_NUM
    return result


# ══════════════════════════════════════════════
# ASSEMBLAGE DE TOUTES LES FEATURES
# ══════════════════════════════════════════════

def build_all_features(df, df_full=None):
    """
    Construit l'ensemble complet des features à partir du DataFrame brut des tirages.
    df: DataFrame avec num0-num4, chance
    df_full: DataFrame avec day, month_year, num0-num4, chance (optionnel, pour jour/mois)
    """
    result = df[ALL_DRAW_COLS].copy()

    # --- Features originales ---
    result['freq_num0'] = freq_val(df, 'num0')
    result['freq_num1'] = freq_val(df, 'num1')
    result['freq_num2'] = freq_val(df, 'num2')
    result['freq_num3'] = freq_val(df, 'num3')
    result['freq_num4'] = freq_val(df, 'num4')
    result['freq_chance'] = freq_val(df, 'chance')
    result['sum_diff'] = sum_diff(df)
    result['pair_chance'] = is_pair_etoile(df)
    result['impair_chance'] = is_impair_etoile(df)
    result['pair'] = is_pair(df)
    result['impair'] = is_impair(df)
    result['is_under_24'] = is_under(df, 24)
    result['is_under_40'] = is_under(df, 40)

    # --- Sommes ---
    result['somme_tirage'] = somme_tirage(df)
    result['somme_totale'] = somme_chance_incluse(df)

    # --- Numérologie ---
    numer = numerologie_cols(df)
    for c in numer.columns:
        result[c] = numer[c]
    result['somme_numer'] = somme_numerologique(df)
    result['reduction_somme'] = reduction_somme_tirage(df)

    # --- Écarts entre tirages ---
    ecarts = ecart_entre_tirages(df)
    for c in ecarts.columns:
        result[c] = ecarts[c]
    result['ecart_moyen'] = ecart_moyen_tirage(df)

    # --- Moyennes mobiles (sans look-ahead) ---
    ma = moyennes_mobiles(df)
    for c in ma.columns:
        result[c] = ma[c]

    # --- Entropie ---
    result['entropie'] = entropie_tirage(df)
    result['entropie_glissante'] = entropie_glissante(df)

    # --- Statistiques du tirage ---
    result['amplitude'] = amplitude_tirage(df)
    result['retard_moyen'] = retard_numeros(df)
    result['consecutifs'] = consecutifs(df)

    # --- Distribution par dizaine ---
    decades = decade_distribution(df)
    for c in decades.columns:
        result[c] = decades[c]

    # ═══ NOUVELLES FEATURES V2 ═══

    # --- Terminaisons ---
    terms = terminaisons(df)
    for c in terms.columns:
        result[c] = terms[c]
    result['nb_term_identiques'] = nb_terminaisons_identiques(df)

    # --- Tendance récente (hot/cold glissant) ---
    trend = tendance_recente(df)
    for c in trend.columns:
        result[c] = trend[c]

    # --- Retard par numéro tiré ---
    ret_nums = retard_par_numero(df)
    for c in ret_nums.columns:
        result[c] = ret_nums[c]

    # --- Répétitions avec le tirage précédent ---
    result['repetitions_precedent'] = repetitions_avec_precedent(df)

    # --- Position relative (petit/grand) ---
    pos = ratio_position(df)
    for c in pos.columns:
        result[c] = pos[c]

    # --- Jour et Mois (si disponible) ---
    if df_full is not None and len(df_full) == len(df):
        result['jour'] = encode_jour(df_full)
        result['mois'] = encode_mois(df_full)
    else:
        result['jour'] = 0
        result['mois'] = 0

    # Remplir les NaN restants (premiers tirages)
    result = result.fillna(0)

    print(f"[Features] {len(result.columns)} features construites pour {len(result)} tirages")
    return result