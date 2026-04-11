"""
🎰 Agent IA Intelligent pour le Loto
═════════════════════════════════════

Un agent conversationnel qui combine :
- Prédictions LSTM (ton modèle entraîné)
- Scoring statistique des combinaisons
- Analyse des fréquences historiques
- Filtres intelligents (numérologie, écarts, entropie)
- Réduction des 19M combinaisons

Lance avec : python agent.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from collections import Counter
from config import *
from utils import *
from loto_functions import load_draws_from_csv

# Suppression des warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ══════════════════════════════════════════════

def load_model_safe():
    """Charge le modèle et le scaler en silence."""
    from keras.models import load_model
    if not os.path.exists(MODEL_PATH):
        print("❌ Aucun modèle trouvé. Lance d'abord l'entraînement sur Colab.")
        return None, None
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def load_data():
    """Charge et prépare les données."""
    df_raw = load_draws_from_csv()
    if df_raw is None:
        print("❌ Aucun fichier CSV trouvé dans data/")
        return None, None
    df = df_raw.iloc[::-1].reset_index(drop=True)
    df_draws = df[ALL_DRAW_COLS].copy()
    df_features = build_all_features(df_draws)
    return df_draws, df_features


# ══════════════════════════════════════════════
# ANALYSE STATISTIQUE HISTORIQUE
# ══════════════════════════════════════════════

class LotoAnalyzer:
    """Analyse statistique complète des tirages historiques."""

    def __init__(self, df_draws):
        self.df = df_draws
        self.total_tirages = len(df_draws)
        self._compute_stats()

    def _compute_stats(self):
        """Calcule toutes les statistiques de base."""
        # Fréquence de chaque numéro (1-49)
        all_nums = []
        for c in NUM_COLS:
            all_nums.extend(self.df[c].values.tolist())
        self.freq_nums = Counter(all_nums)

        # Fréquence du numéro chance (1-10)
        self.freq_chance = Counter(self.df[CHANCE_COL].values.tolist())

        # Retard de chaque numéro (nb tirages depuis dernière apparition)
        self.retards = {}
        last_seen = {}
        for idx, row in self.df.iterrows():
            for c in NUM_COLS:
                last_seen[int(row[c])] = idx
        for n in range(1, LOTO_MAX_NUM + 1):
            if n in last_seen:
                self.retards[n] = self.total_tirages - 1 - last_seen[n]
            else:
                self.retards[n] = self.total_tirages

        # Retard chance
        self.retards_chance = {}
        last_seen_ch = {}
        for idx, row in self.df.iterrows():
            last_seen_ch[int(row[CHANCE_COL])] = idx
        for n in range(1, LOTO_MAX_CHANCE + 1):
            if n in last_seen_ch:
                self.retards_chance[n] = self.total_tirages - 1 - last_seen_ch[n]
            else:
                self.retards_chance[n] = self.total_tirages

        # Sommes historiques
        self.sommes = (self.df[NUM_COLS].sum(axis=1)).values
        self.somme_moyenne = np.mean(self.sommes)
        self.somme_std = np.std(self.sommes)

        # Paires les plus fréquentes (top 20)
        pair_counter = Counter()
        for _, row in self.df.iterrows():
            nums = sorted([int(row[c]) for c in NUM_COLS])
            for pair in combinations(nums, 2):
                pair_counter[pair] += 1
        self.top_paires = pair_counter.most_common(20)

    def get_hot_numbers(self, n=10):
        """Numéros les plus fréquents (chauds)."""
        return self.freq_nums.most_common(n)

    def get_cold_numbers(self, n=10):
        """Numéros les moins fréquents (froids)."""
        return self.freq_nums.most_common()[-n:]

    def get_overdue_numbers(self, n=10):
        """Numéros avec le plus grand retard."""
        sorted_ret = sorted(self.retards.items(), key=lambda x: x[1], reverse=True)
        return sorted_ret[:n]

    def get_recent_numbers(self, n=10):
        """Numéros sortis récemment."""
        sorted_ret = sorted(self.retards.items(), key=lambda x: x[1])
        return sorted_ret[:n]


# ══════════════════════════════════════════════
# SYSTÈME DE SCORING DES COMBINAISONS
# ══════════════════════════════════════════════

class CombiScorer:
    """
    Donne un score à une combinaison (5 nums + chance) basé sur
    plusieurs critères pondérés.
    """

    def __init__(self, analyzer, model_predictions=None):
        self.analyzer = analyzer
        self.predictions = model_predictions or []

        # Poids de chaque critère (ajustables)
        self.weights = {
            'lstm_match': 15,
            'frequency': 15,
            'retard': 20,
            'sum_range': 10,
            'even_odd': 10,
            'decade_spread': 15,
            'consecutive': 5,
            'amplitude': 10,
        }

    def score(self, nums, chance):
        """
        Calcule un score entre 0 et 100 pour une combinaison.
        Plus le score est élevé, plus la combinaison est "intéressante".
        """
        scores = {}

        # 1. Correspondance LSTM
        if self.predictions:
            matches = 0
            for pred in self.predictions:
                pred_nums = set(pred[:5])
                matches += len(set(nums) & pred_nums)
                if pred[5] == chance:
                    matches += 1
            max_matches = len(self.predictions) * 6
            scores['lstm_match'] = (matches / max_matches) * 100
        else:
            scores['lstm_match'] = 50  # Neutre si pas de prédiction

        # 2. Fréquence historique (on favorise les numéros fréquents mais pas trop)
        freq_scores = []
        max_freq = max(self.analyzer.freq_nums.values())
        for n in nums:
            f = self.analyzer.freq_nums.get(n, 0) / max_freq
            # Score en cloche : ni trop rare ni trop fréquent
            freq_scores.append(1 - abs(f - 0.5) * 2)
        scores['frequency'] = np.mean(freq_scores) * 100

        # 3. Retard (bonus pour numéros en retard modéré)
        retard_scores = []
        avg_retard = np.mean(list(self.analyzer.retards.values()))
        for n in nums:
            r = self.analyzer.retards.get(n, 0)
            # Score optimal autour du retard moyen
            ratio = r / (avg_retard + 1)
            retard_scores.append(min(ratio, 2) / 2)
        scores['retard'] = np.mean(retard_scores) * 100

        # 4. Somme dans la plage normale
        s = sum(nums)
        z_score = abs(s - self.analyzer.somme_moyenne) / (self.analyzer.somme_std + 1)
        scores['sum_range'] = max(0, (1 - z_score / 3)) * 100

        # 5. Répartition pair/impair
        n_pairs = sum(1 for n in nums if n % 2 == 0)
        # 2-3 pairs est optimal
        if n_pairs in [2, 3]:
            scores['even_odd'] = 100
        elif n_pairs in [1, 4]:
            scores['even_odd'] = 50
        else:
            scores['even_odd'] = 10

        # 6. Répartition par dizaine
        decades = set()
        for n in nums:
            decades.add((n - 1) // 10)
        scores['decade_spread'] = (len(decades) / 5) * 100

        # 7. Consécutifs (pénalité)
        sorted_nums = sorted(nums)
        n_consec = sum(1 for i in range(4) if sorted_nums[i+1] - sorted_nums[i] == 1)
        scores['consecutive'] = max(0, (1 - n_consec / 3)) * 100

        # 8. Amplitude
        amp = max(nums) - min(nums)
        # Amplitude idéale entre 25 et 40
        if 25 <= amp <= 40:
            scores['amplitude'] = 100
        elif 15 <= amp <= 45:
            scores['amplitude'] = 60
        else:
            scores['amplitude'] = 20

        # Score final pondéré
        total = sum(scores[k] * self.weights[k] for k in scores)
        max_total = sum(self.weights.values()) * 100
        final_score = (total / max_total) * 100

        return round(final_score, 1), scores

    def generate_smart_grids(self, n_grids=10):
        """
        Génère des grilles intelligentes et DIVERSIFIÉES en combinant :
        - Les prédictions LSTM comme base
        - Les stats historiques (chauds, froids, en retard)
        - Des stratégies variées pour couvrir différents scénarios
        - Un filtre de diversité pour éviter les doublons proches
        """
        candidates = []

        hot = [n for n, _ in self.analyzer.get_hot_numbers(25)]
        cold = [n for n, _ in self.analyzer.get_cold_numbers(15)]
        overdue = [n for n, _ in self.analyzer.get_overdue_numbers(20)]
        recent = [n for n, _ in self.analyzer.get_recent_numbers(10)]

        # ── Stratégie 1 : Variations LSTM (remplacer 1-2 numéros) ──
        if self.predictions:
            for pred in self.predictions:
                base = list(pred[:5])
                base_ch = pred[5]
                candidates.append((base, base_ch))

                # Remplacer 1 numéro par un en retard
                for pos in range(5):
                    for od in overdue[:10]:
                        if od not in base:
                            new = base.copy()
                            new[pos] = od
                            candidates.append((sorted(new), base_ch))

                # Remplacer 2 numéros
                for p1 in range(4):
                    for p2 in range(p1+1, 5):
                        for _ in range(5):
                            new = base.copy()
                            pool = [n for n in overdue + hot if n not in new]
                            if len(pool) >= 2:
                                picks = np.random.choice(pool, 2, replace=False)
                                new[p1] = picks[0]
                                new[p2] = picks[1]
                                if len(set(new)) == 5:
                                    candidates.append((sorted(new), base_ch))

                # Varier le chance
                for ch in range(1, LOTO_MAX_CHANCE + 1):
                    candidates.append((base, ch))

        # ── Stratégie 2 : Mix chauds + en retard ──
        for _ in range(200):
            n_hot = np.random.randint(2, 5)
            n_overdue = 5 - n_hot
            pool_h = [n for n in hot if n not in overdue[:5]]
            pool_o = overdue.copy()
            np.random.shuffle(pool_h)
            np.random.shuffle(pool_o)
            nums = pool_h[:n_hot] + pool_o[:n_overdue]
            if len(set(nums)) == 5:
                ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
                candidates.append((sorted(nums), ch))

        # ── Stratégie 3 : 1 numéro par dizaine (bonne répartition) ──
        decades = [list(range(1,11)), list(range(11,21)), list(range(21,31)),
                   list(range(31,41)), list(range(41,50))]
        for _ in range(200):
            nums = [np.random.choice(d) for d in decades]
            ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
            candidates.append((sorted(nums), ch))

        # ── Stratégie 4 : Basée sur les paires fréquentes ──
        for (a, b), _ in self.analyzer.top_paires[:10]:
            for _ in range(20):
                remaining = [n for n in range(1, LOTO_MAX_NUM+1) if n != a and n != b]
                extra = np.random.choice(remaining, 3, replace=False)
                nums = sorted([a, b] + list(extra))
                ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
                candidates.append((nums, ch))

        # ── Stratégie 5 : Somme ciblée (plage optimale) ──
        target_low = int(self.analyzer.somme_moyenne - self.analyzer.somme_std)
        target_high = int(self.analyzer.somme_moyenne + self.analyzer.somme_std)
        for _ in range(300):
            nums = sorted(np.random.choice(range(1, LOTO_MAX_NUM+1), 5, replace=False))
            s = sum(nums)
            if target_low <= s <= target_high:
                ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
                candidates.append((list(nums), ch))

        # ── Stratégie 6 : Numéros froids (contrarian) ──
        for _ in range(50):
            pool = cold + overdue
            np.random.shuffle(pool)
            unique = list(dict.fromkeys(pool))
            if len(unique) >= 5:
                nums = sorted(unique[:5])
                ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
                candidates.append((nums, ch))

        # ── Stratégie 7 : Aléatoire pur (exploration) ──
        for _ in range(500):
            nums = sorted(np.random.choice(range(1, LOTO_MAX_NUM+1), 5, replace=False))
            ch = np.random.randint(LOTO_MIN_CHANCE, LOTO_MAX_CHANCE + 1)
            candidates.append((list(nums), ch))

        # ── Scoring de tous les candidats ──
        scored = []
        seen = set()
        for nums, chance in candidates:
            key = tuple(sorted(nums) + [chance])
            if key not in seen:
                seen.add(key)
                score, details = self.score(nums, chance)
                scored.append({
                    'nums': sorted(nums),
                    'chance': chance,
                    'score': score,
                    'details': details
                })

        scored.sort(key=lambda x: x['score'], reverse=True)
        self.total_evaluated = len(scored)

        # ── Filtre de diversité : au moins 2 numéros différents entre chaque grille ──
        diverse = []
        for grid in scored:
            is_diverse = True
            for selected in diverse:
                common = len(set(grid['nums']) & set(selected['nums']))
                if common >= 4:  # Trop similaire
                    is_diverse = False
                    break
            if is_diverse:
                diverse.append(grid)
            if len(diverse) >= n_grids:
                break

        return diverse


# ══════════════════════════════════════════════
# AGENT CONVERSATIONNEL
# ══════════════════════════════════════════════

class LotoAgent:
    """Agent IA interactif pour le Loto."""

    def __init__(self):
        print("\n⏳ Chargement de l'agent...")

        # Charger modèle
        self.model, self.scaler = load_model_safe()

        # Charger données
        self.df_draws, self.df_features = load_data()

        if self.df_draws is None:
            print("❌ Impossible de charger les données.")
            sys.exit(1)

        # Analyser l'historique
        self.analyzer = LotoAnalyzer(self.df_draws)

        # Générer les prédictions LSTM
        self.predictions = []
        if self.model is not None:
            self.predictions = self._generate_predictions()

        # Créer le scorer
        self.scorer = CombiScorer(self.analyzer, self.predictions)

        # Générer les grilles intelligentes
        self.smart_grids = self.scorer.generate_smart_grids(20)

        print("✅ Agent prêt !\n")

    def _generate_predictions(self, n=5):
        """Génère les prédictions LSTM."""
        from models_functions import predict_next_draw
        preds = []
        # Prédiction principale
        pred = predict_next_draw(self.model, self.scaler, self.df_features)
        preds.append(pred)

        # Variations
        for _ in range(n - 1):
            last = self.df_features.tail(WINDOW_LENGTH).copy()
            noise = np.random.normal(0, 0.5, size=last.shape)
            noisy = self.scaler.transform(last.values) + noise * 0.5
            scaled_pred = self.model.predict(np.array([noisy]), verbose=0)
            nb_f = self.df_features.shape[1]
            padded = np.zeros((1, nb_f))
            padded[0, :NB_LABEL_FEATURES] = scaled_pred[0]
            raw = self.scaler.inverse_transform(padded)[0, :NB_LABEL_FEATURES]
            p = np.round(raw).astype(int)
            for i in range(5):
                p[i] = np.clip(p[i], LOTO_MIN_NUM, LOTO_MAX_NUM)
            p[5] = np.clip(p[5], LOTO_MIN_CHANCE, LOTO_MAX_CHANCE)
            nums = list(p[:5])
            seen = set()
            for i, nn in enumerate(nums):
                while nn in seen:
                    nn = nn + 1 if nn < LOTO_MAX_NUM else LOTO_MIN_NUM
                seen.add(nn)
                nums[i] = nn
            p[:5] = sorted(nums)
            preds.append(p)

        return preds

    def show_welcome(self):
        """Message d'accueil."""
        print("═" * 60)
        print("  🎰  AGENT IA LOTO - Assistant Intelligent")
        print("═" * 60)
        print(f"\n  📊 {self.analyzer.total_tirages} tirages analysés")
        print(f"  🧠 Modèle LSTM : {'chargé' if self.model else 'non disponible'}")
        print(f"  🎯 {len(self.smart_grids)} grilles optimisées prêtes")
        print(f"\n  Commandes disponibles :")
        print(f"  ─────────────────────────────────────")
        print(f"  grilles      → Mes meilleures grilles")
        print(f"  lstm         → Prédictions brutes du LSTM")
        print(f"  chauds       → Numéros les plus fréquents")
        print(f"  froids       → Numéros les moins fréquents")
        print(f"  retard       → Numéros en retard")
        print(f"  recents      → Numéros sortis récemment")
        print(f"  paires       → Paires les plus fréquentes")
        print(f"  somme        → Analyse des sommes")
        print(f"  score 1 2 3  → Score d'une combinaison")
        print(f"  analyser     → Analyse complète")
        print(f"  numérologie  → Analyse numérologique")
        print(f"  quitter      → Fermer l'agent")
        print(f"  ─────────────────────────────────────\n")

    def handle_command(self, cmd):
        """Traite une commande utilisateur."""
        cmd = cmd.strip().lower()

        if cmd in ['grilles', 'g', 'meilleures']:
            self.show_smart_grids()
        elif cmd in ['lstm', 'predictions', 'prédictions']:
            self.show_lstm_predictions()
        elif cmd in ['chauds', 'hot', 'frequents']:
            self.show_hot()
        elif cmd in ['froids', 'cold', 'rares']:
            self.show_cold()
        elif cmd in ['retard', 'retards', 'overdue']:
            self.show_overdue()
        elif cmd in ['recents', 'récents', 'recent']:
            self.show_recent()
        elif cmd in ['paires', 'pairs', 'duos']:
            self.show_pairs()
        elif cmd in ['somme', 'sommes', 'sum']:
            self.show_sum_analysis()
        elif cmd.startswith('score'):
            self.score_combination(cmd)
        elif cmd in ['analyser', 'analyse', 'complet']:
            self.full_analysis()
        elif cmd in ['numérologie', 'numerologie', 'numer']:
            self.show_numerology()
        elif cmd in ['aide', 'help', '?']:
            self.show_welcome()
        elif cmd in ['quitter', 'quit', 'exit', 'q']:
            print("\n👋 À bientôt ! Bonne chance au Loto !\n")
            sys.exit(0)
        else:
            print(f"\n  ❓ Commande inconnue : '{cmd}'")
            print(f"     Tape 'aide' pour voir les commandes disponibles.\n")

    def show_smart_grids(self):
        """Affiche les grilles optimisées avec scores."""
        print("\n" + "═" * 60)
        print("  🎯 MES MEILLEURES GRILLES (classées par score)")
        print("═" * 60)

        for i, grid in enumerate(self.smart_grids[:10], 1):
            nums = ' - '.join(f'{n:2d}' for n in grid['nums'])
            score = grid['score']

            # Barre de score visuelle
            bar_len = int(score / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)

            print(f"\n  Grille {i:2d} │ {nums} │ ⭐ {grid['chance']} │ Score: {score}/100")
            print(f"           │ {bar} │", end="")

            # Détail rapide
            d = grid['details']
            tags = []
            if d.get('lstm_match', 0) > 60:
                tags.append("🧠LSTM")
            if d.get('retard', 0) > 60:
                tags.append("⏰Retard")
            if d.get('even_odd', 0) > 80:
                tags.append("⚖Équilibré")
            if d.get('decade_spread', 0) > 80:
                tags.append("📊Réparti")
            print(f" {' '.join(tags)}")

        total_tested = getattr(self.scorer, 'total_evaluated', 0)
        print(f"\n  💡 {total_tested} combinaisons évaluées sur 19 068 840 possibles")
        print(f"     Réduction : 99.99% des combinaisons éliminées\n")

    def show_lstm_predictions(self):
        """Affiche les prédictions brutes du LSTM."""
        if not self.predictions:
            print("\n  ❌ Pas de modèle LSTM chargé.\n")
            return

        print("\n" + "═" * 60)
        print("  🧠 PRÉDICTIONS LSTM BRUTES")
        print("═" * 60)

        for i, pred in enumerate(self.predictions, 1):
            nums = ' - '.join(f'{n:2d}' for n in pred[:5])
            print(f"  Grille {i} : {nums} | ⭐ {pred[5]}")

        # Numéros consensus
        all_nums = [n for p in self.predictions for n in p[:5]]
        freq = Counter(all_nums)
        print(f"\n  📊 Consensus LSTM :")
        for num, count in freq.most_common(8):
            pct = count / len(self.predictions) * 100
            bar = '█' * int(pct / 5)
            print(f"     {num:2d} → {bar} ({pct:.0f}%)")
        print()

    def show_hot(self):
        """Numéros chauds."""
        print("\n" + "═" * 60)
        print("  🔥 NUMÉROS CHAUDS (les plus fréquents)")
        print("═" * 60)
        for num, count in self.analyzer.get_hot_numbers(15):
            pct = count / self.analyzer.total_tirages * 100
            bar = '█' * int(pct / 2)
            print(f"  {num:2d} → {bar} {count}x ({pct:.1f}%)")
        print()

    def show_cold(self):
        """Numéros froids."""
        print("\n" + "═" * 60)
        print("  ❄️  NUMÉROS FROIDS (les moins fréquents)")
        print("═" * 60)
        for num, count in self.analyzer.get_cold_numbers(15):
            pct = count / self.analyzer.total_tirages * 100
            bar = '░' * int(pct / 2)
            print(f"  {num:2d} → {bar} {count}x ({pct:.1f}%)")
        print()

    def show_overdue(self):
        """Numéros en retard."""
        print("\n" + "═" * 60)
        print("  ⏰ NUMÉROS EN RETARD (pas sortis depuis longtemps)")
        print("═" * 60)
        for num, retard in self.analyzer.get_overdue_numbers(15):
            bar = '█' * min(retard // 2, 30)
            print(f"  {num:2d} → {retard} tirages │ {bar}")
        print()

    def show_recent(self):
        """Numéros récents."""
        print("\n" + "═" * 60)
        print("  🆕 NUMÉROS RÉCENTS (sortis il y a peu)")
        print("═" * 60)
        for num, retard in self.analyzer.get_recent_numbers(15):
            print(f"  {num:2d} → sorti il y a {retard} tirage(s)")
        print()

    def show_pairs(self):
        """Paires fréquentes."""
        print("\n" + "═" * 60)
        print("  👯 PAIRES LES PLUS FRÉQUENTES")
        print("═" * 60)
        for (a, b), count in self.analyzer.top_paires:
            bar = '█' * (count // 2)
            print(f"  {a:2d} - {b:2d} → {bar} {count}x")
        print()

    def show_sum_analysis(self):
        """Analyse des sommes."""
        print("\n" + "═" * 60)
        print("  Σ  ANALYSE DES SOMMES")
        print("═" * 60)
        print(f"  Somme moyenne    : {self.analyzer.somme_moyenne:.1f}")
        print(f"  Écart-type       : {self.analyzer.somme_std:.1f}")
        print(f"  Plage optimale   : {self.analyzer.somme_moyenne - self.analyzer.somme_std:.0f} - {self.analyzer.somme_moyenne + self.analyzer.somme_std:.0f}")
        print(f"  Plage étendue    : {self.analyzer.somme_moyenne - 2*self.analyzer.somme_std:.0f} - {self.analyzer.somme_moyenne + 2*self.analyzer.somme_std:.0f}")

        # Distribution des sommes
        print(f"\n  Distribution :")
        bins = [(50, 80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 200)]
        for low, high in bins:
            count = sum(1 for s in self.analyzer.sommes if low <= s < high)
            pct = count / len(self.analyzer.sommes) * 100
            bar = '█' * int(pct / 2)
            print(f"  {low:3d}-{high:3d} → {bar} {pct:.1f}%")
        print()

    def score_combination(self, cmd):
        """Score une combinaison entrée par l'utilisateur."""
        parts = cmd.replace('score', '').strip().split()
        try:
            if len(parts) == 6:
                nums = [int(x) for x in parts[:5]]
                chance = int(parts[5])
            elif len(parts) == 5:
                nums = [int(x) for x in parts[:5]]
                chance = 1
                print(f"  (Numéro chance non spécifié, utilisation de 1 par défaut)")
            else:
                print("\n  ❓ Usage : score 5 14 25 35 42 6")
                print("     (5 numéros + numéro chance)\n")
                return
        except ValueError:
            print("\n  ❓ Utilise des nombres : score 5 14 25 35 42 6\n")
            return

        score, details = self.scorer.score(nums, chance)
        nums_str = ' - '.join(f'{n:2d}' for n in sorted(nums))

        print(f"\n" + "═" * 60)
        print(f"  📊 SCORE DE : {nums_str} | ⭐ {chance}")
        print(f"═" * 60)
        print(f"\n  Score global : {score}/100\n")

        for key, value in details.items():
            label = {
                'lstm_match': '🧠 LSTM',
                'frequency': '📈 Fréquence',
                'retard': '⏰ Retard',
                'sum_range': 'Σ  Somme',
                'even_odd': '⚖  Pair/Impair',
                'decade_spread': '📊 Dizaines',
                'consecutive': '🔗 Consécutifs',
                'amplitude': '↔  Amplitude',
            }.get(key, key)
            bar_len = int(value / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            weight = self.scorer.weights[key]
            print(f"  {label:16s} │ {bar} │ {value:.0f}/100 (poids: {weight})")

        # Numérologie
        somme = sum(nums)
        red = reduction_numerologique(somme)
        print(f"\n  🔮 Numérologie :")
        print(f"     Somme = {somme} → réduction = {red}")
        for n in sorted(nums):
            print(f"     {n:2d} → {reduction_numerologique(n)}")
        print()

    def show_numerology(self):
        """Analyse numérologique."""
        print("\n" + "═" * 60)
        print("  🔮 ANALYSE NUMÉROLOGIQUE")
        print("═" * 60)

        # Réduction des derniers tirages
        print("\n  Derniers tirages et leur réduction :")
        for _, row in self.df_draws.tail(10).iterrows():
            nums = [int(row[c]) for c in NUM_COLS]
            reds = [reduction_numerologique(n) for n in nums]
            somme = sum(nums)
            red_somme = reduction_numerologique(somme)
            nums_str = ' - '.join(f'{n:2d}' for n in nums)
            reds_str = ' - '.join(f'{r}' for r in reds)
            print(f"  {nums_str} │ réd: {reds_str} │ Σ{somme} → {red_somme}")

        # Fréquence des réductions de somme
        print(f"\n  Fréquence des réductions de somme :")
        red_counter = Counter()
        for _, row in self.df_draws.iterrows():
            s = sum(int(row[c]) for c in NUM_COLS)
            red_counter[reduction_numerologique(s)] += 1

        for red in range(1, 10):
            count = red_counter.get(red, 0)
            pct = count / self.analyzer.total_tirages * 100
            bar = '█' * int(pct)
            print(f"  {red} → {bar} {pct:.1f}%")
        print()

    def full_analysis(self):
        """Analyse complète."""
        self.show_lstm_predictions()
        self.show_hot()
        self.show_overdue()
        self.show_sum_analysis()
        self.show_numerology()
        self.show_smart_grids()


# ══════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════

def main():
    agent = LotoAgent()
    agent.show_welcome()

    while True:
        try:
            cmd = input("  🎰 > ").strip()
            if cmd:
                agent.handle_command(cmd)
        except KeyboardInterrupt:
            print("\n\n👋 À bientôt !\n")
            break
        except Exception as e:
            print(f"\n  ⚠️ Erreur : {e}\n")


if __name__ == '__main__':
    main()