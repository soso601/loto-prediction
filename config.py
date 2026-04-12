"""
Configuration centrale du projet Loto Prediction.
Tous les hyperparamètres et constantes sont définis ici.
"""

# ──────────────────────────────────────────────
# Paramètres du Loto Français
# ──────────────────────────────────────────────
LOTO_MIN_NUM = 1
LOTO_MAX_NUM = 49
LOTO_MIN_CHANCE = 1
LOTO_MAX_CHANCE = 10
LOTO_NUM_BALLS = 5
TOTAL_COMBINATIONS = 19_068_840  # C(49,5) * 10

# ──────────────────────────────────────────────
# Paramètres du modèle LSTM
# ──────────────────────────────────────────────
NB_LABEL_FEATURES = 6       # 5 numéros + 1 chance
UNITS = 32                 # Neurones par couche LSTM
BATCHSIZE = 64
EPOCH = 2000
OPTIMIZER = 'adam'
LOSS = 'mae'
DROPOUT = 0.45
WINDOW_LENGTH = 8          # Fenêtre glissante (≈1 mois de tirages)
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.15

# ──────────────────────────────────────────────
# Paramètres des features
# ──────────────────────────────────────────────
MOVING_AVG_WINDOWS = [5, 10, 20]   # Fenêtres pour moyennes mobiles
ECART_LOOKBACK = 50                # Lookback pour calcul d'écarts
NUM_COLS = ['num0', 'num1', 'num2', 'num3', 'num4']
CHANCE_COL = 'chance'
ALL_DRAW_COLS = NUM_COLS + [CHANCE_COL]

# ──────────────────────────────────────────────
# Chemins de sauvegarde
# ──────────────────────────────────────────────
MODEL_DIR = 'saved_models'
MODEL_PATH = f'{MODEL_DIR}/loto_lstm_model.keras'
MODEL_BIDIRECTIONAL_PATH = f'{MODEL_DIR}/loto_bidirectional_model.keras'
MODEL_AUTOENCODER_PATH = f'{MODEL_DIR}/loto_autoencoder_model.keras'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'
HISTORY_PATH = f'{MODEL_DIR}/training_history.pkl'
DRAWS_CSV_PATH = 'data/tirages_loto.csv'