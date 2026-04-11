"""
Définition, entraînement, sauvegarde, chargement et mise à jour
des modèles LSTM pour la prédiction du Loto.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import (LSTM, Dense, Bidirectional,
                          TimeDistributed, RepeatVector, Flatten, Dropout)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from config import *


# ══════════════════════════════════════════════
# ARCHITECTURES DE MODÈLES
# ══════════════════════════════════════════════

def define_model(nb_features):
    """LSTM standard amélioré."""
    model = Sequential([
        LSTM(UNITS, input_shape=(WINDOW_LENGTH, nb_features), return_sequences=True),
        Dropout(DROPOUT),
        LSTM(UNITS, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(UNITS // 2, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(NB_LABEL_FEATURES)
    ])
    model.compile(
        loss=LOSS,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['mae']
    )
    return model


def define_bidirectional_model(nb_features):
    """LSTM bidirectionnel."""
    model = Sequential([
        Bidirectional(LSTM(UNITS, dropout=DROPOUT, return_sequences=True),
                      input_shape=(WINDOW_LENGTH, nb_features)),
        LSTM(UNITS // 2, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(UNITS // 2, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(NB_LABEL_FEATURES)
    ])
    model.compile(loss=LOSS, optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['mae'])
    return model


def define_autoencoder_model(nb_features):
    """LSTM autoencodeur."""
    model = Sequential([
        LSTM(UNITS, input_shape=(WINDOW_LENGTH, nb_features), return_sequences=True),
        LSTM(UNITS // 2, return_sequences=False),
        RepeatVector(WINDOW_LENGTH),
        LSTM(UNITS, dropout=DROPOUT, return_sequences=True),
        LSTM(UNITS // 2, return_sequences=True),
        TimeDistributed(Dense(nb_features)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(NB_LABEL_FEATURES)
    ])
    model.compile(loss=LOSS, optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['mae'])
    return model


MODEL_REGISTRY = {
    'standard': define_model,
    'bidirectional': define_bidirectional_model,
    'autoencoder': define_autoencoder_model,
}


# ══════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════

def create_lstm_dataset(df, scaler=None, fit_scaler=True):
    """
    Prépare les données pour le LSTM.
    - Utilise un seul scaler cohérent
    - Sépare train/validation proprement
    - Pas de fuite de données

    Retourne: train_X, train_y, val_X, val_y, scaler
    """
    number_of_rows = df.shape[0]
    nb_features = df.shape[1]

    # Scaler unique
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        scaler.fit(df.values)
    elif scaler is None:
        raise ValueError("Un scaler doit être fourni si fit_scaler=False")

    transformed = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed, index=df.index)

    # Construction des fenêtres glissantes
    X = np.empty([number_of_rows - WINDOW_LENGTH, WINDOW_LENGTH, nb_features], dtype=float)
    y = np.empty([number_of_rows - WINDOW_LENGTH, NB_LABEL_FEATURES], dtype=float)

    for i in range(number_of_rows - WINDOW_LENGTH):
        X[i] = transformed_df.iloc[i:i + WINDOW_LENGTH].values
        y[i] = transformed_df.iloc[i + WINDOW_LENGTH, :NB_LABEL_FEATURES].values

    # Séparation train/validation (les plus récents en validation)
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]

    print(f"[Dataset] Train: {train_X.shape}, Validation: {val_X.shape}")
    print(f"[Dataset] Features: {nb_features}, Window: {WINDOW_LENGTH}")

    return train_X, train_y, val_X, val_y, scaler


# ══════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════

def train_model(df, model_type='standard'):
    """
    Pipeline complet d'entraînement.
    Retourne: model, scaler, history
    """
    # Préparation des données
    train_X, train_y, val_X, val_y, scaler = create_lstm_dataset(df)

    nb_features = df.shape[1]

    # Création du modèle
    model_fn = MODEL_REGISTRY.get(model_type, define_model)
    model = model_fn(nb_features)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Entraînement
    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=BATCHSIZE,
        epochs=EPOCH,
        verbose=2,
        callbacks=callbacks
    )

    print(f"\n[Training] Terminé. Best val_loss: {min(history.history['val_loss']):.4f}")

    return model, scaler, history


# ══════════════════════════════════════════════
# SAUVEGARDE / CHARGEMENT
# ══════════════════════════════════════════════

def save_model_and_scaler(model, scaler, history=None):
    """Sauvegarde le modèle, le scaler et l'historique."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    print(f"[Save] Modèle sauvegardé : {MODEL_PATH}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[Save] Scaler sauvegardé : {SCALER_PATH}")

    if history is not None:
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"[Save] Historique sauvegardé : {HISTORY_PATH}")


def load_model_and_scaler():
    """Charge le modèle et le scaler sauvegardés."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Load] Aucun modèle trouvé à {MODEL_PATH}")
        return None, None

    model = load_model(MODEL_PATH)
    print(f"[Load] Modèle chargé : {MODEL_PATH}")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[Load] Scaler chargé : {SCALER_PATH}")

    return model, scaler


def model_exists():
    """Vérifie si un modèle sauvegardé existe."""
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


# ══════════════════════════════════════════════
# MISE À JOUR INCRÉMENTALE
# ══════════════════════════════════════════════

def update_model(df, epochs=200):
    """
    Met à jour un modèle existant avec de nouvelles données.
    Re-fitte le scaler sur toutes les données, puis fine-tune le modèle.
    """
    model, old_scaler = load_model_and_scaler()

    if model is None:
        print("[Update] Aucun modèle existant. Entraînement complet...")
        return train_model(df)

    # Nouveau scaler sur toutes les données (anciennes + nouvelles)
    train_X, train_y, val_X, val_y, new_scaler = create_lstm_dataset(df)

    # Fine-tuning avec learning rate réduit
    from keras.optimizers import Adam
    model.compile(
        loss=LOSS,
        optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),  # LR réduit pour fine-tuning
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=BATCHSIZE,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks
    )

    print(f"[Update] Fine-tuning terminé. Best val_loss: {min(history.history['val_loss']):.4f}")

    # Sauvegarde avec le nouveau scaler
    save_model_and_scaler(model, new_scaler, history)

    return model, new_scaler, history


# ══════════════════════════════════════════════
# PRÉDICTION
# ══════════════════════════════════════════════

def predict_next_draw(model, scaler, df):
    """
    Prédit le prochain tirage en utilisant les WINDOW_LENGTH derniers tirages.
    Utilise le même scaler que l'entraînement (cohérence garantie).
    """
    last_draws = df.tail(WINDOW_LENGTH)
    scaled_input = scaler.transform(last_draws.values)

    # Prédiction
    scaled_prediction = model.predict(np.array([scaled_input]), verbose=0)

    # Inverse transform : on a besoin d'un vecteur de la même taille que le scaler
    # On pad avec des zéros pour les features non-prédites
    nb_features = df.shape[1]
    padded = np.zeros((1, nb_features))
    padded[0, :NB_LABEL_FEATURES] = scaled_prediction[0]
    full_inverse = scaler.inverse_transform(padded)
    raw_prediction = full_inverse[0, :NB_LABEL_FEATURES]

    # Arrondir et contraindre aux bornes du Loto
    prediction = np.round(raw_prediction).astype(int)

    # Contraindre les 5 numéros entre 1 et 49
    for i in range(5):
        prediction[i] = np.clip(prediction[i], LOTO_MIN_NUM, LOTO_MAX_NUM)

    # Contraindre le numéro chance entre 1 et 10
    prediction[5] = np.clip(prediction[5], LOTO_MIN_CHANCE, LOTO_MAX_CHANCE)

    # Assurer l'unicité des 5 numéros
    nums = list(prediction[:5])
    seen = set()
    for i, n in enumerate(nums):
        while n in seen:
            n = n + 1 if n < LOTO_MAX_NUM else LOTO_MIN_NUM
        seen.add(n)
        nums[i] = n
    prediction[:5] = sorted(nums)

    return prediction


def predict_multiple(model, scaler, df, n_predictions=5):
    """
    Génère plusieurs prédictions en ajoutant du bruit léger.
    Utile pour avoir un ensemble de combinaisons candidates.
    """
    predictions = []
    base = predict_next_draw(model, scaler, df)
    predictions.append(base)

    for _ in range(n_predictions - 1):
        # Ajouter un léger bruit aux dernières entrées
        last_draws = df.tail(WINDOW_LENGTH).copy()
        noise = np.random.normal(0, 0.5, size=last_draws.shape)
        noisy_input = scaler.transform(last_draws.values) + noise * 0.1
        scaled_pred = model.predict(np.array([noisy_input]), verbose=0)

        nb_features = df.shape[1]
        padded = np.zeros((1, nb_features))
        padded[0, :NB_LABEL_FEATURES] = scaled_pred[0]
        full_inverse = scaler.inverse_transform(padded)
        raw = full_inverse[0, :NB_LABEL_FEATURES]

        pred = np.round(raw).astype(int)
        for i in range(5):
            pred[i] = np.clip(pred[i], LOTO_MIN_NUM, LOTO_MAX_NUM)
        pred[5] = np.clip(pred[5], LOTO_MIN_CHANCE, LOTO_MAX_CHANCE)

        nums = list(pred[:5])
        seen = set()
        for i, n in enumerate(nums):
            while n in seen:
                n = n + 1 if n < LOTO_MAX_NUM else LOTO_MIN_NUM
            seen.add(n)
            nums[i] = n
        pred[:5] = sorted(nums)
        predictions.append(pred)

    return predictions


def compare_models(df):
    """
    Entraîne les 3 architectures et compare leurs performances.
    Retourne un dictionnaire {nom: (model, scaler, val_loss)}.
    """
    results = {}

    for name, model_fn in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"  Entraînement du modèle : {name.upper()}")
        print(f"{'='*60}")

        model, scaler, history = train_model(df, model_type=name)
        best_val_loss = min(history.history['val_loss'])
        results[name] = {
            'model': model,
            'scaler': scaler,
            'val_loss': best_val_loss,
            'history': history,
        }
        print(f"[{name}] Best val_loss: {best_val_loss:.4f}")

    # Résumé
    print(f"\n{'='*60}")
    print("  COMPARAISON DES MODÈLES")
    print(f"{'='*60}")
    for name, res in sorted(results.items(), key=lambda x: x[1]['val_loss']):
        print(f"  {name:20s} → val_loss = {res['val_loss']:.4f}")

    return results
