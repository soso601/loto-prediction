"""
Script principal du projet Loto Prediction.

Modes d'utilisation :
  python main.py train       → Entraîne un nouveau modèle
  python main.py predict     → Prédit le prochain tirage (modèle sauvegardé)
  python main.py update      → Met à jour le modèle avec les nouveaux tirages
  python main.py compare     → Compare les 3 architectures
  python main.py add         → Ajouter un tirage manuellement
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from config import *
from loto_functions import *
from utils import *
from models_functions import *


def prepare_data(force_scrape=False):
    """Pipeline complet : récupération + feature engineering."""
    df_raw = get_draws(force_scrape=force_scrape)

    # Inverser pour avoir les plus anciens en premier
    df = df_raw.iloc[::-1].reset_index(drop=True)
    df = df[ALL_DRAW_COLS]

    # Construction des features
    df_features = build_all_features(df)
    return df_features


def mode_train(model_type='standard'):
    """Entraîne un nouveau modèle depuis zéro."""
    print("\n" + "="*60)
    print("  MODE : ENTRAÎNEMENT COMPLET")
    print("="*60)

    df = prepare_data(force_scrape=True)
    model, scaler, history = train_model(df, model_type=model_type)
    save_model_and_scaler(model, scaler, history)

    # Prédiction de test
    prediction = predict_next_draw(model, scaler, df)
    print(f"\n🎰 Prédiction du prochain tirage :")
    print(f"   Numéros : {prediction[:5]}")
    print(f"   Chance  : {prediction[5]}")

    # Graphique
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Évolution de la perte')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.legend()
    plt.title('Évolution du MAE')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/training_plot.png', dpi=150)
    plt.show()


def mode_predict(n=5):
    """Prédit en utilisant le modèle sauvegardé."""
    print("\n" + "="*60)
    print("  MODE : PRÉDICTION")
    print("="*60)

    if not model_exists():
        print("❌ Aucun modèle sauvegardé. Lance d'abord : python main.py train")
        return

    model, scaler = load_model_and_scaler()
    df = prepare_data()

    predictions = predict_multiple(model, scaler, df, n_predictions=n)

    print(f"\n🎰 {n} prédictions pour le prochain tirage :\n")
    for i, pred in enumerate(predictions, 1):
        nums = ' - '.join(f'{n:2d}' for n in pred[:5])
        print(f"   Grille {i} : {nums}  | Chance : {pred[5]}")

    # Statistiques sur les prédictions
    all_nums = [p[:5] for p in predictions]
    flat = [n for sublist in all_nums for n in sublist]
    from collections import Counter
    freq = Counter(flat)
    print(f"\n   Numéros les plus fréquents dans les prédictions :")
    for num, count in freq.most_common(10):
        print(f"     {num:2d} → {count} fois")

    return predictions


def mode_update():
    """Met à jour le modèle avec les dernières données."""
    print("\n" + "="*60)
    print("  MODE : MISE À JOUR DU MODÈLE")
    print("="*60)

    df = prepare_data(force_scrape=True)
    model, scaler, history = update_model(df)

    prediction = predict_next_draw(model, scaler, df)
    print(f"\n🎰 Prédiction après mise à jour :")
    print(f"   Numéros : {prediction[:5]}")
    print(f"   Chance  : {prediction[5]}")


def mode_compare():
    """Compare les 3 architectures."""
    print("\n" + "="*60)
    print("  MODE : COMPARAISON DES MODÈLES")
    print("="*60)

    df = prepare_data()
    results = compare_models(df)

    # Sauvegarder le meilleur
    best_name = min(results, key=lambda k: results[k]['val_loss'])
    best = results[best_name]
    print(f"\n✅ Meilleur modèle : {best_name} (val_loss={best['val_loss']:.4f})")
    save_model_and_scaler(best['model'], best['scaler'], best['history'])

    return results


def mode_add_draw():
    """Ajoute un tirage manuellement."""
    print("\n" + "="*60)
    print("  MODE : AJOUT D'UN TIRAGE")
    print("="*60)

    day = input("Jour (ex: Lundi) : ")
    month_year = input("Date (ex: 07 avril 2025) : ")
    nums_str = input("5 numéros séparés par des espaces : ")
    nums = [int(x) for x in nums_str.split()]
    chance = int(input("Numéro chance : "))

    add_single_draw(day, month_year, nums, chance)
    print("✅ Tirage ajouté avec succès.")

    # Proposer la mise à jour du modèle
    if model_exists():
        rep = input("\nMettre à jour le modèle ? (o/n) : ")
        if rep.lower() == 'o':
            mode_update()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage : python main.py [train|predict|update|compare|add]")
        print("\nModes :")
        print("  train    → Entraîne un nouveau modèle")
        print("  predict  → Prédit le prochain tirage")
        print("  update   → Met à jour avec les derniers tirages")
        print("  compare  → Compare les 3 architectures")
        print("  add      → Ajoute un tirage manuellement")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'train':
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'standard'
        mode_train(model_type)
    elif mode == 'predict':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        mode_predict(n)
    elif mode == 'update':
        mode_update()
    elif mode == 'compare':
        mode_compare()
    elif mode == 'add':
        mode_add_draw()
    else:
        print(f"Mode inconnu : {mode}")
