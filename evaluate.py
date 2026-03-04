"""
PULSE - Motor Predictivo | Etapa 4: Evaluación y Explicabilidad
===============================================================
Evaluación final en test set + SHAP para interpretabilidad de predicciones.
"""

import pandas as pd
import numpy as np
import logging
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

import xgboost as xgb
import shap
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)
import joblib

from config import (
    TARGET_COL, ACTIVE_THRESHOLD, THRESHOLDS,
    MIN_PR_AUC, MIN_PRECISION, MIN_RECALL,
    OUTPUTS_DIR, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MÉTRICAS EN TEST SET
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Evaluación completa en el test set (datos más recientes = simulación real).
    El test set nunca debe tocarse hasta este momento.
    """
    log.info("=" * 60)
    log.info("PULSE - Evaluación Final en Test Set")
    log.info("=" * 60)

    y_prob = model.predict_proba(X_test)[:, 1]

    results = {}

    # Métricas para cada threshold
    for name, thresh in THRESHOLDS.items():
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        results[f"threshold_{name}"] = {
            "threshold": thresh,
            "pr_auc":    float(average_precision_score(y_test, y_prob)),
            "roc_auc":   float(roc_auc_score(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "tasa_deteccion": float(tp / max(tp + fn, 1)),  # = recall
            "tasa_falsa_alarma": float(fp / max(fp + tn, 1)),
        }

        log.info(f"\nThreshold {name} ({thresh}):")
        log.info(f"  Precision: {results[f'threshold_{name}']['precision']:.3f}")
        log.info(f"  Recall:    {results[f'threshold_{name}']['recall']:.3f}")
        log.info(f"  F1:        {results[f'threshold_{name}']['f1']:.3f}")
        log.info(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

    # Métricas globales (independientes del threshold)
    pr_auc  = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    results["global"] = {"pr_auc": pr_auc, "roc_auc": roc_auc}

    log.info(f"\n{'='*40}")
    log.info(f"PR-AUC (global): {pr_auc:.3f} | Target: ≥{MIN_PR_AUC}")
    log.info(f"ROC-AUC:         {roc_auc:.3f}")

    # Recomendación de threshold para producción
    active = results["threshold_default"]
    if active["recall"] >= MIN_RECALL and active["precision"] >= MIN_PRECISION:
        log.info("✓ Threshold default (0.5) cumple targets del PoC")
    else:
        log.warning("⚠ Threshold default no cumple todos los targets — revisar calibración")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SHAP - EXPLICABILIDAD GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shap_global(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    feature_names: List[str],
    max_samples: int = 2000,
) -> shap.Explainer:
    """
    Calcula SHAP values sobre una muestra del training set.
    Genera el gráfico de importancia global.
    """
    log.info("Calculando SHAP values globales...")

    # Muestra aleatoria para eficiencia
    n_samples = X_train.shape[0]
    idx = np.random.choice(n_samples, min(max_samples, n_samples), replace=False)
    X_sample = X_train[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Importancia media |SHAP| por feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    log.info(f"Top 10 features por SHAP:")
    for _, row in importance_df.head(10).iterrows():
        log.info(f"  {row['feature']:40s}: {row['mean_abs_shap']:.4f}")

    # Gráfico de beeswarm (resumen global)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("PULSE - Importancia Global de Features (SHAP)")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Gráfico SHAP guardado: {OUTPUTS_DIR / 'shap_global.png'}")

    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    return explainer, importance_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHAP - EXPLICACIÓN POR PREDICCIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════════

def explain_prediction(
    explainer: shap.TreeExplainer,
    X_instance: np.ndarray,
    feature_names: List[str],
    top_n: int = 3,
) -> List[Dict]:
    """
    Genera los top-N factores contribuyentes para una predicción individual.

    Esta función es la que usa el Motor Predictivo en inference:
    retorna los 'contributing_factors' del output del Componente 2.
    """
    shap_vals = explainer.shap_values(X_instance)[0]

    factors = []
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]

    for idx in top_indices:
        shap_val = float(shap_vals[idx])
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"

        factors.append({
            "feature":     feat_name,
            "shap_value":  round(shap_val, 4),
            "direction":   "incrementa_riesgo" if shap_val > 0 else "reduce_riesgo",
            "description": _human_readable_factor(feat_name, shap_val),
        })

    return factors


def _human_readable_factor(feature: str, shap_value: float) -> str:
    """
    Traduce nombre técnico de feature a descripción legible para operadores.
    """
    direction = "incrementa" if shap_value > 0 else "reduce"

    descriptions = {
        "carrier":               f"Transportista seleccionado {direction} el riesgo",
        "region":                f"Región de entrega {direction} el riesgo",
        "comuna":                f"Comuna de destino {direction} el riesgo",
        "tasa_fallo_carrier":    f"Historial del transportista {direction} el riesgo",
        "tasa_fallo_region":     f"Historial de la región {direction} el riesgo",
        "complexity_score":      f"Complejidad de la ruta {direction} el riesgo",
        "num_transfers":         f"Cantidad de transbordos {direction} el riesgo",
        "total_span_hours":      f"Ventana total de proceso {direction} el riesgo",
        "picking_window_hours":  f"Ventana de picking {direction} el riesgo",
        "has_crossdocking":      f"Operación con crossdocking {direction} el riesgo",
        "dia_semana":            f"Día de la semana {direction} el riesgo",
        "metodo_despacho":       f"Método de despacho {direction} el riesgo",
        "tipo_transporte":       f"Tipo de transporte {direction} el riesgo",
        "porcentaje_eta_consumido": f"Porcentaje de ETA consumido {direction} el riesgo",
        "tiempo_ultimo_evento":  f"Tiempo sin actividad {direction} el riesgo",
        "desviacion_vs_itinerario": f"Desvío vs itinerario {direction} el riesgo",
    }

    # Buscar coincidencia parcial
    for key, desc in descriptions.items():
        if key in feature:
            return desc

    return f"{feature} {direction} el riesgo"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CURVAS PR Y ROC
# ═══════════════════════════════════════════════════════════════════════════════

def plot_evaluation_curves(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Genera curvas PR y ROC para el test set.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Curva PR ─────────────────────────────────────────────────────────
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    axes[0].plot(recall_vals, precision_vals, "b-", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[0].axhline(y=MIN_PRECISION, color="r", linestyle="--", alpha=0.5, label=f"Min Precision ({MIN_PRECISION})")
    axes[0].axvline(x=MIN_RECALL,    color="g", linestyle="--", alpha=0.5, label=f"Min Recall ({MIN_RECALL})")
    axes[0].fill_between(recall_vals, precision_vals, alpha=0.1)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Curva Precision-Recall (Test Set)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Curva ROC ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    axes[1].plot(fpr, tpr, "b-", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    axes[1].fill_between(fpr, tpr, alpha=0.1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Curva ROC (Test Set)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("PULSE - Motor Predictivo | Evaluación Final", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = OUTPUTS_DIR / "evaluation_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Curvas guardadas: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Cargar modelo y artefactos
    model = joblib.load(MODELS_DIR / "model_latest.pkl")
    with open(MODELS_DIR / "features_latest.json") as f:
        feat_data = json.load(f)
    feature_names = feat_data["features"]

    # Cargar test set
    import scipy.sparse as sp
    
    if (OUTPUTS_DIR / "X_test.npz").exists():
        X_test = sp.load_npz(OUTPUTS_DIR / "X_test.npz")
    else:
        X_test = np.load(OUTPUTS_DIR / "X_test.npy", allow_pickle=True)
        
    y_test = np.load(OUTPUTS_DIR / "y_test.npy", allow_pickle=True)
    
    if (OUTPUTS_DIR / "X_train.npz").exists():
        X_train = sp.load_npz(OUTPUTS_DIR / "X_train.npz")
    else:
        X_train = np.load(OUTPUTS_DIR / "X_train.npy", allow_pickle=True) if \
                  (OUTPUTS_DIR / "X_train.npy").exists() else X_test  # fallback

    # Evaluar
    test_results = evaluate_on_test(model, X_test, y_test, feature_names)

    # SHAP global
    explainer, importance_df = compute_shap_global(model, X_train, feature_names)
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")

    # Curvas
    plot_evaluation_curves(model, X_test, y_test)

    # Guardar resultados
    with open(OUTPUTS_DIR / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Ejemplo de explicación individual (primera predicción del test)
    sample_explanation = explain_prediction(explainer, X_test[:1], feature_names, top_n=3)
    log.info(f"\nEjemplo de explicación individual:")
    for factor in sample_explanation:
        log.info(f"  {factor['description']} (SHAP: {factor['shap_value']:.4f})")

    print(f"\n{'='*50}")
    print("Evaluación completa.")
    print(f"PR-AUC test: {test_results['global']['pr_auc']:.3f}")
    print(f"Siguiente paso: python inference_api.py")
