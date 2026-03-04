"""
PULSE - Motor Predictivo | Etapa 3: Entrenamiento del Modelo
============================================================
XGBoost con TimeSeriesSplit cross-validation, manejo de desbalance
y serialización del modelo entrenado.
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve,
)
import joblib

from config import (
    TARGET_COL, XGBOOST_PARAMS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    BOOLEAN_FEATURES, TRAIN_RATIO, VAL_RATIO, OUTPUTS_DIR, MODELS_DIR,
    MIN_PR_AUC, MIN_PRECISION, MIN_RECALL, ACTIVE_THRESHOLD,
)

from feature_engineering import temporal_split, prepare_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CÁLCULO DE scale_pos_weight (desbalance de clases)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    XGBoost maneja desbalance con scale_pos_weight = n_negativos / n_positivos.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    log.info(f"  Positivos (fallo): {n_pos:,} | Negativos (éxito): {n_neg:,}")
    log.info(f"  scale_pos_weight: {spw:.2f}")
    return spw


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATION TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_temporal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    TimeSeriesSplit CV sobre el conjunto de training.

    Por qué TimeSeriesSplit y no KFold:
      - KFold mezcla pasado y futuro → data leakage
      - TimeSeriesSplit siempre entrena en datos anteriores y valida en posteriores
    """
    log.info(f"Cross-validation temporal ({n_splits} folds)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_f_train = X_train[train_idx]
        y_f_train = y_train[train_idx]
        X_f_val   = X_train[val_idx]
        y_f_val   = y_train[val_idx]

        # scale_pos_weight para este fold
        spw = compute_scale_pos_weight(y_f_train)
        fold_params = {**params, "scale_pos_weight": spw}

        model = xgb.XGBClassifier(**fold_params)
        model.fit(
            X_f_train, y_f_train,
            eval_set=[(X_f_val, y_f_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_f_val)[:, 1]
        y_pred = (y_prob >= ACTIVE_THRESHOLD).astype(int)

        metrics = {
            "pr_auc":    average_precision_score(y_f_val, y_prob),
            "roc_auc":   roc_auc_score(y_f_val, y_prob),
            "precision": precision_score(y_f_val, y_pred, zero_division=0),
            "recall":    recall_score(y_f_val, y_pred, zero_division=0),
            "f1":        f1_score(y_f_val, y_pred, zero_division=0),
        }
        fold_metrics.append(metrics)

        log.info(
            f"  Fold {fold+1}: PR-AUC={metrics['pr_auc']:.3f} | "
            f"Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f}"
        )

    # Promedios
    avg_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
    }
    std_metrics = {
        f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
    }

    log.info(f"  CV promedio: PR-AUC={avg_metrics['pr_auc']:.3f} ± {std_metrics['pr_auc_std']:.3f}")

    return {**avg_metrics, **std_metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRACIÓN DE THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
) -> Dict[str, float]:
    """
    Encuentra el threshold óptimo según la estrategia elegida.

    Estrategias:
      - "f1": maximiza F1-score (balance precision/recall)
      - "recall": maximiza recall con precision mínima aceptable (≥ MIN_PRECISION)
      - "precision": maximiza precision con recall mínimo aceptable (≥ MIN_RECALL)

    En logística, generalmente "recall" es preferible:
    mejor detectar todos los problemas aunque haya algunas falsas alarmas.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve retorna n+1 valores, el último sin threshold
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    results = {}

    # F1 óptimo
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    results["threshold_f1"] = float(thresholds[best_f1_idx])
    results["f1_at_threshold_f1"] = float(f1_scores[best_f1_idx])

    # Recall máximo con precision mínima
    valid_prec = precisions >= MIN_PRECISION
    if valid_prec.any():
        best_rec_idx = np.argmax(recalls * valid_prec)
        results["threshold_recall"] = float(thresholds[best_rec_idx])
        results["recall_at_threshold_recall"] = float(recalls[best_rec_idx])
        results["precision_at_threshold_recall"] = float(precisions[best_rec_idx])
    else:
        results["threshold_recall"] = ACTIVE_THRESHOLD
        log.warning(f"  No hay threshold que logre precision ≥ {MIN_PRECISION}")

    log.info(f"  Threshold por F1:     {results['threshold_f1']:.3f}")
    log.info(f"  Threshold por Recall: {results.get('threshold_recall', 'N/A')}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    params:  Dict = None,
    run_cv:  bool = True,
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Entrena el modelo final con todos los datos de train.
    Valida sobre el conjunto de validación.
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()

    log.info("=" * 60)
    log.info("PULSE - Entrenamiento XGBoost")
    log.info("=" * 60)

    # ── Cross-validation ──────────────────────────────────────────────────
    cv_metrics = {}
    if run_cv:
        cv_metrics = cross_validate_temporal(X_train, y_train, params)

    # ── Entrenamiento final ───────────────────────────────────────────────
    spw = compute_scale_pos_weight(y_train)
    final_params = {**params, "scale_pos_weight": spw}

    log.info(f"\nEntrenando modelo final...")
    log.info(f"  Parámetros: {final_params}")

    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── Evaluación en Validación ──────────────────────────────────────────
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= ACTIVE_THRESHOLD).astype(int)

    threshold_info = calibrate_threshold(y_val, y_val_prob)

    val_metrics = {
        "pr_auc":    float(average_precision_score(y_val, y_val_prob)),
        "roc_auc":   float(roc_auc_score(y_val, y_val_prob)),
        "precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
        "recall":    float(recall_score(y_val, y_val_pred, zero_division=0)),
        "f1":        float(f1_score(y_val, y_val_pred, zero_division=0)),
        "threshold_used": ACTIVE_THRESHOLD,
        **threshold_info,
    }

    log.info(f"\nMétricas en Validación:")
    log.info(f"  PR-AUC:    {val_metrics['pr_auc']:.3f}   (mínimo: {MIN_PR_AUC})")
    log.info(f"  Precision: {val_metrics['precision']:.3f}  (mínimo: {MIN_PRECISION})")
    log.info(f"  Recall:    {val_metrics['recall']:.3f}   (mínimo: {MIN_RECALL})")
    log.info(f"  ROC-AUC:   {val_metrics['roc_auc']:.3f}")
    log.info(f"  F1:        {val_metrics['f1']:.3f}")

    # Verificar targets del PoC
    targets_ok = True
    if val_metrics["pr_auc"] < MIN_PR_AUC:
        log.warning(f"  ⚠ PR-AUC {val_metrics['pr_auc']:.3f} < {MIN_PR_AUC} (target PoC)")
        targets_ok = False
    if val_metrics["precision"] < MIN_PRECISION:
        log.warning(f"  ⚠ Precision {val_metrics['precision']:.3f} < {MIN_PRECISION} (target PoC)")
        targets_ok = False
    if val_metrics["recall"] < MIN_RECALL:
        log.warning(f"  ⚠ Recall {val_metrics['recall']:.3f} < {MIN_RECALL} (target PoC)")
        targets_ok = False

    if targets_ok:
        log.info("  ✓ Todos los targets del PoC cumplidos")
    else:
        log.warning("  → Considerar: más datos, feature engineering, ajuste de hiperparámetros")

    all_metrics = {
        "validation": val_metrics,
        "cross_validation": cv_metrics,
        "params": {k: str(v) for k, v in final_params.items()},
        "scale_pos_weight": spw,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "n_features": X_train.shape[1],
        "targets_met": targets_ok,
    }

    return model, all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SERIALIZACIÓN Y VERSIONADO
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(
    model: xgb.XGBClassifier,
    metrics: Dict,
    feature_names: list,
    version: str = None,
) -> Path:
    """
    Guarda el modelo con versionado temporal.
    Formato: model_v{N}_{fecha}.pkl + metrics_v{N}_{fecha}.json
    """
    MODELS_DIR.mkdir(exist_ok=True)

    if version is None:
        date_str = datetime.today().strftime("%Y-%m-%d")
        # Determinar número de versión
        existing = list(MODELS_DIR.glob("model_v*.pkl"))
        v_num = len(existing) + 1
        version = f"v{v_num}_{date_str}"

    model_path    = MODELS_DIR / f"model_{version}.pkl"
    metrics_path  = MODELS_DIR / f"metrics_{version}.json"
    features_path = MODELS_DIR / f"features_{version}.json"

    joblib.dump(model, model_path)
    log.info(f"Modelo guardado: {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Métricas guardadas: {metrics_path}")

    with open(features_path, "w") as f:
        json.dump({"features": feature_names, "version": version}, f, indent=2)
    log.info(f"Features guardadas: {features_path}")

    # Copiar como "latest" (en Windows no funcionan symlinks sin privilegios)
    import shutil
    for src, dst_name in [
        (model_path, "model_latest.pkl"),
        (metrics_path, "metrics_latest.json"),
        (features_path, "features_latest.json"),
    ]:
        dst = MODELS_DIR / dst_name
        shutil.copy2(src, dst)

    log.info(f"→ model_latest.pkl actualizado a {version}")
    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from build_dataset import build_dataset

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 1. Construir dataset
    df, feature_cols = build_dataset()

    # 2. Split temporal
    train, val, test = temporal_split(df)

    # 3. Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, prep, feat_names, rates = prepare_features(
        train, val, test, feature_cols
    )

    # 4. Guardar X_train para SHAP después
    import scipy.sparse as sp
    if sp.issparse(X_train):
        sp.save_npz(OUTPUTS_DIR / "X_train.npz", X_train)
    else:
        np.save(OUTPUTS_DIR / "X_train.npy", X_train)

    # 5. Entrenar
    model, metrics = train_model(X_train, y_train, X_val, y_val, run_cv=True)

    # 6. Guardar
    model_path = save_model(model, metrics, feat_names)

    # 7. Guardar X_test/y_test para evaluación final
    if sp.issparse(X_test):
        sp.save_npz(OUTPUTS_DIR / "X_test.npz", X_test)
    else:
        np.save(OUTPUTS_DIR / "X_test.npy", X_test)
    
    if sp.issparse(X_val):
        sp.save_npz(OUTPUTS_DIR / "X_val.npz", X_val)
    else:
        np.save(OUTPUTS_DIR / "X_val.npy", X_val)
        
    np.save(OUTPUTS_DIR / "y_test.npy", y_test)
    np.save(OUTPUTS_DIR / "y_val.npy", y_val)

    print(f"\n{'='*50}")
    print(f"Entrenamiento completo. Modelo: {model_path}")
    print(f"PR-AUC (val): {metrics['validation']['pr_auc']:.3f}")
    print(f"Recall  (val): {metrics['validation']['recall']:.3f}")
    print(f"Siguiente paso: python evaluate.py")
