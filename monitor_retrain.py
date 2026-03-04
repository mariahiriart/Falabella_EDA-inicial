"""
PULSE - Motor Predictivo | Etapa 6: Detección de Drift y Reentrenamiento
========================================================================
Monitorea el modelo en producción y dispara reentrenamiento automático
cuando detecta degradación de performance o drift en los datos.

Ejecutar periódicamente (cron semanal recomendado):
  python monitor_retrain.py
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from scipy import stats
import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score, recall_score

from config import (
    TARGET_COL, RETRAIN_TRIGGERS, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR,
    MIN_PR_AUC, MIN_RECALL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LOGS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE PREDICCIONES RECIENTES (ventana deslizante)
# ═══════════════════════════════════════════════════════════════════════════════

def load_recent_predictions(
    days: int = None,
    predictions_log_path: Path = LOGS_DIR / "predictions_log.csv",
) -> Optional[pd.DataFrame]:
    """
    Carga el log de predicciones de los últimos N días.

    El log debe ser generado por la API de inferencia con este formato:
    order_id | timestamp | risk_score | predicted_outcome | actual_outcome | features...

    actual_outcome se llena retrospectivamente cuando se confirma el resultado.
    """
    if days is None:
        days = RETRAIN_TRIGGERS["sliding_window_days"]

    if not predictions_log_path.exists():
        log.warning(f"No existe log de predicciones en {predictions_log_path}")
        log.info("→ Asegurarse de que inference_api.py está logueando predicciones")
        return None

    df = pd.read_csv(predictions_log_path, parse_dates=["timestamp"])
    cutoff = datetime.now() - timedelta(days=days)
    recent = df[df["timestamp"] >= cutoff].copy()

    log.info(f"Predicciones recientes ({days}d): {len(recent):,}")
    if "actual_outcome" in recent.columns:
        labeled = recent["actual_outcome"].notna().sum()
        log.info(f"  Con resultado real (labeled): {labeled:,} ({labeled/max(len(recent),1)*100:.1f}%)")

    return recent


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MONITOREO DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def check_performance_drift(
    recent_df: pd.DataFrame,
    baseline_metrics: Dict,
) -> Tuple[bool, Dict]:
    """
    Compara métricas recientes vs baseline del modelo.

    Trigger de reentrenamiento si:
      - PR-AUC cae > 10% vs baseline
      - Recall cae por debajo de MIN_RECALL (0.70)
    """
    results = {"trigger": False, "reasons": [], "metrics": {}}

    labeled = recent_df.dropna(subset=["actual_outcome"])
    if len(labeled) < 100:
        log.info(f"  Insuficientes datos etiquetados ({len(labeled)}). Mínimo: 100")
        return False, results

    y_true = labeled["actual_outcome"].astype(int)
    y_prob = labeled["risk_score"]
    y_pred = (y_prob >= 0.5).astype(int)

    current_pr_auc = average_precision_score(y_true, y_prob)
    current_recall = recall_score(y_true, y_pred, zero_division=0)

    baseline_pr_auc = baseline_metrics.get("pr_auc", MIN_PR_AUC)
    pr_auc_drop = (baseline_pr_auc - current_pr_auc) / max(baseline_pr_auc, 1e-8)

    results["metrics"] = {
        "current_pr_auc":    round(current_pr_auc, 4),
        "baseline_pr_auc":   round(baseline_pr_auc, 4),
        "pr_auc_drop_pct":   round(pr_auc_drop * 100, 2),
        "current_recall":    round(current_recall, 4),
        "n_labeled_samples": len(labeled),
    }

    log.info(f"  PR-AUC actual: {current_pr_auc:.3f} | baseline: {baseline_pr_auc:.3f} | drop: {pr_auc_drop*100:.1f}%")
    log.info(f"  Recall actual: {current_recall:.3f} | mínimo: {MIN_RECALL}")

    if pr_auc_drop > RETRAIN_TRIGGERS["pr_auc_drop_pct"]:
        msg = f"PR-AUC cayó {pr_auc_drop*100:.1f}% (> {RETRAIN_TRIGGERS['pr_auc_drop_pct']*100}%)"
        results["reasons"].append(msg)
        results["trigger"] = True
        log.warning(f"  ⚠ TRIGGER: {msg}")

    if current_recall < RETRAIN_TRIGGERS["min_recall"]:
        msg = f"Recall {current_recall:.3f} < {RETRAIN_TRIGGERS['min_recall']}"
        results["reasons"].append(msg)
        results["trigger"] = True
        log.warning(f"  ⚠ TRIGGER: {msg}")

    return results["trigger"], results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETECCIÓN DE DRIFT EN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def check_feature_drift(
    recent_df: pd.DataFrame,
    reference_stats_path: Path = MODELS_DIR / "reference_stats.json",
) -> Tuple[bool, Dict]:
    """
    Kolmogorov-Smirnov test para detectar drift en distribución de features.
    """
    results = {"trigger": False, "drifted_features": [], "ks_results": {}}

    if not reference_stats_path.exists():
        log.warning(f"No hay estadísticas de referencia en {reference_stats_path}")
        log.info("→ Ejecutar save_reference_stats() después del entrenamiento")
        return False, results

    with open(reference_stats_path) as f:
        ref_stats = json.load(f)

    drifted = []
    numeric_cols = recent_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col not in ref_stats:
            continue
        if recent_df[col].notna().sum() < 30:
            continue

        ref = ref_stats[col]
        recent_vals = recent_df[col].dropna().values

        ref_sample = np.interp(
            np.linspace(0, 1, 1000),
            np.linspace(0, 1, len(ref["percentiles"])),
            ref["percentiles"]
        )

        ks_stat, p_value = stats.ks_2samp(ref_sample, recent_vals)
        results["ks_results"][col] = {
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < RETRAIN_TRIGGERS["ks_pvalue_threshold"],
        }

        if p_value < RETRAIN_TRIGGERS["ks_pvalue_threshold"]:
            drifted.append(col)
            log.warning(f"  ⚠ Drift en {col}: KS={ks_stat:.3f}, p={p_value:.4f}")

    results["drifted_features"] = drifted

    critical = ["complexity_score", "num_transfers", "total_span_hours",
                "tasa_fallo_carrier", "porcentaje_eta_consumido"]
    critical_drifted = [f for f in drifted if f in critical]

    if len(critical_drifted) >= 2:
        results["trigger"] = True
        msg = f"Drift crítico en: {critical_drifted}"
        log.warning(f"  ⚠ TRIGGER: {msg}")

    log.info(f"  Features con drift: {len(drifted)} | Críticas: {len(critical_drifted)}")
    return results["trigger"], results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GUARDAR ESTADÍSTICAS DE REFERENCIA
# ═══════════════════════════════════════════════════════════════════════════════

def save_reference_stats(
    X_train: np.ndarray,
    feature_names: list,
    save_path: Path = MODELS_DIR / "reference_stats.json",
) -> None:
    """
    Guarda estadísticas descriptivas del training set como referencia
    para detección de drift futura.
    """
    stats_dict = {}
    df = pd.DataFrame(X_train, columns=feature_names)

    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        stats_dict[col] = {
            "mean":        float(vals.mean()),
            "std":         float(vals.std()),
            "min":         float(vals.min()),
            "max":         float(vals.max()),
            "percentiles": [float(p) for p in np.percentile(vals, np.linspace(0, 100, 20))],
        }

    with open(save_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    log.info(f"Estadísticas de referencia guardadas: {save_path} ({len(stats_dict)} features)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DETECCIÓN DE NUEVAS CATEGORÍAS
# ═══════════════════════════════════════════════════════════════════════════════

def check_new_categories(
    recent_df: pd.DataFrame,
    known_categories_path: Path = MODELS_DIR / "known_categories.json",
) -> Tuple[bool, Dict]:
    """Detecta si aparecieron nuevos carriers o regiones que el modelo no conoce."""
    results = {"trigger": False, "new_values": {}}

    if not known_categories_path.exists():
        return False, results

    with open(known_categories_path) as f:
        known = json.load(f)

    for col in ["carrier", "region", "comuna"]:
        if col not in recent_df.columns or col not in known:
            continue
        current_vals = set(recent_df[col].dropna().unique())
        known_vals = set(known[col])
        new_vals = current_vals - known_vals

        if new_vals:
            results["new_values"][col] = list(new_vals)
            log.warning(f"  ⚠ Nuevos valores en {col}: {new_vals}")

    if results["new_values"]:
        results["trigger"] = True
        log.warning(f"  ⚠ TRIGGER: Categorías desconocidas detectadas")

    return results["trigger"], results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORQUESTADOR DE MONITOREO
# ═══════════════════════════════════════════════════════════════════════════════

def run_monitoring_check() -> Dict:
    """Ejecuta todos los checks de monitoreo y decide si reentrenar."""
    log.info("=" * 60)
    log.info("PULSE - Monitoreo del Modelo")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info("=" * 60)

    metrics_path = MODELS_DIR / "metrics_latest.json"
    baseline_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        baseline_metrics = data.get("validation", {})

    recent_df = load_recent_predictions()

    should_retrain = False
    all_results = {
        "timestamp":    datetime.now().isoformat(),
        "checks":       {},
        "should_retrain": False,
        "retrain_reasons": [],
    }

    if recent_df is not None and len(recent_df) > 0:
        log.info("\n[1/3] Verificando performance...")
        perf_trigger, perf_results = check_performance_drift(recent_df, baseline_metrics)
        all_results["checks"]["performance"] = perf_results
        if perf_trigger:
            should_retrain = True
            all_results["retrain_reasons"].extend(perf_results.get("reasons", []))

        log.info("\n[2/3] Verificando drift de features...")
        drift_trigger, drift_results = check_feature_drift(recent_df)
        all_results["checks"]["feature_drift"] = drift_results
        if drift_trigger:
            should_retrain = True
            all_results["retrain_reasons"].append(
                f"Drift en features: {drift_results.get('drifted_features', [])}"
            )

        log.info("\n[3/3] Verificando nuevas categorías...")
        cat_trigger, cat_results = check_new_categories(recent_df)
        all_results["checks"]["new_categories"] = cat_results
        if cat_trigger:
            should_retrain = True
            all_results["retrain_reasons"].append(
                f"Nuevas categorías: {cat_results.get('new_values', {})}"
            )
    else:
        log.info("Sin datos recientes disponibles para monitoreo")

    all_results["should_retrain"] = should_retrain

    monitor_log = LOGS_DIR / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(monitor_log, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n{'='*60}")
    if should_retrain:
        log.warning(f"⚠ REENTRENAMIENTO RECOMENDADO")
        for reason in all_results["retrain_reasons"]:
            log.warning(f"  → {reason}")
        log.info("Ejecutar: python train_model.py para actualizar el modelo")
    else:
        log.info("✓ Modelo estable. No se requiere reentrenamiento.")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_monitoring_check()

    if results["should_retrain"]:
        print("\n⚠ ACCIÓN REQUERIDA: Reentrenar el modelo")
        print(f"Razones: {results['retrain_reasons']}")
        print("Ejecutar: python train_model.py")
    else:
        print("\n✓ Modelo OK. Próximo check en 7 días.")
