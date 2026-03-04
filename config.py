"""
PULSE - Motor Predictivo (Componente 2)
Configuración central del proyecto.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR    = BASE_DIR / "logs"

# Inputs — archivos en la raíz del proyecto (nombres reales)
MUESTRA_FO_PATH     = BASE_DIR / "Muestra FO (1).xlsb"     # 263k packages
PULSE_FEATURES_PATH = BASE_DIR / "falabella_eda_features.csv"  # 5.591 orders
EVENT_LOG_PATH      = BASE_DIR / "data" / "event_log.csv"   # ← enchufar cuando llegue

# ─── Columnas Muestra FO (nombres reales del archivo .xlsb) ──────────────────
# Mapeo: nombre_interno → nombre_real_en_xlsb
FO_COLS = {
    "order_id":        "order_compra",
    "producto":        "producto",
    "package_id":      "package_id",
    "estado_producto": "estado_producto",
    "fecha_estado_producto": "fecha_estado_producto",
    "estado_package":  "estado_package",
    "fecha_estado_package":  "fecha_estado_package",
    "eta_desde":       "fecha_pactada_desde",
    "eta_promised":    "fecha_pactada_hasta",       # fin de ventana = ETA comprometida
    "fecha_retiro":    "fecha_pactada_retiro",
    "comuna":          "comuna",
    "region":          "region",
    "tienda_retiro":   "tienda_retiro",
    "carrier":         "nombre_transporte",
    "tipo_transporte": "tipo_transporte",
    "estado_entrega":  "estado_entrega",
    "metodo_despacho": "metodo_despacho",
    "origen":          "origen_despacho",
}

# Estados que cuentan como FALLO (incumplimiento ETA)
ESTADOS_FALLO = [
    "Entregada atrasada",
]

# Estados que cuentan como ÉXITO
ESTADOS_EXITO = [
    "Entregada antes de tiempo",
    "Entregada a tiempo",
]

# ─── Target ───────────────────────────────────────────────────────────────────
TARGET_COL = "target_incumplimiento"
# 1 = incumplió ETA (Entregada atrasada o sin entregar)
# 0 = cumplió ETA (Entregada a tiempo o antes)

# ─── Features baseline (sin event log) ───────────────────────────────────────
CATEGORICAL_FEATURES = [
    "carrier",
    "region",
    "comuna",
    "metodo_despacho",
    "tipo_transporte",
    "origen",
]

NUMERICAL_FEATURES = [
    # Temporales derivados de fecha
    "dia_semana",
    "hora_creacion",
    "mes",
    # De PULSE features (cuando se pueda mergear)
    "num_ops",
    "num_processes",
    "num_legs",
    "num_transfers",
    "num_operators",
    "total_span_hours",
    "picking_window_hours",
    "total_process_time_mins",
    "complexity_score",
    "span_ratio",
]

BOOLEAN_FEATURES = [
    "es_fin_semana",
    # De PULSE features (cuando se pueda mergear)
    "has_crossdocking",
    "has_first_mile",
    "short_picking_window",
    "anomalous_span",
    "complex_route",
    "seller_origin",
]

# ─── Features temporales (event log) ─────────────────────────────────────────
# Se agregan automáticamente cuando EVENT_LOG_PATH existe
TEMPORAL_FEATURES = [
    "tiempo_desde_creacion",
    "tiempo_hasta_eta",
    "porcentaje_eta_consumido",
    "velocidad_flujo",
    "tiempo_ultimo_evento",
    "desviacion_vs_itinerario",
    "num_eventos",
    "ultimo_estado_codigo",
]

# ─── Modelo XGBoost ───────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     100,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "aucpr",
    "random_state":     42,
    "n_jobs":           -1,
}

# ─── Umbrales de riesgo ───────────────────────────────────────────────────────
THRESHOLDS = {
    "low":     0.3,   # más alertas, menos precision
    "default": 0.5,   # balance
    "high":    0.7,   # menos alertas, más precision
}
ACTIVE_THRESHOLD = THRESHOLDS["default"]

# ─── Split temporal ───────────────────────────────────────────────────────────
# No usar random split — respetar orden temporal
TRAIN_RATIO = 0.67   # primeros 67% meses
VAL_RATIO   = 0.17   # siguientes 17%
TEST_RATIO  = 0.16   # últimos 16%

# ─── Targets mínimos PoC ─────────────────────────────────────────────────────
MIN_PR_AUC   = 0.75
MIN_PRECISION = 0.70
MIN_RECALL   = 0.80

# ─── Reentrenamiento ─────────────────────────────────────────────────────────
RETRAIN_TRIGGERS = {
    "pr_auc_drop_pct":  0.10,   # si cae >10%
    "min_recall":       0.70,
    "ks_pvalue_threshold": 0.05,  # Kolmogorov-Smirnov drift
    "sliding_window_days": 14,
}
