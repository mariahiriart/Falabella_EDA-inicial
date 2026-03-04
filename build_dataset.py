"""
PULSE - Motor Predictivo | Etapa 1: Carga y Preprocesamiento de Datos
=====================================================================
Lee Muestra_FO + PULSE features, construye el target de incumplimiento
y genera el dataset consolidado para entrenamiento.

Cuando llegue el event log, descomentar la sección EVENT LOG.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

from config import (
    MUESTRA_FO_PATH, PULSE_FEATURES_PATH, EVENT_LOG_PATH,
    FO_COLS, ESTADOS_FALLO, ESTADOS_EXITO, TARGET_COL,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BOOLEAN_FEATURES, TEMPORAL_FEATURES,
    OUTPUTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA MUESTRA FO
# ═══════════════════════════════════════════════════════════════════════════════

def load_muestra_fo(path: Path = MUESTRA_FO_PATH) -> pd.DataFrame:
    """
    Carga el archivo Muestra_FO (.xlsb).
    Requiere pyxlsb: pip install pyxlsb
    """
    log.info(f"Cargando Muestra FO desde {path}...")

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Verificá que el archivo 'Muestra FO (1).xlsb' esté en la raíz del proyecto."
        )

    # xlsb binario → pyxlsb engine
    df = pd.read_excel(path, engine="pyxlsb")
    log.info(f"  Filas cargadas: {len(df):,} | Columnas: {df.shape[1]}")
    log.info(f"  Columnas originales: {df.columns.tolist()}")

    # Renombrar a nombres internos estándar
    col_map = {v: k for k, v in FO_COLS.items()}  # {nombre_real: nombre_interno}

    # Verificar que las columnas existan
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        log.warning(
            f"Columnas no encontradas en Muestra FO: {missing}\n"
            f"Columnas disponibles: {df.columns.tolist()}\n"
            f"→ Actualizar FO_COLS en config.py"
        )
        col_map = {k: v for k, v in col_map.items() if k in df.columns}

    df = df.rename(columns=col_map)

    # ── Parsear fechas ────────────────────────────────────────────────────
    # fecha_estado_producto y fecha_estado_package vienen como strings ISO
    for date_col in ["fecha_estado_producto", "fecha_estado_package"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # fecha_pactada_desde/hasta vienen como números serial de Excel
    for date_col in ["eta_desde", "eta_promised", "fecha_retiro"]:
        if date_col in df.columns:
            # Convertir serial de Excel a datetime
            numeric_mask = pd.to_numeric(df[date_col], errors="coerce").notna()
            df.loc[numeric_mask, date_col] = pd.to_datetime(
                pd.to_numeric(df.loc[numeric_mask, date_col], errors="coerce"),
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
            # Intentar parsear los que ya son strings
            non_numeric_mask = ~numeric_mask & df[date_col].notna()
            if non_numeric_mask.any():
                df.loc[non_numeric_mask, date_col] = pd.to_datetime(
                    df.loc[non_numeric_mask, date_col], errors="coerce"
                )
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Usar fecha_estado_producto como proxy de fecha_creacion
    if "fecha_estado_producto" in df.columns:
        df["fecha_creacion"] = df["fecha_estado_producto"]

    log.info(
        f"  Rango temporal: {df['fecha_creacion'].min()} → {df['fecha_creacion'].max()}"
        if "fecha_creacion" in df.columns else "  Sin fecha_creacion"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCCIÓN DEL TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la variable objetivo binaria:
      1 = incumplió ETA ("Entregada atrasada" o no entregada aún)
      0 = cumplió ETA ("Entregada a tiempo" o "Entregada antes de tiempo")

    Basado en los valores reales de estado_entrega del archivo Muestra FO.
    """
    df = df.copy()

    if "estado_entrega" not in df.columns:
        log.error("Columna 'estado_entrega' no encontrada. No se puede construir target.")
        df[TARGET_COL] = np.nan
        return df

    # Target basado en los estados reales del archivo
    # Éxito = "Entregada antes de tiempo" o "Entregada a tiempo"
    # Fallo = "Entregada atrasada" o cualquier otro estado (no entregado)
    es_exito = df["estado_entrega"].isin(ESTADOS_EXITO)
    es_fallo = df["estado_entrega"].isin(ESTADOS_FALLO)

    # Solo asignar target a filas con estado conocido (éxito o fallo)
    # Filas con estado desconocido (NaN o pendiente) quedan NaN y se filtran
    df[TARGET_COL] = np.nan
    df.loc[es_exito, TARGET_COL] = 0
    df.loc[es_fallo, TARGET_COL] = 1

    # Filtrar solo las filas con target definido
    n_total = len(df)
    n_with_target = df[TARGET_COL].notna().sum()
    n_dropped = n_total - n_with_target
    log.info(f"  Filas con target definido: {n_with_target:,} / {n_total:,} (descartadas: {n_dropped:,})")

    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Métricas del target
    total = len(df)
    n_fallo = df[TARGET_COL].sum()
    n_exito = total - n_fallo
    log.info(f"  Target construido:")
    log.info(f"    Total:    {total:,}")
    log.info(f"    Fallos:   {n_fallo:,} ({n_fallo/total*100:.1f}%)")
    log.info(f"    Éxitos:   {n_exito:,} ({n_exito/total*100:.1f}%)")
    log.info(f"    scale_pos_weight sugerido: {n_exito/max(n_fallo,1):.2f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURES TEMPORALES DERIVADAS (desde Muestra FO)
# ═══════════════════════════════════════════════════════════════════════════════

def add_temporal_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae features temporales derivadas de las fechas disponibles en Muestra FO.
    """
    df = df.copy()

    if "fecha_creacion" in df.columns:
        df["dia_semana"]    = df["fecha_creacion"].dt.dayofweek    # 0=lunes
        df["hora_creacion"] = df["fecha_creacion"].dt.hour
        df["mes"]           = df["fecha_creacion"].dt.month
        df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    else:
        log.warning("Sin fecha_creacion → no se generan features temporales")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CARGA PULSE FEATURES (itinerario)
# ═══════════════════════════════════════════════════════════════════════════════

def load_pulse_features(path: Path = PULSE_FEATURES_PATH) -> pd.DataFrame:
    """
    Carga falabella_eda_features.csv (5.591 orders con 24 features de itinerario).

    NOTA: El merge con Muestra FO no es posible ahora porque:
      - PULSE usa order_id formato FOFCL000028XXXXXX
      - Muestra FO usa order_compra (numérico)
    Se requiere tabla de mapeo de Falabella. Mientras tanto, las features
    PULSE quedan disponibles pero NO se usan en el modelo.
    """
    log.info(f"Cargando PULSE features desde {path}...")

    if not path.exists():
        log.warning(f"No se encontró {path}. Omitiendo PULSE features.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"  PULSE features: {len(df):,} orders | {df.shape[1]} columnas")

    # Limpiar span_ratio (1 null detectado en EDA)
    if "span_ratio" in df.columns:
        df["span_ratio"] = df["span_ratio"].fillna(df["span_ratio"].median())

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVENT LOG FEATURES (ENCHUFAR CUANDO LLEGUE)
# ═══════════════════════════════════════════════════════════════════════════════

def add_event_log_features(
    df: pd.DataFrame,
    event_log_path: Path = EVENT_LOG_PATH
) -> pd.DataFrame:
    """
    Agrega features del event log cuando esté disponible.

    Formato esperado del event log (una fila por evento):
    ┌──────────────┬──────────────────┬─────────────────────┬──────────────┐
    │ order_id     │ event_type       │ event_timestamp     │ event_data   │
    │ FOFCL000...  │ packed           │ 2025-10-15 14:32:00 │ {...}        │
    └──────────────┴──────────────────┴─────────────────────┴──────────────┘
    """
    if not event_log_path.exists():
        log.info("Event log no disponible todavía — omitiendo features temporales.")
        log.info("→ Cuando llegue, copiarlo a data/event_log.csv y re-ejecutar.")
        return df

    log.info(f"Procesando event log desde {event_log_path}...")

    events = pd.read_csv(event_log_path, parse_dates=["event_timestamp"])
    log.info(f"  Eventos cargados: {len(events):,}")

    # ── Agregaciones por order_id ─────────────────────────────────────────
    agg = events.groupby("order_id").agg(
        num_eventos       = ("event_type", "count"),
        primer_evento_ts  = ("event_timestamp", "min"),
        ultimo_evento_ts  = ("event_timestamp", "max"),
        ultimo_tipo       = ("event_type", "last"),
    ).reset_index()

    # ── Join con datos principales ────────────────────────────────────────
    df = df.merge(agg, on="order_id", how="left")
    now = pd.Timestamp.now()

    if "fecha_creacion" in df.columns:
        df["tiempo_desde_creacion"] = (
            (df["ultimo_evento_ts"] - df["fecha_creacion"])
            .dt.total_seconds() / 3600
        )

    if "eta_promised" in df.columns:
        df["tiempo_hasta_eta"] = (
            (df["eta_promised"] - df["ultimo_evento_ts"])
            .dt.total_seconds() / 3600
        ).clip(lower=0)

        total_window = (df["eta_promised"] - df["fecha_creacion"]).dt.total_seconds() / 3600
        df["porcentaje_eta_consumido"] = (
            df["tiempo_desde_creacion"] / total_window.replace(0, np.nan)
        ).clip(0, 1)

    df["tiempo_ultimo_evento"] = (
        (now - df["ultimo_evento_ts"]).dt.total_seconds() / 3600
    )

    # Velocidad: eventos / horas transcurridas
    span = (df["ultimo_evento_ts"] - df["primer_evento_ts"]).dt.total_seconds() / 3600
    df["velocidad_flujo"] = df["num_eventos"] / span.replace(0, np.nan)

    # Desviación vs itinerario (requiere umbrales por segmento — completar con datos reales)
    df["desviacion_vs_itinerario"] = 0.0
    log.warning("  desviacion_vs_itinerario = 0 (placeholder — ajustar con umbrales reales)")

    # Encoding último tipo evento
    event_order = [
        "order_creation", "confirmed", "picking", "packed",
        "dispatch", "in_transit", "out_for_delivery",
        "delivered", "cancelled", "returned"
    ]
    tipo_map = {e: i for i, e in enumerate(event_order)}
    df["ultimo_estado_codigo"] = df["ultimo_tipo"].map(tipo_map).fillna(-1).astype(int)

    log.info(f"  Features del event log agregadas: {TEMPORAL_FEATURES}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MERGE Y DATASET FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(
    save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, list]:
    """
    Pipeline completo de construcción del dataset:
      1. Carga Muestra FO
      2. Construye target
      3. Agrega features temporales derivadas
      4. (Futuro) Merge con PULSE features cuando haya tabla de mapeo
      5. Agrega event log features (si existe)
      6. Retorna DataFrame limpio + lista de features

    NOTA: NO calculamos agregados históricos aquí para evitar data leakage.
    Eso se hace DESPUÉS del split temporal en feature_engineering.py.

    Returns:
        df: DataFrame con features y target
        feature_cols: lista de columnas a usar en el modelo
    """
    log.info("=" * 60)
    log.info("PULSE - Construcción del Dataset")
    log.info("=" * 60)

    # ── 1. Carga base ────────────────────────────────────────────────────
    df = load_muestra_fo()
    log.info(f"Dataset base: {len(df):,} filas")

    # ── 2. Target ────────────────────────────────────────────────────────
    df = build_target(df)

    # ── 3. Features temporales derivadas ─────────────────────────────────
    df = add_temporal_derived_features(df)

    # ── 4. PULSE features ────────────────────────────────────────────────
    # INFO: El merge no es posible todavía porque los IDs son incompatibles.
    # Muestra FO usa order_compra (numérico), PULSE usa FOFCL000028XXXXXX.
    # Cuando llegue la tabla de mapeo, descomentar este bloque:
    #
    # pulse_df = load_pulse_features()
    # if not pulse_df.empty and "order_id" in df.columns:
    #     mapping = pd.read_csv("data/mapping_fofcl_oc.csv")
    #     pulse_df = pulse_df.merge(mapping, on="order_id", how="left")
    #     df = df.merge(pulse_df, left_on="order_id", right_on="order_compra", how="left")
    log.info("  PULSE features: omitidas (sin tabla de mapeo FOFCL ↔ order_compra)")

    # ── 5. Event log features (opcional) ─────────────────────────────────
    df = add_event_log_features(df)

    # ── 6. Definir feature set final ──────────────────────────────────────
    feature_cols = []

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    # Temporales del event log (si llegaron)
    for col in TEMPORAL_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
            log.info(f"  ✓ Feature del event log disponible: {col}")

    feature_cols = list(dict.fromkeys(feature_cols))  # dedup preservando orden
    log.info(f"\nFeatures totales: {len(feature_cols)}")
    log.info(f"  Categóricas: {[c for c in feature_cols if c in CATEGORICAL_FEATURES]}")
    log.info(f"  Numéricas:   {[c for c in feature_cols if c in NUMERICAL_FEATURES]}")
    log.info(f"  Booleanas:   {[c for c in feature_cols if c in BOOLEAN_FEATURES]}")

    # ── 7. Guardar ────────────────────────────────────────────────────────
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"\nDataset guardado en {save_path}")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df, features = build_dataset(
        save_path=OUTPUTS_DIR / "dataset_processed.csv"
    )

    print(f"\n{'='*50}")
    print(f"Dataset listo: {df.shape}")
    print(f"Features ({len(features)}): {features}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True)}")
