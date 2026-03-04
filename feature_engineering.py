"""
PULSE - Motor Predictivo | Etapa 2: Feature Engineering y Preprocessing
========================================================================
Encodings, imputación, scaling y split temporal para entrenamiento.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import joblib

from config import (
    TARGET_COL, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BOOLEAN_FEATURES,
    TEMPORAL_FEATURES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    OUTPUTS_DIR, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPLIT TEMPORAL (no random — respetar orden cronológico)
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    date_col: str = "fecha_creacion",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset respetando el orden temporal.
    Train: primeros 67% | Val: siguientes 17% | Test: últimos 16%

    CRÍTICO: No usar train_test_split con shuffle=True en series temporales.
    Eso introduciría data leakage (el modelo "vería el futuro").
    """
    if date_col not in df.columns:
        log.warning(f"Columna {date_col} no encontrada. Usando orden de filas como proxy.")
        df = df.reset_index(drop=True)
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df_sorted.iloc[:train_end].copy()
    val   = df_sorted.iloc[train_end:val_end].copy()
    test  = df_sorted.iloc[val_end:].copy()

    log.info(f"Split temporal:")
    log.info(f"  Train: {len(train):,} filas ({train[date_col].min()} → {train[date_col].max()})")
    log.info(f"  Val:   {len(val):,} filas ({val[date_col].min()} → {val[date_col].max()})")
    log.info(f"  Test:  {len(test):,} filas ({test[date_col].min()} → {test[date_col].max()})")

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_pos = split[TARGET_COL].sum()
        pct = n_pos / len(split) * 100
        log.info(f"  {name} fallos: {n_pos:,} ({pct:.1f}%)")

    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    """
    Construye el pipeline de sklearn para transformar features.

    Estrategia por tipo:
      - Numéricas: imputar mediana → escalar (StandardScaler)
      - Categóricas: imputar 'unknown' → OneHotEncoder (handle_unknown='ignore')
      - Booleanas: imputar 0 → pasar tal cual (XGBoost maneja bools bien)

    El ColumnTransformer aplica transformaciones en paralelo y las concatena.
    """
    # Separar columnas por tipo según las que estén efectivamente en feature_cols
    num_cols  = [c for c in feature_cols if c in NUMERICAL_FEATURES +
                 TEMPORAL_FEATURES +
                 ["tasa_fallo_carrier", "tasa_fallo_region", "tasa_fallo_carrier_region"]]
    cat_cols  = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    bool_cols = [c for c in feature_cols if c in BOOLEAN_FEATURES]

    log.info(f"Preprocessor:")
    log.info(f"  Numéricas ({len(num_cols)}):  {num_cols}")
    log.info(f"  Categóricas ({len(cat_cols)}): {cat_cols}")
    log.info(f"  Booleanas ({len(bool_cols)}):  {bool_cols}")

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=10,  # categorías con <10 ocurrencias → 'infrequent'
            )),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if bool_cols:
        bool_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ])
        transformers.append(("bool", bool_pipeline, bool_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # descartar columnas no declaradas
    )

    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CÁLCULO DE AGREGADOS HISTÓRICOS (SOLO SOBRE TRAIN)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_apply_historical_rates(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Calcula tasas de fallo históricas SOLO sobre train y las aplica a val/test.

    Esto evita data leakage: val/test no "saben" sus propias tasas.
    Retorna los DataFrames actualizados y el diccionario de tasas (para inference).
    """
    rate_tables = {}
    global_rate = train[TARGET_COL].mean()
    log.info(f"  Tasa global de fallo (train): {global_rate:.3f}")

    def _apply_rate(df_in, col_name, rate_map, global_fallback):
        return df_in[col_name].map(rate_map).fillna(global_fallback)

    for col in ["carrier", "region"]:
        if col in train.columns:
            rates = train.groupby(col)[TARGET_COL].mean().to_dict()
            rate_tables[f"tasa_{col}"] = rates

            for split in [train, val, test]:
                new_col = f"tasa_fallo_{col}"
                split[new_col] = _apply_rate(split, col, rates, global_rate)

            log.info(f"  Tasas por {col}: {len(rates)} valores únicos")

    if "carrier" in train.columns and "region" in train.columns:
        rates_cr = train.groupby(["carrier", "region"])[TARGET_COL].mean()
        rate_tables["tasa_carrier_region"] = {str(k): v for k, v in rates_cr.to_dict().items()}

        for split in [train, val, test]:
            split["tasa_fallo_carrier_region"] = (
                split.set_index(["carrier", "region"])
                .index.map(rates_cr.to_dict())
            )
            split["tasa_fallo_carrier_region"] = split["tasa_fallo_carrier_region"].fillna(global_rate)

    rate_tables["global"] = global_rate
    return train, val, test, rate_tables


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, list, Dict]:
    """
    Aplica el preprocessor completo:
      1. Calcula agregados históricos sobre train (sin data leakage)
      2. Fit del ColumnTransformer sobre train
      3. Transform de train/val/test

    Returns:
        X_train, X_val, X_test: arrays numpy transformados
        y_train, y_val, y_test: arrays de target
        preprocessor: fitted ColumnTransformer (guardar para inference)
        feature_names: nombres de features post-encoding
        rate_tables: diccionario de tasas históricas (guardar para inference)
    """
    # Agregar tasa features (calculadas solo sobre train)
    df_train, df_val, df_test, rate_tables = compute_and_apply_historical_rates(
        df_train.copy(), df_val.copy(), df_test.copy()
    )

    # Actualizar feature_cols con las tasas agregadas
    rate_cols = [c for c in ["tasa_fallo_carrier", "tasa_fallo_region", "tasa_fallo_carrier_region"]
                 if c in df_train.columns]
    all_features = list(dict.fromkeys(feature_cols + rate_cols))
    # Filtrar solo las que existen
    all_features = [c for c in all_features if c in df_train.columns]

    log.info(f"Features finales para el modelo: {len(all_features)}")

    # Extraer X e y
    y_train = df_train[TARGET_COL].values
    y_val   = df_val[TARGET_COL].values
    y_test  = df_test[TARGET_COL].values

    # Fit preprocessor sobre train únicamente
    preprocessor = build_preprocessor(all_features)
    X_train = preprocessor.fit_transform(df_train[all_features])
    X_val   = preprocessor.transform(df_val[all_features])
    X_test  = preprocessor.transform(df_test[all_features])

    log.info(f"Shapes post-encoding:")
    log.info(f"  X_train: {X_train.shape} | y_train positivos: {y_train.sum():,}")
    log.info(f"  X_val:   {X_val.shape}   | y_val positivos:   {y_val.sum():,}")
    log.info(f"  X_test:  {X_test.shape}  | y_test positivos:  {y_test.sum():,}")

    # Nombres de features post-OHE (para SHAP)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]

    # Guardar preprocessor y rate_tables para inference
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    with open(MODELS_DIR / "rate_tables.json", "w") as f:
        # Convertir keys de tuple a string para JSON
        serializable = {}
        for k, v in rate_tables.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)

    log.info(f"Preprocessor guardado en {MODELS_DIR / 'preprocessor.pkl'}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor, feature_names, rate_tables
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Cargar dataset procesado (output de build_dataset.py)
    dataset_path = OUTPUTS_DIR / "dataset_processed.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Ejecutar primero: python build_dataset.py"
        )

    df = pd.read_csv(dataset_path, parse_dates=["fecha_creacion"])

    # Feature list desde config (actualizar según lo que esté disponible)
    all_available = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BOOLEAN_FEATURES
    )
    feature_cols = [c for c in all_available if c in df.columns]

    # Split temporal
    train, val, test = temporal_split(df)

    # Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, prep, feat_names, rates = prepare_features(
        train, val, test, feature_cols
    )

    print(f"\n{'='*50}")
    print(f"Preprocessing completo")
    print(f"X_train: {X_train.shape} | Positivos: {y_train.sum():,}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Features ({len(feat_names)}): {feat_names[:10]}...")
