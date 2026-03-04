"""
PULSE - Motor Predictivo | Etapa 5: API de Inferencia (FastAPI)
===============================================================
Endpoint stateless que recibe queries normalizadas del Orquestador
y retorna predicciones de riesgo con factores contribuyentes (SHAP).

Ejecutar: uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import joblib
import xgboost as xgb
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    MODELS_DIR, ACTIVE_THRESHOLD, THRESHOLDS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BOOLEAN_FEATURES, TEMPORAL_FEATURES,
)
from evaluate import explain_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="PULSE - Motor Predictivo",
    description="Componente 2: Predicción de riesgo de incumplimiento ETA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC - Contrato de datos
# ═══════════════════════════════════════════════════════════════════════════════

class OrderContext(BaseModel):
    """Context de la orden (viene del Componente 1 normalizado)."""
    region:        Optional[str] = None
    comuna:        Optional[str] = None
    carrier:       Optional[str] = None
    metodo_despacho: Optional[str] = None
    tipo_transporte: Optional[str] = None
    origen:        Optional[str] = None
    # PULSE features (del itinerario — cuando llegue tabla de mapeo)
    num_ops:               Optional[int]   = None
    num_processes:         Optional[int]   = None
    num_legs:              Optional[int]   = None
    num_transfers:         Optional[int]   = None
    num_operators:         Optional[int]   = None
    total_span_hours:      Optional[float] = None
    picking_window_hours:  Optional[float] = None
    total_process_time_mins: Optional[int] = None
    complexity_score:      Optional[int]   = None
    span_ratio:            Optional[float] = None
    has_crossdocking:      Optional[bool]  = None
    has_first_mile:        Optional[bool]  = None
    short_picking_window:  Optional[bool]  = None
    anomalous_span:        Optional[bool]  = None
    complex_route:         Optional[bool]  = None
    seller_origin:         Optional[bool]  = None
    service_category:      Optional[str]   = None
    delivery_type:         Optional[str]   = None
    origin_node_type:      Optional[str]   = None
    dest_node_type:        Optional[str]   = None
    origin_operator:       Optional[str]   = None


class PredictionQuery(BaseModel):
    """Query normalizada del Componente 1 → Componente 2."""
    order_id:       str = Field(..., description="ID interno de la orden")
    source_order_id: Optional[str] = None
    package_id:     Optional[str] = None
    current_status: Optional[str] = None
    status_history: Optional[List[str]] = []
    eta_promised:   Optional[str] = None   # ISO datetime string
    fecha_creacion: Optional[str] = None   # ISO datetime string
    context:        Optional[OrderContext] = None
    # Event log features (opcionales)
    tiempo_desde_creacion:    Optional[float] = None
    tiempo_hasta_eta:         Optional[float] = None
    porcentaje_eta_consumido: Optional[float] = None
    velocidad_flujo:          Optional[float] = None
    tiempo_ultimo_evento:     Optional[float] = None
    desviacion_vs_itinerario: Optional[float] = None
    num_eventos:              Optional[int]   = None
    ultimo_estado_codigo:     Optional[int]   = None


class ContributingFactor(BaseModel):
    feature:     str
    shap_value:  float
    direction:   str
    description: str


class PredictionResult(BaseModel):
    """Output del Componente 2 → Componente 1."""
    order_id:             str
    risk_score:           float = Field(..., ge=0, le=1)
    predicted_outcome:    str
    confidence:           str
    disruption_type:      Optional[str] = None
    contributing_factors: List[ContributingFactor] = []
    recommended_action:   str
    timestamp:            str
    model_version:        str
    threshold_used:       float


class BatchPredictionRequest(BaseModel):
    queries: List[PredictionQuery]
    threshold: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    predictions:     List[PredictionResult]
    model_version:   str
    processed_at:    str
    total_queries:   int
    high_risk_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE ARTEFACTOS AL INICIAR
# ═══════════════════════════════════════════════════════════════════════════════

class ModelArtifacts:
    model: xgb.XGBClassifier = None
    preprocessor = None
    explainer: shap.TreeExplainer = None
    feature_names: List[str] = []
    rate_tables: Dict = {}
    model_version: str = "unknown"

artifacts = ModelArtifacts()


@app.on_event("startup")
def load_artifacts():
    """Carga todos los artefactos del modelo al iniciar la API."""
    log.info("Cargando artefactos del modelo...")

    try:
        artifacts.model = joblib.load(MODELS_DIR / "model_latest.pkl")
        log.info("  ✓ Modelo cargado")
    except FileNotFoundError:
        log.error("  ✗ model_latest.pkl no encontrado. Ejecutar train_model.py primero.")
        return

    try:
        artifacts.preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
        log.info("  ✓ Preprocessor cargado")
    except FileNotFoundError:
        log.warning("  ✗ preprocessor.pkl no encontrado")

    try:
        artifacts.explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        log.info("  ✓ SHAP explainer cargado")
    except FileNotFoundError:
        log.warning("  ✗ shap_explainer.pkl no encontrado — explicaciones no disponibles")

    try:
        with open(MODELS_DIR / "features_latest.json") as f:
            feat_data = json.load(f)
        artifacts.feature_names = feat_data["features"]
        artifacts.model_version = feat_data.get("version", "unknown")
        log.info(f"  ✓ Features cargadas: {len(artifacts.feature_names)}")
    except FileNotFoundError:
        log.warning("  ✗ features_latest.json no encontrado")

    try:
        with open(MODELS_DIR / "rate_tables.json") as f:
            artifacts.rate_tables = json.load(f)
        log.info("  ✓ Rate tables cargadas")
    except FileNotFoundError:
        log.warning("  ✗ rate_tables.json no encontrado")

    log.info(f"API lista. Modelo: {artifacts.model_version}")


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARACIÓN DE FEATURES PARA INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def query_to_features(query: PredictionQuery) -> pd.DataFrame:
    """Convierte una PredictionQuery al DataFrame de features."""
    ctx = query.context or OrderContext()
    row = {}

    # ── Features del context (columnas reales de Muestra FO) ──────────────
    row["carrier"]         = ctx.carrier
    row["region"]          = ctx.region
    row["comuna"]          = ctx.comuna
    row["metodo_despacho"] = ctx.metodo_despacho
    row["tipo_transporte"] = ctx.tipo_transporte
    row["origen"]          = ctx.origen

    # ── PULSE features (cuando estén disponibles) ─────────────────────────
    row["num_ops"]              = ctx.num_ops
    row["num_processes"]        = ctx.num_processes
    row["num_legs"]             = ctx.num_legs
    row["num_transfers"]        = ctx.num_transfers
    row["num_operators"]        = ctx.num_operators
    row["total_span_hours"]     = ctx.total_span_hours
    row["picking_window_hours"] = ctx.picking_window_hours
    row["total_process_time_mins"] = ctx.total_process_time_mins
    row["complexity_score"]     = ctx.complexity_score
    row["span_ratio"]           = ctx.span_ratio
    row["has_crossdocking"]     = ctx.has_crossdocking
    row["has_first_mile"]       = ctx.has_first_mile
    row["short_picking_window"] = ctx.short_picking_window
    row["anomalous_span"]       = ctx.anomalous_span
    row["complex_route"]        = ctx.complex_route
    row["seller_origin"]        = ctx.seller_origin

    # ── Temporales de la fecha ─────────────────────────────────────────────
    if query.fecha_creacion:
        try:
            dt = pd.to_datetime(query.fecha_creacion)
            row["dia_semana"]    = dt.dayofweek
            row["hora_creacion"] = dt.hour
            row["mes"]           = dt.month
            row["es_fin_semana"] = int(dt.dayofweek >= 5)
        except Exception:
            row["dia_semana"] = row["hora_creacion"] = row["mes"] = row["es_fin_semana"] = None

    # ── Event log features ────────────────────────────────────────────────
    row["tiempo_desde_creacion"]    = query.tiempo_desde_creacion
    row["tiempo_hasta_eta"]         = query.tiempo_hasta_eta
    row["porcentaje_eta_consumido"] = query.porcentaje_eta_consumido
    row["velocidad_flujo"]          = query.velocidad_flujo
    row["tiempo_ultimo_evento"]     = query.tiempo_ultimo_evento
    row["desviacion_vs_itinerario"] = query.desviacion_vs_itinerario
    row["num_eventos"]              = query.num_eventos
    row["ultimo_estado_codigo"]     = query.ultimo_estado_codigo

    # ── Tasas históricas ──────────────────────────────────────────────────
    global_rate = artifacts.rate_tables.get("global", 0.2)
    
    # Asegurar que estas columnas EXISTAN siempre para el preprocessor
    row["tasa_fallo_carrier"] = global_rate
    if ctx.carrier and "tasa_carrier" in artifacts.rate_tables:
        row["tasa_fallo_carrier"] = artifacts.rate_tables["tasa_carrier"].get(
            ctx.carrier, global_rate
        )
        
    row["tasa_fallo_region"] = global_rate
    if ctx.region and "tasa_region" in artifacts.rate_tables:
        row["tasa_fallo_region"] = artifacts.rate_tables["tasa_region"].get(
            ctx.region, global_rate
        )
        
    row["tasa_fallo_carrier_region"] = global_rate
    if ctx.carrier and ctx.region and "tasa_carrier_region" in artifacts.rate_tables:
        key = str((ctx.carrier, ctx.region))
        row["tasa_fallo_carrier_region"] = artifacts.rate_tables["tasa_carrier_region"].get(
            key, global_rate
        )

    # ── Asegurar consistencia con artifacts.feature_names ─────────────────
    # Si faltan columnas que el preprocessor espera, añadirlas con None
    for col in artifacts.feature_names:
        if col not in row:
            row[col] = None

    return pd.DataFrame([row])


def determine_disruption_type(query: PredictionQuery, shap_factors: list) -> Optional[str]:
    """Clasifica el tipo de disrupción probable basado en los factores SHAP."""
    if not shap_factors:
        return None
    top_feature = shap_factors[0]["feature"] if shap_factors else ""
    if "carrier" in top_feature or "transporte" in top_feature.lower():
        return "carrier_risk"
    if "region" in top_feature or "comuna" in top_feature:
        return "zone_risk"
    if "tiempo" in top_feature or "eta" in top_feature:
        return "delay_risk"
    if "complexity" in top_feature or "transfers" in top_feature:
        return "routing_risk"
    if "crossdocking" in top_feature:
        return "crossdock_risk"
    return "general_risk"


def determine_confidence(risk_score: float, threshold: float) -> str:
    distance = abs(risk_score - threshold)
    if distance >= 0.3:
        return "high"
    elif distance >= 0.15:
        return "medium"
    return "low"


def determine_action(risk_score: float, threshold: float) -> str:
    if risk_score >= THRESHOLDS["high"]:
        return "ACCIÓN_INMEDIATA: Contactar transportista y notificar cliente"
    elif risk_score >= THRESHOLDS["default"]:
        return "MONITOREO_ACTIVO: Revisar en próximo ciclo y preparar contingencia"
    elif risk_score >= THRESHOLDS["low"]:
        return "ALERTA_PREVENTIVA: Marcar para seguimiento manual"
    return "SIN_ACCIÓN: Orden dentro de parámetros normales"


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {
        "status":          "ok" if artifacts.model is not None else "degraded",
        "model_version":   artifacts.model_version,
        "model_loaded":    artifacts.model is not None,
        "shap_available":  artifacts.explainer is not None,
        "features_count":  len(artifacts.feature_names),
        "timestamp":       datetime.now().isoformat(),
    }


@app.post("/predict", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Predicción batch de riesgo de incumplimiento ETA."""
    if artifacts.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Verificar que train_model.py se ejecutó correctamente."
        )

    threshold = request.threshold or ACTIVE_THRESHOLD
    predictions = []

    for query in request.queries:
        try:
            df_row = query_to_features(query)

            if artifacts.preprocessor is not None:
                for col in artifacts.feature_names:
                    if col not in df_row.columns:
                        df_row[col] = None
                X = artifacts.preprocessor.transform(df_row)
            else:
                log.warning(f"Preprocessor no disponible para order {query.order_id}")
                X = df_row.select_dtypes(include=[np.number]).fillna(0).values

            risk_score = float(artifacts.model.predict_proba(X)[0, 1])
            outcome = "fail" if risk_score >= threshold else "success"

            shap_factors = []
            if artifacts.explainer is not None:
                try:
                    shap_factors = explain_prediction(
                        artifacts.explainer, X, artifacts.feature_names, top_n=3
                    )
                except Exception as e:
                    log.warning(f"SHAP falló para {query.order_id}: {e}")

            disruption = determine_disruption_type(query, shap_factors)
            confidence = determine_confidence(risk_score, threshold)
            action = determine_action(risk_score, threshold)

            predictions.append(PredictionResult(
                order_id=query.order_id,
                risk_score=round(risk_score, 4),
                predicted_outcome=outcome,
                confidence=confidence,
                disruption_type=disruption,
                contributing_factors=[ContributingFactor(**f) for f in shap_factors],
                recommended_action=action,
                timestamp=datetime.now().isoformat(),
                model_version=artifacts.model_version,
                threshold_used=threshold,
            ))

        except Exception as e:
            log.error(f"Error procesando {query.order_id}: {e}")
            predictions.append(PredictionResult(
                order_id=query.order_id,
                risk_score=0.0,
                predicted_outcome="error",
                confidence="low",
                disruption_type="processing_error",
                contributing_factors=[],
                recommended_action=f"ERROR: {str(e)[:100]}",
                timestamp=datetime.now().isoformat(),
                model_version=artifacts.model_version,
                threshold_used=threshold,
            ))

    high_risk = sum(1 for p in predictions if p.predicted_outcome == "fail")

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=artifacts.model_version,
        processed_at=datetime.now().isoformat(),
        total_queries=len(predictions),
        high_risk_count=high_risk,
    )


@app.get("/model/info")
def model_info():
    metrics_path = MODELS_DIR / "metrics_latest.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "version":        artifacts.model_version,
        "features_count": len(artifacts.feature_names),
        "features":       artifacts.feature_names,
        "thresholds":     THRESHOLDS,
        "active_threshold": ACTIVE_THRESHOLD,
        "metrics":        metrics.get("validation", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
