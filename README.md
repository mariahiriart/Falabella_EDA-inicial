# Proyecto EDA Logistico Falabella — Conectamos 2025

> Analisis Exploratorio de Datos sobre dos fuentes del sistema de Fulfilment de Falabella Chile: itinerarios de ordenes logisticas (PULSE / foorch_temp_aj) y muestra de estado de entregas (Muestra FO).

---

## Estructura del Proyecto

```
.
├── foorch_temp_aj.xlsx          # Fuente 1: itinerarios de 5.591 Fulfilment Orders
├── Muestra_FO__1_.xlsb          # Fuente 2: muestra de 263.130 packages (oct-dic 2025)
├── eda_falabella.py             # Script EDA principal — fuente PULSE (foorch_temp_aj)
├── falabella_eda_features.csv   # Features extraidas por orden (output del script)
├── dispatch.json                # Ejemplo evento: paquetes despachados
├── order_creation.json          # Ejemplo evento: creacion de orden
├── packed.json                  # Ejemplo evento: paquete empacado
├── fig1_distribucion_general.png
├── fig2_analisis_temporal.png
├── fig3_senales_riesgo.png
├── fig4_actores_flujos.png
├── fig5_resumen_ejecutivo.png
├── eda_fig1_overview.png
├── eda_fig2_transportistas_regiones.png
├── eda_fig3_tiempos_atrasos.png
├── eda_fig4_estados.png
└── README.md
```

---

## Origen de los Datos

Los datos provienen del sistema de **Fulfilment Orders** de Falabella Chile. Falabella envio tres ejemplos de eventos JSON que ilustran el ciclo de vida de una orden:

### order_creation.json — Creacion de Orden
Se genera cuando una orden de compra es registrada en el sistema logistico. Campos clave:
- : ID de la orden logistica (FOFCL000028430379)
- : ID de la orden de compra del cliente (2911368940)
- : lista de items con SKU, seller, canal y modo de fulfillment (FBS, CROSSDOCK)
- : ID del itinerario logistico asignado al item
- : define como se va a cumplir la orden

### packed.json — Empaquetado
Se genera cuando el warehouse empaqueta los items. Campos clave:
- : FOFCL000028428544
- : PKG0054477102 — el package ya tiene ID unico
- : PACKED
- : warehouse de origen
- : fecha prometida con service_category
- : BATCH — procesado en lote

### dispatch.json — Despacho
Se genera cuando el package sale del nodo origen hacia el transportista. Campos clave:
- : FOFCL000028466958
- : SHIPMENT_CONFIRMED
- : FULFILMENT_ORDER_PACKAGES_DISPATCHED
- : timestamp exacto del despacho
- : link al pedido original del cliente

**Relacion clave entre archivos:**  (formato FOFCL000028XXXXXX) une los eventos JSON con las filas de foorch_temp_aj.xlsx. El  numerico corresponde al  de Muestra_FO__1_.xlsb, aunque ambos archivos cubren periodos distintos y no se solapan directamente sin una tabla de mapeo.

---

## Fuente 1 — PULSE EDA (foorch_temp_aj.xlsx)

**Script:** eda_falabella.py
**Dataset:** 5.591 Fulfilment Orders con itinerario logistico completo
**Periodo:** agosto – septiembre 2025

### Que contiene el archivo?

Cada fila es una Fulfilment Order (FOFCL) con tres campos:
- : ID de la orden (FOFCL000028XXXXXX)
- : JSON con fecha prometida y categoria de servicio
- : JSON array con todas las operaciones logisticas (procesos, legs, transfers)

### Pipeline del Script

```
foorch_temp_aj.xlsx
       |
       ▼
  parse_record()          <- parsea JSON de promiseddate y rawlogisticoperations
       |
       ▼
  feature engineering     <- extrae 24 features por orden
       |
       ▼
falabella_eda_features.csv  +  5 figuras PNG
```

### Features Extraidas (falabella_eda_features.csv)

| Feature | Descripcion |
|---|---|
| service_category | MESON / REGULAR / DATE_RANGE / TO_CAR / SAME_DAY |
| delivery_type | COLLECT / HOME_DELIVERY |
| origin_node_type | SELLER / STORE / WAREHOUSE |
| origin_node_name | Nombre del nodo de origen |
| origin_operator | Operador del nodo de origen |
| dest_node_type | Tipo de nodo destino final |
| final_carrier | Transportista final de entrega |
| num_ops | Total de operaciones en el itinerario |
| num_processes | Numero de procesos (nodos fisicos) |
| num_legs | Numero de tramos de transporte |
| num_transfers | Numero de transfers (cambios de hub) |
| has_crossdocking | Booleano: pasa por crossdock? |
| has_first_mile | Booleano: tiene etapa first mile? |
| num_operators | Cantidad de operadores distintos en la ruta |
| total_span_hours | Duracion total planificada del itinerario (horas) |
| picking_window_hours | Ventana disponible para picking (horas) |
| total_process_time_mins | Suma de tiempos de proceso (minutos) |
| complexity_score | Score 0-100+ de complejidad de la ruta |
| short_picking_window | Alerta: window menor a 3h |
| anomalous_span | Alerta: span mayor a 3x la mediana de la categoria |
| complex_route | Alerta: mayor o igual a 8 operaciones en la ruta |
| seller_origin | Booleano: origen es SELLER (FBS) |
| span_ratio | span_real / mediana_categoria |

### Hallazgos Principales PULSE

**Distribucion de ordenes:**
- 46.8% son MESON (retiro en tienda) — mediana de duracion 26h
- 52.6% son HOME_DELIVERY (REGULAR + DATE_RANGE)
- 64.8% tienen origen SELLER (canal FBS — menor visibilidad y control operativo)

**Complejidad de rutas:**
- 25.3% de las rutas son simples (2 operaciones: directo sin hub)
- 39.0% tienen 4 operaciones (ruta estandar con un hub intermedio)
- 14.8% son rutas complejas con 8 o mas operaciones
- 35.7% pasan por crossdocking

**Senales de riesgo detectadas:**
- 39.8% de las ordenes (2.223) tienen picking window menor a 3h — umbral critico
- 4.2% (234 ordenes) tienen span anomalo (mayor a 3x la mediana de su categoria)
- 2.6% (143 ordenes) involucran 3 o mas operadores en la misma ruta

**Score de complejidad por carrier:**

| Carrier | Score promedio | n ordenes |
|---|---|---|
| SODIMAC | 85.6 | 271 |
| CHILEXPRESS | 80.5 | 235 |
| TOTTUS | 77.9 | 157 |
| HOME DELIVERY CORP | 46.7 | 1.431 |
| FALABELLA | 33.6 | 1.935 |

**Horario de inicio del itinerario:**
- Pico operativo entre las 7h y 9h (turno manana)
- Segundo lote a las 0h — procesamiento batch nocturno

### Figuras Generadas PULSE

| Figura | Contenido |
|---|---|
| fig1_distribucion_general.png | Service category, delivery type, complejidad de ruta, tipo nodo origen, carrier final, crossdocking |
| fig2_analisis_temporal.png | Duracion por categoria (boxplot), picking window, percentiles de duracion, hora de inicio |
| fig3_senales_riesgo.png | Senales de riesgo, score de complejidad, score por carrier, span ratio, matriz window vs complejidad |
| fig4_actores_flujos.png | Top 12 nodos origen, combos service+delivery, operadores por ruta, tipo nodo destino |
| fig5_resumen_ejecutivo.png | Dashboard ejecutivo con 8 KPIs principales |

---

## Fuente 2 — Muestra FO EDA (Muestra_FO__1_.xlsb)

**Dataset:** 263.130 packages con estado de entrega real
**Periodo:** octubre – diciembre 2025
**Pestanas:** Entregadas (206.534 filas) + Aun no entregadas (56.596 filas)

### Estructura del Dataset

Cada fila representa un package individual con 18 columnas:

| Campo | Descripcion |
|---|---|
| order_compra | ID numerico de la orden de compra del cliente |
| producto | ID del producto |
| package_id | ID unico del package (PKG00XXXXXXXX) |
| estado_producto | Estado del producto (SHIPMENT_CONFIRMED, SHIPPED, LABELLED...) |
| fecha_estado_producto | Timestamp del ultimo estado del producto |
| estado_package | Estado del package (DELIVERED, IN_TRANSIT, AVAILABLE_FOR_PICKUP...) |
| fecha_estado_package | Timestamp del ultimo estado del package |
| fecha_pactada_desde | Inicio de ventana de entrega pactada |
| fecha_pactada_hasta | Fin de ventana de entrega pactada |
| fecha_pactada_retiro | Fecha pactada de retiro (solo retiro en tienda) |
| comuna | Comuna de destino |
| region | Region de destino |
| tienda_retiro | Tienda de retiro (cuando aplica) |
| nombre_transporte | Nombre del transportista |
| tipo_transporte | TRANSPORTE_PROPIO / THREE_PL / SELLER |
| estado_entrega | Estado final de entrega |
| metodo_despacho | DESPACHO_DOMICILIO / RETIRO_EN_TIENDA / RETIRO_EN_COLL |
| origen_despacho | Identificador del origen logistico del despacho |

### Hallazgos Principales Muestra FO

**Volumen general:**
- 263.130 packages totales — 227.735 ordenes unicas
- 78.5% entregados — 21.5% pendientes o sin entrega

**Estado de entrega (sobre paquetes entregados):**
- Entregada antes de tiempo: 113.274 (54.8%)
- Entregada a tiempo: 51.643 (25.0%)
- Entregada atrasada: 41.617 (20.2%)

**Metodo de despacho:**
- DESPACHO_DOMICILIO: 51.8%
- RETIRO_EN_TIENDA: 37.8%
- RETIRO_EN_COLL: 10.4%

**Tipo de transporte:**
- TRANSPORTE_PROPIO: 66.1% (173.982 packages)
- THREE_PL: 25.5% (67.120 packages)
- SELLER: 7.3% (19.123 packages)

**Transportistas — volumen y tasa de atraso:**

| Transportista | Packages | Tasa de Atraso |
|---|---|---|
| HOME DELIVERY CORP | 96.190 | 11.0% |
| CHILEXPRESS | 27.012 | 16.7% |
| BLUEEXPRESS | 24.578 | 11.5% |
| FALAFLEX | 18.205 | 50.2% ALERTA |
| FALABELLA_GROUP | 17.218 | 21.6% |
| COURIER INTERNACIONAL | 10.433 | 69.8% CRITICO |

**Cobertura geografica y tasa de atraso:**

| Region | Packages | Tasa de Atraso |
|---|---|---|
| RM | 114.004 | 34.3% CRITICO |
| BIOBIO | 77.792 | 9.0% |
| VALPARAISO | 71.334 | 15.0% |

**Distribucion de atrasos:**
- Mediana de atraso en entregas tardias: 1 dia
- Mayoria de atrasos: 1-3 dias; cola hasta los 30+ dias

**Ordenes con multiples packages:**
- 1 package: 202.064 ordenes (88.7%)
- 2 packages: 20.394 ordenes (9.0%)
- 3 o mas packages: 5.277 ordenes (2.3%)

### Figuras Generadas Muestra FO

| Figura | Contenido |
|---|---|
| eda_fig1_overview.png | Entregadas vs No Entregadas, estado de entrega, metodo de despacho, tipo de transporte, nulos por columna, packages por orden |
| eda_fig2_transportistas_regiones.png | Top 15 transportistas por volumen, packages por region |
| eda_fig3_tiempos_atrasos.png | Distribucion dias de atraso, % atraso por transportista, volumen semanal, % atraso por region |
| eda_fig4_estados.png | Estados de producto Top 10, estados de package Top 10 |

---

## Relacion entre las Dos Fuentes

| Aspecto | PULSE (foorch_temp_aj) | Muestra FO |
|---|---|---|
| Granularidad | 1 fila = 1 Fulfilment Order | 1 fila = 1 Package |
| ID principal | FOFCL000028XXXXXX | order_compra (numerico) |
| Periodo cubierto | ago-sep 2025 | oct-dic 2025 |
| Foco | Itinerario planificado | Estado de entrega real |
| N registros | 5.591 ordenes | 263.130 packages |
| Vinculacion | src_order_id del evento JSON | order_compra |

Para cruzar ambas fuentes se requiere una tabla de mapeo FOFCL <-> order_compra disponible en el sistema origen de Falabella. Con ese cruce seria posible comparar el itinerario planificado con el resultado real de entrega y construir un modelo predictivo de riesgo de atraso.

---

## Como Ejecutar el EDA

### Requisitos

```bash
pip install pandas openpyxl numpy matplotlib
```

### Ejecutar script PULSE

```bash
python eda_falabella.py
```

**Input requerido:** foorch_temp_aj.xlsx en el mismo directorio

**Outputs:**
```
falabella_eda_features.csv
fig1_distribucion_general.png
fig2_analisis_temporal.png
fig3_senales_riesgo.png
fig4_actores_flujos.png
fig5_resumen_ejecutivo.png
```

---

## Senales de Riesgo Prioritarias

| Prioridad | Senal | Detalle |
|---|---|---|
| CRITICA | COURIER INTERNACIONAL 69.8% atraso | Revisar contrato y asignacion de rutas |
| CRITICA | RM con 34.3% de atraso | Mayor volumen + mayor tasa de fallo |
| ALTA | FALAFLEX 50.2% atraso | 18.205 packages afectados |
| ALTA | 39.8% picking window menor a 3h | Presion operativa en primer nodo |
| ALTA | 64.8% origen SELLER | Menor control sobre tiempos de preparacion |
| MEDIA | 234 spans anomalos (mayor a 3x mediana) | Candidatos a revision manual del itinerario |
| MEDIA | 14.8% rutas complejas (8 o mas ops) | Mayor probabilidad de error en transferencia |

---

## Proximos Pasos

- Obtener tabla de mapeo FOFCL <-> order_compra para cruzar itinerario con estado de entrega real
- Construir modelo predictivo de riesgo de atraso usando features del CSV + estado final de Muestra FO
- Incorporar datos de peso/volumen y comuna destino para mejorar el score de complejidad
- Monitorear COURIER INTERNACIONAL y FALAFLEX con KPI mensual de tasa de atraso
- Analizar el pico de 0h en inicio de itinerarios — confirmar si es procesamiento batch o error de datos
- Ampliar la muestra FO para incluir periodos anteriores y detectar estacionalidad

---

*Elaborado por Conectamos — Logistica Inteligente | Marzo 2026*