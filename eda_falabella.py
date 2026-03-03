"""
========================================================
 EDA de Operaciones Logísticas
Archivo: foorch_temp_aj.xlsx
========================================================
"""

import json
import warnings
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paleta de colores consistente ──────────────────────
C = {
    "blue":    "#1565C0",
    "teal":    "#00695C",
    "orange":  "#E65100",
    "purple":  "#6A1B9A",
    "red":     "#C62828",
    "green":   "#2E7D32",
    "gray":    "#455A64",
    "yellow":  "#F9A825",
    "light":   "#ECEFF1",
    "dark":    "#0D1B2A",
}
PALETTE = [C["blue"], C["teal"], C["orange"], C["purple"],
           C["red"],  C["green"], C["gray"],  C["yellow"]]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#DEE2E6",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ══════════════════════════════════════════════════════
# 1. CARGA Y PARSEO
# ══════════════════════════════════════════════════════
print("Cargando datos…")

wb = openpyxl.load_workbook("foorch_temp_aj.xlsx", read_only=True)
ws = wb["in"]
raw_rows = list(ws.iter_rows(values_only=True))
wb.close()

# Filtrar header y fila basura
data_rows = [r for r in raw_rows[1:]
             if r[0] and str(r[0]).startswith("FOFCL")]
print(f"  → {len(data_rows):,} órdenes válidas")


def parse_record(row):
    order_id, promised_raw, ops_raw = row
    rec = {"order_id": order_id}

    # ── Promised date ───────────────────────────────────
    try:
        p = json.loads(promised_raw)
        rec["service_category"]   = p.get("serviceCategory", "UNKNOWN")
        rec["collect_date"]       = (p.get("collectAvailabilityDate")
                                     or p.get("dateTo"))
        rec["time_range"]         = p.get("timeRange")
        rec["time_range_from"]    = p.get("timeRangeFrom")
        rec["time_range_to"]      = p.get("timeRangeTo")
    except Exception:
        pass

    # ── Raw logistic operations ─────────────────────────
    try:
        ops = json.loads(ops_raw)

        processes = [o for o in ops if o["type"] == "PROCESS"]
        legs      = [o for o in ops if o["type"] == "LEG"]

        rec["num_ops"]       = len(ops)
        rec["num_processes"] = len(processes)
        rec["num_legs"]      = len(legs)
        rec["num_transfers"] = sum(1 for o in legs
                                   if o.get("legType") == "TRANSFER")

        lp_set = {o.get("logisticProcess") for o in ops
                  if o.get("logisticProcess")}
        rec["has_crossdocking"] = "CROSS_DOCKING" in lp_set
        rec["has_first_mile"]   = "FIRST_MILE"    in lp_set
        rec["num_logistic_processes"] = len(lp_set)

        lt_set = {o.get("legType") for o in legs if o.get("legType")}
        if "HOME_DELIVERY" in lt_set:
            rec["delivery_type"] = "HOME_DELIVERY"
        elif "COLLECT" in lt_set:
            rec["delivery_type"] = "COLLECT"
        else:
            rec["delivery_type"] = "OTHER"

        # Nodo origen (primer PROCESS con nodo)
        first_proc = next(
            (o for o in ops if o["type"] == "PROCESS" and o.get("node")),
            None)
        if first_proc:
            rec["origin_node_type"] = first_proc["node"].get("type")
            rec["origin_node_name"] = first_proc["node"].get("nodeName")
            rec["origin_operator"]  = first_proc["node"].get("operatorName")

        # Nodo destino final
        last_leg = next(
            (o for o in reversed(ops)
             if o["type"] == "LEG" and o.get("dispatchNode")),
            None)
        if last_leg:
            rec["dest_node_type"] = last_leg["dispatchNode"].get("type")
            rec["dest_node_name"] = last_leg["dispatchNode"].get("nodeName")

        # Operadores
        operators = {o["transportOperator"]["name"]
                     for o in ops
                     if o.get("transportOperator", {}).get("name")}
        rec["num_operators"]  = len(operators)
        rec["operators_list"] = "|".join(sorted(operators))

        # Carrier final
        if ops:
            top = ops[-1].get("transportOperator") or {}
            rec["final_carrier"] = top.get("name")

        # Tiempos
        all_times = []
        for o in ops:
            for field in ("startTime", "endTime"):
                t = o.get(field)
                if t and len(t) >= 5:
                    try:
                        all_times.append(datetime(*t[:5]))
                    except Exception:
                        pass
        if all_times:
            span_hs = (max(all_times) - min(all_times)).total_seconds() / 3600
            rec["total_span_hours"] = round(span_hs, 2)
            rec["start_hour"]       = min(all_times).hour

        rec["total_process_time_mins"] = sum(
            o.get("processTimeInMins") or 0 for o in ops)

        # Picking window
        pick_op = next(
            (o for o in ops if o.get("processType") == "PICKING"), None)
        if pick_op:
            tl = pick_op.get("timeline", {})
            pd_t = tl.get("pickingDeadline")
            st_t = pick_op.get("startTime")
            if pd_t and st_t and len(pd_t) >= 5 and len(st_t) >= 5:
                try:
                    dl  = datetime(*pd_t[:5])
                    st  = datetime(*st_t[:5])
                    rec["picking_window_hours"] = round(
                        (dl - st).total_seconds() / 3600, 2)
                except Exception:
                    pass

    except Exception:
        pass

    return rec


records = [parse_record(r) for r in data_rows]
df = pd.DataFrame(records)

print(f"  → DataFrame: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"  → Columnas: {list(df.columns)}")


# ══════════════════════════════════════════════════════
# 2. SCORE DE COMPLEJIDAD
# ══════════════════════════════════════════════════════
def complexity_score(row):
    score = 0
    score += (row.get("num_ops",    2) - 2) * 10
    score += 15 if row.get("has_crossdocking") else 0
    score += 10 if row.get("origin_node_type") == "SELLER" else 0
    score += 5  if row.get("num_operators", 1) >= 2 else 0
    score += 5  if (row.get("picking_window_hours") or 99) < 3 else 0
    score += 10 if row.get("num_transfers", 0) >= 3 else 0
    return score


df["complexity_score"] = df.apply(complexity_score, axis=1)

# Señales de riesgo derivadas
cat_medians = {"MESON": 26, "REGULAR": 34, "DATE_RANGE": 297, "TO_CAR": 29}
df["span_ratio"] = df.apply(
    lambda r: (r["total_span_hours"] / cat_medians[r["service_category"]])
    if (pd.notna(r.get("total_span_hours"))
        and r.get("service_category") in cat_medians) else np.nan,
    axis=1)

df["short_picking_window"] = df["picking_window_hours"] < 3
df["complex_route"]        = df["num_ops"] >= 8
df["anomalous_span"]       = df["span_ratio"] > 3
df["seller_origin"]        = df["origin_node_type"] == "SELLER"


# ══════════════════════════════════════════════════════
# 3. FIGURA 1 — DISTRIBUCIÓN GENERAL
# ══════════════════════════════════════════════════════
print("\nGenerando Figura 1 — Distribución general…")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "PULSE EDA — Distribución General de Órdenes",
    fontsize=15, fontweight="bold", color=C["dark"], y=1.01)

# 1a. Service category
ax = axes[0, 0]
sc = df["service_category"].value_counts()
bars = ax.barh(sc.index, sc.values, color=PALETTE[:len(sc)])
ax.set_title("Service Category", fontweight="bold")
ax.set_xlabel("Cantidad de órdenes")
for bar, val in zip(bars, sc.values):
    pct = val / len(df) * 100
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
ax.set_xlim(0, sc.max() * 1.25)

# 1b. Delivery type
ax = axes[0, 1]
dt = df["delivery_type"].value_counts()
wedges, texts, autotexts = ax.pie(
    dt.values, labels=dt.index, autopct="%1.1f%%",
    colors=PALETTE[:len(dt)], startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2})
for t in autotexts:
    t.set_fontsize(11); t.set_fontweight("bold")
ax.set_title("Delivery Type", fontweight="bold")

# 1c. Complejidad de ruta
ax = axes[0, 2]
ops_cnt = df["num_ops"].value_counts().sort_index()
bars = ax.bar(ops_cnt.index.astype(str), ops_cnt.values,
              color=PALETTE[:len(ops_cnt)], edgecolor="white", linewidth=1.5)
ax.set_title("Complejidad de Ruta (# operaciones)", fontweight="bold")
ax.set_xlabel("Número de operaciones")
ax.set_ylabel("Órdenes")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 20,
            f"{h:,}\n({h/len(df)*100:.1f}%)",
            ha="center", fontsize=8)

# 1d. Origin node type
ax = axes[1, 0]
ont = df["origin_node_type"].value_counts()
bars = ax.bar(ont.index, ont.values,
              color=[C["orange"], C["blue"], C["teal"]], edgecolor="white")
ax.set_title("Tipo de Nodo Origen", fontweight="bold")
ax.set_ylabel("Órdenes")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 20,
            f"{h:,}\n({h/len(df)*100:.1f}%)", ha="center", fontsize=9)

# 1e. Carrier final
ax = axes[1, 1]
cf = df["final_carrier"].value_counts().head(8)
bars = ax.barh(cf.index[::-1], cf.values[::-1], color=PALETTE[:len(cf)])
ax.set_title("Carrier Final (Top 8)", fontweight="bold")
ax.set_xlabel("Órdenes")
for bar, val in zip(bars, cf.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlim(0, cf.max() * 1.2)

# 1f. Crossdocking
ax = axes[1, 2]
cd_vals = df["has_crossdocking"].value_counts()
labels = ["Con Crossdocking", "Sin Crossdocking"]
colors = [C["purple"], C["teal"]]
wedges, texts, autotexts = ax.pie(
    [cd_vals.get(True, 0), cd_vals.get(False, 0)],
    labels=labels, autopct="%1.1f%%", colors=colors,
    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
for t in autotexts:
    t.set_fontsize(11); t.set_fontweight("bold")
ax.set_title("Presencia de Crossdocking", fontweight="bold")

plt.tight_layout()
plt.savefig("fig1_distribucion_general.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig1_distribucion_general.png ✓")


# ══════════════════════════════════════════════════════
# 4. FIGURA 2 — ANÁLISIS TEMPORAL
# ══════════════════════════════════════════════════════
print("Generando Figura 2 — Análisis temporal…")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("PULSE EDA — Análisis Temporal de Órdenes",
             fontsize=15, fontweight="bold", color=C["dark"])

# 2a. Total span hours por categoría (boxplot)
ax = axes[0, 0]
cats_order = ["MESON", "REGULAR", "DATE_RANGE", "TO_CAR"]
data_box = [df[df["service_category"] == c]["total_span_hours"].dropna().values
            for c in cats_order]
bp = ax.boxplot(data_box, labels=cats_order, patch_artist=True,
                medianprops={"color": "white", "linewidth": 2.5},
                flierprops={"marker": "o", "markersize": 3,
                            "alpha": 0.3, "color": C["gray"]})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_title("Duración Total Planificada por Categoría", fontweight="bold")
ax.set_ylabel("Horas")
ax.set_yscale("log")
ax.set_ylim(1, 1000)

# Annotate medians
medians_by_cat = df.groupby("service_category")["total_span_hours"].median()
for i, cat in enumerate(cats_order):
    med = medians_by_cat.get(cat, np.nan)
    if not np.isnan(med):
        ax.text(i + 1, med * 1.5, f"med={med:.0f}h",
                ha="center", fontsize=8, fontweight="bold",
                color=PALETTE[i])

# 2b. Picking window distribution
ax = axes[0, 1]
pw = df["picking_window_hours"].dropna()
pw_clipped = pw.clip(upper=100)
ax.hist(pw_clipped, bins=40, color=C["blue"], edgecolor="white",
        linewidth=0.5, alpha=0.85)
ax.axvline(3, color=C["red"], linestyle="--", linewidth=2,
           label="Umbral riesgo (3h)")
ax.axvline(pw.median(), color=C["orange"], linestyle="--", linewidth=2,
           label=f"Mediana ({pw.median():.0f}h)")
n_short = (pw < 3).sum()
ax.text(3.5, ax.get_ylim()[1] * 0.85,
        f"{n_short:,} órdenes\n({n_short/len(df)*100:.1f}%) con\nwindow < 3h",
        color=C["red"], fontsize=9, fontweight="bold")
ax.set_title("Distribución Picking Window (cap. 100h)", fontweight="bold")
ax.set_xlabel("Horas"); ax.set_ylabel("Órdenes")
ax.legend(fontsize=9)

# 2c. Span percentiles por categoría — bar chart
ax = axes[1, 0]
pct_data = {}
for cat in ["MESON", "REGULAR", "DATE_RANGE"]:
    vals = df[df["service_category"] == cat]["total_span_hours"].dropna()
    pct_data[cat] = {
        "P25": vals.quantile(0.25),
        "P50": vals.quantile(0.50),
        "P75": vals.quantile(0.75),
        "P90": vals.quantile(0.90),
    }

x = np.arange(4)
width = 0.25
for i, (cat, pcts) in enumerate(pct_data.items()):
    vals = list(pcts.values())
    bars = ax.bar(x + i * width, vals, width, label=cat,
                  color=PALETTE[i], edgecolor="white", alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(["P25", "P50", "P75", "P90"])
ax.set_title("Percentiles de Duración por Categoría", fontweight="bold")
ax.set_ylabel("Horas")
ax.legend(fontsize=9)

# 2d. Start hour heatmap-style
ax = axes[1, 1]
sh = df["start_hour"].value_counts().sort_index()
colors_bar = [C["red"] if h in [7, 8, 9] else C["blue"] for h in sh.index]
bars = ax.bar(sh.index, sh.values, color=colors_bar, edgecolor="white")
ax.set_title("Hora de Inicio del Itinerario", fontweight="bold")
ax.set_xlabel("Hora del día"); ax.set_ylabel("Órdenes")
ax.set_xticks(range(0, 24))

peak_patch  = mpatches.Patch(color=C["red"],  label="Pico operativo (7–9h)")
other_patch = mpatches.Patch(color=C["blue"], label="Resto del día")
ax.legend(handles=[peak_patch, other_patch], fontsize=9)

plt.tight_layout()
plt.savefig("fig2_analisis_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig2_analisis_temporal.png ✓")


# ══════════════════════════════════════════════════════
# 5. FIGURA 3 — SEÑALES DE RIESGO
# ══════════════════════════════════════════════════════
print("Generando Figura 3 — Señales de riesgo…")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("PULSE EDA — Señales de Riesgo Detectadas",
             fontsize=15, fontweight="bold", color=C["dark"])

# 3a. Resumen señales de riesgo
ax = axes[0, 0]
risk_signals = {
    "Picking window < 3h": df["short_picking_window"].sum(),
    "Span > 3x mediana":   df["anomalous_span"].sum(),
    "Ruta compleja (≥8 ops)": df["complex_route"].sum(),
    "Origen SELLER\n(menos control)": df["seller_origin"].sum(),
    "3+ operadores en ruta": (df["num_operators"] >= 3).sum(),
}
labels = list(risk_signals.keys())
values = list(risk_signals.values())
pcts   = [v / len(df) * 100 for v in values]
colors_risk = [C["red"], C["orange"], C["purple"],
               C["yellow"], C["teal"]]
bars = ax.barh(labels, values, color=colors_risk, edgecolor="white")
for bar, val, pct in zip(bars, values, pcts):
    ax.text(bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
ax.set_title("Señales de Riesgo Operativo", fontweight="bold")
ax.set_xlabel("Número de órdenes")
ax.set_xlim(0, max(values) * 1.3)

# 3b. Complexity score distribución
ax = axes[0, 1]
bins = [0, 20, 40, 60, 80, 100, 130]
labels_bins = ["0-20\n(Simple)", "20-40", "40-60",
               "60-80", "80-100", "100+\n(Complejo)"]
counts, _ = np.histogram(df["complexity_score"], bins=bins)
colors_cs = [C["teal"], C["blue"], C["yellow"],
             C["orange"], C["red"], C["purple"]]
bars = ax.bar(labels_bins, counts, color=colors_cs, edgecolor="white")
for bar, val in zip(bars, counts):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=8)
ax.set_title("Distribución Score de Complejidad", fontweight="bold")
ax.set_ylabel("Órdenes")

# 3c. Carrier vs complexity score
ax = axes[0, 2]
carrier_stats = (df.groupby("final_carrier")["complexity_score"]
                   .agg(["mean", "count"])
                   .query("count >= 50")
                   .sort_values("mean", ascending=True))
colors_c = [C["red"] if v > 70 else C["orange"] if v > 40 else C["teal"]
            for v in carrier_stats["mean"]]
bars = ax.barh(carrier_stats.index, carrier_stats["mean"],
               color=colors_c, edgecolor="white")
ax.axvline(df["complexity_score"].mean(), color=C["gray"],
           linestyle="--", linewidth=1.5,
           label=f"Media global ({df['complexity_score'].mean():.0f})")
for bar, (_, row) in zip(bars, carrier_stats.iterrows()):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{row['mean']:.1f}  (n={int(row['count'])})",
            va="center", fontsize=8)
ax.set_title("Score Complejidad Promedio por Carrier", fontweight="bold")
ax.set_xlabel("Score promedio")
ax.set_xlim(0, carrier_stats["mean"].max() * 1.3)
ax.legend(fontsize=8)

# Patch carrier colors
high  = mpatches.Patch(color=C["red"],    label="Alto riesgo (>70)")
mid   = mpatches.Patch(color=C["orange"], label="Medio (40-70)")
low   = mpatches.Patch(color=C["teal"],   label="Bajo (<40)")
ax.legend(handles=[high, mid, low], fontsize=8, loc="lower right")

# 3d. Span ratio distribution (anomalías)
ax = axes[1, 0]
sr = df["span_ratio"].dropna().clip(upper=15)
ax.hist(sr, bins=50, color=C["blue"], edgecolor="white",
        linewidth=0.5, alpha=0.85)
ax.axvline(3, color=C["red"], linestyle="--", linewidth=2,
           label="Umbral anomalía (3x)")
ax.axvline(1, color=C["green"], linestyle="--", linewidth=2,
           label="Normal (1x mediana)")
n_anom = (df["span_ratio"] > 3).sum()
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500],
                 3, 15, alpha=0.12, color=C["red"])
ax.set_title("Span Ratio vs Mediana de Categoría\n(cap. 15x)",
             fontweight="bold")
ax.set_xlabel("Ratio (span real / mediana categoría)")
ax.set_ylabel("Órdenes")
ax.legend(fontsize=9)

# Recalculate after histogram
ax.set_ylim(bottom=0)
ymax = ax.get_ylim()[1]
ax.fill_betweenx([0, ymax], 3, 15, alpha=0.12, color=C["red"])
ax.text(8, ymax * 0.7,
        f"{n_anom:,} órdenes\nanómalas\n({n_anom/len(df)*100:.1f}%)",
        color=C["red"], fontsize=10, fontweight="bold", ha="center")

# 3e. Risk matrix: complejidad vs picking window
ax = axes[1, 1]
df_plot = df[["complexity_score", "picking_window_hours",
              "service_category"]].dropna()

cat_colors = {"MESON": C["blue"], "REGULAR": C["teal"],
              "DATE_RANGE": C["orange"], "TO_CAR": C["purple"],
              "SAME_DAY": C["red"]}

for cat, grp in df_plot.groupby("service_category"):
    ax.scatter(grp["picking_window_hours"].clip(upper=100),
               grp["complexity_score"],
               c=cat_colors.get(cat, C["gray"]),
               alpha=0.35, s=15, label=cat)

ax.axvline(3,  color=C["red"],    linestyle="--", linewidth=1.5,
           label="Window < 3h")
ax.axhline(60, color=C["orange"], linestyle="--", linewidth=1.5,
           label="Score > 60")
ax.fill_betweenx([60, 130], 0, 3,  alpha=0.10, color=C["red"])
ax.text(1.5, 110, "ZONA\nALTO RIESGO",
        ha="center", color=C["red"], fontsize=9, fontweight="bold")
ax.set_title("Matriz de Riesgo:\nWindow vs Complejidad", fontweight="bold")
ax.set_xlabel("Picking window (horas, cap. 100h)")
ax.set_ylabel("Complexity score")
ax.legend(fontsize=8, markerscale=2)

# 3f. Origin node type x risk signals
ax = axes[1, 2]
risk_cols = {
    "short_picking_window": "Window < 3h",
    "anomalous_span":       "Span anómalo",
    "complex_route":        "Ruta compleja",
}
origin_types = ["SELLER", "STORE", "WAREHOUSE"]
x = np.arange(len(origin_types))
width = 0.28

for i, (col, label) in enumerate(risk_cols.items()):
    rates = []
    for ot in origin_types:
        subset = df[df["origin_node_type"] == ot]
        rate = subset[col].mean() * 100 if len(subset) > 0 else 0
        rates.append(rate)
    bars = ax.bar(x + i * width, rates, width,
                  label=label, color=PALETTE[i], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, rates):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(origin_types)
ax.set_title("Tasa de Señales de Riesgo\npor Tipo de Origen", fontweight="bold")
ax.set_ylabel("% de órdenes con señal")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig3_senales_riesgo.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig3_senales_riesgo.png ✓")


# ══════════════════════════════════════════════════════
# 6. FIGURA 4 — ACTORES Y FLUJOS
# ══════════════════════════════════════════════════════
print("Generando Figura 4 — Actores y flujos…")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("PULSE EDA — Actores Operativos y Flujos",
             fontsize=15, fontweight="bold", color=C["dark"])

# 4a. Top origin nodes
ax = axes[0, 0]
top_nodes = df["origin_node_name"].value_counts().head(12)
bars = ax.barh(top_nodes.index[::-1], top_nodes.values[::-1],
               color=C["blue"], edgecolor="white", alpha=0.85)
for bar, val in zip(bars, top_nodes.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8)
ax.set_title("Top 12 Nodos de Origen", fontweight="bold")
ax.set_xlabel("Órdenes")

# 4b. Combos service_cat + delivery_type
ax = axes[0, 1]
combos = (df.groupby(["service_category", "delivery_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False))
combo_labels = [f"{r.service_category}\n+ {r.delivery_type}"
                for _, r in combos.iterrows()]
bars = ax.bar(combo_labels, combos["count"],
              color=PALETTE[:len(combos)], edgecolor="white")
for bar, val in zip(bars, combos["count"]):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=8)
ax.set_title("Combinaciones Service Category + Delivery", fontweight="bold")
ax.set_ylabel("Órdenes")
ax.tick_params(axis="x", labelsize=8)

# 4c. Número de operadores en ruta
ax = axes[1, 0]
nop = df["num_operators"].value_counts().sort_index()
bars = ax.bar(nop.index.astype(str), nop.values,
              color=[C["teal"], C["orange"], C["red"]],
              edgecolor="white")
for bar, val in zip(bars, nop.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=10)
ax.set_title("Número de Operadores por Ruta\n(potencial punto de quiebre)",
             fontweight="bold")
ax.set_xlabel("Operadores en la ruta")
ax.set_ylabel("Órdenes")

# 4d. Destination node type
ax = axes[1, 1]
dnt = df["dest_node_type"].value_counts()
bars = ax.barh(dnt.index[::-1], dnt.values[::-1],
               color=PALETTE[:len(dnt)], edgecolor="white", alpha=0.85)
for bar, val in zip(bars, dnt.values[::-1]):
    pct = val / len(df) * 100
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
ax.set_title("Tipo de Nodo Destino", fontweight="bold")
ax.set_xlabel("Órdenes")
ax.set_xlim(0, dnt.max() * 1.25)

plt.tight_layout()
plt.savefig("fig4_actores_flujos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig4_actores_flujos.png ✓")


# ══════════════════════════════════════════════════════
# 7. FIGURA 5 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════
print("Generando Figura 5 — Resumen ejecutivo…")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(C["dark"])

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# KPIs principales (tarjetas)
kpis = [
    ("5,591",  "Órdenes\nanalizadas",    C["blue"]),
    ("46.8%",  "Son MESON\n(retiro tienda)", C["teal"]),
    ("52.6%",  "Home\nDelivery",         C["orange"]),
    ("64.8%",  "Origen\nSELLER",         C["purple"]),
    ("39.8%",  "Picking window\n< 3h",   C["red"]),
    ("234",    "Spans\nanómalos",         C["yellow"]),
    ("14.8%",  "Rutas\ncomplejas (≥8)",  C["gray"]),
    ("35.7%",  "Con\ncrossdocking",      C["green"]),
]

for i, (val, label, color) in enumerate(kpis):
    row, col = divmod(i, 4)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.62, val, ha="center", va="center",
            fontsize=26, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.22, label, ha="center", va="center",
            fontsize=10, color="white", alpha=0.9,
            transform=ax.transAxes)

    # Borde
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("white")
        ax.spines[spine].set_linewidth(1.5)

# Título
fig.text(0.5, 1.01,
         "PULSE — Resumen Ejecutivo EDA",
         ha="center", fontsize=17, fontweight="bold", color="white")
fig.text(0.5, 0.97,
         "foorch_temp_aj.xlsx  |  5,591 Fulfilment Orders  |  Conectamos 2025",
         ha="center", fontsize=10, color="#AAAAAA")

plt.savefig("fig5_resumen_ejecutivo.png", dpi=150,
            bbox_inches="tight", facecolor=C["dark"])
plt.close()
print("  → fig5_resumen_ejecutivo.png ✓")


# ══════════════════════════════════════════════════════
# 8. EXPORT CSV
# ══════════════════════════════════════════════════════
export_cols = [
    "order_id", "service_category", "delivery_type",
    "origin_node_type", "origin_node_name", "origin_operator",
    "dest_node_type", "final_carrier",
    "num_ops", "num_processes", "num_legs", "num_transfers",
    "has_crossdocking", "has_first_mile", "num_operators",
    "total_span_hours", "picking_window_hours",
    "total_process_time_mins", "complexity_score",
    "short_picking_window", "anomalous_span",
    "complex_route", "seller_origin", "span_ratio",
]
df[export_cols].to_csv("pulse_eda_features.csv", index=False)
print("\n  → pulse_eda_features.csv ✓")


# ══════════════════════════════════════════════════════
# 9. PRINT SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("HALLAZGOS PRINCIPALES")
print("="*55)

findings = [
    ("Distribución", [
        f"MESON (retiro tienda): {(df['service_category']=='MESON').mean()*100:.1f}% — mediana 26h",
        f"HOME DELIVERY (REGULAR+DATE_RANGE): {((df['service_category'].isin(['REGULAR','DATE_RANGE']))).mean()*100:.1f}%",
        f"64.8% de órdenes con origen SELLER (FBS, menor control)",
    ]),
    ("Complejidad", [
        f"Rutas simples (2 ops): {(df['num_ops']==2).mean()*100:.1f}% — directas sin hub",
        f"Rutas complejas (≥8 ops): {(df['num_ops']>=8).mean()*100:.1f}% — múltiples hubs",
        f"Con crossdocking: {df['has_crossdocking'].mean()*100:.1f}%",
    ]),
    ("Riesgo detectado", [
        f"Picking window < 3h: {df['short_picking_window'].mean()*100:.1f}% ({df['short_picking_window'].sum():,} órdenes)",
        f"Span anómalo (>3x mediana): {df['anomalous_span'].mean()*100:.1f}% ({df['anomalous_span'].sum():,} órdenes)",
        f"SODIMAC avg complexity: {df[df['final_carrier']=='SODIMAC']['complexity_score'].mean():.1f} vs FALABELLA: {df[df['final_carrier']=='FALABELLA']['complexity_score'].mean():.1f}",
    ]),
    ("Próximo paso", [
        "CRÍTICO: conseguir estado final de cada orden (DELIVERED/FAILED/RETURNED)",
        "Útil: peso/volumen, comuna destino, historial de reintentos",
        "Abrir Muestra_FO__1_.xlsb para cruzar con estados finales",
    ]),
]

for section, items in findings:
    print(f"\n  [{section}]")
    for item in items:
        print(f"    • {item}")

print("\n" + "="*55)
print("Archivos generados:")
print("  fig1_distribucion_general.png")
print("  fig2_analisis_temporal.png")
print("  fig3_senales_riesgo.png")
print("  fig4_actores_flujos.png")
print("  fig5_resumen_ejecutivo.png")
print("  pulse_eda_features.csv")
print("="*55)
