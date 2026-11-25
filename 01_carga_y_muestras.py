# -*- coding: utf-8 -*-
# PROYECTO FINAL - ECONOMETRÍA II (UDLAP)
# Paso 1: Carga de la serie FIX, limpieza y definición de muestras.
# Serie: Tipo de cambio FIX, pesos por dólar, promedio mensual (Banxico, SIE, SF17908).
# Periodo de estudio: enero de 2015 a septiembre de 2026.
#   Entrenamiento     : ene-2015 a sep-2025
#   Validación        : oct-2025 a sep-2026 (12 meses fuera de la estimación)
#   Estimación final  : ene-2015 a sep-2026
#   Observación fuera : oct-2026

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ---------------- Estilo de gráficas ----------------
GOLD      = '#B6A269'
NAVY      = '#041E42'
CHARCOAL  = '#222A35'
TEAL      = '#339989'
BLUE      = '#0F6FF5'
RED       = '#F8696B'
GRID_GRIS = '#E2E8F0'

MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]

def mes_anio(fecha):
    return f"{MESES[fecha.month - 1]}-{fecha.year}"

plt.rcParams.update({
    "savefig.dpi": 300, "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelcolor": CHARCOAL, "axes.edgecolor": CHARCOAL,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID_GRIS,
    "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

def formato_fechas(ax, cada_anios=None, con_mes=False):
    if cada_anios is not None:
        ax.xaxis.set_major_locator(mdates.YearLocator(cada_anios))
    def _f(x, pos=None):
        f = mdates.num2date(x)
        return f"{MESES[f.month - 1]}-{str(f.year)[2:]}" if con_mes else str(f.year)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_f))

# ---------------- Parámetros ----------------
RUTA_EXCEL = "Serie FIX.xlsx"
INICIO_ENTRENAMIENTO = "2015-01-01"
FIN_ENTRENAMIENTO    = "2025-09-01"
INICIO_VALIDACION    = "2025-10-01"
FIN_VALIDACION       = "2026-09-01"
OBS_FUERA            = "2026-10-01"

# ---------------- 1. Lectura del Excel ----------------
# Las primeras 18 filas son encabezados del SIE de Banxico
crudo = pd.read_excel(RUTA_EXCEL, header=None, skiprows=18, names=["fecha", "fix"])
crudo["fecha"] = pd.to_datetime(crudo["fecha"], errors="coerce")
crudo["fix"] = pd.to_numeric(crudo["fix"], errors="coerce")
crudo = crudo.dropna()
crudo = crudo[crudo["fix"] > 0]          # ene-1991 a oct-1991 vienen en 0

fix = crudo.set_index("fecha")["fix"].asfreq("MS")
print(f"Serie descargada: {mes_anio(fix.index[0])} a {mes_anio(fix.index[-1])}, N = {len(fix)}")

# ---------------- 2. Partición de la muestra ----------------
entrenamiento = fix[INICIO_ENTRENAMIENTO:FIN_ENTRENAMIENTO]
validacion    = fix[INICIO_VALIDACION:FIN_VALIDACION]
estim_final   = fix[INICIO_ENTRENAMIENTO:FIN_VALIDACION]
obs_fuera     = fix[OBS_FUERA:OBS_FUERA]

def periodo(s):
    return f"{mes_anio(s.index[0])} a {mes_anio(s.index[-1])}"

print("\nTamaño de cada muestra")
print(f"  Entrenamiento     : {periodo(entrenamiento)}, N = {len(entrenamiento)}")
print(f"  Validación        : {periodo(validacion)}, N = {len(validacion)}")
print(f"  Estimación final  : {periodo(estim_final)}, N = {len(estim_final)}")
print(f"  Observación fuera : {mes_anio(obs_fuera.index[0])} = {obs_fuera.iloc[0]:.4f}")

# ---------------- 3. Estadística descriptiva ----------------
def descriptivos(s):
    return pd.Series({
        "N": len(s), "Media": s.mean(), "Desv. est.": s.std(),
        "Mínimo": s.min(), "Máximo": s.max(),
        "Var. % mensual media": 100 * s.pct_change().mean(),
    })

cuadro1 = pd.DataFrame({
    "Estimación final": descriptivos(estim_final),
    "Entrenamiento": descriptivos(entrenamiento),
    "Validación": descriptivos(validacion),
}).round(4)
print("\nCuadro 1. Estadística descriptiva del FIX (pesos por dólar)")
print(cuadro1.to_string())

cambio_oct = 100 * (fix[OBS_FUERA] / fix[FIN_VALIDACION] - 1)
print(f"\nVariación sep-2026 a oct-2026: {cambio_oct:.2f}%")

var_mensual = 100 * fix[INICIO_ENTRENAMIENTO:OBS_FUERA].pct_change().dropna()
print(f"Meses desde ene-2015 con depreciación igual o mayor: {100 * (var_mensual >= cambio_oct).mean():.2f}%")
print("Las 8 mayores depreciaciones mensuales desde ene-2015 (%):")
for f, v in var_mensual.sort_values(ascending=False).head(8).round(2).items():
    print(f"  {mes_anio(f)}: {v}")

# Franja sombreada de la validación (sin tapar la observación de octubre)
FRANJA_VALIDACION = (pd.Timestamp(INICIO_VALIDACION) - pd.Timedelta(days=15),
                     pd.Timestamp(FIN_VALIDACION) + pd.Timedelta(days=15))

# ---------------- 4. Gráfica 1: periodo de estudio ----------------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(entrenamiento.index, entrenamiento.values, color=NAVY, lw=1.4,
        label=f"Entrenamiento ({periodo(entrenamiento)})")
puente = fix[FIN_ENTRENAMIENTO:INICIO_VALIDACION]          # une ambas líneas sin hueco
ax.plot(puente.index, puente.values, color=GOLD, lw=2.2)
ax.plot(validacion.index, validacion.values, color=GOLD, lw=2.2,
        label=f"Validación ({periodo(validacion)})")
salto = fix[FIN_VALIDACION:OBS_FUERA]
ax.plot(salto.index, salto.values, color=RED, lw=1.5, ls="--")
ax.scatter(obs_fuera.index, obs_fuera.values, color=RED, zorder=5, s=40,
           label=f"Observación fuera ({mes_anio(obs_fuera.index[0])})")
ax.axvspan(*FRANJA_VALIDACION, color=GOLD, alpha=0.15, lw=0)
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
ax.set_title(f"Tipo de cambio FIX, promedio mensual, {mes_anio(entrenamiento.index[0])} a {mes_anio(obs_fuera.index[0])}")
formato_fechas(ax, cada_anios=1)
ax.legend(loc="upper left")
fig.tight_layout()
os.makedirs("figuras", exist_ok=True)
fig.savefig(os.path.join("figuras", "fig01_serie_fix.png"))

# ---------------- Gráfica 2: acercamiento 2024-2026 ----------------
fig, ax = plt.subplots(figsize=(10, 4))
reciente = fix["2024-01-01":OBS_FUERA]
ax.plot(reciente.index, reciente.values, marker="o", ms=3.5, color=NAVY, lw=1.4)
ax.scatter(obs_fuera.index, obs_fuera.values, color=RED, zorder=5, s=50,
           label="Observación fuera")
ax.axvspan(*FRANJA_VALIDACION, color=GOLD, alpha=0.2, lw=0, label="Validación")
ax.annotate(f"{mes_anio(obs_fuera.index[0])}: {obs_fuera.iloc[0]:.2f}",
            (obs_fuera.index[0], obs_fuera.iloc[0]),
            textcoords="offset points", xytext=(-85, 6), fontsize=9, color=CHARCOAL)
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
ax.set_title(f"Tipo de cambio FIX, promedio mensual, ene-2024 a {mes_anio(obs_fuera.index[0])}")
formato_fechas(ax, con_mes=True)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join("figuras", "fig02_fix_reciente.png"))


print("\nListo")
plt.close("all")