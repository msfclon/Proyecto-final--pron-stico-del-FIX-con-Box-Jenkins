# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 2. ¿ES ESTACIONARIA LA SERIE ORIGINAL?
 Criterio de la rúbrica: verificar la estacionariedad con pruebas de hipótesis.
=======================================================================
Qué hace (con la muestra de entrenamiento):
  1. Gráfica de la serie con su media: ¿la media es constante?
  2. Prueba de Bartlett para cada autocorrelación r_k:
         H0: rho_k = 0   vs   H1: rho_k distinto de 0
     Se rechaza H0 si |r_k| > z * ee(r_k).
  3. Prueba de Dickey-Fuller aumentada (ADF):
         Delta Z_t = alfa + beta t + gamma Z_{t-1} + sum delta_i Delta Z_{t-i} + e_t
         H0: gamma = 0 (raíz unitaria, NO estacionaria)
         H1: gamma < 0 (estacionaria)
     Se rechaza H0 si tau < valor crítico (prueba de cola izquierda).
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO
# =====================================================================
ALFA = 0.01                  # nivel de significancia 
K_RETRASOS = 36              # retrasos de la FAC (36 = 3 años)
ESPECIFICACION_ADF = "c"     # especificación que decide la prueba:
                             #   "n"  = sin constante ni tendencia
                             #   "c"  = con constante  <- la adecuada para el FIX
                             #   "ct" = con constante y tendencia
NIVEL = f"{ALFA * 100:g}%"   # texto del nivel, por ejemplo "1%"


# =====================================================================
# 1. DATOS (muestra de entrenamiento)
# =====================================================================
encabezado(f"PASO 2. ESTACIONARIEDAD DE LA SERIE ORIGINAL (alfa = {ALFA})")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
N = len(Z)
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {N}")

cruces = int(np.sum(np.diff(np.sign(Z.values - Z.mean())) != 0))
print(f"Media muestral = {Z.mean():.4f}; la serie cruza su media {cruces} veces")


# =====================================================================
# 2. PRUEBA DE BARTLETT SOBRE LA FAC
# =====================================================================
r = fac(Z, K_RETRASOS)
banda = bandas_bartlett(r, N, ALFA)
significativos = [k for k in range(1, K_RETRASOS + 1) if abs(r[k - 1]) > banda[k - 1]]

cuadro_fac = pd.DataFrame({
    "k": np.arange(1, K_RETRASOS + 1),
    "r_k": r.round(4),
    f"Banda ({valor_z(ALFA):.4f} ee)": banda.round(4),
    "¿Se rechaza H0?": np.where(np.abs(r) > banda, "Sí", "No"),
})

encabezado(f"Cuadro. FAC de la serie original y prueba de Bartlett al {NIVEL}")
print(cuadro_fac.head(12).to_string(index=False))
print(f"Retrasos significativos (de {K_RETRASOS}): {len(significativos)} -> {significativos}")
print(f"r_1 = {r[0]:.4f}, r_6 = {r[5]:.4f}, r_12 = {r[11]:.4f}, "
      f"r_24 = {r[23]:.4f}, r_36 = {r[35]:.4f}")


# =====================================================================
# 3. PRUEBA DE DICKEY-FULLER AUMENTADA
# =====================================================================
cuadro_adf = prueba_adf(Z, ALFA)

encabezado(f"Cuadro. Prueba de Dickey-Fuller aumentada al {NIVEL} (rezagos con AIC)")
print(cuadro_adf.drop(columns="Clave").to_string(index=False))


# =====================================================================
# 4. CONCLUSIÓN
# =====================================================================
fila = cuadro_adf[cuadro_adf["Clave"] == ESPECIFICACION_ADF].iloc[0]
es_estacionaria = fila[f"Decisión ({NIVEL})"] == "Se rechaza H0"

encabezado("Conclusión")
print(f"Especificación: {fila['Especificación']}")
print(f"tau = {fila['tau']}, valor p = {fila['Valor p']}, VC 1% = {fila['VC 1%']}, "
      f"VC 5% = {fila['VC 5%']}")
if es_estacionaria:
    print(f"Se rechaza H0 al {NIVEL}: la serie original NO tiene raíz unitaria, es estacionaria.")
else:
    print(f"No se rechaza H0 al {NIVEL}: la serie original tiene raíz unitaria, NO es estacionaria.")
    print("Hay que aplicar el operador diferencia (Paso 4).")


# =====================================================================
# 5. GRÁFICAS
# =====================================================================
encabezado("Gráficas")

# --- Figura 3: serie original con su media ---
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(Z.index, Z.values, color=NAVY, lw=1.3, label="FIX")
ax.axhline(Z.mean(), color=GOLD, lw=1.2, ls="--", label=f"Media = {Z.mean():.2f}")
ax.set_title(f"Tipo de cambio FIX, promedio mensual, {periodo(Z)}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, cada_anios=1)
ax.legend(loc="upper left")
guardar_figura(fig, "fig03_serie_original.png")

# --- Figura 4: FAC con bandas de Bartlett ---
fig, ax = plt.subplots(figsize=(10, 4.5))
k = np.arange(1, K_RETRASOS + 1)
ax.vlines(k, 0, r, color=NAVY, lw=2)
ax.fill_between(k, -banda, banda, color=TEAL, alpha=0.2, lw=0,
                label=f"Bandas de Bartlett ({(1 - ALFA) * 100:g}%)")
ax.axhline(0, color=CHARCOAL, lw=0.8)
ax.set_ylim(-1, 1)
ax.set_title(f"FAC muestral del tipo de cambio FIX, {periodo(Z)}")
ax.set_xlabel("Retraso k (meses)")
ax.set_ylabel(r"$r_k$")
ax.legend(loc="lower left")
guardar_figura(fig, "fig04_fac_original.png")


print("Listo") 