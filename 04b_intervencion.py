# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 4b. ANÁLISIS DE INTERVENCIÓN (choque de la COVID-19, marzo de 2020)
 Referencia: Guerrero, Análisis estadístico de series de tiempo económicas,
             cap. 7 (Box y Tiao, 1975).
=======================================================================
Motivo: en el Paso 7 el residuo de marzo de 2020 resultó atípico (e* = 6.0)
y hacía fallar la normalidad. No se elimina el dato: se modela el efecto
del evento externo que lo causó.

Modelo con intervención (Guerrero, ec. 7.1.1):
      Z_t = epsilon_{I,t} + N_t
   N_t ............ parte estocástica: el ARIMA de siempre
   epsilon_{I,t} .. función de intervención: efecto del evento ocurrido en t = I

Pasos de este script:
  1. Estadístico C (ec. 7.1.14): ¿el evento tuvo un efecto significativo?
       Se estima el modelo SÓLO con datos anteriores al evento (hasta I - 1) y
       se calculan los errores de pronóstico a un paso de los H meses siguientes:
           C = suma_{h=1..H} e^2 / sigma^2_a   ~   ji-cuadrada con H gl
       H0: el evento no cambió el comportamiento de la serie.
  2. Errores de pronóstico h = 1..H pasos desde el origen I - 1 (figura 7.4
     del libro): su forma sugiere el tipo de función de intervención.
  3. Comparación de funciones de intervención (cuadro 7.1 del libro):
       a) Escalón en marzo:        nabla eps = omega_0 P_I
       b) Cambio de nivel en dos meses (marzo y abril):
                                   nabla eps = (omega_0 - omega_1 B) P_I
     Se estiman junto con el ARIMA (máxima verosimilitud).
  4. Primera diferencia corregida: nabla Z_t sin los saltos de marzo y abril
     de 2020. Con ella se repite la identificación en el Paso 5.
=======================================================================
"""

from configuracion import *
from statsmodels.stats.diagnostic import normal_ad

# =====================================================================
# AJUSTES DE ESTE PASO
# =====================================================================
ALFA = 0.05                    # nivel de significancia del estadístico C y de la regla empírica
FECHA_EVENTO = "2020-03-01"    # I: mes en que ocurre la intervención (COVID-19)
H = 12                         # meses después del evento que se usan en C y en la gráfica
MODELO_BASE = MODELO_A         # modelo con el que se prueba y se compara la intervención

# Funciones de intervención a comparar: nombre -> fechas de los escalones
FORMAS = {
    "Sin intervención": [],
    "Escalón en mar-2020": ["2020-03-01"],
    "Cambio de nivel en dos meses (mar y abr-2020)": ["2020-03-01", "2020-04-01"],
}


def nombre(orden):
    return f"ARIMA({orden[0]},{orden[1]},{orden[2]})"


# =====================================================================
# 1. DATOS
# =====================================================================
encabezado("PASO 4b. ANÁLISIS DE INTERVENCIÓN")
Z = obtener_muestras(cargar_fix())["entrenamiento"]
I = pd.Timestamp(FECHA_EVENTO)
antes = Z[Z.index < I]
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {len(Z)}")
print(f"Evento: {mes_anio(I)} (COVID-19). Modelo base: {nombre(MODELO_BASE)}")
print(f"Datos previos al evento: {periodo(antes)}, N = {len(antes)}")


# =====================================================================
# 2. ESTADÍSTICO C DE BOX Y TIAO
# =====================================================================
encabezado(f"1. Estadístico C (H = {H} meses después del evento)")

res_antes = estimar_arima(antes, MODELO_BASE, fechas=[])      # sin intervención
sigma2_antes = res_antes.params["sigma2"]
fin_H = Z.index[Z.index.get_loc(I) + H - 1]
# Errores de pronóstico a un paso: se aplican los parámetros estimados con datos
# previos al evento (sin volver a estimar) a la serie que incluye los H meses.
res_ext = res_antes.apply(diferenciar(Z[:fin_H], MODELO_BASE[1]))
e1 = res_ext.resid[I:fin_H]
C = np.sum(e1 ** 2) / sigma2_antes
vc = stats.chi2.ppf(1 - ALFA, H)
p_C = 1 - stats.chi2.cdf(C, H)
print(f"Modelo estimado con datos hasta {mes_anio(antes.index[-1])}: "
      f"sigma^2_a = {sigma2_antes:.4f}")
print("Errores de pronóstico a un paso:")
for f, v in e1.items():
    print(f"   {mes_anio(f)}: {v:8.4f}")
print(f"\nH0: el evento no tuvo efecto  vs  H1: sí tuvo efecto")
print(f"C = {C:.3f}   valor crítico ji-cuadrada({H}) al {ALFA * 100:g}% = {vc:.3f}   "
      f"valor p = {p_C:.6f}")
print(f"-> {'Se rechaza H0: el evento SÍ tuvo un efecto significativo' if C > vc else 'No se rechaza H0: no hay evidencia de efecto'}")


# =====================================================================
# 3. ERRORES DE PRONÓSTICO h PASOS DESDE EL ORIGEN I - 1
# =====================================================================
encabezado(f"2. Errores de pronóstico desde {mes_anio(antes.index[-1])} (h = 1, ..., {H})")
pron = pronostico_niveles(res_antes, antes, H)
media = pron["Pronóstico"]
ee = pron["Error estándar"]
real = Z[I:fin_H]
errores = real.values - media.values
z = valor_z(ALFA)
tabla_err = pd.DataFrame({"Fecha": [mes_anio(f) for f in real.index], "h": range(1, H + 1),
                          "Observado": real.values.round(4), "Pronóstico": media.values.round(4),
                          "Error": errores.round(4), f"Límite ±{z:.2f} ee": (z * ee.values).round(4)})
print(tabla_err.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(real.index, -z * ee.values, z * ee.values, color=TEAL, alpha=0.2, lw=0,
                label=rf"$\pm {z:.2f}$ error estándar")
ax.plot(real.index, errores, color=NAVY, lw=1.6, marker="o", ms=4, label="Error de pronóstico")
ax.axhline(0, color=CHARCOAL, lw=0.8)
ax.set_title(f"Errores de pronóstico desde {mes_anio(antes.index[-1])}, "
             f"{nombre(MODELO_BASE)} estimado sin el periodo de la pandemia")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Observado - pronóstico (pesos por dólar)")
formato_fechas(ax, con_mes=True)
ax.legend(loc="upper right")
guardar_figura(fig, "fig20_errores_pronostico_covid.png")


# =====================================================================
# 4. COMPARACIÓN DE FUNCIONES DE INTERVENCIÓN (estimadas junto con el ARIMA)
# =====================================================================
encabezado(f"3. Funciones de intervención con el modelo {nombre(MODELO_BASE)}")
filas = []
resultados = {}
for etiqueta, fechas in FORMAS.items():
    res = estimar_arima(Z, MODELO_BASE, fechas=fechas)
    resultados[etiqueta] = res
    e = res.resid
    sigma = np.sqrt(res.params["sigma2"])
    regla, regla_ok = regla_empirica(e, ALFA)
    pct = regla["% observado"].values
    print(f"\n{etiqueta}")
    for param in res.params.index:
        if param.startswith("escalon"):
            print(f"   omega ({param}) = {res.params[param]:8.4f}   ee = {res.bse[param]:.4f}   "
                  f"t = {res.tvalues[param]:6.2f}   valor p = {res.pvalues[param]:.4f}")
    print(f"   sigma_a = {sigma:.4f}   AIC = {res.aic:.3f}   AICc = {res.aicc:.3f}   BIC = {res.bic:.3f}")
    print(f"   Máximo |e*| = {np.max(np.abs(e)) / sigma:.2f}.  Regla empírica "
          f"(rango del {(1 - ALFA) * 100:g}% de 20000 muestras normales simuladas):")
    for _, f in regla.iterrows():
        print(f"      {f['Banda']} desv. est.: observado {f['% observado']:5.1f}%  teórico "
              f"{f['% teórico']:5.2f}%  rango [{f['Límite inferior']:.1f}, {f['Límite superior']:.1f}]  "
              f"valor p = {f['Valor p']:.4f}  {'compatible' if f['¿Compatible?'] == 'Sí' else 'NO compatible'}")
    print(f"      -> la regla empírica {'se cumple' if regla_ok else 'NO se cumple'}")
    filas.append({"Función de intervención": etiqueta, "Escalones": len(fechas),
                  "sigma_a": round(sigma, 4), "AIC": round(res.aic, 3), "AICc": round(res.aicc, 3),
                  "BIC": round(res.bic, 3), "% ±1": round(pct[0], 1), "% ±2": round(pct[1], 1),
                  "% ±3": round(pct[2], 1), "Regla empírica": "Se cumple" if regla_ok else "No se cumple",
                  "Valor p A-D": round(normal_ad(e.values)[1], 4),
                  "Máximo |e*|": round(np.max(np.abs(e)) / sigma, 3)})
comparacion = pd.DataFrame(filas)

encabezado("Cuadro. Comparación de funciones de intervención (menor AIC/BIC es mejor)")
print(comparacion.to_string(index=False))

sigma_sin = np.sqrt(resultados["Sin intervención"].params["sigma2"])
res_elegida = estimar_arima(Z, MODELO_BASE)            # la de configuracion.py
sigma_con = np.sqrt(res_elegida.params["sigma2"])
print(f"\nIntervención usada en el proyecto (configuracion.py): escalones en "
      f"{', '.join(mes_anio(pd.Timestamp(f)) for f in INTERVENCIONES)}")
print(f"Reducción de la desviación estándar residual: {sigma_con:.4f} / {sigma_sin:.4f} = "
      f"{sigma_con / sigma_sin:.3f}  ->  {100 * (1 - sigma_con / sigma_sin):.1f}% menos")


# =====================================================================
# 5. PRIMERA DIFERENCIA CORREGIDA POR LA INTERVENCIÓN
#    Como d = 1, un escalón en Z_t es un pulso en nabla Z_t: el modelo sólo
#    quita los saltos extraordinarios de marzo y abril de 2020.
#    nabla Z corregida = nabla Z - nabla(suma de omega_hat * S_t)
# =====================================================================
X = variables_intervencion(Z.index)
omegas = res_elegida.params[X.columns]
efecto = (X @ omegas).diff().dropna()          # efecto sobre nabla Z (pulsos)
W = Z.diff().dropna()
W_corr = W - efecto
print("\nEfecto estimado de la intervención sobre la primera diferencia (se resta):")
for c, f in zip(X.columns, INTERVENCIONES):
    print(f"   {mes_anio(pd.Timestamp(f))}: nabla Z observada = {W[f]:.4f}, "
          f"omega = {omegas[c]:.4f}, nabla Z corregida = {W_corr[f]:.4f}")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(W.index, W.values, color=GOLD, lw=1.3, label="Primera diferencia observada")
ax.plot(W_corr.index, W_corr.values, color=NAVY, lw=1.5,
        label="Primera diferencia corregida por la intervención")
ax.axhline(0, color=CHARCOAL, lw=0.8)
ax.axvline(I, color=RED, lw=1, ls="--", label=f"Intervención ({mes_anio(I)})")
ax.set_title(f"Primera diferencia del tipo de cambio FIX, promedio mensual, {periodo(W)}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel(r"$\nabla Z_t$ (pesos por dólar)")
formato_fechas(ax, cada_anios=1)
ax.legend(loc="upper left")
guardar_figura(fig, "fig21_diferencia_corregida.png")


print("Listo")