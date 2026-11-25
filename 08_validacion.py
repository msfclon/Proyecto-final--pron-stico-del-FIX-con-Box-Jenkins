# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 8. VALIDACIÓN FUERA DE LA MUESTRA
 ¿Qué tan bien pronostican los modelos datos que NO usaron para estimarse?
=======================================================================
Muestra de entrenamiento: ene-2015 a sep-2025 (con ella se estimaron los modelos).
Muestra de validación:    oct-2025 a sep-2026 (12 meses que el modelo no conoce).

Modelos (todos con la intervención de marzo y abril de 2020):
   A: ARIMA(0,1,1)   B: ARIMA(1,1,0)   C: ARIMA(0,1,0) = paseo aleatorio
   El paseo aleatorio es la REFERENCIA (Meese y Rogoff, 1983): su pronóstico
   es "el último valor observado", Z_hat_t(h) = Z_t.

Dos formas de validar:
  1. Pronóstico desde un solo origen (sep-2025) para h = 1, ..., 12 meses,
     como si en sep-2025 se pronosticara el año siguiente completo.
  2. Origen móvil a un mes: en cada mes t de sep-2025 a
     ago-2026 se reestima el modelo con los datos hasta t y se pronostica
     t + 1. Después se agrega el dato real y se repite. Son 12 pronósticos a
     un mes, que es como se usaría el modelo en la práctica.

Medidas de error (e = observado - pronóstico; menor es mejor):
   RMSE = raíz( promedio de e^2 )         castiga más los errores grandes
   MAE  = promedio de |e|                 error típico en pesos
   MAPE = 100 * promedio de |e / Z|       error típico en porcentaje
   U de Theil = RMSE del modelo / RMSE del paseo aleatorio
                U < 1: el modelo supera al paseo aleatorio; U > 1: no lo supera
Cobertura: % de datos reales que caen dentro del intervalo de predicción al 95%
           (si el modelo es correcto, debería ser cercano a 95%).
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
CONFIANZA = 0.95             # nivel de los intervalos de predicción
MODELOS = {"A": MODELO_A, "B": MODELO_B, "C": MODELO_C}   # C = paseo aleatorio (referencia)
REFERENCIA = "C"             # modelo contra el que se calcula la U de Theil
MODELO_GRAFICA = "B"         # modelo seleccionado en el Paso 7 (el que se grafica con su intervalo)
INICIO_GRAFICA = "2024-01-01"  # desde dónde se muestra la historia en las gráficas


def nombre(letra):
    o = MODELOS[letra]
    return f"{letra}: ARIMA({o[0]},{o[1]},{o[2]})"


def medidas(error, real):
    """RMSE, MAE y MAPE de un vector de errores."""
    error, real = np.asarray(error), np.asarray(real)
    return {"RMSE": np.sqrt(np.mean(error ** 2)), "MAE": np.mean(np.abs(error)),
            "MAPE (%)": 100 * np.mean(np.abs(error / real))}


# =====================================================================
# 1. DATOS
# =====================================================================
encabezado(f"PASO 8. VALIDACIÓN FUERA DE LA MUESTRA (intervalos al {CONFIANZA * 100:g}%)")
m = obtener_muestras(cargar_fix())
Z_ent = m["entrenamiento"]
Z_val = m["validacion"]
Z_todo = m["estim_final"]                      # ene-2015 a sep-2026
H = len(Z_val)
z = stats.norm.ppf(1 - (1 - CONFIANZA) / 2)
print(f"Entrenamiento: {periodo(Z_ent)} (N = {len(Z_ent)})")
print(f"Validación:    {periodo(Z_val)} (N = {H})")
print(f"Valor crítico normal: z = {z:.4f}")


# =====================================================================
# 2. VALIDACIÓN 1: PRONÓSTICO DESDE UN SOLO ORIGEN (sep-2025, h = 1..12)
# =====================================================================
encabezado(f"Validación 1. Pronóstico desde {mes_anio(Z_ent.index[-1])} para h = 1, ..., {H}")
pron_fijo = {}
filas_fijo = []
for letra, orden in MODELOS.items():
    res = estimar_arima(Z_ent, orden)
    p = pronostico_niveles(res, Z_ent, H)
    p.index = Z_val.index
    p["Observado"] = Z_val.values
    p["Error"] = p["Observado"] - p["Pronóstico"]
    p["Límite inferior"] = p["Pronóstico"] - z * p["Error estándar"]
    p["Límite superior"] = p["Pronóstico"] + z * p["Error estándar"]
    p["¿Dentro?"] = np.where((p["Observado"] >= p["Límite inferior"]) &
                             (p["Observado"] <= p["Límite superior"]), "Sí", "No")
    pron_fijo[letra] = p
    fila = {"Modelo": nombre(letra), **medidas(p["Error"], p["Observado"]),
            "Cobertura (%)": 100 * np.mean(p["¿Dentro?"] == "Sí")}
    filas_fijo.append(fila)

    print(f"\n{nombre(letra)}")
    tabla = pd.DataFrame({"Fecha": [mes_anio(f) for f in p.index], "h": range(1, H + 1),
                          "Observado": p["Observado"].round(4), "Pronóstico": p["Pronóstico"].round(4),
                          "Error": p["Error"].round(4),
                          "Lím. inf.": p["Límite inferior"].round(4),
                          "Lím. sup.": p["Límite superior"].round(4), "¿Dentro?": p["¿Dentro?"]})
    print(tabla.to_string(index=False))

cuadro_fijo = pd.DataFrame(filas_fijo)
rmse_ref = cuadro_fijo.loc[list(MODELOS).index(REFERENCIA), "RMSE"]
cuadro_fijo["U de Theil"] = cuadro_fijo["RMSE"] / rmse_ref
encabezado("Cuadro. Error de pronóstico desde un solo origen (h = 1 a 12)")
print(cuadro_fijo.round(4).to_string(index=False))


# =====================================================================
# 3. VALIDACIÓN 2: ORIGEN MÓVIL A UN MES
#    Origen t = sep-2025, ..., ago-2026: se estima con datos hasta t y se
#    pronostica t + 1 (oct-2025, ..., sep-2026).
# =====================================================================
encabezado("Validación 2. Origen móvil a un mes (se reestima el modelo en cada origen)")
pron_movil = {letra: [] for letra in MODELOS}
for fecha in Z_val.index:
    hist = Z_todo[Z_todo.index < fecha]                 # datos hasta el mes anterior
    for letra, orden in MODELOS.items():
        res = estimar_arima(hist, orden)                # se reestima con los datos disponibles
        p = pronostico_niveles(res, hist, 1).iloc[0]
        pron_movil[letra].append({"Fecha": fecha, "Origen": hist.index[-1],
                                  "Pronóstico": p["Pronóstico"], "Error estándar": p["Error estándar"]})

filas_movil = []
errores_abs = {}
for letra in MODELOS:
    p = pd.DataFrame(pron_movil[letra]).set_index("Fecha")
    p["Observado"] = Z_val.values
    p["Error"] = p["Observado"] - p["Pronóstico"]
    p["Límite inferior"] = p["Pronóstico"] - z * p["Error estándar"]
    p["Límite superior"] = p["Pronóstico"] + z * p["Error estándar"]
    p["¿Dentro?"] = np.where((p["Observado"] >= p["Límite inferior"]) &
                             (p["Observado"] <= p["Límite superior"]), "Sí", "No")
    pron_movil[letra] = p
    errores_abs[letra] = p["Error"].abs()
    filas_movil.append({"Modelo": nombre(letra), **medidas(p["Error"], p["Observado"]),
                        "Cobertura (%)": 100 * np.mean(p["¿Dentro?"] == "Sí")})

    print(f"\n{nombre(letra)}")
    tabla = pd.DataFrame({"Origen": [mes_anio(f) for f in p["Origen"]],
                          "Pronostica": [mes_anio(f) for f in p.index],
                          "Observado": p["Observado"].round(4), "Pronóstico": p["Pronóstico"].round(4),
                          "Error": p["Error"].round(4),
                          "Lím. inf.": p["Límite inferior"].round(4),
                          "Lím. sup.": p["Límite superior"].round(4), "¿Dentro?": p["¿Dentro?"]})
    print(tabla.to_string(index=False))

cuadro_movil = pd.DataFrame(filas_movil)
rmse_ref_m = cuadro_movil.loc[list(MODELOS).index(REFERENCIA), "RMSE"]
cuadro_movil["U de Theil"] = cuadro_movil["RMSE"] / rmse_ref_m
encabezado("Cuadro. Error de pronóstico con origen móvil a un mes")
print(cuadro_movil.round(4).to_string(index=False))


# =====================================================================
# 4. CONCLUSIÓN
# =====================================================================
encabezado("Conclusión")
for titulo, cuadro in [("Desde un solo origen", cuadro_fijo), ("Origen móvil a un mes", cuadro_movil)]:
    mejor = cuadro.sort_values("RMSE").iloc[0]
    print(f"{titulo}: menor RMSE = {mejor['Modelo']} (RMSE = {mejor['RMSE']:.4f}, "
          f"U = {mejor['U de Theil']:.4f})")
    for _, f in cuadro.iterrows():
        if f["Modelo"].startswith(REFERENCIA):
            continue
        veredicto = "supera" if f["U de Theil"] < 1 else "NO supera"
        print(f"   {f['Modelo']}: U = {f['U de Theil']:.4f} -> {veredicto} al paseo aleatorio")


# =====================================================================
# 5. GRÁFICAS
# =====================================================================
encabezado("Gráficas")
historia = Z_todo[INICIO_GRAFICA:]
colores = {"A": GOLD, "B": NAVY, "C": TEAL}

# Figura 34: pronósticos desde sep-2025 de los tres modelos contra lo observado
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
for letra, p in pron_fijo.items():
    ax.plot(p.index, p["Pronóstico"], color=colores[letra], lw=1.6, ls="--", marker="o", ms=3,
            label=f"Pronóstico {nombre(letra)}")
ax.axvline(Z_ent.index[-1], color=RED, lw=1, ls=":", label=f"Origen ({mes_anio(Z_ent.index[-1])})")
ax.set_title(f"Tipo de cambio FIX, promedio mensual, {periodo(historia)}: pronóstico desde "
             f"{mes_anio(Z_ent.index[-1])}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="upper right")
guardar_figura(fig, "fig34_validacion_un_origen.png")

# Figura 35: modelo seleccionado con su intervalo de predicción (un solo origen)
p = pron_fijo[MODELO_GRAFICA]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
ax.fill_between(p.index, p["Límite inferior"], p["Límite superior"], color=TEAL, alpha=0.25, lw=0,
                label=f"Intervalo de predicción al {CONFIANZA * 100:g}%")
ax.plot(p.index, p["Pronóstico"], color=NAVY, lw=1.8, ls="--", marker="o", ms=3,
        label=f"Pronóstico {nombre(MODELO_GRAFICA)}")
ax.axvline(Z_ent.index[-1], color=RED, lw=1, ls=":")
ax.set_title(f"Pronóstico del modelo {nombre(MODELO_GRAFICA)} desde {mes_anio(Z_ent.index[-1])} "
             f"e intervalo de predicción")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="upper right")
guardar_figura(fig, f"fig35_validacion_intervalo_{MODELO_GRAFICA}.png")

# Figura 36: origen móvil a un mes del modelo seleccionado
p = pron_movil[MODELO_GRAFICA]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
ax.errorbar(p.index, p["Pronóstico"], yerr=z * p["Error estándar"], fmt="o", color=NAVY,
            ecolor=TEAL, elinewidth=2, capsize=3, ms=4,
            label=f"Pronóstico a un mes e intervalo al {CONFIANZA * 100:g}%")
ax.set_title(f"Origen móvil a un mes, modelo {nombre(MODELO_GRAFICA)}, {periodo(Z_val)}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="upper right")
guardar_figura(fig, f"fig36_origen_movil_{MODELO_GRAFICA}.png")

# Figura 37: diagrama de caja de los errores absolutos a un mes
fig, ax = plt.subplots(figsize=(8, 4.8))
datos = [errores_abs[letra].values for letra in MODELOS]
caja = ax.boxplot(datos, patch_artist=True, widths=0.5,
                  medianprops={"color": RED, "lw": 2}, flierprops={"markerfacecolor": GOLD})
for parche, letra in zip(caja["boxes"], MODELOS):
    parche.set_facecolor(colores[letra])
    parche.set_alpha(0.6)
ax.set_xticks(range(1, len(MODELOS) + 1))
ax.set_xticklabels([nombre(l) + (" (referencia)" if l == REFERENCIA else "") for l in MODELOS])
ax.set_title(f"Errores absolutos de pronóstico a un mes, {periodo(Z_val)}")
ax.set_ylabel("|Observado - pronóstico| (pesos por dólar)")
guardar_figura(fig, "fig37_caja_errores.png")



print("Listo")