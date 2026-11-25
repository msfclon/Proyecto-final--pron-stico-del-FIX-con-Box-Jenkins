# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 4. OPERADOR DIFERENCIA Y ORDEN DE DIFERENCIACIÓN d
 Criterio de la rúbrica: usar el operador diferencia para volver estacionaria
 la serie, justificando el orden d en cada caso y el orden final.
=======================================================================
Operador diferencia:   (nabla Z)_t   = Z_t - Z_{t-1}
                       (nabla^2 Z)_t = (nabla Z)_t - (nabla Z)_{t-1}

Qué hace (con la muestra de entrenamiento):
  1. Diferencia la serie d = 0, 1, 2 veces.
  2. Aplica Dickey-Fuller aumentada a cada una:
         H0: raíz unitaria   vs   H1: estacionaria
     El orden elegido es el MENOR d con el que se rechaza H0.
  3. Revisa si diferenciar una vez más sería excesivo (sobrediferenciación):
     señales -> la desviación estándar SUBE y r_1 se vuelve negativo (tiende a -0.5).
  4. Prueba si la media de la serie diferenciada es cero:
         H0: mu = 0   vs   H1: mu distinto de 0
     Decide si el modelo ARIMA lleva constante (deriva) o no.
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
ALFA = 0.01                  # nivel de significancia de Dickey-Fuller (estacionariedad: 0.01)
ALFA_MEDIA = 0.05            # nivel de significancia de la prueba de media cero
MAX_D = 2                    # máximo número de diferencias que se revisan
ESPECIFICACION_ADF = "c"     # "n" sin constante | "c" con constante | "ct" con tendencia
NIVEL = f"{ALFA * 100:g}%"


# =====================================================================
# 1. DATOS (muestra de entrenamiento)
# =====================================================================
encabezado(f"PASO 4. ORDEN DE DIFERENCIACIÓN (alfa = {ALFA})")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {len(Z)}")

# Series diferenciadas: d = 0 es la original; cada d aplica el operador una vez más
series = {0: Z}
for d in range(1, MAX_D + 1):
    series[d] = series[d - 1].diff().dropna()
NOMBRE = {0: "Z (original)", 1: "∇Z (1 diferencia)", 2: "∇²Z (2 diferencias)",
          3: "∇³Z (3 diferencias)"}


# =====================================================================
# 2. DICKEY-FULLER A CADA SERIE
# =====================================================================
tablas_adf = []
resumen = []
for d, s in series.items():
    t_adf = prueba_adf(s, ALFA)
    t_adf.insert(0, "Serie", NOMBRE[d])
    tablas_adf.append(t_adf)

    fila = t_adf[t_adf["Clave"] == ESPECIFICACION_ADF].iloc[0]
    r = fac(s, 2)
    resumen.append({"d": d, "Serie": NOMBRE[d], "N": len(s),
                    "Media": round(s.mean(), 4), "Desv. est.": round(s.std(), 4),
                    "r_1": round(r[0], 4), "r_2": round(r[1], 4),
                    "tau": fila["tau"], "Valor p": fila["Valor p"],
                    f"Decisión ADF ({NIVEL})": fila[f"Decisión ({NIVEL})"]})

cuadro_adf = pd.concat(tablas_adf, ignore_index=True)
cuadro_resumen = pd.DataFrame(resumen)

encabezado(f"Cuadro. Dickey-Fuller aumentada por orden de diferenciación (alfa = {ALFA})")
print(cuadro_adf.drop(columns="Clave").to_string(index=False))

encabezado(f"Cuadro. Resumen por orden d (especificación: "
           f"{ESPECIFICACIONES_ADF[ESPECIFICACION_ADF]})")
print(cuadro_resumen.to_string(index=False))


# =====================================================================
# 3. ORDEN ELEGIDO: el menor d con el que se rechaza H0
# =====================================================================
estacionarias = cuadro_resumen[cuadro_resumen[f"Decisión ADF ({NIVEL})"] == "Se rechaza H0"]
d_elegido = int(estacionarias["d"].min()) if len(estacionarias) else None

encabezado("Orden de diferenciación")
for _, f in cuadro_resumen.iterrows():
    print(f"  d = {f['d']}: tau = {f['tau']}, valor p = {f['Valor p']} -> {f[f'Decisión ADF ({NIVEL})']}")
print(f"\nOrden elegido: d = {d_elegido}")
if d_elegido != ORDEN_D:
    print(f"ATENCIÓN: en configuracion.py está ORDEN_D = {ORDEN_D}; cámbialo a {d_elegido}.")


# =====================================================================
# 4. ¿DIFERENCIAR UNA VEZ MÁS SERÍA EXCESIVO? (sobrediferenciación)
# =====================================================================
encabezado("Revisión de sobrediferenciación")
if d_elegido is not None and d_elegido + 1 in series:
    actual = cuadro_resumen.iloc[d_elegido]
    siguiente = cuadro_resumen.iloc[d_elegido + 1]
    sube = siguiente["Desv. est."] > actual["Desv. est."]
    print(f"  Con d = {d_elegido}: desv. est. = {actual['Desv. est.']}, r_1 = {actual['r_1']}")
    print(f"  Con d = {d_elegido + 1}: desv. est. = {siguiente['Desv. est.']}, r_1 = {siguiente['r_1']}")
    print(f"  ¿La desviación estándar sube con d = {d_elegido + 1}? {'Sí' if sube else 'No'}")
    if sube and siguiente["r_1"] < 0:
        print(f"  Diferenciar {d_elegido + 1} veces sería excesivo: se confirma d = {d_elegido}.")


# =====================================================================
# 5. ¿LA MEDIA DE LA SERIE DIFERENCIADA ES CERO?
#    t = media / (s / raíz(n)), con n - 1 grados de libertad
# =====================================================================
W = series[d_elegido]
n = len(W)
t_media = W.mean() / (W.std() / np.sqrt(n))
p_media = 2 * (1 - stats.t.cdf(abs(t_media), df=n - 1))

encabezado(f"Prueba de media cero de la serie diferenciada (alfa = {ALFA_MEDIA})")
print(f"  Media = {W.mean():.4f}, desv. est. = {W.std():.4f}, n = {n}")
print(f"  t = {t_media:.4f}, valor p = {p_media:.4f}")
if p_media > ALFA_MEDIA:
    print("  No se rechaza H0: mu = 0. El modelo ARIMA NO lleva constante (sin deriva).")
else:
    print("  Se rechaza H0: el modelo ARIMA SÍ lleva constante (deriva).")


# =====================================================================
# 6. GRÁFICAS
# =====================================================================
encabezado("Gráficas")
for d, archivo in [(1, "fig07_primera_diferencia.png"), (2, "fig08_segunda_diferencia.png")]:
    if d not in series:
        continue
    s = series[d]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, s.values, color=NAVY, lw=1.2)
    ax.axhline(0, color=CHARCOAL, lw=0.8)
    ax.axhline(s.mean(), color=GOLD, lw=1.2, ls="--", label=f"Media = {s.mean():.3f}")
    nombre = "Primera" if d == 1 else "Segunda"
    ax.set_title(f"{nombre} diferencia del tipo de cambio FIX, {periodo(s)}")
    ax.set_xlabel("Fecha (mensual)")
    ax.set_ylabel("Pesos por dólar")
    formato_fechas(ax, cada_anios=1)
    ax.legend(loc="upper left")
    guardar_figura(fig, archivo)


print("Listo")