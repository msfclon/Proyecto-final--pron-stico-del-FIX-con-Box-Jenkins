# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 6. MODELOS PROPUESTOS Y ESTIMACIÓN DE PARÁMETROS
 Criterios de la rúbrica:
   - Proponer al menos tres modelos, dos de ellos sin selección automática.
   - Estimar los parámetros con software estadístico.
=======================================================================
Modelos (definidos en configuracion.py):
   Modelo A: sugerido por la FAC  (se corta en q)   -> sin selección automática
   Modelo B: sugerido por la FACP (se corta en p)   -> sin selección automática
   Modelo C: selección automática: se estiman todos los ARIMA(p,d,q) con
             p, q <= 3 y se elige el de menor criterio de información entre los
             ADMISIBLES (raíces AR y MA fuera del círculo unitario, supuesto 6).

Estimación: máxima verosimilitud exacta, sin constante, con la función estimar_arima
de configuracion.py: se estima el ARMA(p,q) de nabla Z incluyendo la intervención del
Paso 4b (pulsos de marzo y abril de 2020, coeficientes omega).

NOTACIÓN (importante):
   La guía escribe el modelo como
       (1 - phi_1 B - ... - phi_p B^p) nabla Z_t = (1 - theta_1 B - ... - theta_q B^q) a_t
   statsmodels escribe la parte MA con signo "+": (1 + theta_1 B + ...).
   Por eso este script CAMBIA EL SIGNO de los theta para reportarlos como en la guía.

Pruebas de significancia de cada parámetro:
   H0: parámetro = 0   vs   H1: parámetro distinto de 0
   t = estimación / error estándar; se rechaza H0 si valor p < alfa.
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
ALFA = 0.05                  # nivel de significancia de los parámetros
MAX_P = 3                    # máximo orden AR en la búsqueda automática
MAX_Q = 3                    # máximo orden MA en la búsqueda automática
CRITERIO = "AICc"            # criterio de la búsqueda automática: "AIC", "AICc" o "BIC"
MARGEN_CIRCULO = 1.05        # raíz AR o MA con módulo menor: sobre el círculo unitario (no admisible)

# Estimaciones preliminares del Paso 5 (método de momentos), en notación de la guía
MOMENTOS = {"A": {"theta_1": -0.1860},
            "B": {"phi_1": 0.1798}}


# =====================================================================
# 1. DATOS (muestra de entrenamiento, en niveles: el modelo diferencia con d)
# =====================================================================
encabezado("PASO 6. MODELOS PROPUESTOS Y ESTIMACIÓN DE PARÁMETROS")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {len(Z)}")


def estimar(orden):
    """Estima un ARIMA(p,d,q) sin constante por máxima verosimilitud, con la intervención.
       Regresa el resultado y si el optimizador convergió."""
    res = estimar_arima(Z, orden)
    convergio = bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True
    return res, convergio


def nombre(orden):
    return f"ARIMA({orden[0]},{orden[1]},{orden[2]})"


def modulo_minimo(res):
    """Módulo de la raíz AR o MA más cercana al círculo unitario (infinito si no hay)."""
    ar = [res.params[n] for n in res.params.index if n.startswith("ar.")]
    ma = [res.params[n] for n in res.params.index if n.startswith("ma.")]
    raices = []
    if ar:
        raices += list(np.roots(np.r_[[-c for c in ar[::-1]], 1.0]))    # 1 - phi_1 B - ...
    if ma:
        raices += list(np.roots(np.r_[ma[::-1], 1.0]))                  # 1 + theta_1 B + ... (statsmodels)
    return min(np.abs(raices)) if raices else np.inf


# =====================================================================
# 2. SELECCIÓN AUTOMÁTICA (modelo C): todos los ARIMA(p,d,q) con p, q <= 3
# =====================================================================
encabezado(f"Búsqueda automática: ARIMA(p,{ORDEN_D},q) con p <= {MAX_P}, q <= {MAX_Q}")

filas = []
for p in range(MAX_P + 1):
    for q in range(MAX_Q + 1):
        res, convergio = estimar((p, ORDEN_D, q))
        filas.append({"Modelo": nombre((p, ORDEN_D, q)), "p": p, "q": q,
                      "k": len(res.params), "Log-verosimilitud": round(res.llf, 3),
                      "AIC": round(res.aic, 3), "AICc": round(res.aicc, 3),
                      "BIC": round(res.bic, 3), "¿Convergió?": "Sí" if convergio else "No",
                      "Módulo mínimo de raíces": round(modulo_minimo(res), 4),
                      "¿Admisible?": "Sí" if modulo_minimo(res) >= MARGEN_CIRCULO else "No"})
busqueda = pd.DataFrame(filas).sort_values(CRITERIO)
print(busqueda.head(8).to_string(index=False))

# El automático es el de menor criterio entre los que convergieron y son admisibles
no_adm = busqueda[busqueda["¿Admisible?"] == "No"]["Modelo"].tolist()
if no_adm:
    print(f"\nNo admisibles (raíz con módulo < {MARGEN_CIRCULO}, casi sobre el círculo unitario): "
          f"{', '.join(no_adm)}")
validos = busqueda[(busqueda["¿Convergió?"] == "Sí") & (busqueda["¿Admisible?"] == "Sí")]
mejor = validos.iloc[0]
orden_auto = (int(mejor["p"]), ORDEN_D, int(mejor["q"]))
print(f"\nMenor {CRITERIO} entre los admisibles: {nombre(orden_auto)}")
print(f"Para comparar: el menor BIC es "
      f"{busqueda.sort_values('BIC').iloc[0]['Modelo']}")
if orden_auto in (MODELO_A, MODELO_B):
    print("La búsqueda automática eligió uno de los modelos ya propuestos; "
          "se toma el segundo mejor como modelo C.")
    otros = validos[~validos.apply(lambda f: (int(f["p"]), ORDEN_D, int(f["q"]))
                                   in (MODELO_A, MODELO_B), axis=1)]
    orden_auto = (int(otros.iloc[0]["p"]), ORDEN_D, int(otros.iloc[0]["q"]))
if orden_auto != MODELO_C:
    print(f"ATENCIÓN: en configuracion.py está MODELO_C = {MODELO_C}; cámbialo a {orden_auto}.")


# =====================================================================
# 3. ESTIMACIÓN DE LOS TRES MODELOS
# =====================================================================
modelos = {"A": MODELO_A, "B": MODELO_B, "C": orden_auto}
origen = {"A": "FAC (se corta en q)", "B": "FACP (se corta en p)",
          "C": f"Selección automática (mínimo {CRITERIO})"}

tabla_param = []
tabla_comp = []
for letra, orden in modelos.items():
    res, convergio = estimar(orden)

    encabezado(f"Modelo {letra}: {nombre(orden)}  [{origen[letra]}]")
    for param in res.params.index:
        if param == "sigma2":
            continue
        if param.startswith("escalon"):                 # intervención: omega del escalón
            signo, simbolo = 1, param.replace("escalon", "omega")
        else:
            tipo, retraso = param.split(".L")          # "ar.L1" -> ("ar", "1")
            signo = -1 if tipo == "ma" else 1          # statsmodels
            simbolo = f"{'phi' if tipo == 'ar' else 'theta'}_{retraso}"
        est = signo * res.params[param]
        t = signo * res.tvalues[param]
        pval = res.pvalues[param]
        fila = {"Modelo": f"{letra}: {nombre(orden)}", "Parámetro": simbolo,
                "Estimación": round(est, 4), "Error est.": round(res.bse[param], 4),
                "t": round(t, 4), "Valor p": round(pval, 4),
                f"¿Significativo ({ALFA * 100:g}%)?": "Sí" if pval < ALFA else "No",
                "Momentos (Paso 5)": MOMENTOS.get(letra, {}).get(simbolo, "")}
        tabla_param.append(fila)
        print(f"  {simbolo:8s} = {est:8.4f}   ee = {res.bse[param]:.4f}   t = {t:7.3f}   "
              f"valor p = {pval:.4f}   {'significativo' if pval < ALFA else 'NO significativo'}")
    sigma2 = res.params["sigma2"]
    print(f"  sigma^2_a = {sigma2:.4f}  (varianza del ruido blanco)")
    print(f"  Log-verosimilitud = {res.llf:.3f}, AIC = {res.aic:.3f}, "
          f"AICc = {res.aicc:.3f}, BIC = {res.bic:.3f}")
    if not convergio:
        print("  ATENCIÓN: el optimizador no convergió para este modelo.")

    tabla_comp.append({"Modelo": f"{letra}: {nombre(orden)}", "Origen": origen[letra],
                       "Parámetros (k)": len(res.params),
                       "sigma^2_a": round(sigma2, 4), "Log-verosimilitud": round(res.llf, 3),
                       "AIC": round(res.aic, 3), "AICc": round(res.aicc, 3),
                       "BIC": round(res.bic, 3)})

    # Ecuación:
    #   (1 - phi_1 B - ...) nabla Z_t = (1 - theta_1 B - ...) a_t
    def termino(coef, i):
        potencia = "B" if i == 1 else f"B^{i}"
        return f" {'+' if coef >= 0 else '-'} {abs(coef):.4f}{potencia}"
    ar = "".join(termino(-res.params[f"ar.L{i}"], i) for i in range(1, orden[0] + 1))
    ma = "".join(termino(res.params[f"ma.L{i}"], i) for i in range(1, orden[2] + 1))
    izq = f"(1{ar}) " if ar else ""
    der = f"(1{ma}) " if ma else ""
    pulsos = [f"{res.params[n]:.4f} P[{n.replace('escalon_', '')}]"
              for n in res.params.index if n.startswith("escalon")]
    if pulsos and ar:
        print(f"  Ecuación: {izq}(∇Z_t - {' - '.join(pulsos)}) = {der}a_t")
    else:
        print(f"  Ecuación: {izq}∇Z_t = {''.join(x + ' + ' for x in pulsos)}{der}a_t")
    print("     (P[mes] = pulso: 1 en ese mes y 0 en los demás)" if pulsos else "")

cuadro_param = pd.DataFrame(tabla_param)
cuadro_comp = pd.DataFrame(tabla_comp)

encabezado("Cuadro. Comparación de los tres modelos (menor es mejor)")
print(cuadro_comp.drop(columns="Origen").to_string(index=False))


print("Listo")