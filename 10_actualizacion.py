# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 11. ACTUALIZACIÓN DEL PRONÓSTICO CON EL DATO DE OCTUBRE DE 2026
 Criterio de la rúbrica: actualizar el pronóstico al conocer una nueva
 observación y pronosticar los tres periodos siguientes.
=======================================================================
En el Paso 9 se pronosticó desde T = sep-2026. Ahora se conoce Z_(T+1) = oct-2026.

Fórmula de actualización (apuntes, guía de series estacionales, Proposición 5.1),
SIN reestimar los parámetros:
   1. Error del pronóstico a un paso (la "sorpresa" de octubre):
          a_hat_(T+1) = Z_(T+1) - Z_hat_T(1)
   2. Pronóstico actualizado desde el nuevo origen T + 1:
          Z_hat_(T+1)(h) = Z_hat_T(h + 1) - psi_h * a_hat_(T+1)
      con los pesos psi de los apuntes (psi_0 = -1,
      psi_j = theta_j + Phi_1 psi_(j-1) + Phi_2 psi_(j-2), Phi(B) = (1 - phi_1 B)(1 - B)).
      Es decir: el pronóstico viejo para ese mes, corregido por la sorpresa de
      octubre multiplicada por su peso psi_h (como psi_h < 0, la corrección suma).
   3. Varianza del error desde el nuevo origen (no cambia la fórmula):
          Var[e_(T+1)(h)] = sigma^2_a (psi_0^2 + ... + psi_(h-1)^2)

Comprobación: con los mismos parámetros, pronosticar directamente desde T + 1
(con el dato de octubre) debe dar exactamente lo mismo.

Además se reestima el modelo incluyendo octubre (N = 142) para ver cuánto
cambian los parámetros y el pronóstico.
=======================================================================
"""

from configuracion import *
from openpyxl import load_workbook
from openpyxl.styles import Font

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
CONFIANZA = 0.95              # nivel de los intervalos de predicción
MODELO_FINAL = MODELO_B       # modelo seleccionado en el Paso 7
LETRA_FINAL = "B"
PASOS = 3                     # periodos que se pronostican después de la actualización
INICIO_GRAFICA = "2025-01-01"

p_, d_, q_ = MODELO_FINAL
NOMBRE = f"{LETRA_FINAL}: ARIMA({p_},{d_},{q_})"


def pesos_psi(phi, theta, d, n):
    """Pesos psi con la convención de los apuntes (guía de series estacionales, sec. 4):
       Phi(B) = (1 - phi_1 B - ...)(1 - B)^d = 1 - Phi_1 B - Phi_2 B^2 - ...
       psi_0 = -1  y  psi_j = theta_j + Phi_1 psi_(j-1) + ... + Phi_(p+d) psi_(j-p-d)
       (theta_j = 0 si j > q).  Regresa los psi y los Phi."""
    pol = np.r_[1.0, -np.array(phi, dtype=float)]
    for _ in range(d):
        pol = np.convolve(pol, [1.0, -1.0])
    Phi = -pol[1:]
    psi = [-1.0]
    for j in range(1, n):
        val = theta[j - 1] if j - 1 < len(theta) else 0.0
        val += sum(Phi[i - 1] * psi[j - i] for i in range(1, min(j, len(Phi)) + 1))
        psi.append(val)
    return np.array(psi), Phi


def parametros(res):
    phi = [res.params[f"ar.L{i}"] for i in range(1, p_ + 1)]
    theta = [-res.params[f"ma.L{i}"] for i in range(1, q_ + 1)]    
    return phi, theta, res.params["sigma2"]


# =====================================================================
# 1. DATOS
# =====================================================================
encabezado(f"PASO 11. ACTUALIZACIÓN DEL PRONÓSTICO, MODELO {NOMBRE} CON INTERVENCIÓN")
m = obtener_muestras(cargar_fix())
Z = m["estim_final"]                              # ene-2015 a sep-2026
nuevo = m["obs_fuera"]                            # oct-2026
Z_nuevo = pd.concat([Z, nuevo]).asfreq("MS")      # ene-2015 a oct-2026
T, T1 = Z.index[-1], nuevo.index[0]
z = stats.norm.ppf(1 - (1 - CONFIANZA) / 2)
print(f"Origen anterior: T = {mes_anio(T)}. Nuevo dato: {mes_anio(T1)} = {nuevo.iloc[0]:.4f}")


# =====================================================================
# 2. PRONÓSTICO ANTERIOR (desde T = sep-2026, parámetros del Paso 9)
# =====================================================================
res = estimar_arima(Z, MODELO_FINAL)
phi, theta, sigma2 = parametros(res)
pron_ant = pronostico_niveles(res, Z, PASOS + 1)          # Z_hat_T(1), ..., Z_hat_T(4)
psi, Phi = pesos_psi(phi, theta, d_, PASOS + 1)
encabezado(f"1. Pronóstico anterior desde {mes_anio(T)} (Paso 9)")
print(f"phi = {[round(v, 4) for v in phi]}, theta = {[round(v, 4) for v in theta]}, "
      f"sigma^2_a = {sigma2:.4f}")
for h in range(1, PASOS + 2):
    print(f"   Z_hat_T({h}) = {pron_ant['Pronóstico'].iloc[h - 1]:.4f}   "
          f"({mes_anio(pron_ant.index[h - 1])})")


# =====================================================================
# 3. ACTUALIZACIÓN CON LA FÓRMULA DE BOX-JENKINS (sin reestimar)
# =====================================================================
encabezado(f"2. Actualización con el dato de {mes_anio(T1)} (fórmula de Box-Jenkins)")
a_nuevo = nuevo.iloc[0] - pron_ant["Pronóstico"].iloc[0]
print(f"a_hat_(T+1) = Z_(T+1) - Z_hat_T(1) = {nuevo.iloc[0]:.4f} - {pron_ant['Pronóstico'].iloc[0]:.4f} "
      f"= {a_nuevo:.4f}")
print("Phi(B) = (1 - phi_1 B)(1 - B): " + ", ".join(f"Phi_{i + 1} = {v:.4f}" for i, v in enumerate(Phi)))
print("Pesos psi (psi_0 = -1, psi_j = theta_j + Phi_1 psi_(j-1) + Phi_2 psi_(j-2)): " +
      ", ".join(f"psi_{j} = {psi[j]:.4f}" for j in range(PASOS + 1)))
fechas = pd.date_range(T1 + pd.offsets.MonthBegin(1), periods=PASOS, freq="MS")
actualizado = []
for h in range(1, PASOS + 1):
    viejo = pron_ant["Pronóstico"].iloc[h]                 # Z_hat_T(h + 1)
    valor = viejo - psi[h] * a_nuevo
    actualizado.append(valor)
    print(f"   Z_hat_(T+1)({h}) = Z_hat_T({h + 1}) - psi_{h} a_hat_(T+1) = {viejo:.4f} - "
          f"({psi[h]:.4f})({a_nuevo:.4f}) = {valor:.4f}   ({mes_anio(fechas[h - 1])})")
var_act = sigma2 * np.cumsum(psi[:PASOS] ** 2)
ee_act = np.sqrt(var_act)
tabla_act = pd.DataFrame({"h": range(1, PASOS + 1),
                          "Pronóstico anterior": pron_ant["Pronóstico"].values[1:PASOS + 1],
                          "Pronóstico actualizado": actualizado,
                          "Cambio": np.array(actualizado) - pron_ant["Pronóstico"].values[1:PASOS + 1],
                          "Error estándar": ee_act,
                          "Límite inferior": np.array(actualizado) - z * ee_act,
                          "Límite superior": np.array(actualizado) + z * ee_act}, index=fechas)
print(f"\nCuadro. Pronóstico actualizado e intervalo de predicción al {CONFIANZA * 100:g}%")
t_imp = tabla_act.copy(); t_imp.index = [mes_anio(f) for f in t_imp.index]
print(t_imp.round(4).to_string())

# Comprobación: pronosticar directamente desde T + 1 con los mismos parámetros,
# con la recursión de Box-Jenkins (W = nabla Z sin los pulsos de 2020; a futuros = 0)
X = variables_intervencion(Z.index)
W = diferenciar(Z, 1)
if X is not None:
    W = W - diferenciar(X @ res.params[X.columns], 1)
w_hist = list(W.values) + [nuevo.iloc[0] - Z.iloc[-1]]
a_hist = list(res.resid.values) + [a_nuevo]
z_prev, directo = nuevo.iloc[0], []
for h in range(1, PASOS + 1):
    w_hat = sum(phi[i] * w_hist[-1 - i] for i in range(p_)) - \
            sum(theta[j] * a_hist[-1 - j] for j in range(q_))
    w_hist.append(w_hat)
    a_hist.append(0.0)
    z_prev = z_prev + w_hat
    directo.append(z_prev)
directo = np.array(directo)
print("\nComprobación (pronóstico directo desde oct-2026 con los mismos parámetros): " +
      ", ".join(f"{v:.4f}" for v in directo) +
      f"  -> diferencia máxima = {np.max(np.abs(directo - np.array(actualizado))):.6f}")


# =====================================================================
# 4. REESTIMACIÓN INCLUYENDO EL DATO DE OCTUBRE (N = 142)
# =====================================================================
encabezado(f"3. Reestimación del modelo con {periodo(Z_nuevo)} (N = {len(Z_nuevo)})")
res_re = estimar_arima(Z_nuevo, MODELO_FINAL)
phi_r, theta_r, sigma2_r = parametros(res_re)
for n_ in res_re.params.index:
    if n_ == "sigma2":
        continue
    sim = (n_.replace("escalon", "omega") if n_.startswith("escalon")
           else ("phi_" if n_.startswith("ar") else "theta_") + n_.split(".L")[1])
    signo = -1 if n_.startswith("ma") else 1
    print(f"  {sim:14s} = {signo * res_re.params[n_]:8.4f} (antes {signo * res.params[n_]:8.4f})   "
          f"ee = {res_re.bse[n_]:.4f}   valor p = {res_re.pvalues[n_]:.4f}")
print(f"  sigma^2_a = {sigma2_r:.4f} (antes {sigma2:.4f})")
pron_re = pronostico_niveles(res_re, Z_nuevo, PASOS)
tabla_re = pd.DataFrame({"h": range(1, PASOS + 1),
                         "Actualizado (fórmula)": actualizado,
                         "Reestimado": pron_re["Pronóstico"].values,
                         "ee reestimado": pron_re["Error estándar"].values,
                         "Lím. inf. reestimado": pron_re["Pronóstico"].values - z * pron_re["Error estándar"].values,
                         "Lím. sup. reestimado": pron_re["Pronóstico"].values + z * pron_re["Error estándar"].values},
                        index=[mes_anio(f) for f in fechas])
tabla_re["Diferencia"] = tabla_re["Reestimado"] - tabla_re["Actualizado (fórmula)"]
print(tabla_re.round(4).to_string())


# =====================================================================
# 5. GRÁFICA: PRONÓSTICO ANTERIOR CONTRA ACTUALIZADO
# =====================================================================
encabezado("Gráficas")
historia = Z_nuevo[INICIO_GRAFICA:]
pron_ant_10 = pronostico_niveles(res, Z, PASOS + 1)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
ax.fill_between(pron_ant_10.index, pron_ant_10["Pronóstico"] - z * pron_ant_10["Error estándar"],
                pron_ant_10["Pronóstico"] + z * pron_ant_10["Error estándar"], color=GOLD, alpha=0.2,
                lw=0, label=f"Intervalo al {CONFIANZA * 100:g}% desde {mes_anio(T)}")
ax.plot(np.r_[[T], pron_ant_10.index], np.r_[[Z.iloc[-1]], pron_ant_10["Pronóstico"].values],
        color=GOLD, lw=1.6, ls="--", marker="o", ms=3, label=f"Pronóstico desde {mes_anio(T)}")
ax.fill_between(fechas, tabla_act["Límite inferior"], tabla_act["Límite superior"], color=TEAL,
                alpha=0.3, lw=0, label=f"Intervalo al {CONFIANZA * 100:g}% actualizado")
ax.plot(np.r_[[T1], fechas], np.r_[[nuevo.iloc[0]], tabla_act["Pronóstico actualizado"].values],
        color=NAVY, lw=1.8, ls="--", marker="o", ms=3, label=f"Pronóstico actualizado desde {mes_anio(T1)}")
ax.scatter([T1], [nuevo.iloc[0]], color=RED, s=55, zorder=5,
           label=f"Observado {mes_anio(T1)} = {nuevo.iloc[0]:.4f}")
ax.set_title(f"Tipo de cambio FIX, promedio mensual: actualización del pronóstico con el dato de "
             f"{mes_anio(T1)}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="lower left", fontsize=8)
guardar_figura(fig, f"fig40_actualizacion_{LETRA_FINAL}.png")


print("Listo")