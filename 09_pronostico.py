# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 9. PRONÓSTICO E INTERVALOS DE PREDICCIÓN
 (incluye el Paso 10: comparación del pronóstico propio contra el de software)
 Criterios de la rúbrica:
   - Pronóstico óptimo e intervalos de predicción de la serie original:
     3 periodos sin software (a mano) y 10 periodos con software.
   - Comparar el pronóstico propio contra el del software.
=======================================================================
Modelo final: el seleccionado en el Paso 7, reestimado con TODA la muestra
de estimación (ene-2015 a sep-2026, N = 141). Origen del pronóstico: T = sep-2026.

Pronóstico "a mano" (fórmulas y notación de los apuntes):
   Modelo:  Phi(B) Z_t = Theta(B) a_t,  con  Phi(B) = (1 - phi_1 B)(1 - B) = 1 - Phi_1 B - Phi_2 B^2
            Phi_1 = 1 + phi_1,  Phi_2 = -phi_1,  Theta(B) = 1 - theta_1 B
            (los pulsos de 2020 ya no actúan en el futuro)
   Ecuación en diferencias:  Z_t = Phi_1 Z_(t-1) + Phi_2 Z_(t-2) + a_t - theta_1 a_(t-1)
   Pronóstico óptimo (esperanza condicional): lo pasado se conserva, lo futuro se sustituye
   por su pronóstico y los a futuros valen 0:
       Z_hat_T(1) = Phi_1 Z_T + Phi_2 Z_(T-1) - theta_1 a_T
       Z_hat_T(2) = Phi_1 Z_hat_T(1) + Phi_2 Z_T
       Z_hat_T(3) = Phi_1 Z_hat_T(2) + Phi_2 Z_hat_T(1)
   Pesos psi (convención de los apuntes, psi_0 = -1):
       psi_j = theta_j + Phi_1 psi_(j-1) + Phi_2 psi_(j-2)
   Varianza del error de pronóstico:  Var[e_T(h)] = sigma^2_a (psi_0^2 + ... + psi_(h-1)^2)
   Intervalo de predicción al (1 - alfa):  Z_hat_T(h) +- z_(alfa/2) * sigma_a * raíz(suma psi_j^2)

Pronóstico "con software": statsmodels estima el ARIMA directamente sobre Z (en niveles,
con los escalones de la intervención) y calcula el pronóstico con su función get_forecast.

Contraste de la pregunta de investigación: ¿el dato real de oct-2026 cae dentro del
intervalo de predicción al 95% de Z_hat_T(1)?
=======================================================================
"""

from configuracion import *
from statsmodels.tsa.arima.model import ARIMA
from openpyxl import load_workbook
from openpyxl.styles import Font

# =====================================================================
# AJUSTES DE ESTE PASO
# =====================================================================
CONFIANZA = 0.95                    # nivel del intervalo de predicción principal
NIVELES_ABANICO = [0.50, 0.80, 0.95]  # intervalos de la gráfica de abanico
MODELO_FINAL = MODELO_B             # modelo seleccionado en el Paso 7
LETRA_FINAL = "B"
PASOS_MANO = 3                      # periodos pronosticados a mano
PASOS_SOFTWARE = 10                 # periodos pronosticados con software
INICIO_GRAFICA = "2025-01-01"       # desde dónde se muestra la historia en las gráficas

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




# =====================================================================
# 1. DATOS
# =====================================================================
encabezado(f"PASO 9. PRONÓSTICO CON EL MODELO {NOMBRE} CON INTERVENCIÓN")
m = obtener_muestras(cargar_fix())
Z = m["estim_final"]                          # ene-2015 a sep-2026
obs_fuera = m["obs_fuera"]                    # oct-2026
T = Z.index[-1]
z = stats.norm.ppf(1 - (1 - CONFIANZA) / 2)
print(f"Muestra de estimación final: {periodo(Z)}, N = {len(Z)}")
print(f"Origen del pronóstico: T = {mes_anio(T)}, Z_T = {Z.iloc[-1]:.4f}; "
      f"Z_(T-1) = {Z.iloc[-2]:.4f} ({mes_anio(Z.index[-2])})")
print(f"Observación fuera de la muestra: {mes_anio(obs_fuera.index[0])} = {obs_fuera.iloc[0]:.4f}")


# =====================================================================
# 2. REESTIMACIÓN DEL MODELO FINAL CON TODA LA MUESTRA
# =====================================================================
encabezado("1. Reestimación del modelo con ene-2015 a sep-2026")
res = estimar_arima(Z, MODELO_FINAL)
phi = [res.params[f"ar.L{i}"] for i in range(1, p_ + 1)]
theta = [-res.params[f"ma.L{i}"] for i in range(1, q_ + 1)]
sigma2 = res.params["sigma2"]
for n_ in res.params.index:
    if n_ == "sigma2":
        continue
    sim = (n_.replace("escalon", "omega") if n_.startswith("escalon")
           else ("phi_" if n_.startswith("ar") else "theta_") + n_.split(".L")[1])
    signo = -1 if n_.startswith("ma") else 1
    print(f"  {sim:14s} = {signo * res.params[n_]:8.4f}   ee = {res.bse[n_]:.4f}   "
          f"t = {signo * res.tvalues[n_]:7.3f}   valor p = {res.pvalues[n_]:.4f}")
print(f"  sigma^2_a = {sigma2:.4f}   sigma_a = {np.sqrt(sigma2):.4f}")

# Residuos y primera diferencia sin los pulsos (para la recursión a mano)
X = variables_intervencion(Z.index)
W = diferenciar(Z, 1)
if X is not None:
    W = W - diferenciar(X @ res.params[X.columns], 1)
a = res.resid


# =====================================================================
# 3. PRONÓSTICO A MANO (fórmulas de Box-Jenkins)
# =====================================================================
encabezado(f"2. Pronóstico a mano: {PASOS_MANO} periodos")

# Pronóstico de la diferencia: W_hat = sum phi_i W_(t-i) - sum theta_j a_(t-j), a futuros = 0
w_hist = list(W.values)
a_hist = list(a.values)
z_mano = []
z_prev = Z.iloc[-1]
for h in range(1, PASOS_MANO + 1):
    w_hat = sum(phi[i] * w_hist[-1 - i] for i in range(p_)) - \
            sum(theta[j] * a_hist[-1 - j] for j in range(q_))
    w_hist.append(w_hat)
    a_hist.append(0.0)
    z_prev = z_prev + w_hat
    z_mano.append(z_prev)

# Pesos psi (convención de los apuntes, psi_0 = -1)
psi, Phi = pesos_psi(phi, theta, d_, max(PASOS_MANO, PASOS_SOFTWARE))
var_mano = sigma2 * np.cumsum(psi[:PASOS_MANO] ** 2)
ee_mano = np.sqrt(var_mano)

fechas_mano = pd.date_range(T + pd.offsets.MonthBegin(1), periods=PASOS_MANO, freq="MS")
tabla_mano = pd.DataFrame({"h": range(1, PASOS_MANO + 1),
                           "Pronóstico": z_mano, "psi_(h-1)": psi[:PASOS_MANO],
                           "Var e(h)": var_mano, "Error estándar": ee_mano,
                           "Límite inferior": np.array(z_mano) - z * ee_mano,
                           "Límite superior": np.array(z_mano) + z * ee_mano},
                          index=fechas_mano)

# Cálculos con números (para el reporte), en el caso ARIMA(p,1,q) con p, q <= 1
if p_ <= 1 and q_ <= 1:
    f1 = phi[0] if p_ else 0.0
    t1 = theta[0] if q_ else 0.0
    P1, P2 = 1 + f1, -f1
    ZT, ZT1, aT = Z.iloc[-1], Z.iloc[-2], a.iloc[-1]
    print(f"  phi_1 = {f1:.4f}, theta_1 = {t1:.4f}, sigma^2_a = {sigma2:.4f}, a_T = {aT:.4f}")
    print(f"  Phi(B) = (1 - phi_1 B)(1 - B):  Phi_1 = 1 + phi_1 = {P1:.4f},  Phi_2 = -phi_1 = {P2:.4f}")
    print(f"  Z_hat_T(1) = Phi_1 Z_T + Phi_2 Z_(T-1) - theta_1 a_T")
    print(f"             = {P1:.4f} ({ZT:.4f}) + ({P2:.4f}) ({ZT1:.4f}) - {t1:.4f} ({aT:.4f}) = {z_mano[0]:.4f}")
    print(f"  Z_hat_T(2) = Phi_1 Z_hat_T(1) + Phi_2 Z_T")
    print(f"             = {P1:.4f} ({z_mano[0]:.4f}) + ({P2:.4f}) ({ZT:.4f}) = {z_mano[1]:.4f}")
    print(f"  Z_hat_T(3) = Phi_1 Z_hat_T(2) + Phi_2 Z_hat_T(1)")
    print(f"             = {P1:.4f} ({z_mano[1]:.4f}) + ({P2:.4f}) ({z_mano[0]:.4f}) = {z_mano[2]:.4f}")
    print(f"  psi_0 = -1")
    print(f"  psi_1 = theta_1 + Phi_1 psi_0 = {t1:.4f} + {P1:.4f} (-1) = {psi[1]:.4f}")
    print(f"  psi_2 = Phi_1 psi_1 + Phi_2 psi_0 = {P1:.4f} ({psi[1]:.4f}) + ({P2:.4f}) (-1) = {psi[2]:.4f}")
    print(f"  Var e(1) = sigma^2 psi_0^2                        = {var_mano[0]:.4f}")
    print(f"  Var e(2) = sigma^2 (psi_0^2 + psi_1^2)            = {var_mano[1]:.4f}")
    print(f"  Var e(3) = sigma^2 (psi_0^2 + psi_1^2 + psi_2^2)  = {var_mano[2]:.4f}")
print(f"\nCuadro. Pronóstico a mano e intervalo de predicción al {CONFIANZA * 100:g}% (z = {z:.4f})")
t_imp = tabla_mano.copy()
t_imp.index = [mes_anio(f) for f in t_imp.index]
print(t_imp.round(4).to_string())


# =====================================================================
# 4. PRONÓSTICO CON SOFTWARE (statsmodels, modelo en niveles)
# =====================================================================
encabezado(f"3. Pronóstico con software (statsmodels): {PASOS_SOFTWARE} periodos")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_sw = ARIMA(Z, exog=X, order=MODELO_FINAL, trend="n").fit()
exog_futuro = None if X is None else np.ones((PASOS_SOFTWARE, X.shape[1]))   # escalones siguen en 1
pron_sw = res_sw.get_forecast(steps=PASOS_SOFTWARE, exog=exog_futuro)
tabla_sw = pd.DataFrame({"h": range(1, PASOS_SOFTWARE + 1),
                         "Pronóstico": pron_sw.predicted_mean.values,
                         "Error estándar": pron_sw.se_mean.values})
tabla_sw.index = pd.date_range(T + pd.offsets.MonthBegin(1), periods=PASOS_SOFTWARE, freq="MS")
for nivel in sorted(set(NIVELES_ABANICO + [CONFIANZA])):
    zz = stats.norm.ppf(1 - (1 - nivel) / 2)
    tabla_sw[f"Lím. inf. {nivel * 100:g}%"] = tabla_sw["Pronóstico"] - zz * tabla_sw["Error estándar"]
    tabla_sw[f"Lím. sup. {nivel * 100:g}%"] = tabla_sw["Pronóstico"] + zz * tabla_sw["Error estándar"]
print("Parámetros de statsmodels (modelo en niveles): " +
      ", ".join(f"{n_} = {v:.4f}" for n_, v in res_sw.params.items()))
col_ci = [f"Lím. inf. {CONFIANZA * 100:g}%", f"Lím. sup. {CONFIANZA * 100:g}%"]
t_imp = tabla_sw[["h", "Pronóstico", "Error estándar"] + col_ci].copy()
t_imp.index = [mes_anio(f) for f in t_imp.index]
print(t_imp.round(4).to_string())


# =====================================================================
# 5. PASO 10: COMPARACIÓN DEL PRONÓSTICO A MANO CONTRA EL DE SOFTWARE
# =====================================================================
encabezado("4. Comparación: pronóstico a mano contra software")
comp = pd.DataFrame({"h": range(1, PASOS_MANO + 1),
                     "A mano": tabla_mano["Pronóstico"].values,
                     "Software": tabla_sw["Pronóstico"].values[:PASOS_MANO],
                     "ee a mano": tabla_mano["Error estándar"].values,
                     "ee software": tabla_sw["Error estándar"].values[:PASOS_MANO]},
                    index=[mes_anio(f) for f in fechas_mano])
comp["Diferencia pronóstico"] = comp["A mano"] - comp["Software"]
comp["Diferencia ee"] = comp["ee a mano"] - comp["ee software"]
print(comp.round(4).to_string())
print(f"Máxima diferencia absoluta en el pronóstico: {comp['Diferencia pronóstico'].abs().max():.6f}")


# =====================================================================
# 6. CONTRASTE: ¿EL DATO REAL DE OCT-2026 CAE DENTRO DEL INTERVALO?
# =====================================================================
encabezado(f"5. Pregunta de investigación: {mes_anio(obs_fuera.index[0])}")
real = obs_fuera.iloc[0]
pron1 = tabla_mano["Pronóstico"].iloc[0]
ee1 = tabla_mano["Error estándar"].iloc[0]
li, ls = pron1 - z * ee1, pron1 + z * ee1
z_obs = (real - pron1) / ee1
p_dos = 2 * (1 - stats.norm.cdf(abs(z_obs)))
p_mayor = 1 - stats.norm.cdf(z_obs)
dentro = li <= real <= ls
print(f"Pronóstico Z_hat_T(1) = {pron1:.4f}, error estándar = {ee1:.4f}")
print(f"Intervalo de predicción al {CONFIANZA * 100:g}%: [{li:.4f}, {ls:.4f}]")
print(f"Dato real: {real:.4f}  -> {'DENTRO' if dentro else 'FUERA'} del intervalo")
print(f"Error de pronóstico = {real - pron1:.4f} pesos ({100 * (real - pron1) / pron1:.2f}%)")
print(f"Error estandarizado: z = {z_obs:.3f}")
print(f"Probabilidad, según el modelo, de un valor tan alejado o más (dos colas): {p_dos:.4f}")
print(f"Probabilidad, según el modelo, de un valor igual o mayor: {p_mayor:.4f}")
print(f"El dato quedaría dentro de un intervalo de predicción de al menos {100 * (1 - p_dos):.2f}%")
contraste = pd.DataFrame({"Concepto": ["Pronóstico", "Error estándar", f"Lím. inf. {CONFIANZA * 100:g}%",
                                       f"Lím. sup. {CONFIANZA * 100:g}%", "Dato real", "¿Dentro?",
                                       "z del dato", "Valor p (dos colas)", "P(valor >= dato)"],
                          "Valor": [round(pron1, 4), round(ee1, 4), round(li, 4), round(ls, 4),
                                    round(real, 4), "Sí" if dentro else "No", round(z_obs, 4),
                                    round(p_dos, 4), round(p_mayor, 4)]})


# =====================================================================
# 7. GRÁFICAS
# =====================================================================
encabezado("Gráficas")
historia = Z[INICIO_GRAFICA:]
fecha_obs = obs_fuera.index[0]

# Figura 38: gráfica de abanico (intervalos al 50, 80 y 95%) con oct-2026 marcado
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
for nivel, alfa_color in zip(sorted(NIVELES_ABANICO, reverse=True), [0.15, 0.28, 0.42]):
    ax.fill_between(tabla_sw.index, tabla_sw[f"Lím. inf. {nivel * 100:g}%"],
                    tabla_sw[f"Lím. sup. {nivel * 100:g}%"], color=TEAL, alpha=alfa_color, lw=0,
                    label=f"Intervalo de predicción al {nivel * 100:g}%")
ax.plot(np.r_[[T], tabla_sw.index], np.r_[[Z.iloc[-1]], tabla_sw["Pronóstico"].values],
        color=NAVY, lw=1.8, ls="--", label=f"Pronóstico {NOMBRE}")
ax.scatter([fecha_obs], [real], color=RED, s=55, zorder=5,
           label=f"Observado {mes_anio(fecha_obs)} = {real:.4f}")
ax.axvline(T, color=GOLD, lw=1, ls=":")
ax.set_title(f"Tipo de cambio FIX, promedio mensual: pronóstico desde {mes_anio(T)} "
             f"con el modelo {NOMBRE}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="lower left", fontsize=8)
guardar_figura(fig, f"fig38_abanico_{LETRA_FINAL}.png")

# Figura 39: contraste de oct-2026 contra intervalos de distinto nivel
niveles_c = [0.50, 0.80, 0.95, 0.99]
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, nivel in enumerate(niveles_c):
    zz = stats.norm.ppf(1 - (1 - nivel) / 2)
    ax.plot([pron1 - zz * ee1, pron1 + zz * ee1], [i, i], color=TEAL, lw=6, alpha=0.4 + 0.15 * i,
            solid_capstyle="butt")
    ax.text(pron1 + zz * ee1 + 0.03, i, f"{pron1 + zz * ee1:.2f}", va="center", fontsize=8.5,
            color=CHARCOAL)
ax.axvline(pron1, color=NAVY, lw=1.5, ls="--", label=f"Pronóstico = {pron1:.4f}")
ax.axvline(real, color=RED, lw=2, label=f"Observado {mes_anio(fecha_obs)} = {real:.4f}")
ax.set_yticks(range(len(niveles_c)))
ax.set_yticklabels([f"Intervalo al {n_ * 100:g}%" for n_ in niveles_c])
ax.set_xlabel("Pesos por dólar")
ax.set_title(f"Tipo de cambio FIX de {mes_anio(fecha_obs)}: dato observado contra los intervalos "
             f"de predicción")
ax.legend(loc="lower right", fontsize=8)
guardar_figura(fig, "fig39_contraste_octubre.png")


print("Listo")