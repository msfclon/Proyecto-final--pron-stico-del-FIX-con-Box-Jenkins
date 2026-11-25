# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 9b. ROBUSTEZ: INTERVALOS DE PREDICCIÓN CON BOOTSTRAP

=======================================================================
El problema:
   Los intervalos del Paso 9 son  Z_hat_T(h) +- z_(alfa/2) * ee[e_T(h)],  y el
   z_(alfa/2) = 1.96 sale de la distribución NORMAL. Pero los residuos del modelo
   no pasan normalidad (Paso 7). ¿Están bien calibrados los intervalos?

La solución: bootstrap de los residuos (Efron, 1979; Thombs y Schucany, 1990;
Pascual, Romo y Ruiz, 2004). En lugar de suponer choques normales, se usan como
choques futuros los residuos REALES del modelo, sorteados con reemplazo.

   Versión 1. Bootstrap de residuos (parámetros fijos):
      1. Se estima el modelo y se guardan los residuos a_hat_t, centrados en cero.
      2. Para cada réplica b = 1, ..., B se sortean choques a*_(T+1), ..., a*_(T+H)
         de esos residuos y se simula la trayectoria con la ecuación del modelo:
            nabla Z*_(T+h) = phi_1 nabla Z*_(T+h-1) + ... + a*_(T+h) - theta_1 a*_(T+h-1) - ...
            Z*_(T+h)       = Z*_(T+h-1) + nabla Z*_(T+h)
      3. El intervalo al (1 - alfa) es el percentil alfa/2 y 1 - alfa/2 de las
         B trayectorias en cada h. Ya no se usa el 1.96.

   Versión 2. Bootstrap con reestimación (incluye la incertidumbre de los parámetros):
      1. Se genera una serie artificial de nabla Z (corregida por la intervención)
         con el modelo estimado y residuos sorteados.
      2. Se reestima el ARMA en esa serie artificial -> phi*, theta*.
      3. Con phi*, theta* y choques sorteados se pronostica DESDE LOS DATOS REALES
         (varias trayectorias por cada reestimación).
      4. Se repite R veces y se toman los percentiles.

Contraste: ¿el dato real de oct-2026 cae dentro del intervalo bootstrap al 95%?
=======================================================================
"""

from configuracion import *
from statsmodels.tsa.arima.model import ARIMA

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
CONFIANZA = 0.95                 # nivel de los intervalos de predicción
MODELO_FINAL = MODELO_B          # modelo seleccionado en el Paso 7
LETRA_FINAL = "B"
PASOS = 10                       # horizonte del pronóstico (igual que con software en el Paso 9)
REPLICAS = 100_000               # réplicas del bootstrap de residuos (versión 1)
REPLICAS_REESTIMACION = 500      # series artificiales reestimadas (versión 2; es lo más lento)
CHOQUES_POR_REPLICA = 200        # trayectorias futuras por cada reestimación (500 x 200 = 100,000)
CALENTAMIENTO = 50               # datos iniciales que se descartan al simular series artificiales
SEMILLA = 2026                   # semilla para que los resultados se puedan repetir
INICIO_GRAFICA = "2025-01-01"    # desde dónde se muestra la historia en las gráficas

p_, d_, q_ = MODELO_FINAL
NOMBRE = f"{LETRA_FINAL}: ARIMA({p_},{d_},{q_})"
alfa = 1 - CONFIANZA
z_alfa = stats.norm.ppf(1 - alfa / 2)
rng = np.random.default_rng(SEMILLA)


def simular_diferencias(ar, ma, w_previos, a_previos, choques):
    """Simula nabla Z futuro con la ecuación del ARMA (signos de statsmodels:
       nabla Z_t = sum ar_i nabla Z_(t-i) + a_t + sum ma_j a_(t-j)).
       w_previos y a_previos: últimos valores observados (el más reciente al final).
       choques: matriz (réplicas x pasos). Regresa una matriz del mismo tamaño."""
    B, H = choques.shape
    p, q = len(ar), len(ma)
    w = np.zeros((B, max(p, 1) + H))
    a = np.zeros((B, max(q, 1) + H))
    if p:
        w[:, max(p, 1) - p:max(p, 1)] = w_previos[-p:]
    if q:
        a[:, max(q, 1) - q:max(q, 1)] = a_previos[-q:]
    for h in range(H):
        i_w, i_a = max(p, 1) + h, max(q, 1) + h
        a[:, i_a] = choques[:, h]
        valor = a[:, i_a].copy()
        for i in range(p):
            valor += ar[i] * w[:, i_w - 1 - i]
        for j in range(q):
            valor += ma[j] * a[:, i_a - 1 - j]
        w[:, i_w] = valor
    return w[:, max(p, 1):]


def parametros_arma(res):
    """Coeficientes AR y MA con los signos de statsmodels."""
    ar = np.array([res.params[n] for n in res.params.index if n.startswith("ar.")])
    ma = np.array([res.params[n] for n in res.params.index if n.startswith("ma.")])
    return ar, ma


# =====================================================================
# 1. DATOS Y MODELO
# =====================================================================
encabezado(f"PASO 9b. INTERVALOS BOOTSTRAP CON EL MODELO {NOMBRE} CON INTERVENCIÓN")
m = obtener_muestras(cargar_fix())
Z = m["estim_final"]
obs_fuera = m["obs_fuera"]
real = float(obs_fuera.iloc[0])
T = Z.index[-1]
print(f"Muestra de estimación: {periodo(Z)}, N = {len(Z)}")
print(f"Origen del pronóstico: T = {mes_anio(T)}, Z_T = {Z.iloc[-1]:.4f}")
print(f"Dato fuera de la muestra: {mes_anio(obs_fuera.index[0])} = {real:.4f}")

res = estimar_arima(Z, MODELO_FINAL)
ar, ma = parametros_arma(res)
sigma2 = float(res.params["sigma2"])
print(f"Parámetros: " + ", ".join(f"{n} = {res.params[n]:.4f}" for n in res.params.index))

# Primera diferencia corregida por la intervención (los pulsos de 2020 ya no actúan)
W = diferenciar(Z, d_)
X = diferenciar(variables_intervencion(Z.index), d_)
omegas = np.array([res.params[c] for c in X.columns])
W_corr = W - X.values @ omegas

# Residuos del modelo, centrados en cero (son los choques que se sortean)
residuos = np.asarray(res.resid, dtype=float)
residuos_c = residuos - residuos.mean()
print(f"\nResiduos: {len(residuos)} datos, media = {residuos.mean():.4f}, "
      f"desviación estándar = {residuos.std(ddof=1):.4f}")
print(f"  Residuo mínimo = {residuos_c.min():.4f}, máximo = {residuos_c.max():.4f}")


# =====================================================================
# 2. INTERVALOS CON EL SUPUESTO DE NORMALIDAD
# =====================================================================
encabezado("1. Intervalos con el supuesto de normalidad (Paso 9)")
normal = pronostico_niveles(res, Z, PASOS)
normal["Lím. inf. normal"] = normal["Pronóstico"] - z_alfa * normal["Error estándar"]
normal["Lím. sup. normal"] = normal["Pronóstico"] + z_alfa * normal["Error estándar"]
fechas = normal.index
print(normal.round(4).set_axis([mes_anio(f) for f in fechas]).to_string())


# =====================================================================
# 3. BOOTSTRAP DE RESIDUOS (VERSIÓN 1: PARÁMETROS FIJOS)
# =====================================================================
encabezado(f"2. Bootstrap de residuos: {REPLICAS:,} trayectorias")
choques = rng.choice(residuos_c, size=(REPLICAS, PASOS), replace=True)
w_futuro = simular_diferencias(ar, ma, W_corr.values, residuos, choques)
Z_sim = Z.iloc[-1] + np.cumsum(w_futuro, axis=1)
boot_li = np.percentile(Z_sim, 100 * alfa / 2, axis=0)
boot_ls = np.percentile(Z_sim, 100 * (1 - alfa / 2), axis=0)
boot_med = np.median(Z_sim, axis=0)
print(f"Cada trayectoria sortea {PASOS} choques de los {len(residuos_c)} residuos y aplica "
      f"la ecuación del modelo.")


# =====================================================================
# 4. BOOTSTRAP CON REESTIMACIÓN (VERSIÓN 2)
# =====================================================================
encabezado(f"3. Bootstrap con reestimación: {REPLICAS_REESTIMACION} réplicas (tarda unos segundos)")
Z_sim_r = []
coef_r = []
n_w = len(W_corr)
for r in range(REPLICAS_REESTIMACION):
    # 1) serie artificial de nabla Z con residuos sorteados
    a_art = rng.choice(residuos_c, size=(1, n_w + CALENTAMIENTO), replace=True)
    w_art = simular_diferencias(ar, ma, np.zeros(max(p_, 1)), np.zeros(max(q_, 1)), a_art)[0]
    w_art = w_art[CALENTAMIENTO:]
    # 2) reestimación del ARMA en la serie artificial
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_r = ARIMA(w_art, order=(p_, 0, q_), trend="n").fit()
    except Exception:
        continue
    valores = np.asarray(res_r.params, dtype=float)
    ar_r, ma_r = valores[:p_], valores[p_:p_ + q_]
    coef_r.append(np.r_[ar_r, ma_r])
    # 3) pronóstico desde los datos reales con los parámetros reestimados
    #    (varias trayectorias por réplica para que los percentiles no dependan del azar)
    ch = rng.choice(residuos_c, size=(CHOQUES_POR_REPLICA, PASOS), replace=True)
    w_f = simular_diferencias(ar_r, ma_r, W_corr.values, residuos, ch)
    Z_sim_r.append(Z.iloc[-1] + np.cumsum(w_f, axis=1))
Z_sim_r = np.vstack(Z_sim_r)
coef_r = np.array(coef_r)
reest_li = np.percentile(Z_sim_r, 100 * alfa / 2, axis=0)
reest_ls = np.percentile(Z_sim_r, 100 * (1 - alfa / 2), axis=0)
print(f"Reestimaciones válidas: {len(coef_r)} de {REPLICAS_REESTIMACION}; "
      f"trayectorias simuladas: {len(Z_sim_r):,}")
for k in range(coef_r.shape[1]):
    etiqueta = f"phi_{k + 1}" if k < p_ else f"theta_{k - p_ + 1} (signo de statsmodels)"
    print(f"  {etiqueta}: estimado = {np.r_[ar, ma][k]:.4f}; en las réplicas, media = "
          f"{coef_r[:, k].mean():.4f}, desviación estándar = {coef_r[:, k].std(ddof=1):.4f}")


# =====================================================================
# 5. COMPARACIÓN DE INTERVALOS
# =====================================================================
encabezado(f"4. Intervalos de predicción al {CONFIANZA * 100:g}%: normal contra bootstrap")
tabla = pd.DataFrame({
    "h": range(1, PASOS + 1),
    "Pronóstico": normal["Pronóstico"].values,
    "Lím. inf. normal": normal["Lím. inf. normal"].values,
    "Lím. sup. normal": normal["Lím. sup. normal"].values,
    "Lím. inf. bootstrap": boot_li,
    "Lím. sup. bootstrap": boot_ls,
    "Lím. inf. bootstrap reestimando": reest_li,
    "Lím. sup. bootstrap reestimando": reest_ls,
}, index=fechas)
tabla["Ancho normal"] = tabla["Lím. sup. normal"] - tabla["Lím. inf. normal"]
tabla["Ancho bootstrap"] = tabla["Lím. sup. bootstrap"] - tabla["Lím. inf. bootstrap"]
print(tabla.round(4).set_axis([mes_anio(f) for f in fechas]).to_string())


# =====================================================================
# 6. PREGUNTA DE INVESTIGACIÓN CON BOOTSTRAP
# =====================================================================
encabezado(f"5. Pregunta de investigación: {mes_anio(obs_fuera.index[0])}")
filas = []
for metodo, li, ls, prob in [
        ("Normal (Paso 9)", tabla["Lím. inf. normal"].iloc[0], tabla["Lím. sup. normal"].iloc[0],
         1 - stats.norm.cdf((real - normal["Pronóstico"].iloc[0]) / normal["Error estándar"].iloc[0])),
        ("Bootstrap de residuos", boot_li[0], boot_ls[0], np.mean(Z_sim[:, 0] >= real)),
        ("Bootstrap con reestimación", reest_li[0], reest_ls[0], np.mean(Z_sim_r[:, 0] >= real))]:
    dentro = li <= real <= ls
    filas.append({"Método": metodo, "Límite inferior": round(li, 4), "Límite superior": round(ls, 4),
                  "Dato real": real, "¿Dentro?": "Sí" if dentro else "No",
                  "P(valor >= dato real)": round(prob, 4)})
    print(f"{metodo:28s} [{li:.4f}, {ls:.4f}]  -> {'DENTRO' if dentro else 'FUERA'} del intervalo; "
          f"P(valor >= {real:.4f}) = {prob:.4f}")
contraste = pd.DataFrame(filas)
error_oct = real - normal["Pronóstico"].iloc[0]
n_mayores = int(np.sum(residuos_c >= error_oct))
print(f"\nError de pronóstico de octubre = {error_oct:.4f}. Residuos históricos de ese tamaño o "
      f"mayores: {n_mayores} de {len(residuos_c)}.")
print("Interpretación: si el dato queda fuera también con bootstrap, la conclusión no depende "
      "del supuesto de normalidad.")


# =====================================================================
# 7. GRÁFICAS
# =====================================================================
encabezado("Gráficas")
fecha_obs = obs_fuera.index[0]

# Figura 41: distribución bootstrap de octubre contra la normal
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.hist(Z_sim[:, 0], bins=35, density=True, color=NAVY, alpha=0.75,
        label=f"Bootstrap de residuos ({REPLICAS:,} réplicas)")
xs = np.linspace(Z_sim[:, 0].min(), Z_sim[:, 0].max(), 400)
ax.plot(xs, stats.norm.pdf(xs, normal["Pronóstico"].iloc[0], normal["Error estándar"].iloc[0]),
        color=GOLD, lw=2, label="Densidad normal (Paso 9)")
ax.axvline(tabla["Lím. sup. normal"].iloc[0], color=GOLD, lw=1.5, ls="--",
           label=f"Límite superior normal = {tabla['Lím. sup. normal'].iloc[0]:.2f}")
ax.axvline(boot_ls[0], color=TEAL, lw=1.8, ls="--",
           label=f"Límite superior bootstrap = {boot_ls[0]:.2f}")
ax.axvline(real, color=RED, lw=2.2, label=f"Observado {mes_anio(fecha_obs)} = {real:.4f}")
ax.set_title(f"Tipo de cambio FIX, promedio mensual: distribución del pronóstico de "
             f"{mes_anio(fecha_obs)}, bootstrap contra normal")
ax.set_xlabel("Pesos por dólar")
ax.set_ylabel("Densidad")
ax.legend(loc="upper left", fontsize=8)
guardar_figura(fig, f"fig41_bootstrap_octubre_{LETRA_FINAL}.png")

# Figura 42: intervalos al 95% normal contra bootstrap en todo el horizonte
historia = Z[INICIO_GRAFICA:]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia.index, historia.values, color=CHARCOAL, lw=1.6, label="Observado")
ax.fill_between(fechas, tabla["Lím. inf. normal"], tabla["Lím. sup. normal"], color=TEAL,
                alpha=0.25, lw=0, label=f"Intervalo al {CONFIANZA * 100:g}%, normal")
ax.plot(fechas, boot_li, color=GOLD, lw=1.8, ls="--",
        label=f"Intervalo al {CONFIANZA * 100:g}%, bootstrap")
ax.plot(fechas, boot_ls, color=GOLD, lw=1.8, ls="--")
ax.plot(np.r_[[T], fechas], np.r_[[Z.iloc[-1]], tabla["Pronóstico"].values], color=NAVY, lw=1.8,
        ls="--", label=f"Pronóstico {NOMBRE}")
ax.scatter([fecha_obs], [real], color=RED, s=55, zorder=5,
           label=f"Observado {mes_anio(fecha_obs)} = {real:.4f}")
ax.axvline(T, color=GOLD, lw=1, ls=":")
ax.set_title(f"Tipo de cambio FIX, promedio mensual: intervalos de predicción normal y bootstrap "
             f"desde {mes_anio(T)}")
ax.set_xlabel("Fecha (mensual)")
ax.set_ylabel("Pesos por dólar")
formato_fechas(ax, con_mes=True)
ax.legend(loc="lower left", fontsize=8)
guardar_figura(fig, f"fig42_bootstrap_intervalos_{LETRA_FINAL}.png")


print("Listo")