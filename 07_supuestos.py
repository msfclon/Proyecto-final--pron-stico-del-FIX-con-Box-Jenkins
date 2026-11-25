# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 7. VERIFICACIÓN DE LOS 8 SUPUESTOS Y SELECCIÓN DEL MODELO
 Criterio de la rúbrica: verificar los supuestos de los tres modelos y
 elegir el mejor con base en los supuestos y en AIC, AICc y BIC.
=======================================================================
Los tres modelos incluyen la intervención del Paso 4b (cambio de nivel en
marzo y abril de 2020) y se estiman con estimar_arima de configuracion.py.

Se sigue la guía de estudio (caps. 8 a 10). En todas las pruebas, H0 es que
el supuesto SE CUMPLE: valor p grande -> se verifica; valor p chico -> no.

  1. Media cero ......... prueba t:  H0: mu_e = 0
  2. Varianza constante . Breusch-Pagan (versión de Koenker, residuos^2 contra
                          el tiempo):  H0: homocedasticidad
  3. Normalidad ......... Criterio 1: pruebas de bondad de ajuste, Anderson-Darling
                          y Shapiro-Wilk  (H0: los residuos son normales)
                          Criterio 2, si fallan: regla empírica 68-95-99.7
                          (rangos obtenidos por simulación). Complemento: sesgo,
                          curtosis, histograma y gráfica Q-Q
  4. Independencia ...... Box-Pierce y Ljung-Box con K - p - q grados de libertad
                          (H0: rho_1 = ... = rho_K = 0) y FAC de los residuos en
                          los primeros retrasos
  5. Parsimonia ......... IC de cada parámetro: estimación +- z * ee (no debe
                          contener al 0) y revisión de factores comunes AR-MA
  6. Modelo admisible ... estacionario (raíces AR fuera del círculo unitario),
                          invertible (raíces MA fuera) y región admisible
  7. Modelo estable ..... correlaciones entre los estimadores (no cercanas a +-1)
  8. Atípicos ........... residuos estandarizados: |e*| > 2 sospechoso, > 3 atípico

Notación de la guía:  (1 - phi_1 B - ...) nabla Z_t = (1 - theta_1 B - ...) a_t
(statsmodels usa "+" en la parte MA; el script cambia el signo de los theta).
Los coeficientes de la intervención se llaman omega (omega_mar2020, omega_abr2020).
=======================================================================
"""

from configuracion import *
from statsmodels.stats.diagnostic import normal_ad

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
ALFA = 0.05                  # nivel de significancia de las pruebas de supuestos
K_LJUNG = [12, 24]           # retrasos K de Box-Pierce y Ljung-Box
RETRASOS_CORTOS = 3          # la FAC de los residuos no debe ser significativa en k = 1, 2, 3
K_FAC_RESIDUOS = 36          # retrasos de la gráfica FAC de los residuos
UMBRAL_SOSPECHOSO = 2        # |residuo estandarizado| > 2: observación sospechosa
UMBRAL_ATIPICO = 3           # |residuo estandarizado| > 3: valor atípico
UMBRAL_CORRELACION = 0.8     # |correlación entre estimadores| > 0.8: modelo inestable
DISTANCIA_FACTOR_COMUN = 0.15  # raíces AR y MA más cercanas que esto: casi se cancelan
MARGEN_CIRCULO = 1.05        # módulo de raíz menor a 1.05: "en el límite" del círculo unitario

MODELOS = {"A": MODELO_A, "B": MODELO_B, "C": MODELO_C}   # definidos en configuracion.py
NUM_FIGURA = {"A": 22, "B": 26, "C": 30}                  # primer número de figura de cada modelo


# =====================================================================
# 1. DATOS Y ESTIMACIÓN
# =====================================================================
encabezado(f"PASO 7. VERIFICACIÓN DE SUPUESTOS (alfa = {ALFA})")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
W = diferenciar(Z, ORDEN_D)
X = variables_intervencion(Z.index)
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {len(Z)}")
if X is not None:
    print(f"Intervención incluida: escalones en "
          f"{', '.join(mes_anio(pd.Timestamp(f)) for f in INTERVENCIONES)}")


def nombre(orden):
    return f"ARIMA({orden[0]},{orden[1]},{orden[2]})"


def simbolo(param):
    """Nombre del parámetro en notación de la guía."""
    if param.startswith("escalon"):
        return param.replace("escalon", "omega")
    tipo, retraso = param.split(".L")
    return f"{'phi' if tipo == 'ar' else 'theta'}_{retraso}"


def ajustar(orden):
    """Estima el modelo y regresa todo lo necesario en notación de la guía."""
    res = estimar_arima(Z, orden)
    e = res.resid                                      # residuos de feb-2015 en adelante
    sigma = np.sqrt(res.params["sigma2"])
    nombres_sm = [n for n in res.params.index if n != "sigma2"]
    signos = np.array([-1.0 if n.startswith("ma") else 1.0 for n in nombres_sm])
    simbolos = [simbolo(n) for n in nombres_sm]
    est = pd.Series(signos * res.params[nombres_sm].values, index=simbolos)
    ee = pd.Series(res.bse[nombres_sm].values, index=simbolos)
    cov = res.cov_params().loc[nombres_sm, nombres_sm].values * np.outer(signos, signos)
    cov = pd.DataFrame(cov, index=simbolos, columns=simbolos)
    return res, e, sigma, est, ee, cov


def raices(est, letra):
    """Raíces de 1 - c_1 B - ... - c_k B^k para la parte AR ('phi') o MA ('theta')."""
    c = [est[f"{letra}_{i}"] for i in range(1, 10) if f"{letra}_{i}" in est.index]
    if not c:
        return np.array([])
    return np.roots(np.r_[[-x for x in c[::-1]], 1.0])


resumen_global = []
tablas = {"param": [], "ljung": [], "corr": [], "atipicos": [], "supuestos": [], "normalidad": []}

for letra, orden in MODELOS.items():
    p, d, q = orden
    res, e, sigma, est, ee, cov = ajustar(orden)
    n = len(e)
    etiqueta = f"{letra}: {nombre(orden)}"
    veredicto = {}

    encabezado(f"MODELO {etiqueta} con intervención   (N residuos = {n}, sigma_a = {sigma:.4f})")

    # -----------------------------------------------------------------
    # SUPUESTO 1. MEDIA CERO
    # -----------------------------------------------------------------
    t1, p1 = stats.ttest_1samp(e, 0.0)
    veredicto[1] = "Se verifica" if p1 > ALFA else "No se verifica"
    print(f"\n[1] Media cero: media = {e.mean():.4f}, t = {t1:.4f}, valor p = {p1:.4f} -> {veredicto[1]}")

    # -----------------------------------------------------------------
    # SUPUESTO 2. VARIANZA CONSTANTE (Breusch-Pagan, versión de Koenker)
    #   Regresión auxiliar: e_t^2 = a0 + a1 * t + u_t;  BP = n * R^2 ~ chi^2(1)
    # -----------------------------------------------------------------
    y2 = e.values ** 2
    Xbp = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    beta = np.linalg.lstsq(Xbp, y2, rcond=None)[0]
    r2 = 1 - np.sum((y2 - Xbp @ beta) ** 2) / np.sum((y2 - y2.mean()) ** 2)
    bp = n * r2
    p2 = 1 - stats.chi2.cdf(bp, df=1)
    veredicto[2] = "Se verifica" if p2 > ALFA else "No se verifica"
    print(f"[2] Varianza constante (Breusch-Pagan): BP = {bp:.4f}, gl = 1, valor p = {p2:.4f} -> {veredicto[2]}")

    # -----------------------------------------------------------------
    # SUPUESTO 3. NORMALIDAD
    #   Criterio 1: Anderson-Darling y Shapiro-Wilk. H0: los residuos son normales.
    #               Se cumple si NINGUNA rechaza H0.
    #   Criterio 2 (si falla el 1): regla empírica 68-95-99.7 (función regla_empirica:
    #               rango de cada banda con 20,000 muestras normales simuladas).
    #   Complemento: sesgo (z = sesgo / raíz(6/n), H0: sesgo = 0) y curtosis.
    # -----------------------------------------------------------------
    ad, p_ad = normal_ad(e.values)
    sw, p_sw = stats.shapiro(e.values)
    pruebas_cumplen = (p_ad > ALFA) and (p_sw > ALFA)
    print("[3] Normalidad:")
    print("    Criterio 1. Pruebas de bondad de ajuste:")
    print(f"      Anderson-Darling: A = {ad:.4f}, valor p = {p_ad:.4f} -> "
          f"{'no se rechaza H0' if p_ad > ALFA else 'se rechaza H0'}")
    print(f"      Shapiro-Wilk:     W = {sw:.4f}, valor p = {p_sw:.4f} -> "
          f"{'no se rechaza H0' if p_sw > ALFA else 'se rechaza H0'}")
    print(f"      -> las pruebas {'se cumplen' if pruebas_cumplen else 'NO se cumplen'}")

    regla, regla_cumple = regla_empirica(e, ALFA)
    print(f"    Criterio 2. Regla empírica (rango del {(1 - ALFA) * 100:g}% en 20,000 muestras normales simuladas):")
    for _, f in regla.iterrows():
        print(f"      {f['Banda']} desv. est.: observado {f['% observado']:5.1f}%  teórico "
              f"{f['% teórico']:5.2f}%  rango [{f['Límite inferior']:.1f}, {f['Límite superior']:.1f}]  "
              f"valor p = {f['Valor p']:.4f}  {'compatible' if f['¿Compatible?'] == 'Sí' else 'NO compatible'}")
    print(f"      -> la regla empírica {'se cumple' if regla_cumple else 'NO se cumple'}")

    sesgo = stats.skew(e)
    z_sesgo = sesgo / np.sqrt(6 / n)
    p_sesgo = 2 * (1 - stats.norm.cdf(abs(z_sesgo)))
    if p_sesgo > ALFA:
        lectura_sesgo = "simétrico"
    else:
        lectura_sesgo = "sesgo positivo (cola derecha)" if sesgo > 0 else "sesgo negativo (cola izquierda)"
    curtosis = stats.kurtosis(e)
    print(f"    Complemento. Sesgo = {sesgo:.3f}, z = {z_sesgo:.3f}, valor p = {p_sesgo:.4f} -> {lectura_sesgo}")
    print(f"                 Curtosis en exceso = {curtosis:.3f} (normal: 0; positiva = colas pesadas)")

    if pruebas_cumplen:
        veredicto[3] = "Se verifica"
    elif regla_cumple:
        veredicto[3] = "Se verifica (regla empírica)"
    else:
        veredicto[3] = "No se verifica"
    print(f"    -> {veredicto[3]}")
    fila_norm = {"Modelo": etiqueta, "A-D": round(ad, 4), "Valor p A-D": round(p_ad, 4),
                 "S-W": round(sw, 4), "Valor p S-W": round(p_sw, 4)}
    for _, f in regla.iterrows():
        fila_norm[f"% {f['Banda']}"] = f["% observado"]
        fila_norm[f"Rango {f['Banda']}"] = f"{f['Límite inferior']}-{f['Límite superior']}"
    fila_norm.update({"Sesgo": round(sesgo, 4), "Valor p sesgo": round(p_sesgo, 4),
                      "Curtosis en exceso": round(curtosis, 4), "Resultado": veredicto[3]})
    tablas["normalidad"].append(fila_norm)

    # -----------------------------------------------------------------
    # SUPUESTO 4. INDEPENDENCIA
    #   Box-Pierce y Ljung-Box con gl = K - p - q, y FAC de los residuos en los
    #   primeros retrasos (la gráfica no debe tener barras fuera de las bandas).
    # -----------------------------------------------------------------
    r_e = fac(e, max(max(K_LJUNG), K_FAC_RESIDUOS))
    independiente = True
    print("[4] Independencia:")
    for K in K_LJUNG:
        q_bp = n * np.sum(r_e[:K] ** 2)
        q_lb = n * (n + 2) * np.sum(r_e[:K] ** 2 / (n - np.arange(1, K + 1)))
        gl = K - p - q
        p_bp = 1 - stats.chi2.cdf(q_bp, gl)
        p_lb = 1 - stats.chi2.cdf(q_lb, gl)
        independiente &= (p_lb > ALFA) and (p_bp > ALFA)
        print(f"    K = {K}: Box-Pierce = {q_bp:.3f} (p = {p_bp:.4f}), "
              f"Ljung-Box = {q_lb:.3f} (p = {p_lb:.4f}), gl = {gl}")
        tablas["ljung"].append({"Modelo": etiqueta, "K": K, "gl": gl,
                                "Box-Pierce": round(q_bp, 4), "Valor p BP": round(p_bp, 4),
                                "Ljung-Box": round(q_lb, 4), "Valor p LB": round(p_lb, 4)})
    banda_res = banda_facp(n, ALFA)
    fuera = [k for k in range(1, K_FAC_RESIDUOS + 1) if abs(r_e[k - 1]) > banda_res]
    fuera_cortos = [k for k in fuera if k <= RETRASOS_CORTOS]
    print(f"    FAC de residuos: r_1 = {r_e[0]:.4f}, r_2 = {r_e[1]:.4f}, r_3 = {r_e[2]:.4f}; "
          f"fuera de ±{banda_res:.3f}: {fuera if fuera else 'ninguna'}")
    if fuera_cortos:
        print(f"    La FAC de los residuos es significativa en el retraso {fuera_cortos}: "
              "queda autocorrelación sin modelar.")
    veredicto[4] = "Se verifica" if (independiente and not fuera_cortos) else "No se verifica"
    print(f"    -> {veredicto[4]}")

    # -----------------------------------------------------------------
    # SUPUESTO 5. PARSIMONIA E INFERENCIA SOBRE LOS PARÁMETROS
    #   IC: estimación +- z * ee; si no contiene al 0, el parámetro es significativo
    # -----------------------------------------------------------------
    z = valor_z(ALFA)
    print(f"[5] Parsimonia: {p + q} parámetros ARMA y {len(est) - p - q} de intervención; "
          f"IC al {(1 - ALFA) * 100:g}%:")
    todos_signif = True
    for s_ in est.index:
        li, ls = est[s_] - z * ee[s_], est[s_] + z * ee[s_]
        contiene = li <= 0 <= ls
        todos_signif &= not contiene
        print(f"    {s_:14s} = {est[s_]:8.4f}   IC = ({li:.4f}, {ls:.4f})   "
              f"{'contiene 0: NO significativo' if contiene else 'no contiene 0: significativo'}")
        tablas["param"].append({"Modelo": etiqueta, "Parámetro": s_, "Estimación": round(est[s_], 4),
                                "Error est.": round(ee[s_], 4), "Límite inferior": round(li, 4),
                                "Límite superior": round(ls, 4),
                                "¿Significativo?": "No" if contiene else "Sí"})
    raices_ar, raices_ma = raices(est, "phi"), raices(est, "theta")
    factor_comun = False
    if len(raices_ar) and len(raices_ma):
        dist = min(abs(a - m) for a in raices_ar for m in raices_ma)
        factor_comun = dist < DISTANCIA_FACTOR_COMUN
        print(f"    Distancia mínima entre raíces AR y MA = {dist:.4f}"
              f"{'  -> CASI SE CANCELAN (factores comunes): parámetros redundantes' if factor_comun else ''}")
    veredicto[5] = "Se verifica" if (todos_signif and not factor_comun) else "No se verifica"
    print(f"    -> {veredicto[5]}")

    # -----------------------------------------------------------------
    # SUPUESTO 6. MODELO ADMISIBLE (estacionario, invertible y región)
    # -----------------------------------------------------------------
    mod_ar = np.abs(raices_ar)
    mod_ma = np.abs(raices_ma)
    estacionario = bool(np.all(mod_ar > 1)) if len(mod_ar) else True
    invertible = bool(np.all(mod_ma > 1)) if len(mod_ma) else True
    todas = np.r_[mod_ar, mod_ma]
    en_limite = bool(np.any(todas < MARGEN_CIRCULO)) if len(todas) else False
    print("[6] Modelo admisible:")
    print(f"    Raíces AR: {np.round(raices_ar, 4) if len(raices_ar) else 'no hay'}  "
          f"módulos: {np.round(mod_ar, 4) if len(mod_ar) else '-'}  -> "
          f"{'estacionario' if estacionario else 'NO estacionario'}")
    print(f"    Raíces MA: {np.round(raices_ma, 4) if len(raices_ma) else 'no hay'}  "
          f"módulos: {np.round(mod_ma, 4) if len(mod_ma) else '-'}  -> "
          f"{'invertible' if invertible else 'NO invertible'}")

    # FAC muestral de la primera diferencia corregida por la intervención de este modelo
    W_corr = W.copy()
    if X is not None:
        W_corr = W - diferenciar(X @ res.params[X.columns], ORDEN_D)
    r_w = fac(W_corr, 2)
    if (p, q) == (0, 0):
        print("    Ruido blanco para nabla Z: siempre es estacionario e invertible (no tiene raíces).")
    elif (p, q) == (0, 1):
        th = est["theta_1"]
        region = "RA1 (0 < theta < 1)" if th > 0 else "RA2 (-1 < theta < 0)"
        rho1 = -th / (1 + th ** 2)
        print(f"    Región: {region}. rho_1 teórica = {rho1:.4f} vs r_1 muestral = {r_w[0]:.4f}")
    elif (p, q) == (1, 0):
        ph = est["phi_1"]
        region = "RA1 (0 < phi < 1)" if ph > 0 else "RA2 (-1 < phi < 0)"
        print(f"    Región: {region}. rho_1 teórica = phi = {ph:.4f} vs r_1 muestral = {r_w[0]:.4f}; "
              f"rho_2 teórica = phi^2 = {ph ** 2:.4f} vs r_2 muestral = {r_w[1]:.4f}")
    elif (p, q) == (2, 0):
        f1, f2 = est["phi_1"], est["phi_2"]
        disc = f1 ** 2 + 4 * f2
        region = {(True, True): "RA2", (False, True): "RA1",
                  (True, False): "RA4", (False, False): "RA3"}[(f1 > 0, disc > 0)]
        rho1 = f1 / (1 - f2)
        rho2 = f1 * rho1 + f2
        print(f"    Región: {region}. rho_1, rho_2 teóricas = {rho1:.4f}, {rho2:.4f} "
              f"vs muestrales = {r_w[0]:.4f}, {r_w[1]:.4f}")
    else:
        print("    Región: no aplica (orden mayor: sólo estacionariedad e invertibilidad)")
    if estacionario and invertible:
        veredicto[6] = "Se verifica (en el límite)" if en_limite else "Se verifica"
    else:
        veredicto[6] = "No se verifica"
    if en_limite:
        print(f"    ATENCIÓN: hay raíces con módulo menor a {MARGEN_CIRCULO}, casi sobre el círculo unitario.")
    print(f"    -> {veredicto[6]}")

    # -----------------------------------------------------------------
    # SUPUESTO 7. MODELO ESTABLE (correlaciones entre estimadores)
    # -----------------------------------------------------------------
    print("[7] Modelo estable (correlaciones entre los estimadores):")
    if len(est) < 2:
        veredicto[7] = "No aplica (un solo parámetro)"
        print(f"    Un solo parámetro: no hay pares que revisar -> {veredicto[7]}")
    else:
        de = np.sqrt(np.diag(cov.values))
        corr = pd.DataFrame(cov.values / np.outer(de, de), index=cov.index, columns=cov.columns)
        print(corr.round(4).to_string())
        fuera_diag = np.abs(corr.values[~np.eye(len(corr), dtype=bool)])
        max_corr = fuera_diag.max()
        veredicto[7] = "Se verifica" if max_corr < UMBRAL_CORRELACION else "No se verifica"
        print(f"    Máxima |correlación| = {max_corr:.4f} (umbral {UMBRAL_CORRELACION}) -> {veredicto[7]}")
        c2 = corr.copy()
        c2.insert(0, "Modelo", etiqueta)
        tablas["corr"].append(c2.round(4))

    # -----------------------------------------------------------------
    # SUPUESTO 8. VALORES ATÍPICOS (residuos estandarizados)
    # -----------------------------------------------------------------
    e_est = e / sigma
    sospechosos = e_est[np.abs(e_est) > UMBRAL_SOSPECHOSO]
    atipicos = e_est[np.abs(e_est) > UMBRAL_ATIPICO]
    veredicto[8] = "Se verifica" if len(atipicos) == 0 else f"Revisar ({len(atipicos)} atípico(s))"
    print(f"[8] Atípicos: {len(sospechosos)} con |e*| > {UMBRAL_SOSPECHOSO}, "
          f"{len(atipicos)} con |e*| > {UMBRAL_ATIPICO}")
    for f, v in sospechosos.items():
        print(f"    {mes_anio(f)}: e* = {v:.3f}{'   <- ATÍPICO' if abs(v) > UMBRAL_ATIPICO else ''}")
        tablas["atipicos"].append({"Modelo": etiqueta, "Fecha": mes_anio(f), "e*": round(v, 4),
                                   "Clasificación": "Atípico" if abs(v) > UMBRAL_ATIPICO else "Sospechoso"})
    print(f"    -> {veredicto[8]}")

    # -----------------------------------------------------------------
    # GRÁFICAS DEL MODELO (cuatro por modelo, separadas)
    # -----------------------------------------------------------------
    num = NUM_FIGURA[letra]
    titulo = f"modelo {etiqueta} con intervención"

    # (a) Residuos estandarizados contra el tiempo (supuestos 2 y 8)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(e_est.index, e_est.values, color=NAVY, lw=1.1)
    ax.axhline(0, color=CHARCOAL, lw=0.8)
    for u, col in [(UMBRAL_SOSPECHOSO, GOLD), (UMBRAL_ATIPICO, RED)]:
        ax.axhline(u, color=col, lw=1, ls="--")
        ax.axhline(-u, color=col, lw=1, ls="--", label=rf"$\pm${u}")
    for f, v in sospechosos.items():
        ax.annotate(mes_anio(f), (f, v), textcoords="offset points", xytext=(5, 0),
                    fontsize=7.5, color=RED if abs(v) > UMBRAL_ATIPICO else CHARCOAL)
    ax.set_title(f"Residuos estandarizados del {titulo}, {periodo(e_est)}")
    ax.set_xlabel("Fecha (mensual)")
    ax.set_ylabel(r"$e_t^* = \hat a_t / \hat\sigma_a$")
    formato_fechas(ax, cada_anios=1)
    ax.legend(loc="lower right", ncol=2)
    guardar_figura(fig, f"fig{num}_residuos_{letra}.png")

    # (b) Histograma con la curva normal y las bandas de la regla empírica (supuesto 3)
    z_e = (e - e.mean()) / e.std()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(z_e, bins=20, density=True, color=NAVY, alpha=0.85, edgecolor="white",
            label="Residuos estandarizados")
    malla = np.linspace(-4, 4, 300)
    ax.plot(malla, stats.norm.pdf(malla), color=GOLD, lw=2, label="Densidad normal estándar")
    for j, estilo in [(1, "-"), (2, "--"), (3, ":")]:
        f = regla.iloc[j - 1]
        ax.axvline(j, color=TEAL, lw=1, ls=estilo)
        ax.axvline(-j, color=TEAL, lw=1, ls=estilo,
                   label=rf"$\pm{j}$ desv. est.: {f['% observado']:.1f}% (normal {f['% teórico']:.1f}%)")
    ax.set_xlim(-4, 4)
    ax.set_title(f"Histograma de los residuos estandarizados del {titulo}")
    ax.set_xlabel("Residuo estandarizado")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper left")
    guardar_figura(fig, f"fig{num + 1}_histograma_{letra}.png")

    # (c) Gráfica Q-Q normal (supuesto 3)
    fig, ax = plt.subplots(figsize=(7, 6))
    (teo, mues), (pend, inter, _) = stats.probplot(z_e, dist="norm")
    ax.scatter(teo, mues, color=NAVY, s=16, label="Residuos estandarizados")
    ax.plot(teo, pend * teo + inter, color=RED, lw=1.5, label="Recta de referencia (normal)")
    ax.set_title(f"Gráfica Q-Q normal de los residuos del\n{titulo}")
    ax.set_xlabel("Cuantiles teóricos de la normal estándar")
    ax.set_ylabel("Cuantiles de los residuos")
    ax.legend(loc="upper left")
    guardar_figura(fig, f"fig{num + 2}_qq_{letra}.png")

    # (d) FAC de los residuos (supuesto 4)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    kk = np.arange(1, K_FAC_RESIDUOS + 1)
    ax.vlines(kk, 0, r_e[:K_FAC_RESIDUOS], color=NAVY, lw=2)
    ax.axhspan(-banda_res, banda_res, color=TEAL, alpha=0.2, lw=0,
               label=rf"$\pm {valor_z(ALFA):.2f}/\sqrt{{N}}$ = $\pm${banda_res:.3f}")
    ax.axhline(0, color=CHARCOAL, lw=0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"FAC de los residuos del {titulo}")
    ax.set_xlabel("Retraso k (meses)")
    ax.set_ylabel(r"$r_k(\hat a)$")
    ax.legend(loc="upper right")
    guardar_figura(fig, f"fig{num + 3}_fac_residuos_{letra}.png")

    # Resumen del modelo
    nombres_sup = {1: "Media cero", 2: "Varianza constante", 3: "Normalidad", 4: "Independencia",
                   5: "Parsimonia", 6: "Modelo admisible", 7: "Modelo estable", 8: "Atípicos"}
    for i in range(1, 9):
        tablas["supuestos"].append({"Supuesto": f"{i}. {nombres_sup[i]}", "Modelo": letra,
                                    "Resultado": veredicto[i]})
    fallas = sum(v.startswith("No se verifica") for v in veredicto.values())
    revisar = sum(v.startswith("Revisar") for v in veredicto.values())
    resumen_global.append({"Modelo": etiqueta, "Supuestos que fallan": fallas,
                           "Por revisar": revisar, "Parámetros ARMA": p + q,
                           "AIC": round(res.aic, 3), "AICc": round(res.aicc, 3),
                           "BIC": round(res.bic, 3)})


# =====================================================================
# 2. RESUMEN Y SELECCIÓN
# =====================================================================
cuadro_sup = pd.DataFrame(tablas["supuestos"]).pivot(index="Supuesto", columns="Modelo",
                                                     values="Resultado")
cuadro_global = pd.DataFrame(resumen_global)

encabezado("Cuadro. Resumen de supuestos por modelo")
print(cuadro_sup.to_string())
encabezado("Cuadro. Supuestos que fallan y criterios de información")
print(cuadro_global.to_string(index=False))

# Regla de selección: 1) menos supuestos que fallan; 2) en empate, menos parámetros
# (principio de parsimonia de la guía); 3) si sigue el empate, menor BIC.
orden_sel = cuadro_global.sort_values(["Supuestos que fallan", "Parámetros ARMA", "BIC"])
print("\nRegla: menos supuestos que fallan -> menos parámetros (parsimonia) -> menor BIC")
print(f"Modelo seleccionado: {orden_sel.iloc[0]['Modelo']}")


print("Listo")