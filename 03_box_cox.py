# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 3. TRANSFORMACIÓN ESTABILIZADORA DE VARIANZA (BOX-COX)
 Criterio de la rúbrica: aplicar y justificar la transformación, estimando
 lambda puntualmente y por intervalo de confianza.
=======================================================================
Transformación (guía de series estacionales, def. 6.1):
    T(Z) = (Z^lambda - 1) / lambda   si lambda distinto de 0
    T(Z) = log(Z)                    si lambda = 0
    lambda = 1: ninguna | 0.5: raíz cuadrada | 0: logaritmo | -1: inverso

Qué hace (con la muestra de entrenamiento):
  1. Media y desviación estándar por año: ¿la variabilidad crece con el nivel?
  2. Estima lambda por máxima verosimilitud (log-verosimilitud perfil):
         l(lambda) = -(N/2) log(sigma^2(lambda)) + (lambda - 1) sum log(Z_t)
  3. Intervalo de confianza por razón de verosimilitud:
         { lambda : 2 [ l(lambda_hat) - l(lambda) ] <= chi^2 con 1 g.l. }
  4. Prueba los valores comunes de lambda.
     Si el intervalo contiene a 1, no hace falta transformar.
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
CONFIANZA = 0.95                        # nivel de confianza del intervalo de lambda
LAMBDAS_A_PROBAR = [1, 0.5, 0, -1]      # valores comunes que se contrastan
NOMBRES = {1: "Ninguna", 0.5: "Raíz cuadrada", 0: "Logaritmo", -1: "Inverso"}
MALLA_LAMBDA = np.linspace(-3, 4, 14001)  # valores de lambda evaluados (paso de 0.0005)


# =====================================================================
# 1. DATOS (muestra de entrenamiento)
# =====================================================================
encabezado(f"PASO 3. TRANSFORMACIÓN DE BOX-COX (confianza = {CONFIANZA})")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
N = len(Z)
print(f"Muestra de entrenamiento: {periodo(Z)}, N = {N}")


# =====================================================================
# 2. MEDIA Y DESVIACIÓN ESTÁNDAR POR AÑO
# =====================================================================
por_anio = Z.groupby(Z.index.year).agg(["mean", "std", "count"])
por_anio.columns = ["Media", "Desv. est.", "Meses"]
corr, p_corr = stats.pearsonr(por_anio["Media"], por_anio["Desv. est."])

encabezado("Cuadro. Media y desviación estándar del FIX por año")
print(por_anio.round(4).to_string())
print(f"Correlación entre media y desviación anuales: {corr:.4f} (valor p = {p_corr:.4f})")


# =====================================================================
# 3. ESTIMACIÓN DE LAMBDA POR MÁXIMA VEROSIMILITUD
# =====================================================================
def box_cox(z, lam):
    z = np.asarray(z, dtype=float)
    return np.log(z) if abs(lam) < 1e-12 else (z ** lam - 1) / lam

def log_verosimilitud(z, lam):
    """l(lambda) = -(N/2) log(sigma^2(lambda)) + (lambda - 1) sum log(Z_t)"""
    z = np.asarray(z, dtype=float)
    t = box_cox(z, lam)
    sigma2 = np.mean((t - t.mean()) ** 2)
    return -len(z) / 2 * np.log(sigma2) + (lam - 1) * np.sum(np.log(z))

# A mano: se evalúa l(lambda) en la malla y se busca el máximo
l_malla = np.array([log_verosimilitud(Z, lam) for lam in MALLA_LAMBDA])
l_max = l_malla.max()
lam_mano = MALLA_LAMBDA[l_malla.argmax()]
chi2_crit = stats.chi2.ppf(CONFIANZA, df=1)
dentro = MALLA_LAMBDA[2 * (l_max - l_malla) <= chi2_crit]

# Comprobación con scipy
_, lam_hat, ic = stats.boxcox(Z.values, alpha=1 - CONFIANZA)

encabezado("Estimación de lambda por máxima verosimilitud")
print(f"  A mano : lambda = {lam_mano:.4f}, IC = [{dentro.min():.4f}, {dentro.max():.4f}]")
print(f"  scipy  : lambda = {lam_hat:.4f}, IC = [{ic[0]:.4f}, {ic[1]:.4f}]")
print(f"  Valor crítico chi-cuadrada(1) al {CONFIANZA * 100:g}%: {chi2_crit:.4f}")


# =====================================================================
# 4. PRUEBAS DE RAZÓN DE VEROSIMILITUD PARA VALORES COMUNES
#    H0: lambda = lambda_0  vs  H1: lambda distinto de lambda_0
#    RV = 2 [ l(lambda_hat) - l(lambda_0) ] ~ chi^2 con 1 g.l.
# =====================================================================
alfa = 1 - CONFIANZA
filas = []
for lam0 in LAMBDAS_A_PROBAR:
    rv = 2 * (log_verosimilitud(Z, lam_hat) - log_verosimilitud(Z, lam0))
    pval = 1 - stats.chi2.cdf(rv, df=1)
    filas.append({"lambda_0": lam0, "Transformación": NOMBRES.get(lam0, ""),
                  "RV": round(rv, 4), "Valor p": round(pval, 4),
                  "¿En el IC?": "Sí" if ic[0] <= lam0 <= ic[1] else "No",
                  f"Decisión ({alfa * 100:g}%)":
                      "No se rechaza H0" if pval > alfa else "Se rechaza H0"})
cuadro_rv = pd.DataFrame(filas)

encabezado("Cuadro. Pruebas de razón de verosimilitud para lambda")
print(cuadro_rv.to_string(index=False))


# =====================================================================
# 5. CONCLUSIÓN
# =====================================================================
encabezado("Conclusión")
if ic[0] <= 1 <= ic[1]:
    print("El intervalo de confianza contiene a lambda = 1: no se rechaza H0: lambda = 1.")
    print("La varianza ya es estable y NO es necesario transformar la serie.")
    print("Con lambda = 1, T(Z_t) = Z_t - 1 sólo desplaza la serie: se trabaja con Z_t.")
else:
    print("El intervalo NO contiene a 1: hay que transformar la serie.")
    print("Elegir un valor sencillo dentro del intervalo y anotarlo en configuracion.py.")
print(f"(Valor registrado en configuracion.py: LAMBDA_BOX_COX = {LAMBDA_BOX_COX})")


# =====================================================================
# 6. GRÁFICAS
# =====================================================================
encabezado("Gráficas")

# --- Figura 5: log-verosimilitud perfil ---
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(MALLA_LAMBDA, l_malla, color=NAVY, lw=1.6, label=r"$\ell(\lambda)$")
ax.axhline(l_max - chi2_crit / 2, color=CHARCOAL, lw=0.9, ls=":",
           label=r"$\ell(\hat\lambda) - \chi^2_{1}/2$")
ax.axvspan(ic[0], ic[1], color=TEAL, alpha=0.15, lw=0,
           label=f"IC {CONFIANZA * 100:g}%: [{ic[0]:.2f}, {ic[1]:.2f}]")
ax.axvline(lam_hat, color=NAVY, lw=1.2, ls="--", label=rf"$\hat\lambda$ = {lam_hat:.2f}")
ax.axvline(1, color=GOLD, lw=1.6, label=r"$\lambda$ = 1 (sin transformación)")
ax.axvline(0, color=RED, lw=1.2, ls="--", label=r"$\lambda$ = 0 (logaritmo)")
ax.set_xlim(-2, 4)
ax.set_ylim(l_max - 12, l_max + 1.5)
ax.set_title(f"Log-verosimilitud perfil de Box-Cox, tipo de cambio FIX, {periodo(Z)}")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Log-verosimilitud $\ell(\lambda)$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
guardar_figura(fig, "fig05_box_cox_verosimilitud.png")

# --- Figura 6: desviación estándar contra media por año ---
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(por_anio["Media"], por_anio["Desv. est."], color=NAVY, s=40, zorder=3)
desplaza = {2022: (-30, 6)}          # evita que se enciman las etiquetas 2021 y 2022
for anio, f in por_anio.iterrows():
    ax.annotate(str(anio), (f["Media"], f["Desv. est."]), textcoords="offset points",
                xytext=desplaza.get(anio, (6, 4)), fontsize=8.5, color=CHARCOAL)
ax.text(0.02, 0.95, f"Correlación = {corr:.2f} (valor p = {p_corr:.2f})",
        transform=ax.transAxes, fontsize=9, color=CHARCOAL, va="top")
ax.set_title(f"Desviación estándar contra media anual del tipo de cambio FIX, {periodo(Z)}")
ax.set_xlabel("Media anual (pesos por dólar)")
ax.set_ylabel("Desviación estándar anual (pesos por dólar)")
guardar_figura(fig, "fig06_desviacion_media.png")



print("Listo")