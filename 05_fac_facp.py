# -*- coding: utf-8 -*-
"""
=======================================================================
 PASO 5. FAC Y FACP DE LA SERIE DIFERENCIADA: PRUEBAS DE SIGNIFICANCIA
 Criterio de la rúbrica: usar las pruebas de significancia de la FAC y la
 FACP para justificar el modelo que ajusta los datos.
=======================================================================
Se trabaja con W_t = nabla^d Z_t (d se toma de configuracion.py), CORREGIDA por la
intervención del Paso 4b: se le restan los saltos extraordinarios de marzo y abril
de 2020 (omega estimados junto con el modelo A). Así la pandemia no distorsiona
la FAC ni la FACP. Si INTERVENCIONES = [] en configuracion.py, no se corrige nada.

Qué hace:
  1. Prueba de Bartlett secuencial sobre la FAC (guía de estudio, cap. 4):
         H0: rho_k = 0   vs   H1: rho_k distinto de 0
     Se supone MA(q) con las autocorrelaciones ya significativas; la banda es
         z * raíz[(1 + 2 * sum_{j<=q} r_j^2) / N]
     El número de autocorrelaciones significativas al inicio da q.
  2. Prueba de significancia de la FACP (guía de estudio, sec. 3.4):
         H0: phi_kk = 0   vs   H1: phi_kk distinto de 0
     Se rechaza H0 si |phi_kk| > z / raíz(N). Las significativas al inicio dan p.
  3. Revisa la estacionalidad: FAC y FACP en los retrasos 12, 24 y 36.
  4. Identifica los modelos con la regla de la guía (cap. 5):
         FAC se corta en q y FACP decae  -> MA(q)
         FACP se corta en p y FAC decae  -> AR(p)
         ambas decaen                    -> ARMA mixto
  5. Calcula estimaciones preliminares por el método de momentos.
=======================================================================
"""

from configuracion import *

# =====================================================================
# AJUSTES DE ESTE PASO (se pueden cambiar)
# =====================================================================
ALFA = 0.05                            # nivel de significancia de la FAC y la FACP
K_RETRASOS = 36                        # retrasos que se revisan (36 = 3 años)
RETRASOS_ESTACIONALES = [12, 24, 36]   # múltiplos del periodo estacional (E = 12)
NIVEL = f"{ALFA * 100:g}%"


# =====================================================================
# 1. DATOS: serie diferenciada W_t = nabla^d Z_t, corregida por la intervención
# =====================================================================
encabezado(f"PASO 5. FAC Y FACP DE LA SERIE DIFERENCIADA (alfa = {ALFA})")

Z = obtener_muestras(cargar_fix())["entrenamiento"]
W = diferenciar(Z, ORDEN_D)
X = variables_intervencion(Z.index)
if X is not None:
    # Se restan los efectos de la intervención (pulsos en nabla Z), estimados con el modelo A
    omegas = estimar_arima(Z, MODELO_A).params[X.columns]
    efecto = diferenciar(X @ omegas, ORDEN_D)
    print("Corrección por intervención (se resta a la serie diferenciada):")
    for c, f in zip(X.columns, INTERVENCIONES):
        print(f"   {mes_anio(pd.Timestamp(f))}: nabla Z = {W[f]:.4f} - omega {omegas[c]:.4f} "
              f"= {W[f] - efecto[f]:.4f}")
    W = W - efecto
    TEXTO_SERIE = "primera diferencia corregida"
else:
    TEXTO_SERIE = "primera diferencia"
N = len(W)
print(f"Serie: {TEXTO_SERIE} (d = {ORDEN_D}) del FIX, {periodo(W)}, N = {N}")
print(f"Valor crítico normal al {NIVEL} (dos colas): z = {valor_z(ALFA):.4f}")


# =====================================================================
# 2. FAC: PRUEBA DE BARTLETT SECUENCIAL
# =====================================================================
r = fac(W, K_RETRASOS)
banda_fac, q_hat = bartlett_secuencial(r, N, ALFA)
sig_fac = [k for k in range(1, K_RETRASOS + 1) if abs(r[k - 1]) > banda_fac[k - 1]]

cuadro_fac = pd.DataFrame({
    "k": np.arange(1, K_RETRASOS + 1),
    "r_k": r.round(4),
    "q supuesto": [min(k - 1, q_hat) for k in range(1, K_RETRASOS + 1)],
    "Banda": banda_fac.round(4),
    "¿Se rechaza H0?": np.where(np.abs(r) > banda_fac, "Sí", "No"),
})

encabezado(f"Cuadro. FAC de la {TEXTO_SERIE} y prueba de Bartlett al {NIVEL}")
print(cuadro_fac.head(12).to_string(index=False))
print(f"\nRetrasos con FAC significativa (de {K_RETRASOS}): {sig_fac}")
print(f"La FAC se corta en q = {q_hat}")


# =====================================================================
# 3. FACP: PRUEBA DE SIGNIFICANCIA
# =====================================================================
phi = facp(r)
banda_p = banda_facp(N, ALFA)
sig_facp = [k for k in range(1, K_RETRASOS + 1) if abs(phi[k - 1]) > banda_p]
p_hat = 0
while p_hat < K_RETRASOS and abs(phi[p_hat]) > banda_p:   # significativas al inicio
    p_hat += 1

cuadro_facp = pd.DataFrame({
    "k": np.arange(1, K_RETRASOS + 1),
    "phi_kk": phi.round(4),
    "Banda": round(banda_p, 4),
    "¿Se rechaza H0?": np.where(np.abs(phi) > banda_p, "Sí", "No"),
})

encabezado(f"Cuadro. FACP de la {TEXTO_SERIE} al {NIVEL} (banda = ±{banda_p:.4f})")
print(cuadro_facp.head(12).to_string(index=False))
print(f"\nRetrasos con FACP significativa (de {K_RETRASOS}): {sig_facp}")
print(f"La FACP se corta en p = {p_hat}")
aislados = [k for k in sig_fac if k > q_hat + 1] + [k for k in sig_facp if k > p_hat + 1]
if aislados:
    print(f"Retrasos significativos aislados, lejos del corte: {sorted(set(aislados))}. "
          f"Con {K_RETRASOS} retrasos al {NIVEL} se esperan ~{ALFA * K_RETRASOS:.1f} por azar; "
          "sin patrón ni interpretación, no se modelan (Guerrero, p. 287).")


# =====================================================================
# 4. ESTACIONALIDAD: retrasos 12, 24 y 36
# =====================================================================
cuadro_est = pd.DataFrame({
    "k": RETRASOS_ESTACIONALES,
    "r_k": [round(r[k - 1], 4) for k in RETRASOS_ESTACIONALES],
    "Banda FAC": [round(banda_fac[k - 1], 4) for k in RETRASOS_ESTACIONALES],
    "phi_kk": [round(phi[k - 1], 4) for k in RETRASOS_ESTACIONALES],
    "Banda FACP": round(banda_p, 4),
})
cuadro_est["¿Significativa?"] = np.where(
    (cuadro_est["r_k"].abs() > cuadro_est["Banda FAC"]) |
    (cuadro_est["phi_kk"].abs() > cuadro_est["Banda FACP"]), "Sí", "No")
hay_estacionalidad = (cuadro_est["¿Significativa?"] == "Sí").any()

encabezado("Cuadro. Revisión de estacionalidad (retrasos múltiplos de 12)")
print(cuadro_est.to_string(index=False))
if hay_estacionalidad:
    print("Hay autocorrelación significativa en retrasos estacionales: considerar un SARIMA.")
else:
    print("Ninguna autocorrelación estacional es significativa: la serie NO es estacional.")


# =====================================================================
# 5. IDENTIFICACIÓN DE MODELOS
# =====================================================================
encabezado("Identificación de modelos (regla de la guía, cap. 5)")
print(f"  FAC  se corta en q = {q_hat}  -> sugiere un MA({q_hat}):  ARIMA(0,{ORDEN_D},{q_hat})")
print(f"  FACP se corta en p = {p_hat}  -> sugiere un AR({p_hat}):  ARIMA({p_hat},{ORDEN_D},0)")
print("  Como ambas se cortan pronto, también es razonable un modelo mixto ARMA,")
print("  que se buscará con selección automática en el Paso 6.")
if q_hat == 1:
    region = "RA2 (-1 < theta < 0)" if r[0] > 0 else "RA1 (0 < theta < 1)"
    print(f"  r_1 = {r[0]:.4f} {'> 0' if r[0] > 0 else '< 0'}: el MA(1) estaría en la región {region}.")
    print(f"  (En RA2 la FACP teórica alterna signos empezando positiva y decae.) FACP muestral: "
          f"phi_11 = {phi[0]:.3f}, phi_22 = {phi[1]:.3f}, phi_33 = {phi[2]:.3f}, phi_44 = {phi[3]:.3f}")
if p_hat == 1:
    region = "RA1 (0 < phi < 1)" if phi[0] > 0 else "RA2 (-1 < phi < 0)"
    print(f"  phi_11 = {phi[0]:.4f}: el AR(1) estaría en la región {region}.")


# =====================================================================
# 6. ESTIMACIONES PRELIMINARES (método de momentos)
# =====================================================================
encabezado("Estimaciones preliminares por el método de momentos")

# MA(1): W_t = (1 - theta B) a_t  ->  rho_1 = -theta / (1 + theta^2)
#        se despeja r_1 theta^2 + theta + r_1 = 0 y se toma la raíz invertible (|theta| < 1)
theta_mom = np.nan
if q_hat == 1 and abs(r[0]) < 0.5:
    theta_mom = (-1 + np.sqrt(1 - 4 * r[0] ** 2)) / (2 * r[0])
    print(f"  MA(1): theta = {theta_mom:.4f}   (de rho_1 = -theta / (1 + theta^2), raíz invertible)")

# AR(p): ecuaciones de Yule-Walker  R * phi = r,  con R la matriz de autocorrelaciones
phi_yw = np.array([])
if p_hat > 0:
    R = np.array([[1.0 if i == j else r[abs(i - j) - 1] for j in range(p_hat)] for i in range(p_hat)])
    phi_yw = np.linalg.solve(R, r[:p_hat])
    texto = ", ".join(f"phi_{i + 1} = {v:.4f}" for i, v in enumerate(phi_yw))
    print(f"  AR({p_hat}): {texto}   (ecuaciones de Yule-Walker)")


# =====================================================================
# 7. GRÁFICAS
# =====================================================================
encabezado("Gráficas")
k = np.arange(1, K_RETRASOS + 1)
colores = [GOLD if kk in RETRASOS_ESTACIONALES else NAVY for kk in k]

# --- Figura 9: FAC con bandas de Bartlett secuenciales ---
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.vlines(k, 0, r, colors=colores, lw=2.2)
ax.fill_between(k, -banda_fac, banda_fac, color=TEAL, alpha=0.2, lw=0, step="mid",
                label=f"Bandas de Bartlett ({(1 - ALFA) * 100:g}%)")
ax.axhline(0, color=CHARCOAL, lw=0.8)
ax.plot([], [], color=GOLD, lw=2.2, label="Retrasos estacionales (12, 24, 36)")
ax.set_ylim(-0.5, 0.5)
ax.set_title(f"FAC muestral de la {TEXTO_SERIE} del tipo de cambio FIX, {periodo(W)}")
ax.set_xlabel("Retraso k (meses)")
ax.set_ylabel(r"$r_k$")
ax.legend(loc="upper right")
guardar_figura(fig, "fig09_fac_diferencia.png")

# --- Figura 10: FACP con bandas +-z/raíz(N) ---
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.vlines(k, 0, phi, colors=colores, lw=2.2)
ax.axhspan(-banda_p, banda_p, color=TEAL, alpha=0.2, lw=0,
           label=rf"$\pm {valor_z(ALFA):.2f}/\sqrt{{N}}$ = $\pm${banda_p:.3f}")
ax.axhline(0, color=CHARCOAL, lw=0.8)
ax.plot([], [], color=GOLD, lw=2.2, label="Retrasos estacionales (12, 24, 36)")
ax.set_ylim(-0.5, 0.5)
ax.set_title(f"FACP muestral de la {TEXTO_SERIE} del tipo de cambio FIX, {periodo(W)}")
ax.set_xlabel("Retraso k (meses)")
ax.set_ylabel(r"$\hat{\phi}_{kk}$")
ax.legend(loc="upper right")
guardar_figura(fig, "fig10_facp_diferencia.png")

print("Listo")