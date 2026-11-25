# -*- coding: utf-8 -*-
"""
=======================================================================
 CONFIGURACIÓN DEL PROYECTO - Pronóstico del tipo de cambio FIX
 Econometría II, UDLAP
=======================================================================
Este archivo NO se corre solo. Lo usan todos los scripts (01, 02, 03...)
con la línea:   from configuracion import *

Tiene tres partes:
  PARTE 1. Parámetros del proyecto -> datos, fechas y decisiones tomadas.
  PARTE 2. Estilo de las gráficas  -> colores y formato.
  PARTE 3. Funciones de apoyo      -> fórmulas que usan varios scripts.

IMPORTANTE: los NIVELES DE SIGNIFICANCIA no están aquí. Cada script
los tiene al inicio, en su bloque "AJUSTES DE ESTE PASO", para poder
cambiarlos en el mismo archivo que se está revisando.
=======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from scipy import stats

# El Excel de Banxico no trae estilos de formato; este aviso no es un error.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# =====================================================================
# PARTE 1. PARÁMETROS DEL PROYECTO
# =====================================================================

# --- 1.1 Archivo de datos --------------------------------------------
RUTA_EXCEL = "Serie FIX.xlsx"     # Excel descargado de Banxico (SIE, serie SF17908)
FILAS_ENCABEZADO = 18             # filas de títulos del Excel de Banxico antes de los datos

# --- 1.2 Muestras (formato "AAAA-MM-01") ------------------------------
INICIO_ENTRENAMIENTO = "2015-01-01"   # con esta muestra se identifica y estima el modelo
FIN_ENTRENAMIENTO    = "2025-09-01"
INICIO_VALIDACION    = "2025-10-01"   # 12 meses fuera de la estimación para medir el error
FIN_VALIDACION       = "2026-09-01"
OBS_FUERA            = "2026-10-01"   # observación que se deja fuera (contraste y actualización)

# --- 1.3 Decisiones ya tomadas (se actualizan conforme avanzamos) ----
LAMBDA_BOX_COX = 1    # Paso 3: el IC de lambda contiene a 1 -> no se transforma la serie
ORDEN_D        = 1    # Paso 4: con una diferencia la serie es estacionaria

# Paso 4b: análisis de intervención (Guerrero, cap. 7). Fechas donde empieza un
# ESCALÓN (cambio de nivel): S_t = 0 antes de la fecha y 1 desde la fecha.
# Dos escalones = cambio de nivel en dos meses (COVID-19: marzo y abril de 2020).
# Para trabajar SIN intervención, dejar la lista vacía: INTERVENCIONES = []
INTERVENCIONES = ["2020-03-01", "2020-04-01"]

# Modelos candidatos, en formato (p, d, q). Todos sin constante (Paso 4: media de nabla Z = 0)
MODELO_A = (0, 1, 1)  # Paso 5: la FAC se corta en q = 1  -> MA(1)
MODELO_B = (1, 1, 0)  # Paso 5 (serie corregida por la intervención): la FACP se corta en p = 1
MODELO_C = (0, 1, 0)  # Paso 6: selección automática (mínimo AICc entre los admisibles)
# --- 1.4 Carpetas donde se guardan los resultados ---------------------
CARPETA_FIGURAS    = "figuras"      # gráficas .png (las que se suben a Overleaf)
CARPETA_RESULTADOS = "resultados"   # archivos de Excel con cuadros y cálculos


# =====================================================================
# PARTE 2. ESTILO DE LAS GRÁFICAS
# =====================================================================

GOLD      = '#B6A269'   # dorado
NAVY      = '#041E42'   # azul marino (color principal)
CHARCOAL  = '#222A35'   # gris oscuro (texto y ejes)
TEAL      = '#339989'   # verde azulado (bandas de confianza)
BLUE      = '#0F6FF5'   # azul
RED       = '#F8696B'   # rojo (observación fuera, alertas)
GRID_GRIS = '#E2E8F0'   # cuadrícula

MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]

plt.rcParams.update({
    "savefig.dpi": 300, "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelcolor": CHARCOAL, "axes.edgecolor": CHARCOAL,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID_GRIS,
    "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


# =====================================================================
# PARTE 3. FUNCIONES DE APOYO (no hace falta modificarlas)
# =====================================================================

# --- 3.1 Fechas y textos ---------------------------------------------
def mes_anio(fecha):
    """Convierte una fecha en texto tipo 'oct-2026'."""
    return f"{MESES[fecha.month - 1]}-{fecha.year}"


def periodo(serie):
    """Texto con el periodo de una serie, por ejemplo 'ene-2015 a sep-2025'."""
    return f"{mes_anio(serie.index[0])} a {mes_anio(serie.index[-1])}"


def encabezado(texto):
    """Imprime un título visible en la terminal."""
    print("\n" + "=" * 70)
    print(f" {texto}")
    print("=" * 70)


# --- 3.2 Datos ---------------------------------------------------------
def cargar_fix():
    """Lee el Excel de Banxico y regresa la serie mensual completa del FIX."""
    crudo = pd.read_excel(RUTA_EXCEL, header=None, skiprows=FILAS_ENCABEZADO,
                          names=["fecha", "fix"])
    crudo["fecha"] = pd.to_datetime(crudo["fecha"], errors="coerce")
    crudo["fix"] = pd.to_numeric(crudo["fix"], errors="coerce")
    crudo = crudo.dropna()
    crudo = crudo[crudo["fix"] > 0]          # ene-1991 a oct-1991 vienen en 0
    return crudo.set_index("fecha")["fix"].asfreq("MS")


def obtener_muestras(fix):
    """Divide la serie en las muestras definidas en la PARTE 1."""
    return {
        "entrenamiento": fix[INICIO_ENTRENAMIENTO:FIN_ENTRENAMIENTO],
        "validacion":    fix[INICIO_VALIDACION:FIN_VALIDACION],
        "estim_final":   fix[INICIO_ENTRENAMIENTO:FIN_VALIDACION],
        "obs_fuera":     fix[OBS_FUERA:OBS_FUERA],
    }


def variables_intervencion(indice, fechas=None):
    """Matriz de escalones S_t para las fechas de intervención (Paso 4b).
       S_t = 0 antes de la fecha y 1 desde la fecha en adelante.
       En el modelo con d = 1, un escalón en Z equivale a un pulso en nabla Z.
       Regresa None si no hay intervenciones."""
    fechas = INTERVENCIONES if fechas is None else fechas
    if not fechas:
        return None
    columnas = {}
    for f in fechas:
        f = pd.Timestamp(f)
        columnas[f"escalon_{MESES[f.month - 1]}{f.year}"] = (indice >= f).astype(float)
    return pd.DataFrame(columnas, index=indice)


def diferenciar(serie, d):
    """Aplica d diferencias (nabla^d) a una serie o tabla y quita los datos perdidos."""
    for _ in range(d):
        serie = serie.diff()
    return serie.dropna()


class ResultadoARIMA:
    """Resultado de statsmodels con los errores estándar recalculados.
       Con variables de intervención, statsmodels calcula mal el hessiano (errores
       estándar absurdos o cero), aunque las estimaciones y la verosimilitud son
       correctas. Aquí la matriz de covarianzas se obtiene como la inversa del
       hessiano numérico de la log-verosimilitud (diferencias centrales):
           Cov(beta_hat) = [ -d^2 lnL / d beta d beta' ]^(-1)
       Todo lo demás (params, llf, aic, resid, get_forecast, ...) es el de statsmodels."""

    def __init__(self, res):
        self._res = res
        self._cov = None

    def __getattr__(self, nombre):
        return getattr(self._res, nombre)

    def cov_params(self):
        if self._cov is None:
            x = self._res.params.values.astype(float)
            f = lambda v: self._res.model.loglike(v)
            k = len(x)
            h = 1e-4 * np.maximum(np.abs(x), 1.0)
            H = np.zeros((k, k))
            for i in range(k):
                for j in range(i, k):
                    ei, ej = np.zeros(k), np.zeros(k)
                    ei[i], ej[j] = h[i], h[j]
                    H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej)
                                         - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            nombres = self._res.params.index
            self._cov = pd.DataFrame(np.linalg.inv(-H), index=nombres, columns=nombres)
        return self._cov

    @property
    def bse(self):
        return pd.Series(np.sqrt(np.abs(np.diag(self.cov_params().values))),
                         index=self._res.params.index)

    @property
    def tvalues(self):
        return self._res.params / self.bse

    @property
    def pvalues(self):
        return pd.Series(2 * (1 - stats.norm.cdf(np.abs(self.tvalues.values))),
                         index=self._res.params.index)


def estimar_arima(Z, orden, fechas=None):
    """Estima un ARIMA(p,d,q) sin constante por máxima verosimilitud exacta, como en
       Box-Jenkins: se diferencia la serie d veces y se estima el ARMA(p,q) de nabla^d Z.
       Las intervenciones entran como escalones S_t en Z; al diferenciar se vuelven
       pulsos en nabla Z, con el mismo coeficiente omega.
       Los errores estándar se recalculan con el hessiano numérico (ver ResultadoARIMA).
       Los residuos empiezan en el segundo mes (se pierde un dato por diferenciar)."""
    from statsmodels.tsa.arima.model import ARIMA
    p, d, q = orden
    W = diferenciar(Z, d)
    X = variables_intervencion(Z.index, fechas)
    if X is not None:
        X = diferenciar(X, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ResultadoARIMA(ARIMA(W, exog=X, order=(p, 0, q), trend="n").fit())

def pronostico_niveles(res, Z, pasos):
    """Pronóstico de Z_{T+h}, h = 1..pasos, a partir del ARMA estimado para nabla Z (d = 1).
       Media:     Z_T + suma de los pronósticos de nabla Z
       Varianza:  sigma^2_a * suma_{j<h} Psi_j^2, con Psi_j = psi_0 + ... + psi_j
                  (pesos psi del ARMA acumulados por la diferencia).
       Después de la intervención los escalones ya no cambian: los pulsos futuros valen 0."""
    from statsmodels.tsa.arima_process import arma2ma
    nombres_exog = [n for n in res.params.index if n.startswith("escalon")]
    exog_futuro = np.zeros((pasos, len(nombres_exog))) if nombres_exog else None
    media_w = res.get_forecast(steps=pasos, exog=exog_futuro).predicted_mean
    phi = [res.params[n] for n in res.params.index if n.startswith("ar.")]
    theta = [res.params[n] for n in res.params.index if n.startswith("ma.")]
    psi = arma2ma(np.r_[1, -np.array(phi)], np.r_[1, np.array(theta)], lags=pasos)
    Psi = np.cumsum(psi)
    ee = np.sqrt(res.params["sigma2"] * np.cumsum(Psi ** 2))
    return pd.DataFrame({"Pronóstico": Z.iloc[-1] + np.cumsum(media_w.values),
                         "Error estándar": ee}, index=media_w.index)


def regla_empirica(e, alfa, simulaciones=20000, semilla=2026):
    """Regla empírica 68-95-99.7 (guía de estudio, sección 10.3), con un criterio objetivo
       de "qué tan cerca" deben estar los porcentajes:
       se simulan muchas muestras normales del MISMO tamaño que los residuos, se
       estandarizan igual (con su propia media y desviación estándar) y se ve en qué
       rango cae el porcentaje dentro de ±1, ±2 y ±3 desv. est. en el (1 - alfa)% de ellas.
       H0: los residuos son normales. La banda es compatible si el % observado cae
       dentro de ese rango. La regla se cumple si las tres bandas son compatibles."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    z = (e - e.mean()) / e.std(ddof=1)
    rng = np.random.default_rng(semilla)
    x = rng.standard_normal((simulaciones, n))
    zs = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
    filas = []
    for j, teorico in {1: 68.27, 2: 95.45, 3: 99.73}.items():
        obs = 100 * np.mean(np.abs(z) <= j)
        sim = 100 * np.mean(np.abs(zs) <= j, axis=1)
        li, ls = np.percentile(sim, [100 * alfa / 2, 100 * (1 - alfa / 2)])
        p = min(1.0, 2 * min(np.mean(sim >= obs - 1e-9), np.mean(sim <= obs + 1e-9)))
        filas.append({"Banda": f"±{j}", "% observado": round(obs, 1), "% teórico": teorico,
                      "Límite inferior": round(li, 1), "Límite superior": round(ls, 1),
                      "Valor p": round(p, 4),
                      "¿Compatible?": "Sí" if li <= obs <= ls else "No"})
    tabla = pd.DataFrame(filas)
    return tabla, bool((tabla["¿Compatible?"] == "Sí").all())


# --- 3.3 Guardar resultados ---------------------------------------------
def guardar_figura(fig, nombre):
    """Guarda una gráfica en la carpeta de figuras y la cierra."""
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")


def ruta_resultado(nombre):
    """Ruta dentro de la carpeta de resultados (la crea si no existe)."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    return os.path.join(CARPETA_RESULTADOS, nombre)


def formato_fechas(ax, cada_anios=None, con_mes=False):
    """Eje horizontal con fechas en español: '2026' o 'oct-26'."""
    if cada_anios is not None:
        ax.xaxis.set_major_locator(mdates.YearLocator(cada_anios))

    def _f(x, pos=None):
        f = mdates.num2date(x)
        return f"{MESES[f.month - 1]}-{str(f.year)[2:]}" if con_mes else str(f.year)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_f))


# --- 3.4 FAC, FACP y bandas de significancia --------------------------
def fac(x, K):
    """FAC muestral (guía de estudio, cap. 4):
       r_k = sum (x_t - media)(x_{t+k} - media) / sum (x_t - media)^2"""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    den = np.sum(d ** 2)
    return np.array([np.sum(d[:-k] * d[k:]) / den for k in range(1, K + 1)])


def facp(r):
    """FACP muestral con el algoritmo de Durbin-Levinson (ecuaciones de Yule-Walker).
       Recibe las autocorrelaciones r_1, ..., r_K y regresa phi_11, ..., phi_KK."""
    K = len(r)
    phi = np.zeros((K + 1, K + 1))
    salida = np.zeros(K)
    phi[1, 1] = r[0]
    salida[0] = r[0]
    for k in range(2, K + 1):
        num = r[k - 1] - np.sum(phi[k - 1, 1:k] * r[k - 2::-1][:k - 1])
        den = 1 - np.sum(phi[k - 1, 1:k] * r[:k - 1])
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        salida[k - 1] = phi[k, k]
    return salida


def valor_z(alfa):
    """Valor crítico normal de dos colas: 1.96 si alfa = 0.05; 2.576 si alfa = 0.01."""
    return stats.norm.ppf(1 - alfa / 2)


def bandas_bartlett(r, N, alfa):
    """Banda de la prueba de Bartlett para cada r_k:
       z * ee(r_k), con ee(r_k) = raíz[(1 + 2 * sum_{j<k} r_j^2) / N]"""
    ee = np.array([np.sqrt((1 + 2 * np.sum(r[:k - 1] ** 2)) / N)
                   for k in range(1, len(r) + 1)])
    return valor_z(alfa) * ee


def bartlett_secuencial(r, N, alfa):
    """Prueba de Bartlett secuencial para identificar q (guía de estudio, cap. 4):
       Paso 1: se supone q = 0 y se prueba r_1 con la banda z / raíz(N).
       Paso 2: si r_1 es significativa, se supone q = 1 y se prueba r_2 con
               z * raíz[(1 + 2 r_1^2) / N]; y así sucesivamente.
       En cuanto una r_k no es significativa, la FAC "se corta" en q = k - 1
       y para los retrasos siguientes se usa esa misma banda.
       Regresa las bandas de cada retraso y el valor de q identificado."""
    z = valor_z(alfa)
    bandas = np.zeros(len(r))
    q = 0
    cortada = False
    for k in range(1, len(r) + 1):
        bandas[k - 1] = z * np.sqrt((1 + 2 * np.sum(r[:q] ** 2)) / N)
        if not cortada and abs(r[k - 1]) > bandas[k - 1]:
            q = k                      # r_k significativa: se agrega al supuesto MA(q)
        else:
            cortada = True             # primera r_k no significativa: la FAC se corta
    return bandas, q


def banda_facp(N, alfa):
    """Banda de significancia de la FACP: z / raíz(N), con N = número de datos de la serie
       diferenciada (N - d en la notación de la guía)."""
    return valor_z(alfa) / np.sqrt(N)


# --- 3.5 Prueba de Dickey-Fuller aumentada ----------------------------
ESPECIFICACIONES_ADF = {"n": "Sin constante ni tendencia",
                        "c": "Con constante",
                        "ct": "Con constante y tendencia"}


def prueba_adf(serie, alfa):
    """Corre la prueba ADF en las tres especificaciones (rezagos elegidos con AIC).
       H0: raíz unitaria (no estacionaria)  vs  H1: estacionaria.
       Se rechaza H0 si tau < valor crítico al nivel alfa (equivale a valor p < alfa)."""
    from statsmodels.tsa.stattools import adfuller
    nivel = f"{alfa * 100:g}%"
    filas = []
    for clave, nombre in ESPECIFICACIONES_ADF.items():
        tau, pval, rezagos, nobs, criticos, _ = adfuller(
            np.asarray(serie, dtype=float),
            regression=clave,
            autolag="AIC"
            )
        # Con alfa de 1%, 5% o 10% se compara contra el valor crítico de tablas;
        # con cualquier otro alfa se usa el valor p.
        rechaza = tau < criticos[nivel] if nivel in criticos else pval < alfa
        filas.append({"Especificación": nombre, "Clave": clave,
                      "tau": round(tau, 4), "Valor p": round(pval, 4),
                      "Rezagos": rezagos, "Obs.": nobs,
                      "VC 1%": round(criticos["1%"], 4), "VC 5%": round(criticos["5%"], 4),
                      "VC 10%": round(criticos["10%"], 4),
                      f"Decisión ({nivel})": "Se rechaza H0" if rechaza else "No se rechaza H0"})
    return pd.DataFrame(filas)