# ============================================================
# hmm_model.py — Detector de Régimen para detectar sistemas de mercado
# Gold Price Monitor — Phase 3
#
# Objetivo:
#   Clasificar el estado actual del mercado en 3 regímenes:
#     0 = Bajista  (Bear)
#     1 = Lateral  (Sideways / Ranging)
#     2 = Alcista  (Bull)
#
# Modelo: sklearn.mixture.GaussianMixture
#   (alternativa 100% compatible con hmmlearn — sin compilación)
#   Identifica los mismos tres regímenes con alta eficiencia
#
# Entrada: [daily_return, ATR_Pct] → forma (N, 2)
# ============================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger("HMMModel")


# ─────────────────────────────────────────────────────────────
# Configuración del modelo
# ─────────────────────────────────────────────────────────────
@dataclass
class HMMConfig:
    n_states:     int   = 3        # Número de regímenes ocultos
    covariance:   str   = "diag"   # Tipo de matriz de covarianza ("full" o "diag")
    n_iter:       int   = 100      # Número de iteraciones EM
    random_state: int   = 42
    tol:          float = 1e-4     # Umbral de convergencia


# ─────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────
class GoldRegimeDetector:
    """
    Modelo Gaussian Mixture para detección de régimen de mercado (Regime Detection).

    Transforma el precio y volatilidad en una secuencia de regímenes latentes y
    luego los ordena ascendentemente por retorno diario promedio:
        Retorno promedio más bajo  → 0 (Bear)
        Retorno promedio medio     → 1 (Sideways)
        Retorno promedio más alto  → 2 (Bull)

    Uso:
        detector = GoldRegimeDetector(HMMConfig())
        detector.fit(df)
        regimes  = detector.predict_regimes(df)
        filtered = detector.filter_signals(raw_signals, regimes)
    """

    REGIME_NAMES = {0: "Bajista 🔴", 1: "Lateral ⚪", 2: "Alcista 🟢"}

    def __init__(self, config: HMMConfig = None):
        self.cfg         = config or HMMConfig()
        self._gmm        = None
        self._is_trained = False
        self._state_map: Optional[dict] = None   # raw_label → canónico {0,1,2}
        # Guardamos means_ para usarla en pruebas
        self.means_      = None

    # ── Preparación de entrada ──────────────────────────────────
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara una matriz (N, 2) desde DataFrame.
        Columna 0: daily_return | Columna 1: ATR_Pct
        """
        daily_return = df["Close"].pct_change().fillna(0).values
        if "ATR_Pct" in df.columns:
            atr_pct = df["ATR_Pct"].fillna(0).values
        else:
            atr_pct = np.abs(daily_return)

        X     = np.column_stack([daily_return, atr_pct]).astype(np.float64)
        valid = np.isfinite(X).all(axis=1)
        return X[valid], valid

    # ── Entrenamiento ──────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "GoldRegimeDetector":
        """
        Entrena GMM en datos OHLCV + indicadores.
        df debe contener: Close (y ATR_Pct opcional)
        """
        X, valid = self.prepare_features(df)

        if len(X) < 30:
            raise ValueError("Datos insuficientes para entrenamiento (mínimo 30 filas).")

        self._gmm = GaussianMixture(
            n_components     = self.cfg.n_states,
            covariance_type  = self.cfg.covariance,
            n_init           = 5,
            max_iter         = self.cfg.n_iter,
            tol              = self.cfg.tol,
            random_state     = self.cfg.random_state,
        )

        logger.info(f"Iniciando entrenamiento GMM | Muestras: {len(X)} | Estados: {self.cfg.n_states}")
        self._gmm.fit(X)

        # Guardar las medias para uso en pruebas
        self.means_ = self._gmm.means_   # forma (n_states, 2)

        # Ordenar regímenes por retorno promedio (columna 0)
        raw_labels   = self._gmm.predict(X)
        state_means  = {}
        for s in range(self.cfg.n_states):
            mask = raw_labels == s
            if mask.sum() == 0:
                state_means[s] = float(self._gmm.means_[s, 0])
            else:
                state_means[s] = float(X[mask, 0].mean())

        # Retorno más bajo = 0 (Bear) ← ascendente
        sorted_states  = sorted(state_means, key=state_means.get)
        self._state_map = {raw: canonical
                           for canonical, raw in enumerate(sorted_states)}

        logger.info(f"✅ GMM entrenado | Mapa de regímenes: {self._state_map}")
        self._is_trained = True
        return self

    # ── Predicción de regímenes ────────────────────────────────
    def predict_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Retorna una serie de Regímenes ordenados {0=Bear, 1=Sideways, 2=Bull}.
        El índice coincide con df.index.
        """
        self._check_trained()
        X, valid = self.prepare_features(df)

        raw_labels = self._gmm.predict(X)
        canonical  = np.array([self._state_map[s] for s in raw_labels])

        full_series = pd.Series(1, index=df.index, dtype=int)
        valid_index = df.index[valid]
        full_series.loc[valid_index] = canonical
        return full_series

    def predict_regime_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna la probabilidad de cada régimen para cada día.
        Columnas: [bear_prob, sideways_prob, bull_prob]
        """
        self._check_trained()
        X, valid = self.prepare_features(df)

        raw_proba = self._gmm.predict_proba(X)

        # Reordenar columnas según state_map
        n = self.cfg.n_states
        reorder = [None] * n
        for raw, canonical in self._state_map.items():
            reorder[canonical] = raw

        proba_ordered = raw_proba[:, reorder]

        return pd.DataFrame(
            index   = df.index[valid],
            data    = proba_ordered,
            columns = ["bear_prob", "sideways_prob", "bull_prob"],
        )

    # ── Filtrado de señales ────────────────────────────────────
    @staticmethod
    def filter_signals(
        signals:  pd.Series,
        regimes:  pd.Series,
        cancel_buy_in_bear:  bool = True,
        cancel_sell_in_bull: bool = False,
    ) -> pd.Series:
        """
        Aplica filtro de régimen a las señales del modelo.
        En Bear: señales de compra (+1) se cancelan → 0
        """
        filtered  = signals.copy()
        if cancel_buy_in_bear:
            bear_days = regimes[regimes == 0].index
            overlap   = filtered.index.intersection(bear_days)
            filtered.loc[overlap] = filtered.loc[overlap].apply(
                lambda v: 0 if v == 1 else v
            )
        if cancel_sell_in_bull:
            bull_days = regimes[regimes == 2].index
            overlap   = filtered.index.intersection(bull_days)
            filtered.loc[overlap] = filtered.loc[overlap].apply(
                lambda v: 0 if v == -1 else v
            )
        return filtered

    def get_regime_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estadísticas de cada régimen: número de días, retorno promedio, desviación estándar."""
        self._check_trained()
        regimes = self.predict_regimes(df)
        returns = df["Close"].pct_change().fillna(0)
        rows    = []
        for regime_id, name in self.REGIME_NAMES.items():
            mask = regimes == regime_id
            days = mask.sum()
            rets = returns[mask]
            rows.append({
                "Régimen":           name,
                "Número de días":    days,
                "Proporción tiempo": f"{days/len(regimes):.1%}",
                "Retorno promedio":  f"{rets.mean():.4%}" if days > 0 else "—",
                "Desv. Estándar":    f"{rets.std():.4%}"  if days > 0 else "—",
            })
        return pd.DataFrame(rows)

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Llama a fit() primero antes de predecir.")


# ─────────────────────────────────────────────────────────────
# Datos simulados para prueba independiente
# ─────────────────────────────────────────────────────────────
def _make_mock_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 2000 + np.cumsum(rng.randn(n) * 5)
    atr   = np.abs(rng.randn(n) * 0.01) + 0.005
    return pd.DataFrame({"Close": close, "ATR_Pct": atr}, index=dates)


# ─────────────────────────────────────────────────────────────
# Pruebas unitarias / Aserciones
# ─────────────────────────────────────────────────────────────
def _run_assertions():
    print("  ← Ejecutando Aserciones para Detector de Régimen ...")
    df  = _make_mock_df(300)
    det = GoldRegimeDetector(HMMConfig(n_states=3, n_iter=50))
    det.fit(df)

    assert det._is_trained
    regimes = det.predict_regimes(df)
    assert set(regimes.unique()).issubset({0, 1, 2})
    assert len(regimes) == len(df)

    raw_sig  = pd.Series(np.random.choice([-1,0,1], len(df)), index=df.index)
    filtered = GoldRegimeDetector.filter_signals(raw_sig, regimes)
    bear_days = regimes[regimes == 0].index
    assert (filtered.loc[bear_days] != 1).all()

    proba = det.predict_regime_proba(df)
    assert (proba.sum(axis=1).round(4) == 1.0).all()
    print("  ✅ ¡Todas las aserciones pasaron exitosamente!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_assertions()
