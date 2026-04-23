# ============================================================
# feature_engineer.py — Ingeniería de características para el modelo cuantitativo
# Gold Price Monitor — Phase 2
#
# Principio de oro: sin fugas de datos futuros (No Leakage)
# Cada característica depende solo de datos en t-1 o anteriores
# Target depende de t+1 (lo que predeciremos)
# ============================================================

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureEngineer")


# ── Lista de características finales para el modelo ──────────────────────────
FEATURE_COLUMNS = [
    # Retornos de precio (Price Returns)
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    # Ratios EMA (señales de tendencia)
    "Price_vs_EMA20", "Price_vs_EMA50", "Price_vs_EMA200",
    "EMA20_vs_EMA50", "EMA50_vs_EMA200",
    # Indicadores de momentum
    "RSI", "RSI_Lag1", "RSI_Lag2",
    "MACD_Hist", "MACD_Hist_Lag1",
    "StochRSI_K", "WILLR",
    # Indicadores de volatilidad
    "ATR_Pct", "BB_Percent",
    # Indicadores de volumen
    "OBV_Change_Pct", "VWAP_Signal",
    # Confluence
    "Confluence_Score",
]

TARGET_COLUMN  = "Target"
RETURN_HORIZON = 1   # Predecimos el retorno del día siguiente


class FeatureEngineer:
    """
    Transforma un DataFrame de la fase 1 (datos + indicadores)
    en una matriz de características lista para el modelo.

    Reglas estrictas:
     1. shift(1) en cada característica para evitar fuga de datos del día actual
     2. Target = señal de retorno en t+1
     3. dropna() completo antes de la salida
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_threshold: float = 0.003,  # Umbral de clasificación 0.3% (arriba compra, abajo venta)
        return_horizon: int = RETURN_HORIZON,
    ):
        self.df               = df.copy()
        self.target_threshold = target_threshold
        self.return_horizon   = return_horizon
        self._engineered_df: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────
    # Construcción de características
    # ─────────────────────────────────────────────────────────
    def build_features(self) -> pd.DataFrame:
        """
        Función principal — construye todas las características y añade la columna objetivo.
        Devuelve un DataFrame limpio sin NaN.
        """
        df = self.df.copy()

        # ── 1. Retornos de precio ──────────────────────────────────
        df["Return_1d"]  = df["Close"].pct_change(1)
        df["Return_3d"]  = df["Close"].pct_change(3)
        df["Return_5d"]  = df["Close"].pct_change(5)
        df["Return_10d"] = df["Close"].pct_change(10)

        # ── 2. Ratios EMA (posición del precio relativa a promedios) ─────────
        for ema_col, feature_col in [
            ("EMA_20",  "Price_vs_EMA20"),
            ("EMA_50",  "Price_vs_EMA50"),
            ("EMA_200", "Price_vs_EMA200"),
        ]:
            if ema_col in df.columns:
                df[feature_col] = (df["Close"] / df[ema_col] - 1).round(6)

        if "EMA_20" in df.columns and "EMA_50" in df.columns:
            df["EMA20_vs_EMA50"] = (df["EMA_20"] / df["EMA_50"] - 1).round(6)

        if "EMA_50" in df.columns and "EMA_200" in df.columns:
            df["EMA50_vs_EMA200"] = (df["EMA_50"] / df["EMA_200"] - 1).round(6)

        # ── 3. Lag Features (valores de indicadores en t-1, t-2) ───────
        # Principio: shift(1) = valor de ayer → sin fuga de datos del día actual
        if "RSI" in df.columns:
            df["RSI_Lag1"] = df["RSI"].shift(1)
            df["RSI_Lag2"] = df["RSI"].shift(2)

        if "MACD_Hist" in df.columns:
            df["MACD_Hist_Lag1"] = df["MACD_Hist"].shift(1)

        # ── 4. OBV Change (cambio porcentual en volumen compra/venta) ──
        if "OBV" in df.columns:
            df["OBV_Change_Pct"] = df["OBV"].pct_change(1).replace(
                [np.inf, -np.inf], np.nan
            )

        # ── 5. Columna objetivo (Target) ──────────────────────────
        # Futuro: retorno del cierre del día siguiente
        forward_return = df["Close"].shift(-self.return_horizon) / df["Close"] - 1

        df[TARGET_COLUMN] = np.where(
            forward_return >  self.target_threshold,  1,   # compra
            np.where(
                forward_return < -self.target_threshold, -1,  # venta
                0                                             # neutro
            )
        )

        # Mantener el retorno real para el Backtester
        df["Forward_Return"] = forward_return

        # ── 6. Limpieza: eliminar filas con NaN ─────────────────────
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        keep_cols = available_features + [TARGET_COLUMN, "Forward_Return",
                                          "Close", "Open", "High", "Low", "Volume"]
        df_clean = df[keep_cols].dropna()

        # Eliminar últimas n filas (sin objetivo debido a shift(-n))
        df_clean = df_clean.iloc[:-self.return_horizon]

        logger.info(
            f"✅ Ingeniería de características completada | "
            f"Características: {len(available_features)} | "
            f"Filas: {len(df_clean)} | "
            f"Distribución del objetivo:\n"
            f"  Compra (+1): {(df_clean[TARGET_COLUMN]==1).sum()} "
            f"({(df_clean[TARGET_COLUMN]==1).mean()*100:.1f}%)\n"
            f"  Venta  (-1): {(df_clean[TARGET_COLUMN]==-1).sum()} "
            f"({(df_clean[TARGET_COLUMN]==-1).mean()*100:.1f}%)\n"
            f"  Neutro (0): {(df_clean[TARGET_COLUMN]==0).sum()} "
            f"({(df_clean[TARGET_COLUMN]==0).mean()*100:.1f}%)"
        )

        self._engineered_df = df_clean
        return df_clean

    # ─────────────────────────────────────────────────────────
    # División de datos
    # ─────────────────────────────────────────────────────────
    def train_test_split(
        self, test_size: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide los datos en Train/Test en orden temporal (sin shuffle!).
        En datos financieros: el futuro siempre debe estar en Test.

        Devuelve: X_train, X_test, y_train, y_test
        """
        if self._engineered_df is None:
            self.build_features()

        df = self._engineered_df
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]

        X = df[available_features]
        y = df[TARGET_COLUMN]

        split_idx = int(len(df) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test  = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test  = y.iloc[split_idx:]

        logger.info(
            f"División de datos | "
            f"Entrenamiento: {len(X_train)} filas ({X_train.index[0].date()} → {X_train.index[-1].date()}) | "
            f"Prueba:  {len(X_test)} filas ({X_test.index[0].date()} → {X_test.index[-1].date()})"
        )

        return X_train, X_test, y_train, y_test

    def get_full_data(self) -> pd.DataFrame:
        """Devuelve el DataFrame completo con características y objetivo."""
        if self._engineered_df is None:
            self.build_features()
        return self._engineered_df

    def get_feature_names(self) -> List[str]:
        """Devuelve los nombres de las características realmente disponibles."""
        if self._engineered_df is None:
            self.build_features()
        return [c for c in FEATURE_COLUMNS if c in self._engineered_df.columns]
