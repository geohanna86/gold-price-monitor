# ============================================================
# indicators.py — Motor de indicadores técnicos
# Gold Price Monitor — Phase 1
#
# Indicadores soportados:
#   Trend:      EMA(20/50/200), MACD
#   Momentum:   RSI(14), StochRSI, Williams %R
#   Volatility: ATR(14), Bollinger Bands(20)
#   Volume:     OBV, VWAP (diario)
#   Signals:    Sistema de señales integrado (Confluence Score)
#
# Librería utilizada: ta (pip install ta)
# Documentación oficial: https://technical-analysis-library-in-python.readthedocs.io
# ============================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ta  # noqa: F401
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger("TechnicalIndicators")


class TechnicalIndicators:
    """
    Motor de indicadores técnicos — utiliza el patrón Builder.

    Se puede usar:
        df_with_indicators = (
            TechnicalIndicators(df, config)
            .add_trend_indicators()
            .add_momentum_indicators()
            .add_volatility_indicators()
            .add_volume_indicators()
            .get_dataframe()
        )

    O de forma breve:
        df_with_indicators = TechnicalIndicators(df).add_all().get_dataframe()
    """

    def __init__(self, df: pd.DataFrame, config: AppConfig = DEFAULT_CONFIG):
        if not _TA_AVAILABLE:
            raise ImportError(
                "Librería ta no instalada.\n"
                "Instálala: pip install ta"
            )

        self._validate_input(df)
        self.df  = df.copy()
        self.cfg = config.indicator
        logger.info(
            f"TechnicalIndicators initialized | Filas: {len(self.df)} | "
            f"De: {self.df.index[0].date()} a: {self.df.index[-1].date()}"
        )

    # ─────────────────────────────────────────────────────────
    # Verificación de validez de entrada
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas requeridas faltantes: {missing}")
        if len(df) < 30:
            raise ValueError(
                f"Datos insuficientes: solo {len(df)} filas. "
                "Mínimo 30 filas para calcular indicadores."
            )

    # ─────────────────────────────────────────────────────────
    # Sección 1: Indicadores de tendencia (Trend Indicators)
    # ─────────────────────────────────────────────────────────
    def add_trend_indicators(self) -> "TechnicalIndicators":
        """
        Añade: EMA(20), EMA(50), EMA(200), MACD(12,26,9)
        Requisitos mínimos: 26 filas para MACD, 200 filas para EMA200
        """
        close = self.df["Close"]
        cfg   = self.cfg

        # ── EMA ──
        self.df["EMA_20"]  = ta.trend.ema_indicator(close, window=cfg.ema_short)
        self.df["EMA_50"]  = ta.trend.ema_indicator(close, window=cfg.ema_mid)
        self.df["EMA_200"] = ta.trend.ema_indicator(close, window=cfg.ema_long)

        # ── MACD ──
        # ta.trend.macd*: funciones separadas para cada componente (line, signal, histogram)
        self.df["MACD"]        = ta.trend.macd(
            close, window_slow=cfg.macd_slow, window_fast=cfg.macd_fast
        )
        self.df["MACD_Signal"] = ta.trend.macd_signal(
            close,
            window_slow=cfg.macd_slow,
            window_fast=cfg.macd_fast,
            window_sign=cfg.macd_signal,
        )
        self.df["MACD_Hist"] = ta.trend.macd_diff(
            close,
            window_slow=cfg.macd_slow,
            window_fast=cfg.macd_fast,
            window_sign=cfg.macd_signal,
        )

        # ── EMA Trend Signal: posición del precio relativa a EMAs ──
        self.df["EMA_Trend"] = self._compute_ema_trend()

        logger.info("✅ Indicadores de tendencia (Trend) añadidos: EMA_20, EMA_50, EMA_200, MACD")
        return self

    def _compute_ema_trend(self) -> pd.Series:
        """
        Devuelve un valor de {-1, 0, 1} para determinar la tendencia:
          +1 = subida  (Close > EMA20 > EMA50 > EMA200)
          -1 = bajada  (Close < EMA20 < EMA50 < EMA200)
           0 = neutro (no definido)
        """
        c   = self.df["Close"]
        e20 = self.df.get("EMA_20")
        e50 = self.df.get("EMA_50")
        e200= self.df.get("EMA_200")

        if e20 is None or e50 is None or e200 is None:
            return pd.Series(0, index=self.df.index)

        bullish = (c > e20) & (e20 > e50) & (e50 > e200)
        bearish = (c < e20) & (e20 < e50) & (e50 < e200)

        trend = pd.Series(0, index=self.df.index, dtype=int)
        trend[bullish] =  1
        trend[bearish] = -1
        return trend

    # ─────────────────────────────────────────────────────────
    # Sección 2: Indicadores de momentum (Momentum Indicators)
    # ─────────────────────────────────────────────────────────
    def add_momentum_indicators(self) -> "TechnicalIndicators":
        """
        Añade: RSI(14), StochRSI(14), Williams %R(14)
        """
        close = self.df["Close"]
        high  = self.df["High"]
        low   = self.df["Low"]
        cfg   = self.cfg

        # ── RSI ──
        self.df["RSI"] = ta.momentum.rsi(close, window=cfg.rsi_period)

        # ── Stochastic RSI ──
        # ta.momentum.stochrsi_k / stochrsi_d: dos funciones separadas
        self.df["StochRSI_K"] = ta.momentum.stochrsi_k(
            close,
            window=cfg.stochrsi_period,
            smooth1=3,
            smooth2=3,
        )
        self.df["StochRSI_D"] = ta.momentum.stochrsi_d(
            close,
            window=cfg.stochrsi_period,
            smooth1=3,
            smooth2=3,
        )

        # ── Williams %R ──
        self.df["WILLR"] = ta.momentum.williams_r(
            high, low, close, lbp=cfg.willr_period
        )

        # ── Señales de momentum integradas ──
        self.df["Momentum_Signal"] = self._compute_momentum_signal()

        logger.info("✅ Indicadores de momentum (Momentum) añadidos: RSI, StochRSI, Williams %R")
        return self

    def _compute_momentum_signal(self) -> pd.Series:
        """
        Devuelve una señal integrada entre RSI y Williams %R:
          +1 = fuerza compradora (RSI < Oversold o WillR < -80)
          -1 = fuerza vendedora  (RSI > Overbought o WillR > -20)
           0 = neutro
        """
        signal = pd.Series(0, index=self.df.index, dtype=int)

        if "RSI" in self.df.columns:
            signal[self.df["RSI"] < self.cfg.rsi_oversold]    +=  1
            signal[self.df["RSI"] > self.cfg.rsi_overbought]  += -1

        if "WILLR" in self.df.columns:
            signal[self.df["WILLR"] < self.cfg.willr_oversold]    +=  1
            signal[self.df["WILLR"] > self.cfg.willr_overbought]  += -1

        return signal.clip(-2, 2)

    # ─────────────────────────────────────────────────────────
    # Sección 3: Indicadores de volatilidad (Volatility Indicators)
    # ─────────────────────────────────────────────────────────
    def add_volatility_indicators(self) -> "TechnicalIndicators":
        """
        Añade: ATR(14), Bollinger Bands(20, 2σ)
        """
        close = self.df["Close"]
        high  = self.df["High"]
        low   = self.df["Low"]
        cfg   = self.cfg

        # ── ATR (Average True Range) ──
        # Nota: la librería ta pone ceros en el período de calentamiento → los reemplazamos con NaN
        atr_raw = ta.volatility.average_true_range(
            high, low, close, window=cfg.atr_period
        )
        self.df["ATR"] = atr_raw.replace(0, float("nan"))

        # ── ATR normalizado por precio (Normalized) ──
        self.df["ATR_Pct"] = (self.df["ATR"] / close * 100).round(4)

        # ── Bollinger Bands ──
        # La librería ta devuelve cada componente de una función separada
        self.df["BB_Lower"]   = ta.volatility.bollinger_lband(
            close, window=cfg.bb_period, window_dev=cfg.bb_std
        )
        self.df["BB_Mid"]     = ta.volatility.bollinger_mavg(
            close, window=cfg.bb_period
        )
        self.df["BB_Upper"]   = ta.volatility.bollinger_hband(
            close, window=cfg.bb_period, window_dev=cfg.bb_std
        )
        self.df["BB_Width"]   = ta.volatility.bollinger_wband(
            close, window=cfg.bb_period, window_dev=cfg.bb_std
        )
        self.df["BB_Percent"] = ta.volatility.bollinger_pband(
            close, window=cfg.bb_period, window_dev=cfg.bb_std
        )

        # ── Stop Loss propuesto basado en ATR ──
        self.df["Stop_Loss_Long"]  = (close - 1.5 * self.df["ATR"]).round(2)
        self.df["Stop_Loss_Short"] = (close + 1.5 * self.df["ATR"]).round(2)

        logger.info("✅ Indicadores de volatilidad (Volatility) añadidos: ATR, Bollinger Bands, Stop Loss")
        return self

    # ─────────────────────────────────────────────────────────
    # Sección 4: Indicadores de volumen (Volume Indicators)
    # ─────────────────────────────────────────────────────────
    def add_volume_indicators(self) -> "TechnicalIndicators":
        """
        Añade: OBV, VWAP (diario)
        """
        close  = self.df["Close"]
        high   = self.df["High"]
        low    = self.df["Low"]
        volume = self.df["Volume"]

        # ── OBV (On-Balance Volume) ──
        self.df["OBV"] = ta.volume.on_balance_volume(close, volume)

        # ── VWAP diario (Volume Weighted Average Price) ──
        # Calculado manualmente para asegurar precisión con datos diarios
        typical_price    = (high + low + close) / 3
        tpv              = typical_price * volume  # Precio × Volumen
        cumulative_tpv   = tpv.cumsum()
        cumulative_vol   = volume.cumsum()
        self.df["VWAP"]  = (cumulative_tpv / cumulative_vol).round(2)

        # ── Posición del precio relativa al VWAP ──
        self.df["VWAP_Signal"] = np.where(
            close > self.df["VWAP"],  1,   # Precio arriba VWAP = presión compradora
            np.where(
                close < self.df["VWAP"], -1,  # Precio abajo VWAP = presión vendedora
                0
            )
        )

        logger.info("✅ Indicadores de volumen (Volume) añadidos: OBV, VWAP")
        return self

    # ─────────────────────────────────────────────────────────
    # Sección 5: Sistema de señales integrado (Confluence Score)
    # ─────────────────────────────────────────────────────────
    def add_confluence_score(self) -> "TechnicalIndicators":
        """
        Calcula la puntuación de confluencia (Confluence Score) entre todos los indicadores.
        El valor varía de -4 (venta fuerte) a +4 (compra fuerte).

        Lógica:
         - EMA Trend:         +1 subida, -1 bajada
         - Momentum Signal:   +1 sobreventa (oportunidad compra), -1 sobrecompra
         - MACD Histogram:    +1 positivo y ascendente, -1 negativo y descendente
         - VWAP Signal:       +1 arriba VWAP, -1 abajo VWAP
        """
        score = pd.Series(0.0, index=self.df.index)

        if "EMA_Trend" in self.df.columns:
            score += self.df["EMA_Trend"]

        if "Momentum_Signal" in self.df.columns:
            score += self.df["Momentum_Signal"].clip(-1, 1)

        if "MACD_Hist" in self.df.columns:
            macd_hist = self.df["MACD_Hist"]
            score += np.where(
                macd_hist > 0,  1,
                np.where(macd_hist < 0, -1, 0)
            )

        if "VWAP_Signal" in self.df.columns:
            score += self.df["VWAP_Signal"]

        self.df["Confluence_Score"] = score.round(0).astype(int)

        # ── Traducción de puntuación a etiqueta textual ──
        def score_to_label(s: int) -> str:
            if s >= 3:  return "🟢 Compra fuerte"
            if s == 2:  return "🟩 Compra"
            if s == 1:  return "🔵 Inclinación compra"
            if s == -1: return "🟡 Inclinación venta"
            if s == -2: return "🟧 Venta"
            if s <= -3: return "🔴 Venta fuerte"
            return "⚪ Neutro"

        self.df["Signal_Label"] = self.df["Confluence_Score"].apply(score_to_label)

        logger.info("✅ Confluence Score calculado")
        return self

    # ─────────────────────────────────────────────────────────
    # Funciones genéricas
    # ─────────────────────────────────────────────────────────
    def add_all(self) -> "TechnicalIndicators":
        """Añade todos los indicadores de una sola vez."""
        return (
            self.add_trend_indicators()
                .add_momentum_indicators()
                .add_volatility_indicators()
                .add_volume_indicators()
                .add_confluence_score()
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Devuelve el DataFrame completo con todos los indicadores."""
        return self.df

    def get_latest_signals(self) -> pd.Series:
        """Devuelve la última lectura de todos los indicadores (la vela más reciente)."""
        signal_cols = [
            "Close", "EMA_20", "EMA_50", "EMA_200",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "StochRSI_K", "StochRSI_D",
            "WILLR", "ATR", "ATR_Pct",
            "BB_Lower", "BB_Upper", "BB_Percent",
            "OBV", "VWAP",
            "Confluence_Score", "Signal_Label",
            "Stop_Loss_Long", "Stop_Loss_Short",
        ]
        available_cols = [c for c in signal_cols if c in self.df.columns]
        latest = self.df[available_cols].iloc[-1]
        return latest

    def get_indicator_summary(self) -> pd.DataFrame:
        """Devuelve una tabla resumen de la señal actual con interpretación."""
        last = self.get_latest_signals()

        summary_data = []

        # RSI
        if "RSI" in last.index:
            rsi_val  = last["RSI"]
            rsi_status = (
                "Sobrecompra ⚠️" if rsi_val > 70
                else "Sobreventa ✅" if rsi_val < 30
                else "Neutro"
            )
            summary_data.append(("RSI(14)", f"{rsi_val:.1f}", rsi_status))

        # MACD
        if "MACD_Hist" in last.index:
            hist = last["MACD_Hist"]
            summary_data.append((
                "MACD Histogram",
                f"{hist:.2f}",
                "Ascendente 📈" if hist > 0 else "Descendente 📉"
            ))

        # Bollinger Bands
        if "BB_Percent" in last.index:
            bbp = last["BB_Percent"]
            summary_data.append((
                "BB Position",
                f"{bbp:.2%}",
                "Cerca del límite superior" if bbp > 0.8 else
                "Cerca del límite inferior" if bbp < 0.2 else "Dentro del rango"
            ))

        # Williams %R
        if "WILLR" in last.index:
            w = last["WILLR"]
            summary_data.append((
                "Williams %R",
                f"{w:.1f}",
                "Sobrecompra" if w > -20 else
                "Sobreventa" if w < -80 else "Neutro"
            ))

        # ATR
        if "ATR" in last.index and "Close" in last.index:
            atr_pct = last["ATR"] / last["Close"] * 100
            summary_data.append((
                "ATR (volatilidad diaria)",
                f"${last['ATR']:.2f} ({atr_pct:.2f}%)",
                "Alto" if atr_pct > 1.5 else "Moderado"
            ))

        # Confluence
        if "Confluence_Score" in last.index:
            summary_data.append((
                "🎯 Confluence Score",
                str(int(last["Confluence_Score"])),
                last.get("Signal_Label", "—")
            ))

        return pd.DataFrame(
            summary_data,
            columns=["Indicador", "Valor", "Interpretación"]
        )
