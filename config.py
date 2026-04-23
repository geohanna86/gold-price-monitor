# ============================================================
# config.py — Configuración centralizada para el sistema de monitoreo de oro
# Gold Price Monitor — Phase 1 Configuration
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List


# ── Configuración de datos ──────────────────────────────────────
GOLD_TICKER_FUTURES  = "GC=F"      # Contratos de futuros de oro en Yahoo Finance
GOLD_TICKER_FOREX    = "XAUUSD=X"  # Precio de oro al contado (Spot)
DXY_TICKER           = "DX-Y.NYB"  # Índice del dólar estadounidense
SILVER_TICKER        = "SI=F"      # Plata (correlación positiva con el oro)

# Marcos de tiempo soportados en yfinance
VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

# Períodos soportados
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]


# ── Configuración de indicadores técnicos ────────────────────────────────
@dataclass
class IndicatorConfig:
    # ─ Trend ─
    ema_short:  int = 20    # EMA corto plazo
    ema_mid:    int = 50    # EMA plazo medio
    ema_long:   int = 200   # EMA largo plazo (tendencia principal)

    # ─ Momentum ─
    rsi_period:       int = 14  # RSI
    rsi_overbought:   int = 70  # Sobrecompra
    rsi_oversold:     int = 30  # Sobreventa
    stochrsi_period:  int = 14  # Stochastic RSI
    willr_period:     int = 14  # Williams %R
    willr_overbought: int = -20 # Sobrecompra (cerca de 0)
    willr_oversold:   int = -80 # Sobreventa (cerca de -100)

    # ─ MACD ─
    macd_fast:   int = 12
    macd_slow:   int = 26
    macd_signal: int = 9

    # ─ Volatilidad ─
    atr_period:    int = 14  # Average True Range
    bb_period:     int = 20  # Bollinger Bands
    bb_std:        float = 2.0

    # ─ Volumen ─
    vwap_enabled: bool = True


# ── Configuración de datos simulados (Mock) ──────────────────────
@dataclass
class MockConfig:
    n_rows:        int   = 300          # Número de velas simuladas
    start_price:   float = 2050.0       # Precio inicial del oro (dólar/onza)
    daily_drift:   float = 0.0003       # Incremento diario gradual (0.03%)
    daily_vol:     float = 0.010        # Volatilidad diaria (1%)
    seed:          int   = 42           # Para garantizar resultados reproducibles


# ── Configuración de salida ────────────────────────────────────────
@dataclass
class OutputConfig:
    save_csv:      bool  = True
    csv_path:      str   = "output/gold_data_with_indicators.csv"
    show_summary:  bool  = True
    decimal_places: int  = 4


# ── Configuración unificada que reúne todas las configuraciones ──────────────────────────
@dataclass
class AppConfig:
    mode:       str           = "mock"   # "mock" o "live"
    indicator:  IndicatorConfig = field(default_factory=IndicatorConfig)
    mock:       MockConfig      = field(default_factory=MockConfig)
    output:     OutputConfig    = field(default_factory=OutputConfig)


# Configuración predeterminada para uso directo
DEFAULT_CONFIG = AppConfig()
