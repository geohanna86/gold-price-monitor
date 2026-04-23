# ============================================================
# trading_filters.py — Filtros de Trading Avanzados
# Gold Price Monitor — Phase 4
#
# Componentes:
#   1. SessionFilter → Identifica sesión de trading (LONDON/NEW_YORK/OVERLAP/ASIAN/CLOSED)
#   2. MultiTimeframeAnalyzer → Análisis multi-marco (H1, H4, D1)
#   3. DXYFilter → Filtro de sesgo del índice dólar
#   4. TradingFiltersManager → Orquestador principal
#   5. TradingContext → Resultado de evaluación
#
# ============================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("TradingFilters")


# ─────────────────────────────────────────────────────────────
# Dataclasses de resultado
# ─────────────────────────────────────────────────────────────

@dataclass
class TradingContext:
    """
    Contexto de trading consolidado — resultado de la evaluación completa
    de todos los filtros disponibles.
    """
    session: str
    """Sesión actual: LONDON, NEW_YORK, OVERLAP, ASIAN, CLOSED"""

    is_tradeable: bool
    """True solo si es London, New York u Overlap"""

    session_multiplier: float
    """Multiplicador de confianza por sesión: 1.3 (Overlap), 1.1 (LONDON/NY), 0.7 (ASIAN), 0.5 (CLOSED)"""

    mtf_signal: Dict
    """Señal multi-marco: h1_trend, h4_trend, d1_trend, alignment_score, confirmed, signal_strength"""

    dxy_bias: str
    """Sesgo DXY: 'BEARISH_DXY' (favorece oro), 'BULLISH_DXY' (perjudica oro), 'NEUTRAL'"""

    dxy_multiplier: float
    """Multiplicador DXY: 1.2 (DXY down), 0.8 (DXY up), 1.0 (neutral)"""

    overall_multiplier: float
    """Multiplicador final = session_multiplier × dxy_multiplier × alignment_score_modifier"""

    recommendation: str
    """Texto descriptivo para el trader"""


# ─────────────────────────────────────────────────────────────
# Filtro 1: SessionFilter
# ─────────────────────────────────────────────────────────────

class SessionFilter:
    """
    Determina la sesión de trading actual basada en la hora UTC.

    Sesiones:
        - LONDON:  08:00–17:00 UTC
        - NEW_YORK: 13:00–22:00 UTC
        - OVERLAP:  13:00–17:00 UTC (superposición, más fuerte)
        - ASIAN:   22:00–08:00 UTC (del día anterior/actual)
        - CLOSED:  Fuera de horarios (fines de semana, festivos)
    """

    # Horarios en UTC (formato time)
    LONDON_START = time(8, 0)
    LONDON_END = time(17, 0)
    NEWYORK_START = time(13, 0)
    NEWYORK_END = time(22, 0)
    OVERLAP_START = time(13, 0)
    OVERLAP_END = time(17, 0)
    ASIAN_START = time(22, 0)
    ASIAN_END = time(8, 0)  # Cruza medianoche

    def __init__(self, utc_time: Optional[datetime] = None):
        """
        Inicializa el filtro con una hora específica.

        Parámetros:
            utc_time: datetime en UTC. Si es None, usa la hora actual.
        """
        self.utc_time = utc_time or datetime.utcnow()
        self.current_time = self.utc_time.time()

    def get_current_session(self) -> str:
        """
        Retorna la sesión actual.

        Retorna:
            str: 'LONDON', 'NEW_YORK', 'OVERLAP', 'ASIAN', o 'CLOSED'
        """
        # Verificar OVERLAP primero (es un subconjunto de LONDON y NEW_YORK)
        if self.OVERLAP_START <= self.current_time < self.OVERLAP_END:
            return "OVERLAP"

        # Verificar LONDON (sin el overlap)
        if self.LONDON_START <= self.current_time < self.LONDON_END:
            return "LONDON"

        # Verificar NEW_YORK (sin el overlap)
        if self.NEWYORK_START <= self.current_time < self.NEWYORK_END:
            return "NEW_YORK"

        # Verificar ASIAN (cruza medianoche: 22:00 a 08:00)
        # Caso 1: hora >= 22:00 (noche)
        if self.current_time >= self.ASIAN_START:
            return "ASIAN"

        # Caso 2: hora < 08:00 (madrugada del día anterior, sigue siendo ASIAN)
        # PERO: 00:00 a 08:00 es ASIAN solo si NO estamos en horario de Londres
        # En nuestro caso, ASIAN va de 22:00 (hoy) a 08:00 (mañana)
        # Si estamos entre 00:00-08:00, es parte de la sesión ASIAN que comenzó ayer
        if self.current_time < self.ASIAN_END and self.current_time < self.LONDON_START:
            return "ASIAN"

        # Fuera de horarios: entre 08:00 y 13:00 (cierre después de Londres, antes de overlap)
        return "CLOSED"

    def is_tradeable_session(self) -> bool:
        """
        Retorna True solo en sesiones LONDON, NEW_YORK u OVERLAP.

        Retorna:
            bool: True si es tradeable, False en ASIAN o CLOSED
        """
        session = self.get_current_session()
        return session in ("LONDON", "NEW_YORK", "OVERLAP")

    def get_session_multiplier(self) -> float:
        """
        Retorna multiplicador de confianza según la sesión.

        Retorna:
            float: 1.3 (OVERLAP), 1.1 (LONDON/NEW_YORK), 0.7 (ASIAN), 0.5 (CLOSED)
        """
        session = self.get_current_session()
        multipliers = {
            "OVERLAP": 1.3,
            "LONDON": 1.1,
            "NEW_YORK": 1.1,
            "ASIAN": 0.7,
            "CLOSED": 0.5,
        }
        return multipliers.get(session, 0.5)


# ─────────────────────────────────────────────────────────────
# Filtro 2: MultiTimeframeAnalyzer
# ─────────────────────────────────────────────────────────────

class MultiTimeframeAnalyzer:
    """
    Analiza múltiples marcos de tiempo (H1, H4, D1) para detectar
    tendencias y alineación entre timeframes.

    Indicadores calculados:
        - EMA20: media móvil exponencial de 20 períodos
        - EMA50: media móvil exponencial de 50 períodos
        - RSI14: índice de fuerza relativa de 14 períodos
        - Trend: determinado por comparación EMA20 > EMA50
    """

    def __init__(self):
        """Inicializa el analizador multi-marco."""
        pass

    @staticmethod
    def _calculate_ema(series: pd.Series, span: int) -> pd.Series:
        """
        Calcula la Media Móvil Exponencial (EMA).

        Parámetros:
            series: serie de pandas
            span: período de la EMA

        Retorna:
            pd.Series: EMA calculada
        """
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula el Índice de Fuerza Relativa (RSI).

        Parámetros:
            series: serie de pandas (precios de cierre)
            period: período RSI

        Retorna:
            pd.Series: RSI calculada [0, 100]
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # RSI inicial = 50 si no hay datos suficientes

    @staticmethod
    def _get_trend(df: pd.DataFrame) -> int:
        """
        Determina la tendencia comparando EMA20 > EMA50.

        Parámetros:
            df: DataFrame con columnas close, ema20, ema50

        Retorna:
            int: 1 (alcista), -1 (bajista), 0 (neutral)
        """
        if df.empty or len(df) < 50:
            return 0

        close_price = df["close"].iloc[-1]
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]

        # Si hay NaN o falta data
        if pd.isna(ema20) or pd.isna(ema50):
            return 0

        # Tendencia alcista
        if ema20 > ema50:
            return 1

        # Tendencia bajista
        if ema20 < ema50:
            return -1

        # Neutral (emas iguales o muy próximas)
        return 0

    def analyze_timeframe(self, df: pd.DataFrame, timeframe_name: str) -> Dict:
        """
        Analiza un único timeframe (H1, H4, D1).

        Parámetros:
            df: DataFrame con columnas open, high, low, close, volume
            timeframe_name: nombre del timeframe (H1, H4, D1)

        Retorna:
            dict: {
                'timeframe': str,
                'ema20': float,
                'ema50': float,
                'rsi': float,
                'trend': int (1/-1/0),
                'close': float
            }
        """
        if df.empty or len(df) < 50:
            return {
                "timeframe": timeframe_name,
                "ema20": np.nan,
                "ema50": np.nan,
                "rsi": 50.0,
                "trend": 0,
                "close": np.nan,
            }

        # Normalizar nombres de columnas (minúsculas)
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Calcular indicadores
        df["ema20"] = self._calculate_ema(df["close"], 20)
        df["ema50"] = self._calculate_ema(df["close"], 50)
        df["rsi"] = self._calculate_rsi(df["close"], 14)

        # Determinar tendencia
        trend = self._get_trend(df)

        # Valores finales
        ema20_value = float(df["ema20"].iloc[-1]) if not pd.isna(df["ema20"].iloc[-1]) else np.nan
        ema50_value = float(df["ema50"].iloc[-1]) if not pd.isna(df["ema50"].iloc[-1]) else np.nan
        rsi_value = float(df["rsi"].iloc[-1]) if not pd.isna(df["rsi"].iloc[-1]) else 50.0
        close_value = float(df["close"].iloc[-1]) if not pd.isna(df["close"].iloc[-1]) else np.nan

        return {
            "timeframe": timeframe_name,
            "ema20": ema20_value,
            "ema50": ema50_value,
            "rsi": rsi_value,
            "trend": trend,
            "close": close_value,
        }

    @staticmethod
    def get_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula el Average True Range (ATR).

        Parámetros:
            df: DataFrame con columnas high, low, close
            period: período ATR

        Retorna:
            float: valor ATR
        """
        if df.empty or len(df) < period:
            return 0.0

        df = df.copy()
        df.columns = df.columns.str.lower()

        # True Range = max(H-L, abs(H-Cp), abs(L-Cp))
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    def get_mtf_signal(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Evalúa múltiples timeframes y retorna señal consolidada.

        Parámetros:
            mtf_data: dict con claves 'H1', 'H4', 'D1' → DataFrames

        Retorna:
            dict: {
                'h1_trend': int,
                'h4_trend': int,
                'd1_trend': int,
                'alignment_score': float (0.0-1.0),
                'confirmed': bool,
                'signal_strength': str ('STRONG', 'MODERATE', 'WEAK')
            }
        """
        # Obtener datos de cada timeframe
        h1_data = mtf_data.get("H1", pd.DataFrame())
        h4_data = mtf_data.get("H4", pd.DataFrame())
        d1_data = mtf_data.get("D1", pd.DataFrame())

        # Analizar cada timeframe
        h1_result = self.analyze_timeframe(h1_data, "H1")
        h4_result = self.analyze_timeframe(h4_data, "H4")
        d1_result = self.analyze_timeframe(d1_data, "D1")

        h1_trend = h1_result["trend"]
        h4_trend = h4_result["trend"]
        d1_trend = d1_result["trend"]

        # Calcular alineación (todas en la misma dirección)
        trends = [h1_trend, h4_trend, d1_trend]
        non_neutral_trends = [t for t in trends if t != 0]

        if len(non_neutral_trends) == 0:
            alignment_score = 0.0
            confirmed = False
        elif len(non_neutral_trends) == 3:
            # Todas no-neutral
            if h1_trend == h4_trend == d1_trend:
                # Todas alineadas
                alignment_score = 1.0
                confirmed = True
            else:
                # Desalineadas
                matching = sum(1 for i, j in [(h1_trend, h4_trend),
                                               (h4_trend, d1_trend),
                                               (h1_trend, d1_trend)]
                              if i == j and i != 0)
                alignment_score = matching / 3.0
                confirmed = False
        else:
            # Algunas neutral, algunas no
            alignment_score = len(non_neutral_trends) / 3.0
            confirmed = False

        # Determinar fuerza de señal
        if confirmed and alignment_score >= 0.9:
            signal_strength = "STRONG"
        elif alignment_score >= 0.6:
            signal_strength = "MODERATE"
        else:
            signal_strength = "WEAK"

        return {
            "h1_trend": h1_trend,
            "h4_trend": h4_trend,
            "d1_trend": d1_trend,
            "alignment_score": float(alignment_score),
            "confirmed": bool(confirmed),
            "signal_strength": signal_strength,
        }


# ─────────────────────────────────────────────────────────────
# Filtro 3: DXYFilter
# ─────────────────────────────────────────────────────────────

class DXYFilter:
    """
    Filtra el sesgo del índice dólar (DXY).
    - DXY bajista → favorece compra de oro (correlación negativa)
    - DXY alcista → perjudica compra de oro (correlación negativa)

    Si no hay datos disponibles, retorna NEUTRAL sin errores.
    """

    def __init__(self, dxy_data: Optional[pd.DataFrame] = None):
        """
        Inicializa el filtro DXY.

        Parámetros:
            dxy_data: DataFrame con columna 'DXY' o None
        """
        self.dxy_data = dxy_data
        self.has_data = dxy_data is not None and not dxy_data.empty

    def get_dxy_bias(self) -> str:
        """
        Determina el sesgo actual del DXY.

        Retorna:
            str: 'BEARISH_DXY' (DXY bajista, favorece oro),
                 'BULLISH_DXY' (DXY alcista, perjudica oro),
                 'NEUTRAL' (sin datos o transición)
        """
        if not self.has_data:
            return "NEUTRAL"

        try:
            # Normalizar nombres de columnas
            dxy_col = None
            for col in self.dxy_data.columns:
                if col.upper() == "DXY":
                    dxy_col = col
                    break

            if dxy_col is None:
                return "NEUTRAL"

            # Obtener último precio y anterior
            if len(self.dxy_data) < 2:
                return "NEUTRAL"

            dxy_last = self.dxy_data[dxy_col].iloc[-1]
            dxy_prev = self.dxy_data[dxy_col].iloc[-2]

            if pd.isna(dxy_last) or pd.isna(dxy_prev):
                return "NEUTRAL"

            # Determinar dirección (DXY)
            if dxy_last < dxy_prev:
                return "BEARISH_DXY"  # DXY bajista
            elif dxy_last > dxy_prev:
                return "BULLISH_DXY"  # DXY alcista
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.warning(f"Error al procesar DXY: {e} — retornando NEUTRAL")
            return "NEUTRAL"

    def get_dxy_multiplier(self) -> float:
        """
        Retorna multiplicador según el sesgo DXY.

        Retorna:
            float: 1.2 (DXY bajista, favorece oro),
                   0.8 (DXY alcista, perjudica oro),
                   1.0 (neutral)
        """
        bias = self.get_dxy_bias()
        multipliers = {
            "BEARISH_DXY": 1.2,
            "BULLISH_DXY": 0.8,
            "NEUTRAL": 1.0,
        }
        return multipliers.get(bias, 1.0)


# ─────────────────────────────────────────────────────────────
# Orquestador: TradingFiltersManager
# ─────────────────────────────────────────────────────────────

class TradingFiltersManager:
    """
    Orquestador principal que integra todos los filtros:
    SessionFilter, MultiTimeframeAnalyzer, DXYFilter.

    Proporciona una evaluación consolidada en un único objeto TradingContext.
    """

    def __init__(self):
        """Inicializa el gestor de filtros."""
        self.session_filter = None
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.dxy_filter = None

    def evaluate(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        dxy_data: Optional[pd.DataFrame] = None,
        utc_time: Optional[datetime] = None,
    ) -> TradingContext:
        """
        Evalúa todos los filtros y retorna un contexto de trading consolidado.

        Parámetros:
            mtf_data: dict con {'H1': df, 'H4': df, 'D1': df}
            dxy_data: DataFrame con datos DXY o None
            utc_time: datetime en UTC, si es None usa hora actual

        Retorna:
            TradingContext: objeto con todos los resultados
        """
        # Inicializar filtros
        self.session_filter = SessionFilter(utc_time)
        self.dxy_filter = DXYFilter(dxy_data)

        # Obtener datos de sesión
        session = self.session_filter.get_current_session()
        is_tradeable = self.session_filter.is_tradeable_session()
        session_multiplier = self.session_filter.get_session_multiplier()

        # Obtener señal multi-marco
        mtf_signal = self.mtf_analyzer.get_mtf_signal(mtf_data)

        # Obtener sesgo DXY
        dxy_bias = self.dxy_filter.get_dxy_bias()
        dxy_multiplier = self.dxy_filter.get_dxy_multiplier()

        # Calcular multiplicador general
        alignment_modifier = mtf_signal["alignment_score"]
        overall_multiplier = session_multiplier * dxy_multiplier * (0.8 + 0.4 * alignment_modifier)

        # Generar recomendación textual
        recommendation = self._build_recommendation(
            session, is_tradeable, mtf_signal, dxy_bias, overall_multiplier
        )

        return TradingContext(
            session=session,
            is_tradeable=is_tradeable,
            session_multiplier=session_multiplier,
            mtf_signal=mtf_signal,
            dxy_bias=dxy_bias,
            dxy_multiplier=dxy_multiplier,
            overall_multiplier=overall_multiplier,
            recommendation=recommendation,
        )

    @staticmethod
    def _build_recommendation(
        session: str,
        is_tradeable: bool,
        mtf_signal: Dict,
        dxy_bias: str,
        overall_multiplier: float,
    ) -> str:
        """
        Construye un mensaje descriptivo para el trader.

        Parámetros:
            session: sesión actual
            is_tradeable: si es horario tradeable
            mtf_signal: señal multi-marco
            dxy_bias: sesgo DXY
            overall_multiplier: multiplicador final

        Retorna:
            str: recomendación textual
        """
        parts = []

        # Sesión
        if not is_tradeable:
            parts.append(f"⚠️ SESIÓN {session}: Actividad REDUCIDA (multiplicador bajo)")
        else:
            parts.append(f"✅ Sesión {session}: Actividad ACTIVA")

        # Alineación multi-marco
        alignment = mtf_signal["alignment_score"]
        signal_strength = mtf_signal["signal_strength"]

        if mtf_signal["confirmed"]:
            parts.append(f"✅ Marcos ALINEADOS ({signal_strength}) — Confianza alta")
        else:
            parts.append(f"⚠️ Marcos parcialmente alineados ({signal_strength}) — Alineación: {alignment:.2f}")

        # Tendencias
        trends_text = []
        if mtf_signal["h1_trend"] == 1:
            trends_text.append("H1 ↑")
        elif mtf_signal["h1_trend"] == -1:
            trends_text.append("H1 ↓")

        if mtf_signal["h4_trend"] == 1:
            trends_text.append("H4 ↑")
        elif mtf_signal["h4_trend"] == -1:
            trends_text.append("H4 ↓")

        if mtf_signal["d1_trend"] == 1:
            trends_text.append("D1 ↑")
        elif mtf_signal["d1_trend"] == -1:
            trends_text.append("D1 ↓")

        if trends_text:
            parts.append(f"Tendencias: {' | '.join(trends_text)}")

        # DXY
        if dxy_bias == "BEARISH_DXY":
            parts.append("📈 DXY BAJISTA: Favorable para ORO")
        elif dxy_bias == "BULLISH_DXY":
            parts.append("📉 DXY ALCISTA: Desfavorable para ORO")
        else:
            parts.append("➡️ DXY Neutral")

        # Multiplicador final
        parts.append(f"\nMultiplicador Final: {overall_multiplier:.2f}x")

        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
# Tests unitarios
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRUEBAS UNITARIAS — trading_filters.py")
    print("="*70 + "\n")

    # ─── Test 1: SessionFilter ───
    print("📝 TEST 1: SessionFilter")
    print("-" * 70)

    test_times = [
        (datetime(2026, 4, 22, 7, 30), "ASIAN"),       # Antes de Londres (sigue ASIAN del día anterior)
        (datetime(2026, 4, 22, 8, 0), "LONDON"),       # Inicio Londres
        (datetime(2026, 4, 22, 12, 0), "LONDON"),      # Londres plena
        (datetime(2026, 4, 22, 13, 0), "OVERLAP"),     # Inicio overlap
        (datetime(2026, 4, 22, 15, 0), "OVERLAP"),     # Overlap plena
        (datetime(2026, 4, 22, 17, 0), "NEW_YORK"),    # Fin overlap, NY activo
        (datetime(2026, 4, 22, 20, 0), "NEW_YORK"),    # NY plena
        (datetime(2026, 4, 22, 22, 0), "ASIAN"),       # Inicio Asiático
        (datetime(2026, 4, 22, 23, 59), "ASIAN"),      # Fin del día (Asiático)
        (datetime(2026, 4, 23, 1, 0), "ASIAN"),        # Madrugada Asiático
        (datetime(2026, 4, 23, 6, 0), "ASIAN"),        # Antes de Londres (Asiático)
    ]

    for test_time, expected_session in test_times:
        sf = SessionFilter(test_time)
        actual_session = sf.get_current_session()
        is_tradeable = sf.is_tradeable_session()
        multiplier = sf.get_session_multiplier()

        status = "✅" if actual_session == expected_session else "❌"
        print(
            f"{status} {test_time.strftime('%H:%M UTC')} → "
            f"{actual_session} (esperado: {expected_session}) | "
            f"Tradeable: {is_tradeable} | Multiplier: {multiplier}"
        )

    print("\n")

    # ─── Test 2: MultiTimeframeAnalyzer ───
    print("📝 TEST 2: MultiTimeframeAnalyzer")
    print("-" * 70)

    # Generar datos mock
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")

    # H1: tendencia alcista
    h1_close = 100 + np.cumsum(np.random.normal(0.5, 1, n_rows))
    h1_high = h1_close + np.abs(np.random.normal(0.5, 0.3, n_rows))
    h1_low = h1_close - np.abs(np.random.normal(0.5, 0.3, n_rows))
    h1_data = pd.DataFrame(
        {
            "open": h1_close + np.random.normal(0, 0.2, n_rows),
            "high": h1_high,
            "low": h1_low,
            "close": h1_close,
            "volume": np.random.randint(1000, 5000, n_rows),
        },
        index=dates,
    )

    # H4: tendencia alcista
    h4_close = 100 + np.cumsum(np.random.normal(0.3, 0.8, n_rows))
    h4_high = h4_close + np.abs(np.random.normal(0.4, 0.3, n_rows))
    h4_low = h4_close - np.abs(np.random.normal(0.4, 0.3, n_rows))
    h4_data = pd.DataFrame(
        {
            "open": h4_close + np.random.normal(0, 0.2, n_rows),
            "high": h4_high,
            "low": h4_low,
            "close": h4_close,
            "volume": np.random.randint(1000, 5000, n_rows),
        },
        index=dates,
    )

    # D1: tendencia alcista
    d1_close = 100 + np.cumsum(np.random.normal(0.2, 0.6, n_rows))
    d1_high = d1_close + np.abs(np.random.normal(0.3, 0.3, n_rows))
    d1_low = d1_close - np.abs(np.random.normal(0.3, 0.3, n_rows))
    d1_data = pd.DataFrame(
        {
            "open": d1_close + np.random.normal(0, 0.2, n_rows),
            "high": d1_high,
            "low": d1_low,
            "close": d1_close,
            "volume": np.random.randint(1000, 5000, n_rows),
        },
        index=dates,
    )

    mtf_analyzer = MultiTimeframeAnalyzer()
    mtf_signal = mtf_analyzer.get_mtf_signal(
        {"H1": h1_data, "H4": h4_data, "D1": d1_data}
    )

    print(f"✅ H1 Tendencia: {mtf_signal['h1_trend']} (1=alcista, -1=bajista, 0=neutral)")
    print(f"✅ H4 Tendencia: {mtf_signal['h4_trend']}")
    print(f"✅ D1 Tendencia: {mtf_signal['d1_trend']}")
    print(f"✅ Alineación: {mtf_signal['alignment_score']:.2f} (0.0-1.0)")
    print(f"✅ Confirmado: {mtf_signal['confirmed']}")
    print(f"✅ Fuerza: {mtf_signal['signal_strength']}")

    # Test ATR
    atr = mtf_analyzer.get_atr(h1_data, period=14)
    print(f"✅ ATR (H1): {atr:.2f}")

    print("\n")

    # ─── Test 3: DXYFilter ───
    print("📝 TEST 3: DXYFilter")
    print("-" * 70)

    # Caso 1: Con datos DXY (bajista)
    dxy_data_bullish = pd.DataFrame(
        {"DXY": [104.5, 104.0, 103.8]},  # bajista
        index=dates[-3:],
    )
    dxy_filter_1 = DXYFilter(dxy_data_bullish)
    bias_1 = dxy_filter_1.get_dxy_bias()
    mult_1 = dxy_filter_1.get_dxy_multiplier()
    print(f"✅ DXY Bajista → Sesgo: {bias_1}, Multiplicador: {mult_1}")

    # Caso 2: Con datos DXY (alcista)
    dxy_data_bearish = pd.DataFrame(
        {"DXY": [103.5, 104.0, 104.5]},  # alcista
        index=dates[-3:],
    )
    dxy_filter_2 = DXYFilter(dxy_data_bearish)
    bias_2 = dxy_filter_2.get_dxy_bias()
    mult_2 = dxy_filter_2.get_dxy_multiplier()
    print(f"✅ DXY Alcista → Sesgo: {bias_2}, Multiplicador: {mult_2}")

    # Caso 3: Sin datos DXY (no debe lanzar error)
    dxy_filter_3 = DXYFilter(None)
    bias_3 = dxy_filter_3.get_dxy_bias()
    mult_3 = dxy_filter_3.get_dxy_multiplier()
    print(f"✅ Sin datos DXY → Sesgo: {bias_3}, Multiplicador: {mult_3}")

    print("\n")

    # ─── Test 4: TradingFiltersManager ───
    print("📝 TEST 4: TradingFiltersManager")
    print("-" * 70)

    manager = TradingFiltersManager()

    # Contexto: Overlap, marcos alineados, DXY bajista
    context = manager.evaluate(
        mtf_data={"H1": h1_data, "H4": h4_data, "D1": d1_data},
        dxy_data=dxy_data_bullish,
        utc_time=datetime(2026, 4, 22, 15, 30),  # Overlap
    )

    print(f"✅ Sesión: {context.session}")
    print(f"✅ Tradeable: {context.is_tradeable}")
    print(f"✅ Session Multiplier: {context.session_multiplier}")
    print(f"✅ DXY Bias: {context.dxy_bias}")
    print(f"✅ DXY Multiplier: {context.dxy_multiplier}")
    print(f"✅ Overall Multiplier: {context.overall_multiplier:.2f}")
    print(f"\n✅ Recomendación:\n{context.recommendation}")

    print("\n" + "="*70)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("="*70 + "\n")
