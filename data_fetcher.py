# ============================================================
# data_fetcher.py — Módulo de obtención de datos de oro
# Gold Price Monitor — Phase 1
#
# Modos soportados:
#   "mock" → datos simulados realistas (ejecución inmediata sin API)
#   "live" → datos reales de Yahoo Finance a través de yfinance
# ============================================================

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import AppConfig, DEFAULT_CONFIG, GOLD_TICKER_FUTURES, DXY_TICKER

# Suprimir advertencias innecesarias de yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar sistema de registros
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GoldDataFetcher")


class GoldDataFetcher:
    """
    Obtener datos de oro — soporta dos modos: Mock (simulado) y Live (real).

    Ejemplo de uso:
        fetcher = GoldDataFetcher(mode="mock")
        df = fetcher.get_data()

        # Para cambiar a datos reales: cambia solo la siguiente línea
        fetcher = GoldDataFetcher(mode="live")
        df = fetcher.get_data(interval="1d", period="1y")
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: AppConfig = DEFAULT_CONFIG):
        self.config = config
        self.mode   = config.mode
        self._data: Optional[pd.DataFrame] = None
        self._mtf_data: Dict[str, pd.DataFrame] = {}  # Para almacenar datos multi-timeframe
        logger.info(f"GoldDataFetcher initialized — mode='{self.mode}'")

    # ─────────────────────────────────────────────────────────
    # Sección 1: Generación de datos simulados (Mock Data)
    # ─────────────────────────────────────────────────────────
    def _generate_mock_ohlcv(self) -> pd.DataFrame:
        """
        Genera datos OHLCV realistas para el oro usando una caminata aleatoria calibrada
        (Movimiento Geométrico Browniano) — el mismo modelo usado en finanzas cuantitativas.

        Características:
         - Garantiza: High ≥ max(Open, Close) siempre
         - Garantiza: Low  ≤ min(Open, Close) siempre
         - Genera solo días hábiles (Business Days)
        """
        cfg = self.config.mock
        np.random.seed(cfg.seed)
        n = cfg.n_rows

        logger.info(f"Generando {n} velas simuladas — precio inicial: ${cfg.start_price:.2f}")

        # ── Generar precios de cierre (GBM) ──
        daily_returns  = np.random.normal(cfg.daily_drift, cfg.daily_vol, n)
        close_prices   = cfg.start_price * np.cumprod(1 + daily_returns)

        # ── Generar Open del Close anterior (brecha aleatoria pequeña) ──
        open_prices    = np.empty(n)
        open_prices[0] = cfg.start_price
        gap            = np.random.normal(0, cfg.daily_vol * 0.3, n - 1)
        open_prices[1:] = close_prices[:-1] * (1 + gap)

        # ── Generar High y Low basado en rango realista ──
        candle_range_pct = np.abs(np.random.normal(0, cfg.daily_vol * 0.6, n))
        wick_up_pct      = np.abs(np.random.normal(0, cfg.daily_vol * 0.3, n))
        wick_down_pct    = np.abs(np.random.normal(0, cfg.daily_vol * 0.3, n))

        body_high = np.maximum(open_prices, close_prices)
        body_low  = np.minimum(open_prices, close_prices)

        high_prices = body_high * (1 + wick_up_pct)
        low_prices  = body_low  * (1 - wick_down_pct)

        # Garantizar High ≥ Open y High ≥ Close en todos los casos
        high_prices = np.maximum(high_prices, body_high)
        low_prices  = np.minimum(low_prices,  body_low)

        # ── Volumen de negociación — siempre positivo ──
        base_volume = 250_000
        volume = (
            base_volume
            + base_volume * 0.5 * np.abs(np.random.normal(0, 1, n))
            + base_volume * 0.3 * candle_range_pct / cfg.daily_vol
        ).astype(int)

        # ── Construir el DataFrame ──
        dates = pd.bdate_range(
            end=pd.Timestamp.today().normalize(),
            periods=n,
            freq="B",
        )

        df = pd.DataFrame(
            {
                "Open":   np.round(open_prices,  2),
                "High":   np.round(high_prices,  2),
                "Low":    np.round(low_prices,   2),
                "Close":  np.round(close_prices, 2),
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"

        # ── Validación final de OHLC ──
        assert (df["High"] >= df["Open"]).all(),  "Error: High < Open"
        assert (df["High"] >= df["Close"]).all(), "Error: High < Close"
        assert (df["Low"]  <= df["Open"]).all(),  "Error: Low > Open"
        assert (df["Low"]  <= df["Close"]).all(), "Error: Low > Close"
        assert (df["Volume"] > 0).all(),           "Error: Volumen negativo o cero"

        logger.info(
            f"✅ Mock OHLCV listo | Filas: {len(df)} | "
            f"Rango de precios: ${df['Close'].min():.2f} – ${df['Close'].max():.2f}"
        )
        return df

    def _generate_mock_macro(self, gold_df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera datos simulados de macroeconomía (DXY correlacionado inversamente con el oro).
        La correlación natural entre DXY y el oro oscila entre -0.6 y -0.8.
        """
        cfg = self.config.mock
        np.random.seed(cfg.seed + 100)
        n = len(gold_df)

        # DXY = correlacionado inversamente con el oro
        gold_returns = gold_df["Close"].pct_change().fillna(0).values
        dxy_noise    = np.random.normal(0, 0.003, n)
        dxy_returns  = -0.65 * gold_returns + dxy_noise

        dxy_start  = 104.0
        dxy_prices = dxy_start * np.cumprod(1 + dxy_returns)

        macro_df = pd.DataFrame(
            {
                "DXY":         np.round(dxy_prices, 3),
                "DXY_Returns": np.round(dxy_returns, 6),
            },
            index=gold_df.index,
        )
        macro_df.index.name = "Date"
        logger.info(
            f"✅ Mock Macro listo | Rango DXY: "
            f"{macro_df['DXY'].min():.2f} – {macro_df['DXY'].max():.2f}"
        )
        return macro_df

    def _generate_mock_ohlcv_for_timeframe(
        self,
        base_df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Genera datos OHLCV simulados para un timeframe específico basado en datos base.
        Si es H1 o M5, genera nuevos datos. Si es H4 o D1, resamplea desde base.

        Parámetros:
            base_df: DataFrame base con OHLCV (típicamente datos diarios o horarios)
            timeframe: "H1", "H4", "D1", etc.

        Retorna:
            DataFrame con índice DatetimeIndex y columnas: Open, High, Low, Close, Volume
        """
        np.random.seed(self.config.mock.seed + hash(timeframe) % 10000)

        if timeframe == "H1":
            # Para H1, generamos datos horarios realistas
            n_bars = max(len(base_df) * 6, 240)  # ~1 mes de datos horarios
            cfg = self.config.mock

            daily_returns = np.random.normal(cfg.daily_drift, cfg.daily_vol * 0.3, n_bars)
            close_prices = cfg.start_price * np.cumprod(1 + daily_returns)

            open_prices = np.empty(n_bars)
            open_prices[0] = cfg.start_price
            gap = np.random.normal(0, cfg.daily_vol * 0.15, n_bars - 1)
            open_prices[1:] = close_prices[:-1] * (1 + gap)

            candle_range_pct = np.abs(np.random.normal(0, cfg.daily_vol * 0.3, n_bars))
            wick_up_pct = np.abs(np.random.normal(0, cfg.daily_vol * 0.15, n_bars))
            wick_down_pct = np.abs(np.random.normal(0, cfg.daily_vol * 0.15, n_bars))

            body_high = np.maximum(open_prices, close_prices)
            body_low = np.minimum(open_prices, close_prices)

            high_prices = body_high * (1 + wick_up_pct)
            low_prices = body_low * (1 - wick_down_pct)

            high_prices = np.maximum(high_prices, body_high)
            low_prices = np.minimum(low_prices, body_low)

            base_volume = 50_000
            volume = (
                base_volume
                + base_volume * 0.5 * np.abs(np.random.normal(0, 1, n_bars))
                + base_volume * 0.3 * candle_range_pct / cfg.daily_vol
            ).astype(int)

            dates = pd.date_range(
                end=pd.Timestamp.today().normalize(),
                periods=n_bars,
                freq="h"
            )

        elif timeframe == "H4":
            # H4: resamplear desde datos de mayor granularidad
            if len(base_df) < 4:
                # Si no hay datos suficientes, generar nuevos
                return self._generate_mock_ohlcv_for_timeframe(base_df, "H1").resample("4h").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna()

            resampled = base_df.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
            return resampled

        elif timeframe == "D1":
            # D1: usar datos diarios directamente o resamplear
            resampled = base_df.resample("D").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
            return resampled

        else:
            logger.warning(f"Timeframe no soportado: {timeframe} — usando H1 por defecto")
            return self._generate_mock_ohlcv_for_timeframe(base_df, "H1")

        # Construir DataFrame para H1
        df = pd.DataFrame(
            {
                "Open": np.round(open_prices, 2),
                "High": np.round(high_prices, 2),
                "Low": np.round(low_prices, 2),
                "Close": np.round(close_prices, 2),
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"

        return df

    # ─────────────────────────────────────────────────────────
    # Sección 1b: Datos multi-timeframe (Mock y Live)
    # ─────────────────────────────────────────────────────────
    def fetch_multi_timeframe(
        self,
        symbol: str = GOLD_TICKER_FUTURES,
        timeframes: Optional[list] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos para múltiples timeframes (H1, H4, D1, etc).
        Funciona en ambos modos: mock (genera datos simulados) y live (obtiene de yfinance).

        Parámetros:
            symbol: símbolo del instrumento (ej: "GC=F" para futuros de oro)
            timeframes: lista de timeframes (ej: ["H1", "H4", "D1"])
                        Predeterminado: ["H1", "H4", "D1"]

        Retorna:
            Dict con estructura: {"H1": df_h1, "H4": df_h4, "D1": df_d1, ...}
            Cada DataFrame contiene: Open, High, Low, Close, Volume con índice DatetimeIndex

        Ejemplo:
            >>> fetcher = GoldDataFetcher(mode="mock")
            >>> mtf_data = fetcher.fetch_multi_timeframe()
            >>> print(mtf_data.keys())
            dict_keys(['H1', 'H4', 'D1'])
            >>> print(mtf_data['H1'].head())
        """
        if timeframes is None:
            timeframes = ["H1", "H4", "D1"]

        logger.info(f"Obteniendo datos multi-timeframe | Timeframes: {timeframes} | Modo: {self.mode}")

        result = {}

        if self.mode == "mock":
            # Generar datos base (H1)
            base_df = self._generate_mock_ohlcv()

            # Generar datos para cada timeframe
            for tf in timeframes:
                try:
                    result[tf] = self._generate_mock_ohlcv_for_timeframe(base_df, tf)
                    logger.info(
                        f"✅ Mock {tf} listo | Filas: {len(result[tf])} | "
                        f"Rango: {result[tf].index[0].date()} — {result[tf].index[-1].date()}"
                    )
                except Exception as e:
                    logger.error(f"Error generando datos mock para {tf}: {e}")
                    result[tf] = pd.DataFrame()

        elif self.mode == "live":
            # Para datos en vivo, obtener H1 primero y resamplear para otros timeframes
            try:
                h1_df = self._fetch_live_ohlcv(
                    ticker=symbol,
                    interval="1h",
                    period="1mo"
                )

                for tf in timeframes:
                    if tf == "H1":
                        result["H1"] = h1_df.copy()
                    elif tf == "H4":
                        # Resamplear H1 a H4
                        result["H4"] = h1_df.resample("4h").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum"
                        }).dropna()
                    elif tf == "D1":
                        # Resamplear H1 a D1
                        result["D1"] = h1_df.resample("D").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum"
                        }).dropna()
                    else:
                        logger.warning(f"Timeframe no soportado en live: {tf}")

                    if tf in result and not result[tf].empty:
                        logger.info(
                            f"✅ Live {tf} listo | Filas: {len(result[tf])} | "
                            f"Rango: {result[tf].index[0].date()} — {result[tf].index[-1].date()}"
                        )

            except Exception as e:
                logger.error(f"Error obteniendo datos en vivo multi-timeframe: {e}")
                logger.info("Cambiando a mock data como fallback...")
                # Fallback a mock
                return self.fetch_multi_timeframe(symbol=symbol, timeframes=timeframes)

        else:
            raise ValueError(
                f"Modo no soportado: '{self.mode}'. Opciones: 'mock' o 'live'."
            )

        return result

    # ─────────────────────────────────────────────────────────
    # Sección 2: Obtención de datos DXY (Índice del Dólar)
    # ─────────────────────────────────────────────────────────
    def fetch_dxy(self, n_bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Obtiene datos del índice del dólar (DXY).
        Funciona en ambos modos: mock (genera datos correlacionados) y live (obtiene de yfinance).

        Parámetros:
            n_bars: número de barras a obtener (predeterminado: 100)

        Retorna:
            DataFrame con columnas: Open, High, Low, Close, Volume
            O None si la obtención falló (solo en live mode, logged como warning)

        Características:
            - En mock: correlación negativa con el oro (~-0.65)
            - En live: intenta obtener "DX-Y.NYB" o "UUP" como alternativa
            - Si falla en live: retorna None sin lanzar exception

        Ejemplo:
            >>> fetcher = GoldDataFetcher(mode="mock")
            >>> dxy_df = fetcher.fetch_dxy(n_bars=50)
            >>> print(dxy_df.tail())
        """
        logger.info(f"Obteniendo DXY ({n_bars} barras) | Modo: {self.mode}")

        if self.mode == "mock":
            return self._fetch_mock_dxy(n_bars=n_bars)
        elif self.mode == "live":
            return self._fetch_live_dxy(n_bars=n_bars)
        else:
            logger.error(f"Modo no soportado: {self.mode}")
            return None

    def _fetch_mock_dxy(self, n_bars: int = 100) -> pd.DataFrame:
        """
        Genera datos mock realistas del DXY con correlación negativa con el oro.

        Los datos se generan con:
        - Movimiento geométrico browniano
        - Correlación negativa natural con oro (~-0.65)
        - Volatilidad realista del índice dólar
        """
        cfg = self.config.mock
        np.random.seed(cfg.seed + 200)

        # Generar precios DXY con correlación inversa simulada
        dxy_drift = -0.00005  # ligero drift negativo
        dxy_vol = 0.004  # volatilidad típica del DXY (~0.4% diario)

        dxy_returns = np.random.normal(dxy_drift, dxy_vol, n_bars)
        dxy_start = 104.0
        dxy_close = dxy_start * np.cumprod(1 + dxy_returns)

        # Generar Open, High, Low realistas
        dxy_open = np.empty(n_bars)
        dxy_open[0] = dxy_start
        gap = np.random.normal(0, dxy_vol * 0.3, n_bars - 1)
        dxy_open[1:] = dxy_close[:-1] * (1 + gap)

        candle_range_pct = np.abs(np.random.normal(0, dxy_vol * 0.5, n_bars))
        wick_up_pct = np.abs(np.random.normal(0, dxy_vol * 0.25, n_bars))
        wick_down_pct = np.abs(np.random.normal(0, dxy_vol * 0.25, n_bars))

        body_high = np.maximum(dxy_open, dxy_close)
        body_low = np.minimum(dxy_open, dxy_close)

        dxy_high = body_high * (1 + wick_up_pct)
        dxy_low = body_low * (1 - wick_down_pct)

        dxy_high = np.maximum(dxy_high, body_high)
        dxy_low = np.minimum(dxy_low, body_low)

        # Volumen típico del índice (simulado)
        base_volume = 100_000
        volume = (
            base_volume
            + base_volume * 0.5 * np.abs(np.random.normal(0, 1, n_bars))
            + base_volume * 0.3 * candle_range_pct / dxy_vol
        ).astype(int)

        # Crear índice de fechas (últimos n_bars días)
        dates = pd.bdate_range(
            end=pd.Timestamp.today().normalize(),
            periods=n_bars,
            freq="B"
        )

        dxy_df = pd.DataFrame(
            {
                "Open": np.round(dxy_open, 3),
                "High": np.round(dxy_high, 3),
                "Low": np.round(dxy_low, 3),
                "Close": np.round(dxy_close, 3),
                "Volume": volume,
            },
            index=dates,
        )
        dxy_df.index.name = "Date"

        logger.info(
            f"✅ DXY Mock listo | Filas: {len(dxy_df)} | "
            f"Rango: ${dxy_df['Close'].min():.3f} – ${dxy_df['Close'].max():.3f}"
        )
        return dxy_df

    def _fetch_live_dxy(self, n_bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Obtiene datos DXY en vivo desde Yahoo Finance.
        Intenta obtener "DX-Y.NYB" (DXY) o "UUP" (inverso del dólar) como fallback.

        Si falla la obtención: retorna None con log de warning (no lanza exception).
        """
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            logger.warning("yfinance no está instalado — usando mock DXY")
            return self._fetch_mock_dxy(n_bars=n_bars)

        tickers_to_try = ["DX-Y.NYB", "UUP"]
        dxy_df = None

        for ticker in tickers_to_try:
            try:
                logger.info(f"Intentando obtener DXY desde {ticker}...")
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(period="3mo", interval="1d", auto_adjust=True)

                if not df.empty and len(df) > 0:
                    # Limpiar y preparar columnas
                    if "Close" in df.columns:
                        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df.index = pd.to_datetime(df.index)
                        df.index.name = "Date"
                        df.dropna(subset=["Close"], inplace=True)

                        # Tomar últimas n_bars
                        dxy_df = df.tail(n_bars).copy()

                        logger.info(
                            f"✅ DXY en vivo obtenido desde {ticker} | Filas: {len(dxy_df)} | "
                            f"Rango: {dxy_df.index[0].date()} — {dxy_df.index[-1].date()}"
                        )
                        return dxy_df
            except Exception as e:
                logger.warning(f"Fallo al obtener DXY desde {ticker}: {e}")
                continue

        # Si todos los intentos fallaron
        logger.warning(
            f"No se pudo obtener DXY desde ningún ticker ({', '.join(tickers_to_try)}) — "
            "retornando None"
        )
        return None

    # ─────────────────────────────────────────────────────────
    # Sección 3: Obtención de datos en vivo (Live Data)
    # ─────────────────────────────────────────────────────────
    def _fetch_live_ohlcv(
        self,
        ticker: str = GOLD_TICKER_FUTURES,
        interval: str = "1d",
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Obtiene datos reales de Yahoo Finance.
        Para habilitar este modo: cambia mode="live" en AppConfig.

        Parámetros:
            ticker:   símbolo del instrumento financiero (predeterminado: futuros de oro GC=F)
            interval: marco de tiempo (1m, 5m, 15m, 1h, 1d, ...)
            period:   período de tiempo (1d, 5d, 1mo, 3mo, 1y, 2y, ...)
        """
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "yfinance no está instalado. Ejecuta: pip install yfinance"
            )

        logger.info(
            f"Obteniendo datos en vivo | Ticker: {ticker} | "
            f"Interval: {interval} | Period: {period}"
        )

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError(
                f"No se recibieron datos para {ticker}. "
                "Comprueba la conexión a Internet o la validez del ticker."
            )

        # Limpiar columnas — yfinance a veces devuelve columnas adicionales
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.dropna(subset=["Close"], inplace=True)

        logger.info(
            f"✅ OHLCV en vivo listo | Filas: {len(df)} | "
            f"Desde: {df.index[0].date()} hasta: {df.index[-1].date()}"
        )
        return df

    def _fetch_live_macro(self, start: str, end: str) -> pd.DataFrame:
        """Obtiene datos DXY reales (índice del dólar)."""
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise ImportError("yfinance no está instalado.")

        logger.info("Obteniendo datos DXY en vivo...")
        dxy = yf.download(DXY_TICKER, start=start, end=end,
                          auto_adjust=True, progress=False)

        if dxy.empty:
            logger.warning("No se pudo obtener DXY — se usará una columna vacía.")
            return pd.DataFrame()

        macro_df = dxy[["Close"]].rename(columns={"Close": "DXY"})
        macro_df["DXY_Returns"] = macro_df["DXY"].pct_change()
        macro_df.index.name = "Date"
        return macro_df

    # ─────────────────────────────────────────────────────────
    # Sección 4: Interfaz principal
    # ─────────────────────────────────────────────────────────
    def get_data(
        self,
        interval: str = "1d",
        period: str = "1y",
        include_mtf: bool = False,
    ) -> pd.DataFrame:
        """
        Función principal — devuelve un DataFrame completo con datos OHLCV.
        Funciona en ambos modos (mock / live) de la misma forma.

        Parámetros:
            interval: marco de tiempo (1m, 5m, 1h, 1d, etc.)
            period: período de tiempo (1d, 5d, 1mo, 3mo, 1y, etc.)
            include_mtf: si True, también obtiene datos multi-timeframe
                        (se almacenan en self._mtf_data)

        Salida:
            pd.DataFrame con columnas: Open, High, Low, Close, Volume
            y si está disponible: DXY, DXY_Returns

        Nota:
            Si include_mtf=True, accede a los datos multi-timeframe vía:
            self._mtf_data (Dict[str, DataFrame])
        """
        if self.mode == "mock":
            gold_df  = self._generate_mock_ohlcv()
            macro_df = self._generate_mock_macro(gold_df)
            df = gold_df.join(macro_df, how="left")

        elif self.mode == "live":
            gold_df  = self._fetch_live_ohlcv(
                ticker=GOLD_TICKER_FUTURES,
                interval=interval,
                period=period,
            )
            start = str(gold_df.index[0].date())
            end   = str(gold_df.index[-1].date())
            try:
                macro_df = self._fetch_live_macro(start=start, end=end)
                df = gold_df.join(macro_df, how="left")
            except Exception as e:
                logger.warning(f"No se pudo obtener datos macro: {e} — continuaremos sin ellos.")
                df = gold_df.copy()
        else:
            raise ValueError(
                f"Modo no soportado: '{self.mode}'. Opciones: 'mock' o 'live'."
            )

        self._validate_dataframe(df)
        self._data = df

        # Si se solicita, obtener datos multi-timeframe
        if include_mtf:
            logger.info("Obteniendo datos multi-timeframe...")
            try:
                self._mtf_data = self.fetch_multi_timeframe()
                logger.info("✅ Datos multi-timeframe obtenidos exitosamente")
            except Exception as e:
                logger.warning(f"Error al obtener multi-timeframe: {e}")
                self._mtf_data = {}

        return df

    def get_latest_price(self) -> float:
        """Devuelve el último precio de cierre disponible."""
        if self._data is None:
            raise RuntimeError("Llama a get_data() primero.")
        return float(self._data["Close"].iloc[-1])

    def get_price_summary(self) -> pd.Series:
        """Devuelve un resumen estadístico de los precios."""
        if self._data is None:
            raise RuntimeError("Llama a get_data() primero.")
        close = self._data["Close"]
        return pd.Series(
            {
                "Último precio":         round(close.iloc[-1], 2),
                "Precio máximo":        round(close.max(),    2),
                "Precio mínimo":        round(close.min(),    2),
                "Promedio":         round(close.mean(),   2),
                "Desviación estándar": round(close.std(),  2),
                "Número de velas":     len(close),
            }
        )

    # ─────────────────────────────────────────────────────────
    # Sección 5: Métodos auxiliares para acceso a datos multi-timeframe
    # ─────────────────────────────────────────────────────────
    def get_mtf_data(self, timeframe: str = "H1") -> Optional[pd.DataFrame]:
        """
        Accede a datos multi-timeframe almacenados tras llamar a get_data(include_mtf=True).

        Parámetros:
            timeframe: "H1", "H4", "D1", etc.

        Retorna:
            DataFrame del timeframe solicitado, o None si no disponible

        Ejemplo:
            >>> fetcher.get_data(include_mtf=True)
            >>> df_h4 = fetcher.get_mtf_data("H4")
        """
        if timeframe not in self._mtf_data:
            logger.warning(
                f"Timeframe '{timeframe}' no disponible. "
                f"Timeframes disponibles: {list(self._mtf_data.keys())}"
            )
            return None
        return self._mtf_data.get(timeframe)

    # ─────────────────────────────────────────────────────────
    # Sección 6: Validación interna
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Valida el DataFrame antes de pasarlo a los indicadores."""
        missing = [c for c in GoldDataFetcher.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        if df.empty:
            raise ValueError("El DataFrame está vacío.")

        null_counts = df[GoldDataFetcher.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            logger.warning(f"Valores faltantes en los datos:\n{null_counts[null_counts > 0]}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El índice debe ser de tipo DatetimeIndex.")

        logger.info("✅ Validación del DataFrame: exitosa")
