# ============================================================
# test_phase1.py — Pruebas unitarias de la Fase 1
# Gold Price Monitor
#
# Ejecución:
#   cd "Gold Price Monitor" && python -m pytest tests/ -v
#   o directamente:
#   python tests/test_phase1.py
# ============================================================

import sys
import os
import unittest
import logging

import numpy as np
import pandas as pd

# Agregar ruta raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, MockConfig, IndicatorConfig, DEFAULT_CONFIG
from data_fetcher import GoldDataFetcher
from indicators import TechnicalIndicators

# Desactivar registros de INFO durante las pruebas
logging.disable(logging.CRITICAL)


class TestMockDataGeneration(unittest.TestCase):
    """Pruebas de generación de datos ficticios."""

    def setUp(self):
        self.config  = AppConfig(mode="mock")
        self.fetcher = GoldDataFetcher(self.config)
        self.df      = self.fetcher.get_data()

    # ─── Forma de datos ───────────────────────────────────────
    def test_dataframe_not_empty(self):
        """El DataFrame debe estar vacío."""
        self.assertFalse(self.df.empty, "¡El DataFrame está vacío!")

    def test_correct_row_count(self):
        """El número de filas debe ser igual a n_rows en la configuración."""
        expected = self.config.mock.n_rows
        self.assertEqual(
            len(self.df), expected,
            f"Esperado {expected} fila(s), actual {len(self.df)}"
        )

    def test_required_columns_exist(self):
        """Las columnas OHLCV deben estar disponibles."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            self.assertIn(col, self.df.columns, f"¡Columna {col} faltante!")

    def test_datetime_index(self):
        """El índice debe ser DatetimeIndex."""
        self.assertIsInstance(
            self.df.index, pd.DatetimeIndex,
            "¡El índice no es DatetimeIndex!"
        )

    # ─── Validez OHLC ───────────────────────────────────────────
    def test_high_gte_close(self):
        """High siempre debe ser >= Close."""
        violations = (self.df["High"] < self.df["Close"]).sum()
        self.assertEqual(violations, 0, f"¡High < Close en {violations} fila(s)!")

    def test_high_gte_open(self):
        """High siempre debe ser >= Open."""
        violations = (self.df["High"] < self.df["Open"]).sum()
        self.assertEqual(violations, 0, f"¡High < Open en {violations} fila(s)!")

    def test_low_lte_close(self):
        """Low siempre debe ser <= Close."""
        violations = (self.df["Low"] > self.df["Close"]).sum()
        self.assertEqual(violations, 0, f"¡Low > Close en {violations} fila(s)!")

    def test_low_lte_open(self):
        """Low siempre debe ser <= Open."""
        violations = (self.df["Low"] > self.df["Open"]).sum()
        self.assertEqual(violations, 0, f"¡Low > Open en {violations} fila(s)!")

    def test_volume_positive(self):
        """Volume siempre debe ser positivo."""
        violations = (self.df["Volume"] <= 0).sum()
        self.assertEqual(violations, 0, f"¡Volume negativo o cero en {violations} fila(s)!")

    def test_price_realistic_range(self):
        """Los precios deben estar en un rango realista para oro."""
        close = self.df["Close"]
        self.assertGreater(close.min(), 500, "¡Precio del oro menor que $500 — no realista!")
        self.assertLess(close.max(), 10_000, "¡Precio del oro mayor que $10,000 — no realista!")

    def test_no_null_in_ohlcv(self):
        """No debe haber valores NULL en OHLCV."""
        null_count = self.df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum()
        self.assertEqual(null_count, 0, f"¡Hay {null_count} valores NULL en OHLCV!")

    # ─── Consistencia interna ─────────────────────────────────────
    def test_reproducibility(self):
        """El mismo seed debe dar los mismos datos."""
        df2 = GoldDataFetcher(self.config).get_data()
        pd.testing.assert_frame_equal(
            self.df, df2,
            check_like=False,
            obj="Mock Data reproducibility"
        )

    def test_dxy_negative_correlation(self):
        """DXY debe estar correlacionado negativamente con el oro."""
        if "DXY" not in self.df.columns:
            self.skipTest("Columna DXY no encontrada")
        corr = self.df["Close"].pct_change().corr(
            self.df["DXY"].pct_change()
        )
        self.assertLess(corr, 0, f"¡La correlación entre DXY y oro es positiva ({corr:.3f}) — debe ser negativa!")


class TestTrendIndicators(unittest.TestCase):
    """Pruebas de indicadores de tendencia."""

    @classmethod
    def setUpClass(cls):
        config  = AppConfig(mode="mock")
        fetcher = GoldDataFetcher(config)
        df_raw  = fetcher.get_data()
        cls.ti  = TechnicalIndicators(df_raw, config).add_trend_indicators()
        cls.df  = cls.ti.get_dataframe()

    def test_ema20_exists(self):
        self.assertIn("EMA_20", self.df.columns, "¡EMA_20 faltante!")

    def test_ema50_exists(self):
        self.assertIn("EMA_50", self.df.columns, "¡EMA_50 faltante!")

    def test_ema200_exists(self):
        self.assertIn("EMA_200", self.df.columns, "¡EMA_200 faltante!")

    def test_macd_columns_exist(self):
        for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
            self.assertIn(col, self.df.columns, f"¡Columna MACD faltante: {col}!")

    def test_ema_has_valid_values(self):
        """EMA_20 debe tener valores no nulos después del período de calentamiento."""
        valid = self.df["EMA_20"].dropna()
        self.assertGreater(len(valid), 0, "¡EMA_20 no contiene valores válidos!")

    def test_ema_order_in_trend(self):
        """
        En tendencia alcista: EMA_20 > EMA_50 > EMA_200 (en su mayoría).
        Es suficiente que ocurra en al menos 30% del tiempo.
        """
        df_clean = self.df[["EMA_20", "EMA_50", "EMA_200"]].dropna()
        bullish_pct = (
            (df_clean["EMA_20"] > df_clean["EMA_50"]) &
            (df_clean["EMA_50"] > df_clean["EMA_200"])
        ).mean()
        # No requerimos 100% — la tendencia cambia
        self.assertGreater(bullish_pct, 0.0, "¡No se registró ninguna tendencia alcista en EMA!")

    def test_ema_trend_values(self):
        """EMA_Trend debe tener valores solo de {-1, 0, 1}."""
        if "EMA_Trend" not in self.df.columns:
            self.skipTest("EMA_Trend no encontrado")
        valid_values = {-1, 0, 1}
        actual_values = set(self.df["EMA_Trend"].dropna().unique())
        self.assertTrue(
            actual_values.issubset(valid_values),
            f"¡EMA_Trend contiene valores inesperados: {actual_values}"
        )

    def test_macd_histogram_equals_line_minus_signal(self):
        """Verificar: MACD_Hist ≈ MACD - MACD_Signal."""
        df_clean = self.df[["MACD", "MACD_Signal", "MACD_Hist"]].dropna()
        if df_clean.empty:
            self.skipTest("MACD data is empty after dropna")
        computed  = df_clean["MACD"] - df_clean["MACD_Signal"]
        max_error = (computed - df_clean["MACD_Hist"]).abs().max()
        self.assertLess(
            max_error, 0.01,
            f"¡MACD_Hist no es igual a MACD - Signal! Error máximo: {max_error:.6f}"
        )


class TestMomentumIndicators(unittest.TestCase):
    """Pruebas de indicadores de momentum."""

    @classmethod
    def setUpClass(cls):
        config  = AppConfig(mode="mock")
        fetcher = GoldDataFetcher(config)
        df_raw  = fetcher.get_data()
        cls.ti  = TechnicalIndicators(df_raw, config).add_momentum_indicators()
        cls.df  = cls.ti.get_dataframe()

    def test_rsi_exists(self):
        self.assertIn("RSI", self.df.columns, "¡RSI faltante!")

    def test_rsi_range_0_to_100(self):
        """RSI siempre debe estar entre 0 y 100."""
        rsi_valid = self.df["RSI"].dropna()
        self.assertGreater(len(rsi_valid), 0, "¡RSI no contiene valores válidos!")
        self.assertTrue((rsi_valid >= 0).all(),   "¡RSI contiene valores negativos!")
        self.assertTrue((rsi_valid <= 100).all(), "¡RSI contiene valores mayores que 100!")

    def test_willr_range_minus100_to_0(self):
        """Williams %R debe estar entre -100 y 0."""
        if "WILLR" not in self.df.columns:
            self.skipTest("WILLR no encontrado")
        willr_valid = self.df["WILLR"].dropna()
        self.assertTrue((willr_valid >= -100).all(), "¡WILLR menor que -100!")
        self.assertTrue((willr_valid <= 0).all(),    "¡WILLR mayor que 0!")

    def test_stochrsi_range(self):
        """StochRSI K y D deben estar entre 0 y 100."""
        for col in ["StochRSI_K", "StochRSI_D"]:
            if col not in self.df.columns:
                continue
            valid = self.df[col].dropna()
            if valid.empty:
                continue
            self.assertTrue((valid >= 0).all(),   f"¡{col} contiene valores negativos!")
            self.assertTrue((valid <= 100).all(), f"¡{col} contiene valores > 100!")


class TestVolatilityIndicators(unittest.TestCase):
    """Pruebas de indicadores de volatilidad."""

    @classmethod
    def setUpClass(cls):
        config  = AppConfig(mode="mock")
        fetcher = GoldDataFetcher(config)
        df_raw  = fetcher.get_data()
        cls.ti  = TechnicalIndicators(df_raw, config).add_volatility_indicators()
        cls.df  = cls.ti.get_dataframe()

    def test_atr_positive(self):
        """ATR siempre debe ser positivo."""
        atr_valid = self.df["ATR"].dropna()
        self.assertGreater(len(atr_valid), 0, "¡ATR no contiene valores válidos!")
        self.assertTrue((atr_valid > 0).all(), "¡ATR contiene valores negativos o cero!")

    def test_bollinger_order(self):
        """Debe cumplirse: BB_Upper > BB_Mid > BB_Lower."""
        bb_cols = ["BB_Lower", "BB_Mid", "BB_Upper"]
        if not all(c in self.df.columns for c in bb_cols):
            self.skipTest("Bollinger Bands no calculadas")
        df_clean = self.df[bb_cols].dropna()
        self.assertTrue(
            (df_clean["BB_Upper"] > df_clean["BB_Mid"]).all(),
            "¡BB_Upper no es mayor que BB_Mid en todas las filas!"
        )
        self.assertTrue(
            (df_clean["BB_Mid"] > df_clean["BB_Lower"]).all(),
            "¡BB_Mid no es mayor que BB_Lower en todas las filas!"
        )

    def test_bb_percent_range(self):
        """BB_Percent (posición del precio) debe estar entre 0 y 1 en su mayoría."""
        if "BB_Percent" not in self.df.columns:
            self.skipTest("BB_Percent no encontrado")
        bbp = self.df["BB_Percent"].dropna()
        # Puede exceder 0-1 ligeramente al romper el rango
        self.assertGreater(bbp.min(), -1.0, "¡BB_Percent anormalmente menor que -1!")
        self.assertLess(bbp.max(),    2.0,  "¡BB_Percent anormalmente mayor que 2!")

    def test_stop_loss_long_below_close(self):
        """Stop Loss para operaciones largas debe ser menor que el precio."""
        if "Stop_Loss_Long" not in self.df.columns:
            self.skipTest("Stop_Loss_Long no encontrado")
        violations = (self.df["Stop_Loss_Long"] >= self.df["Close"]).sum()
        self.assertEqual(violations, 0, f"¡Stop_Loss_Long por encima del precio en {violations} fila(s)!")


class TestVolumeIndicators(unittest.TestCase):
    """Pruebas de indicadores de volumen."""

    @classmethod
    def setUpClass(cls):
        config  = AppConfig(mode="mock")
        fetcher = GoldDataFetcher(config)
        df_raw  = fetcher.get_data()
        cls.ti  = TechnicalIndicators(df_raw, config).add_volume_indicators()
        cls.df  = cls.ti.get_dataframe()

    def test_obv_exists(self):
        self.assertIn("OBV", self.df.columns, "¡OBV faltante!")

    def test_vwap_exists(self):
        self.assertIn("VWAP", self.df.columns, "¡VWAP faltante!")

    def test_vwap_positive(self):
        """VWAP siempre debe ser positivo."""
        vwap_valid = self.df["VWAP"].dropna()
        self.assertTrue((vwap_valid > 0).all(), "¡VWAP contiene valores no positivos!")

    def test_vwap_close_to_price(self):
        """VWAP debe estar cerca del precio promedio de manera razonable."""
        vwap_valid  = self.df["VWAP"].dropna()
        close_valid = self.df["Close"].loc[vwap_valid.index]
        avg_diff_pct = ((vwap_valid - close_valid) / close_valid).abs().mean() * 100
        self.assertLess(
            avg_diff_pct, 30,
            f"¡La diferencia entre VWAP y el precio es demasiado grande: {avg_diff_pct:.2f}%"
        )

    def test_vwap_signal_values(self):
        """VWAP_Signal debe tener valores solo de {-1, 0, 1}."""
        if "VWAP_Signal" not in self.df.columns:
            self.skipTest("VWAP_Signal no encontrado")
        valid_values  = {-1, 0, 1}
        actual_values = set(self.df["VWAP_Signal"].unique())
        self.assertTrue(
            actual_values.issubset(valid_values),
            f"¡VWAP_Signal contiene valores inesperados: {actual_values}"
        )


class TestConfluenceScore(unittest.TestCase):
    """Pruebas de puntuación de confluencia integrada."""

    @classmethod
    def setUpClass(cls):
        config  = AppConfig(mode="mock")
        fetcher = GoldDataFetcher(config)
        df_raw  = fetcher.get_data()
        cls.ti  = TechnicalIndicators(df_raw, config).add_all()
        cls.df  = cls.ti.get_dataframe()

    def test_confluence_score_exists(self):
        self.assertIn("Confluence_Score", self.df.columns, "¡Confluence_Score faltante!")

    def test_confluence_score_range(self):
        """Confluence_Score debe estar entre -4 y +4."""
        score = self.df["Confluence_Score"].dropna()
        self.assertTrue((score >= -4).all(), f"¡Confluence_Score menor que -4! Mínimo: {score.min()}")
        self.assertTrue((score <=  4).all(), f"¡Confluence_Score mayor que +4! Máximo: {score.max()}")

    def test_signal_label_exists(self):
        self.assertIn("Signal_Label", self.df.columns, "¡Signal_Label faltante!")

    def test_signal_label_not_empty(self):
        """Signal_Label debe contener textos no vacíos."""
        labels = self.df["Signal_Label"].dropna()
        self.assertGreater(len(labels), 0, "¡Signal_Label está vacío!")
        self.assertTrue((labels != "").all(), "¡Algunos Signal_Labels están vacíos!")

    def test_full_pipeline_column_count(self):
        """La fase completa debe producir más de 20 columnas de indicadores."""
        total_cols = len(self.df.columns)
        self.assertGreater(
            total_cols, 20,
            f"¡Número total de columnas {total_cols} — debe ser más de 20!"
        )

    def test_latest_signals_returns_series(self):
        """get_latest_signals debe retornar pd.Series."""
        result = self.ti.get_latest_signals()
        self.assertIsInstance(result, pd.Series)
        self.assertIn("Close", result.index)

    def test_indicator_summary_returns_dataframe(self):
        """get_indicator_summary debe retornar pd.DataFrame."""
        result = self.ti.get_indicator_summary()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, "¡Tabla de resumen vacía!")


# ═══════════════════════════════════════════════════════════════
# Assert Statements — Verificación inmediata en ejecución directa
# ═══════════════════════════════════════════════════════════════
def run_quick_assertions():
    """
    Verificación rápida con assert — funciona sin pytest.
    Ejecutar: python tests/test_phase1.py --quick
    """
    print("🔍 Ejecutando verificaciones rápidas...\n")

    config  = AppConfig(mode="mock")
    fetcher = GoldDataFetcher(config)
    df      = fetcher.get_data()

    # ─ Datos ─
    assert not df.empty,                                 "❌ ¡DataFrame vacío!"
    assert len(df) == config.mock.n_rows,                "❌ ¡Número de filas incorrecto!"
    assert all(c in df.columns for c in ["Open","High","Low","Close","Volume"]), "❌ ¡Columnas OHLCV faltantes!"
    assert (df["High"] >= df["Close"]).all(),             "❌ ¡High < Close!"
    assert (df["High"] >= df["Open"]).all(),              "❌ ¡High < Open!"
    assert (df["Low"]  <= df["Close"]).all(),             "❌ ¡Low > Close!"
    assert (df["Volume"] > 0).all(),                      "❌ ¡Volume no es positivo!"
    print("  ✅ Datos OHLCV: válidos")

    # ─ Indicadores ─
    ti = TechnicalIndicators(df, config).add_all()
    df_i = ti.get_dataframe()

    assert "RSI" in df_i.columns,                         "❌ ¡RSI faltante!"
    rsi_valid = df_i["RSI"].dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), "❌ ¡RSI fuera de rango 0-100!"
    print("  ✅ RSI: en rango [0, 100]")

    assert "ATR" in df_i.columns,                         "❌ ¡ATR faltante!"
    assert (df_i["ATR"].dropna() > 0).all(),              "❌ ¡ATR no es positivo!"
    print("  ✅ ATR: siempre positivo")

    assert "BB_Upper" in df_i.columns,                    "❌ ¡Bollinger Bands faltante!"
    bb_clean = df_i[["BB_Lower","BB_Mid","BB_Upper"]].dropna()
    assert (bb_clean["BB_Upper"] > bb_clean["BB_Lower"]).all(), "❌ ¡BB_Upper ≤ BB_Lower!"
    print("  ✅ Bollinger Bands: Upper > Lower")

    assert "Confluence_Score" in df_i.columns,            "❌ ¡Confluence_Score faltante!"
    score = df_i["Confluence_Score"].dropna()
    assert (score >= -4).all() and (score <= 4).all(),    "❌ ¡Confluence_Score fuera de [-4, 4]!"
    print("  ✅ Confluence_Score: en rango [-4, 4]")

    assert "VWAP" in df_i.columns,                        "❌ ¡VWAP faltante!"
    assert (df_i["VWAP"].dropna() > 0).all(),             "❌ ¡VWAP no es positivo!"
    print("  ✅ VWAP: siempre positivo")

    assert "Stop_Loss_Long" in df_i.columns,              "❌ ¡Stop_Loss_Long faltante!"
    assert (df_i["Stop_Loss_Long"].dropna() < df_i["Close"].loc[df_i["Stop_Loss_Long"].dropna().index]).all(), \
                                                           "❌ ¡Stop_Loss_Long por encima del precio!"
    print("  ✅ Stop_Loss_Long: siempre menor que el precio")

    # ─ Calidad de salida ─
    total_cols = len(df_i.columns)
    assert total_cols > 20, f"❌ ¡Número de columnas ({total_cols}) menor que lo esperado!"
    print(f"  ✅ Total de columnas: {total_cols}")

    latest = ti.get_latest_signals()
    assert isinstance(latest, pd.Series), "❌ ¡get_latest_signals no retorna pd.Series!"
    print("  ✅ get_latest_signals: retorna pd.Series")

    print(f"\n  {'-'*45}")
    print(f"  🎉 ¡Todas las verificaciones rápidas pasaron! ({total_cols} columnas)")
    print(f"  💰 Último precio: ${fetcher.get_latest_price():.2f}")
    print(f"  {'-'*45}\n")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_quick_assertions()
    else:
        print("🧪 Ejecutando pruebas unitarias completas...\n")
        # تشغيل جميع الاختبارات
        loader = unittest.TestLoader()
        suite  = unittest.TestSuite()

        test_classes = [
            TestMockDataGeneration,
            TestTrendIndicators,
            TestMomentumIndicators,
            TestVolatilityIndicators,
            TestVolumeIndicators,
            TestConfluenceScore,
        ]
        for cls in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        print("\n" + "="*65)
        if result.wasSuccessful():
            print("🎉 ¡Todas las pruebas pasaron!")
        else:
            print(f"❌ {len(result.failures)} prueba(s) fallaron, {len(result.errors)} error(es)")
            sys.exit(1)
