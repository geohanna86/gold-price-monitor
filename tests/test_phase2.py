# ============================================================
# test_phase2.py — Pruebas unitarias de la Fase 2
# Gold Price Monitor
#
# Ejecución: python tests/test_phase2.py
# ============================================================

import sys, os, logging, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import AppConfig
from data_fetcher import GoldDataFetcher
from indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN
from ml_model import GoldPredictor, ModelConfig
from backtester import GoldBacktester, BacktestConfig, BacktestResults


# ── Pipeline compartido para todas las pruebas ───────────────────────────
def build_pipeline():
    config = AppConfig(mode="mock")
    df_raw = GoldDataFetcher(config).get_data()
    df_ind = TechnicalIndicators(df_raw, config).add_all().get_dataframe()
    fe     = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    return fe, df_ind


# ════════════════════════════════════════════════════════════
class TestFeatureEngineer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fe, cls.df_ind = build_pipeline()
        cls.full_df = cls.fe.get_full_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.fe.train_test_split(0.20)

    def test_feature_df_not_empty(self):
        self.assertFalse(self.full_df.empty)

    def test_target_column_exists(self):
        self.assertIn(TARGET_COLUMN, self.full_df.columns)

    def test_target_values_valid(self):
        """Target debe ser solo de {-1, 0, +1}."""
        vals = set(self.full_df[TARGET_COLUMN].unique())
        self.assertTrue(vals.issubset({-1, 0, 1}),
                        f"¡Target contiene valores inesperados: {vals}")

    def test_no_null_in_features(self):
        """No debe haber valores NaN en las características."""
        feats = [c for c in FEATURE_COLUMNS if c in self.full_df.columns]
        nulls = self.full_df[feats].isnull().sum().sum()
        self.assertEqual(nulls, 0, f"¡Hay {nulls} valores NaN en las características!")

    def test_no_leakage_in_features(self):
        """
        Verificar sin fuga: Forward_Return (futuro)
        debe existir solo en la columna Target y no en las características.
        """
        feat_names = self.fe.get_feature_names()
        self.assertNotIn("Forward_Return", feat_names,
                         "¡Forward_Return existe en características — fuga del futuro!")

    def test_train_test_chronological_order(self):
        """Train debe preceder a Test cronológicamente."""
        self.assertLess(
            self.X_train.index[-1], self.X_test.index[0],
            "¡Datos de Test preceden a Train cronológicamente — error crítico!"
        )

    def test_test_size_approximately_correct(self):
        """Relación Test aproximadamente 20% del total."""
        total = len(self.X_train) + len(self.X_test)
        ratio = len(self.X_test) / total
        self.assertAlmostEqual(ratio, 0.20, delta=0.05,
                               msg=f"¡Relación Test = {ratio:.2%} lejos de 20%!")

    def test_feature_names_list(self):
        """get_feature_names debe retornar una lista no vacía."""
        names = self.fe.get_feature_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 5,
                           "¡Número de características menor que lo esperado!")

    def test_return_features_no_inf(self):
        """Las características de retorno no deben contener Inf."""
        ret_cols = [c for c in self.full_df.columns if "Return" in c and c != "Forward_Return"]
        for col in ret_cols:
            has_inf = np.isinf(self.full_df[col]).any()
            self.assertFalse(has_inf, f"¡La columna {col} contiene Inf!")


# ════════════════════════════════════════════════════════════
class TestGoldPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fe, _       = build_pipeline()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = fe.train_test_split(0.20)
        cfg         = ModelConfig(n_estimators=50, max_depth=5, n_cv_splits=3)
        cls.pred    = GoldPredictor(cfg)
        cls.pred.train(cls.X_train, cls.y_train, run_cv=True)
        cls.metrics = cls.pred.evaluate(cls.X_test, cls.y_test)

    def test_model_is_trained(self):
        self.assertTrue(self.pred.is_trained())

    def test_predict_returns_array(self):
        preds = self.pred.predict(self.X_test)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predictions_are_valid_classes(self):
        """Las predicciones deben ser de {-1, 0, +1}."""
        preds = self.pred.predict(self.X_test)
        valid = {-1, 0, 1}
        actual = set(np.unique(preds))
        self.assertTrue(actual.issubset(valid),
                        f"¡Predicciones inesperadas: {actual}")

    def test_predict_proba_sums_to_one(self):
        """Las probabilidades de cada muestra deben sumar a 1."""
        probas = self.pred.predict_proba(self.X_test)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="¡Las probabilidades no suman a 1!"
        )

    def test_accuracy_is_valid(self):
        """Accuracy debe estar entre 0 y 1."""
        self.assertGreaterEqual(self.metrics.accuracy, 0.0)
        self.assertLessEqual(self.metrics.accuracy,    1.0)

    def test_f1_is_valid(self):
        self.assertGreaterEqual(self.metrics.f1_weighted, 0.0)
        self.assertLessEqual(self.metrics.f1_weighted,    1.0)

    def test_cv_scores_exist(self):
        """CV Accuracy debe ser calculado."""
        self.assertGreater(self.metrics.cv_accuracy_mean, 0.0)

    def test_feature_importance_covers_all_features(self):
        """Feature Importance debe cubrir todas las características."""
        n_features = len(self.X_train.columns)
        n_imp      = len(self.metrics.feature_importance)
        self.assertEqual(n_features, n_imp,
                         f"¡Feature Importance: {n_imp} vs {n_features} características!")

    def test_feature_importance_sums_to_one(self):
        """La suma de importancia de características debe ser ≈ 1."""
        total = sum(self.metrics.feature_importance.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01,
                               msg=f"¡Suma de Feature Importance = {total:.4f} ≠ 1!")

    def test_predict_with_confidence_shape(self):
        """predict_with_confidence debe retornar la misma longitud de entrada."""
        sigs, conf = self.pred.predict_with_confidence(self.X_test, min_confidence=0.40)
        self.assertEqual(len(sigs), len(self.X_test))
        self.assertEqual(len(conf), len(self.X_test))

    def test_confidence_range(self):
        """El nivel de confianza debe estar entre 0 y 1."""
        _, conf = self.pred.predict_with_confidence(self.X_test)
        self.assertTrue((conf >= 0).all() and (conf <= 1).all(),
                        "¡El nivel de confianza está fuera de [0, 1]!")

    def test_top_features_returns_dict(self):
        top = self.pred.get_top_features(5)
        self.assertIsInstance(top, dict)
        self.assertLessEqual(len(top), 5)

    def test_untrained_raises_error(self):
        """Llamar predict antes de train debe generar RuntimeError."""
        new_pred = GoldPredictor()
        with self.assertRaises(RuntimeError):
            new_pred.predict(self.X_test)


# ════════════════════════════════════════════════════════════
class TestGoldBacktester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Construir pipeline completo
        fe, df_ind    = build_pipeline()
        X_train, X_test, y_train, y_test = fe.train_test_split(0.20)
        full_df       = fe.get_full_data()

        cfg           = ModelConfig(n_estimators=50, max_depth=5, n_cv_splits=3)
        pred          = GoldPredictor(cfg)
        pred.train(X_train, y_train, run_cv=False)

        signals, conf = pred.predict_with_confidence(X_test, min_confidence=0.40)
        test_prices   = full_df.loc[X_test.index, "Close"]

        bt_config     = BacktestConfig(initial_capital=10_000, commission_pct=0.001)
        cls.bt        = GoldBacktester(bt_config)
        cls.results   = cls.bt.run(test_prices, signals)
        cls.prices    = test_prices
        cls.signals   = signals

    def test_results_not_none(self):
        self.assertIsNotNone(self.results)

    def test_final_capital_positive(self):
        """El capital final debe ser siempre positivo."""
        self.assertGreater(self.results.final_capital, 0,
                           "¡Capital final negativo — error en simulación!")

    def test_equity_curve_length(self):
        """La longitud de la curva de capital debe ser igual a la longitud de datos de prueba."""
        self.assertEqual(
            len(self.results.equity_curve), len(self.prices),
            "¡Longitud de curva de capital no coincide con longitud de datos de precio!"
        )

    def test_equity_curve_all_positive(self):
        """La curva de capital debe ser siempre positiva."""
        self.assertTrue(
            (self.results.equity_curve > 0).all(),
            "¡Capital llegó a cero o negativo — error!"
        )

    def test_win_rate_in_valid_range(self):
        """Win Rate debe estar entre 0 y 1."""
        self.assertGreaterEqual(self.results.win_rate, 0.0)
        self.assertLessEqual(self.results.win_rate,    1.0)

    def test_trade_counts_consistent(self):
        """La suma de operaciones ganadoras + perdedoras = operaciones totales."""
        self.assertEqual(
            self.results.winning_trades + self.results.losing_trades,
            self.results.total_trades,
            "¡Número de operaciones inconsistente!"
        )

    def test_commission_non_negative(self):
        """Las comisiones deben ser no negativas."""
        self.assertGreaterEqual(self.results.total_commission, 0)

    def test_max_drawdown_non_positive(self):
        """Max Drawdown debe ser negativo o cero."""
        self.assertLessEqual(self.results.max_drawdown_pct, 0,
                             "¡Max Drawdown positivo — error en cálculo!")

    def test_trades_df_returns_dataframe(self):
        df = self.bt.get_trades_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_bnh_return_correct(self):
        """Verificar precisión del cálculo Buy & Hold."""
        expected_bnh = self.prices.iloc[-1] / self.prices.iloc[0] - 1
        self.assertAlmostEqual(
            self.results.bnh_return_pct, expected_bnh, delta=0.001,
            msg="¡Cálculo Buy & Hold incorrecto!"
        )

    def test_alpha_equals_strategy_minus_bnh(self):
        """Alpha = rendimiento de estrategia - rendimiento Buy & Hold."""
        expected_alpha = self.results.total_return_pct - self.results.bnh_return_pct
        self.assertAlmostEqual(
            self.results.alpha, expected_alpha, delta=0.001,
            msg="¡Cálculo Alpha incorrecto!"
        )

    def test_invalid_input_raises(self):
        """Los datos con longitud diferente deben generar AssertionError."""
        with self.assertRaises(AssertionError):
            self.bt.run(
                self.prices,
                np.ones(len(self.prices) + 5)  # Longitud diferente a propósito
            )


# ════════════════════════════════════════════════════════════
def run_phase2_assertions():
    """Verificación rápida completa con assert — sin pytest."""
    print("🔍 Ejecutando verificaciones de Fase 2...\n")

    # ─ Pipeline ─
    fe, df_ind = build_pipeline()
    X_train, X_test, y_train, y_test = fe.train_test_split(0.20)
    full_df = fe.get_full_data()

    assert not full_df.empty,                       "❌ ¡full_df vacío!"
    assert TARGET_COLUMN in full_df.columns,        "❌ ¡Columna Target faltante!"
    assert set(full_df[TARGET_COLUMN].unique()).issubset({-1,0,1}), "❌ ¡Valores Target incorrectos!"
    assert X_train.index[-1] < X_test.index[0],    "❌ ¡Fuga temporal en división!"
    assert full_df[fe.get_feature_names()].isnull().sum().sum() == 0, "❌ ¡NaN en características!"
    print("  ✅ Feature Engineering: sano sin fuga")

    # ─ Modelo ─
    cfg  = ModelConfig(n_estimators=50, max_depth=5, n_cv_splits=3)
    pred = GoldPredictor(cfg)
    pred.train(X_train, y_train, run_cv=True)
    assert pred.is_trained(),                       "❌ ¡Modelo no entrenado!"

    preds = pred.predict(X_test)
    assert len(preds) == len(X_test),               "❌ ¡Longitud de predicciones incorrecta!"
    assert set(np.unique(preds)).issubset({-1,0,1}),"❌ ¡Predicciones fuera de rango!"

    probas = pred.predict_proba(X_test)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
    print("  ✅ Random Forest: funciona y retorna resultados válidos")

    metrics = pred.evaluate(X_test, y_test)
    assert 0 <= metrics.accuracy <= 1,              "❌ ¡Accuracy fuera de [0,1]!"
    total_imp = sum(metrics.feature_importance.values())
    assert abs(total_imp - 1.0) < 0.01,             f"❌ ¡Feature Importance no suma a 1: {total_imp:.4f}"
    print(f"  ✅ Precisión del modelo: {metrics.accuracy:.2%} | CV: {metrics.cv_accuracy_mean:.2%}")

    # ─ Backtesting ─
    signals, conf = pred.predict_with_confidence(X_test, min_confidence=0.40)
    test_prices   = full_df.loc[X_test.index, "Close"]
    bt            = GoldBacktester(BacktestConfig(initial_capital=10_000))
    res           = bt.run(test_prices, signals)

    assert res.final_capital > 0,                   "❌ ¡Capital negativo!"
    assert len(res.equity_curve) == len(test_prices),"❌ ¡Longitud Equity Curve incorrecta!"
    assert (res.equity_curve > 0).all(),            "❌ ¡Capital llega a cero!"
    assert 0 <= res.win_rate <= 1,                  "❌ ¡Win Rate fuera de [0,1]!"
    assert res.max_drawdown_pct <= 0,               "❌ ¡Max Drawdown positivo!"
    assert res.winning_trades + res.losing_trades == res.total_trades, "❌ ¡Número de operaciones inconsistente!"
    print(f"  ✅ Backtesting: {res.total_trades} operaciones | Rendimiento: {res.total_return_pct:+.2%}")
    print(f"  ✅ Alpha vs Buy&Hold: {res.alpha:+.2%} | Sharpe: {res.sharpe_ratio:.3f}")

    print(f"\n  {'─'*50}")
    print(f"  🎉 ¡Todas las verificaciones de Fase 2 pasaron!")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_phase2_assertions()
    else:
        print("🧪 Ejecutando pruebas unitarias de Fase 2...\n")
        loader = unittest.TestLoader()
        suite  = unittest.TestSuite()
        for cls in [TestFeatureEngineer, TestGoldPredictor, TestGoldBacktester]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print("\n" + "="*65)
        if result.wasSuccessful():
            print("🎉 ¡Todas las pruebas pasaron!")
        else:
            print(f"❌ {len(result.failures)} fallos | {len(result.errors)} errores")
            sys.exit(1)
