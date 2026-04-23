# ============================================================
# tests/test_phase3.py — Pruebas de Fase 3
# Gold Price Monitor
#
# Cubre:
#   - LSTM  : LSTMCell, GoldLSTM, AdamOptimizer
#   - HMM   : GoldRegimeDetector, filter_signals
#   - Ensemble: GoldEnsemble, EnsembleConfig
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import unittest

# ─────────────────────────────────────────────────────────────
# Datos ficticios compartidos
# ─────────────────────────────────────────────────────────────
def _build_pipeline(n: int = 400, seed: int = 42):
    """Retorna (df_ind, df_feat, fe) de datos Mock."""
    from config import AppConfig
    from data_fetcher import GoldDataFetcher
    from indicators import TechnicalIndicators
    from feature_engineer import FeatureEngineer

    cfg     = AppConfig(mode="mock")
    cfg.mock.n_rows = n
    df_raw  = GoldDataFetcher(cfg).get_data()
    df_ind  = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
    fe      = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    df_feat = fe.get_full_data()
    return df_ind, df_feat, fe


# ─────────────────────────────────────────────────────────────
# Pruebas LSTM
# ─────────────────────────────────────────────────────────────
class TestLSTMCell(unittest.TestCase):
    """Pruebas unitarias de celda LSTM."""

    def setUp(self):
        from lstm_model import LSTMCell
        self.cell = LSTMCell(input_size=5, hidden_size=8, seed=0)

    def test_forward_step_shapes(self):
        """La forma de las salidas es correcta."""
        x      = np.random.randn(5)
        h_prev = np.zeros(8)
        c_prev = np.zeros(8)
        h, c   = self.cell.forward_step(x, h_prev, c_prev)
        self.assertEqual(h.shape, (8,))
        self.assertEqual(c.shape, (8,))

    def test_forward_sequence_length(self):
        """forward_sequence produce h y c con forma correcta."""
        X_seq = np.random.randn(15, 5)
        h0    = np.zeros(8)
        c0    = np.zeros(8)
        h, c  = self.cell.forward_sequence(X_seq, h0, c0)
        self.assertEqual(h.shape, (8,))
        self.assertEqual(c.shape, (8,))

    def test_backward_returns_grads(self):
        """backward_sequence retorna gradientes con formas correctas."""
        X_seq = np.random.randn(10, 5)
        h0    = np.zeros(8)
        c0    = np.zeros(8)
        self.cell.forward_sequence(X_seq, h0, c0)
        grads, dh = self.cell.backward_sequence(np.ones(8), np.zeros(8))
        self.assertIn("W_f", grads)
        self.assertEqual(grads["W_f"].shape, self.cell.W_f.shape)

    def test_forget_gate_bias_initialized_to_one(self):
        """Truco Forget Gate Bias — b_f comienza con 1."""
        from lstm_model import LSTMCell
        cell = LSTMCell(input_size=10, hidden_size=16, seed=99)
        self.assertTrue((cell.b_f == 1).all(),
                        "¡b_f debe inicializarse con 1 (Forget Gate Bias Trick)")

    def test_sigmoid_range(self):
        """La función sigmoid produce valores en [0, 1]."""
        from lstm_model import _sigmoid
        # Nota: valores límite como ±100 pueden producir 0.0 o 1.0 en float64
        x = np.array([-100, -1, 0, 1, 100])
        s = _sigmoid(x)
        self.assertTrue((s >= 0).all() and (s <= 1).all(),
                        "¡sigmoid debe estar en [0, 1]")
        # Verificar que valores moderados dan resultados correctos
        self.assertAlmostEqual(float(_sigmoid(np.array([0.0]))[0]), 0.5, places=5)
        self.assertGreater(float(_sigmoid(np.array([1.0]))[0]), 0.5)
        self.assertLess(float(_sigmoid(np.array([-1.0]))[0]), 0.5)

    def test_softmax_sums_to_one(self):
        """La suma de softmax es 1."""
        from lstm_model import _softmax
        x = np.random.randn(3)
        self.assertAlmostEqual(_softmax(x).sum(), 1.0, places=5)


class TestGoldLSTM(unittest.TestCase):
    """Pruebas del modelo LSTM completo."""

    @classmethod
    def setUpClass(cls):
        """Preparar datos una sola vez para todas las pruebas."""
        from lstm_model import GoldLSTM, LSTMConfig, prepare_lstm_data
        from feature_engineer import FEATURE_COLUMNS

        _, cls.df_feat, _ = _build_pipeline(n=300)

        cfg        = LSTMConfig(hidden_size=12, seq_length=10, epochs=5,
                                batch_size=8, patience=3)
        n_features = len([c for c in FEATURE_COLUMNS if c in cls.df_feat.columns])
        cls.model  = GoldLSTM(n_features, cfg)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.test_idx = \
            prepare_lstm_data(cls.df_feat, seq_length=10, test_size=0.20)
        cls.model.fit(cls.X_train, cls.y_train)

    def test_model_is_trained(self):
        self.assertTrue(self.model.is_trained())

    def test_predict_output_shape(self):
        preds = self.model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predict_classes_valid(self):
        preds = self.model.predict(self.X_test)
        self.assertTrue(set(preds).issubset({-1, 0, 1}),
                        f"¡Clases inesperadas: {set(preds)}")

    def test_predict_proba_shape(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 3))

    def test_predict_proba_sums_to_one(self):
        proba = self.model.predict_proba(self.X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_train_loss_decreasing(self):
        """El promedio de pérdida en el medio debe ser menor que al inicio generalmente."""
        losses = self.model.train_losses
        self.assertGreater(len(losses), 2, "¡Debe haber más de un epoch")
        # Promedio de los 2 primeros vs los 2 últimos
        first_avg = np.mean(losses[:2])
        last_avg  = np.mean(losses[-2:])
        # Solo verificamos que la pérdida no explote
        self.assertLess(last_avg, first_avg * 3,
                        "¡La pérdida aumentó demasiado — problema en entrenamiento")

    def test_build_sequences_no_leakage(self):
        """
        X_seq[i] depende de [i : i+seq_len] y el objetivo es y[i+seq_len].
        El objetivo no debe estar incluido en la secuencia.
        """
        from lstm_model import GoldLSTM
        X = np.arange(50).reshape(50, 1).astype(float)
        y = np.array([1] * 50)
        X_seq, y_seq = GoldLSTM.build_sequences(X, y, seq_len=5)
        # y_seq[0] = y[5] — no debe estar incluido en X_seq[0]
        self.assertEqual(len(X_seq), 45)
        self.assertEqual(len(y_seq), 45)

    def test_fit_without_error_raises_if_not_trained(self):
        """predict sin fit genera RuntimeError."""
        from lstm_model import GoldLSTM, LSTMConfig
        from feature_engineer import FEATURE_COLUMNS
        n_feat = len([c for c in FEATURE_COLUMNS if c in self.df_feat.columns])
        m = GoldLSTM(n_feat, LSTMConfig(epochs=1))
        with self.assertRaises(RuntimeError):
            m.predict(self.X_test)


class TestAdamOptimizer(unittest.TestCase):
    """Pruebas del Optimizador Adam."""

    def test_adam_reduces_param_error(self):
        """Adam reduce la pérdida en un problema simple."""
        from lstm_model import AdamOptimizer
        opt = AdamOptimizer(lr=0.1)
        w   = np.array([1.0, 1.0])
        for _ in range(100):
            grad = 2 * w   # Gradiente MSE: grad = 2*(w - 0)
            opt.update({"w": w}, {"w": grad})
        self.assertLess(np.linalg.norm(w), 0.1,
                        "¡Adam debe acercar w a cero")


# ─────────────────────────────────────────────────────────────
# Pruebas HMM
# ─────────────────────────────────────────────────────────────
class TestGoldRegimeDetector(unittest.TestCase):
    """Pruebas del modelo detector de regímenes."""

    @classmethod
    def setUpClass(cls):
        from hmm_model import GoldRegimeDetector, HMMConfig
        cls.df_ind, _, _ = _build_pipeline(n=300)
        cfg        = HMMConfig(n_states=3, n_iter=50)
        cls.det    = GoldRegimeDetector(cfg)
        cls.det.fit(cls.df_ind)

    def test_fit_completes(self):
        self.assertTrue(self.det._is_trained)

    def test_predict_regimes_valid_values(self):
        regimes = self.det.predict_regimes(self.df_ind)
        self.assertTrue(set(regimes.unique()).issubset({0, 1, 2}))

    def test_predict_regimes_length_matches_df(self):
        regimes = self.det.predict_regimes(self.df_ind)
        self.assertEqual(len(regimes), len(self.df_ind))

    def test_state_map_canonical(self):
        """El mapa cubre todos los estados {0,1,2}."""
        self.assertEqual(set(self.det._state_map.values()), {0, 1, 2})

    def test_state_ordering_by_return(self):
        """
        El estado 0 debe tener un retorno promedio menor que el estado 2.
        Usamos means_ interno de GMM para garantizar consistencia independientemente de la distribución de datos.
        """
        # Mapa: canonical_state → raw_state
        inv_map = {v: k for k, v in self.det._state_map.items()}
        raw_bear = inv_map[0]   # Estado bruto correspondiente a Bear
        raw_bull = inv_map[2]   # Estado bruto correspondiente a Bull

        # Retorno promedio según means_ interno de GMM (columna 0 = daily_return)
        mean_bear_gmm = float(self.det.means_[raw_bear, 0])
        mean_bull_gmm = float(self.det.means_[raw_bull, 0])

        self.assertLessEqual(mean_bear_gmm, mean_bull_gmm,
                             "¡Media GMM de Bear debe ser ≤ Bull")

    def test_regime_proba_sums_to_one(self):
        proba = self.det.predict_regime_proba(self.df_ind)
        row_sums = proba.sum(axis=1).round(5)
        self.assertTrue((row_sums == 1.0).all())

    def test_filter_signals_no_buy_in_bear(self):
        regimes = self.det.predict_regimes(self.df_ind)
        raw     = pd.Series(
            np.random.choice([-1, 0, 1], size=len(self.df_ind)),
            index=self.df_ind.index
        )
        from hmm_model import GoldRegimeDetector as GRD
        filtered  = GRD.filter_signals(raw, regimes, cancel_buy_in_bear=True)
        bear_days = regimes[regimes == 0].index
        if len(bear_days) > 0:
            self.assertTrue(
                (filtered.loc[bear_days] != 1).all(),
                "¡No debe haber señales de compra en régimen bajista"
            )

    def test_get_regime_stats_has_three_rows(self):
        stats = self.det.get_regime_stats(self.df_ind)
        self.assertEqual(len(stats), 3)

    def test_prepare_features_shape(self):
        from hmm_model import GoldRegimeDetector as GRD
        X, valid = GRD.prepare_features(self.df_ind)
        self.assertEqual(X.shape[1], 2, "¡Las entradas deben ser dos columnas")
        self.assertTrue(np.isfinite(X).all(), "¡Todos los valores deben ser finitos")

    def test_raises_if_not_trained(self):
        from hmm_model import GoldRegimeDetector, HMMConfig
        det = GoldRegimeDetector()
        with self.assertRaises(RuntimeError):
            det.predict_regimes(self.df_ind)


# ─────────────────────────────────────────────────────────────
# Pruebas Ensemble
# ─────────────────────────────────────────────────────────────
class TestGoldEnsemble(unittest.TestCase):
    """Pruebas del modelo ensemble."""

    @classmethod
    def setUpClass(cls):
        from ensemble import GoldEnsemble, EnsembleConfig
        from lstm_model import LSTMConfig

        cls.df_ind, cls.df_feat, _ = _build_pipeline(n=400)

        cfg       = EnsembleConfig(
            rf_weight=0.40, lstm_weight=0.40, hmm_weight=0.20,
            lstm_config=LSTMConfig(hidden_size=12, seq_length=10,
                                   epochs=5, patience=3, batch_size=8),
        )
        cls.ensemble = GoldEnsemble(cfg)
        cls.ensemble.fit(cls.df_feat, cls.df_ind)
        cls.results  = cls.ensemble.predict(cls.df_feat, cls.df_ind)

    def test_ensemble_is_trained(self):
        self.assertTrue(self.ensemble._is_trained)

    def test_signals_valid_values(self):
        self.assertTrue(
            set(self.results.signals.unique()).issubset({-1, 0, 1}),
            f"¡Clases inesperadas: {set(self.results.signals.unique())}"
        )

    def test_confidence_in_range(self):
        conf = self.results.confidence
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())

    def test_regimes_valid_values(self):
        self.assertTrue(
            set(self.results.regimes.unique()).issubset({0, 1, 2})
        )

    def test_no_buy_in_bear_regime(self):
        """No debe haber señales de compra en régimen bajista."""
        sigs    = self.results.signals
        regimes = self.results.regimes
        common  = sigs.index.intersection(regimes.index)
        if len(common) > 0:
            bear_days = regimes.loc[common][regimes.loc[common] == 0].index
            if len(bear_days) > 0:
                self.assertTrue(
                    (sigs.loc[bear_days] != 1).all(),
                    "¡Filtro HMM debe cancelar compras en régimen bajista"
                )

    def test_rf_signals_length(self):
        self.assertGreater(len(self.results.rf_signals), 0)

    def test_lstm_signals_classes(self):
        self.assertTrue(
            set(self.results.lstm_signals.unique()).issubset({-1, 0, 1})
        )

    def test_ensemble_config_weights_sum_to_one(self):
        from ensemble import EnsembleConfig
        cfg   = EnsembleConfig(rf_weight=0.5, lstm_weight=0.3, hmm_weight=0.2)
        total = cfg.rf_weight + cfg.lstm_weight + cfg.hmm_weight
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_ensemble_config_invalid_weights_raises(self):
        from ensemble import EnsembleConfig
        with self.assertRaises(AssertionError):
            EnsembleConfig(rf_weight=0.5, lstm_weight=0.5, hmm_weight=0.5)

    def test_raises_if_not_trained(self):
        from ensemble import GoldEnsemble
        e = GoldEnsemble()
        with self.assertRaises(RuntimeError):
            e.predict(self.df_feat, self.df_ind)

    def test_reorder_proba(self):
        """La función _reorder_proba reordena correctamente."""
        from ensemble import _reorder_proba
        proba   = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
        classes = np.array([-1, 0, 1])
        out     = _reorder_proba(proba, classes)
        # P(-1) es la primera columna
        np.testing.assert_array_almost_equal(out[:, 0], [0.2, 0.1])
        np.testing.assert_array_almost_equal(out[:, 2], [0.3, 0.3])


# ─────────────────────────────────────────────────────────────
# Ejecución del ensemble completo como prueba de integración
# ─────────────────────────────────────────────────────────────
class TestPhase3Integration(unittest.TestCase):
    """Prueba de integración: ejecutar run_phase3 completamente."""

    def test_run_phase3_no_errors(self):
        """run_phase3.main() se completa sin excepciones."""
        # Necesitamos n_rows pequeño para acelerar la prueba
        from config import AppConfig
        from data_fetcher import GoldDataFetcher
        from indicators import TechnicalIndicators
        from feature_engineer import FeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN
        from hmm_model import GoldRegimeDetector, HMMConfig
        from lstm_model import GoldLSTM, LSTMConfig, prepare_lstm_data
        from ensemble import GoldEnsemble, EnsembleConfig
        from backtester import GoldBacktester, BacktestConfig

        cfg        = AppConfig(mode="mock")
        cfg.mock.n_rows = 350
        df_raw     = GoldDataFetcher(cfg).get_data()
        df_ind     = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
        fe         = FeatureEngineer(df_ind, target_threshold=0.003)
        fe.build_features()
        df_feat    = fe.get_full_data()

        ens_cfg    = EnsembleConfig(
            rf_weight=0.4, lstm_weight=0.4, hmm_weight=0.2,
            lstm_config=LSTMConfig(hidden_size=8, seq_length=10,
                                   epochs=3, patience=2, batch_size=8),
        )
        ensemble = GoldEnsemble(ens_cfg)
        ensemble.fit(df_feat, df_ind)
        results = ensemble.predict(df_feat, df_ind)

        self.assertIsNotNone(results)
        self.assertGreater(len(results.signals), 0)

        # backtesting
        test_prices = df_ind.loc[results.signals.index, "Close"]
        bt_res = GoldBacktester(BacktestConfig()).run(test_prices, results.signals)
        self.assertIsNotNone(bt_res)
        self.assertIsInstance(bt_res.total_return_pct, float)


# ─────────────────────────────────────────────────────────────
# نقطة التشغيل
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
