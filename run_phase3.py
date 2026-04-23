# ============================================================
# run_phase3.py — Corredor completo para Fase 3
# Gold Price Monitor
#
# Componentes:
#   1. LSTM desde cero (numpy)
#   2. HMM para detección de régimen de mercado
#   3. Ensemble: RF(0.4) + LSTM(0.4) + HMM_filter(0.2)
#   4. Backtesting comparado: RF solo vs Ensemble
#
# Ejecución:
#   python run_phase3.py
# ============================================================

import logging
import os
import sys

import numpy as np
import pandas as pd

from config import AppConfig
from data_fetcher import GoldDataFetcher
from indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer
from ml_model import GoldPredictor, ModelConfig
from backtester import GoldBacktester, BacktestConfig
from lstm_model import GoldLSTM, LSTMConfig, prepare_lstm_data
from hmm_model import GoldRegimeDetector, HMMConfig
from ensemble import GoldEnsemble, EnsembleConfig

logging.basicConfig(
    level=logging.WARNING,   # Ocultamos logs detallados — mostramos solo resumen
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Phase3")

# Colores de terminal
GOLD, BLUE, GREEN, RED, RESET, BOLD = (
    "\033[33m", "\033[94m", "\033[92m", "\033[91m", "\033[0m", "\033[1m"
)
SEP = "=" * 65


def print_step(n: int, title: str):
    print(f"\n{BLUE}{BOLD}{SEP}")
    print(f"  Paso {n}: {title}")
    print(f"{SEP}{RESET}")


def print_metric(label: str, value: str, color: str = ""):
    c = color or ""
    print(f"  {label:<30}: {c}{BOLD}{value}{RESET}")


def main():
    print(f"\n{GOLD}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   🧠  Gold Price Monitor — Phase 3                       ║")
    print("  ║   LSTM + HMM + Ensemble Predictor                        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{RESET}")

    # ══ Paso 1: Datos e indicadores ════════════════════════════
    print_step(1, "Obteniendo datos y calculando indicadores")
    config  = AppConfig(mode="mock")
    df_raw  = GoldDataFetcher(config).get_data()
    df_ind  = (
        TechnicalIndicators(df_raw, config)
        .add_all()
        .get_dataframe()
    )
    print(f"  ✅ Datos listos | Filas: {len(df_ind)} | Columnas: {len(df_ind.columns)}")

    # ══ Paso 2: Ingeniería de características ══════════════════════════════════
    print_step(2, "Ingeniería de características")
    fe      = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    df_feat = fe.get_full_data()
    print(f"  ✅ Características: {len(fe.get_feature_names())} | Filas: {len(df_feat)}")

    # ══ Paso 3: Entrenar HMM ══════════════════════════════════════
    print_step(3, "Sistema de detección de régimen (HMM)")
    hmm_cfg  = HMMConfig(n_states=3, n_iter=100)
    detector = GoldRegimeDetector(hmm_cfg)
    detector.fit(df_ind)   # Usa Close + ATR_Pct

    regimes_full = detector.predict_regimes(df_ind)
    stats        = detector.get_regime_stats(df_ind)
    print(f"\n{GOLD}  Estadísticas de regímenes de mercado:{RESET}")
    for _, row in stats.iterrows():
        print(f"    {row['regime']:<20} | Días: {row['num_days']:<4} "
              f"({row['pct_time']}) | Retorno promedio: {row['avg_return']}")

    # ══ Paso 4: Entrenar red neuronal LSTM ══════════════════════════════════════
    print_step(4, "Entrenamiento de red neuronal LSTM")
    lstm_cfg  = LSTMConfig(
        hidden_size=48, seq_length=15, epochs=80,
        batch_size=16, learning_rate=0.001,
        dropout=0.15, patience=15,
    )
    # تحضير بيانات LSTM
    from feature_engineer import FEATURE_COLUMNS
    n_features = len([c for c in FEATURE_COLUMNS if c in df_feat.columns])

    X_train_seq, X_test_seq, y_train_lstm, y_test_lstm, test_lstm_idx = \
        prepare_lstm_data(df_feat, lstm_cfg.seq_length, test_size=0.20)

    lstm_model = GoldLSTM(n_features, lstm_cfg)
    # Activar logging solo para LSTM
    logging.getLogger("LSTMModel").setLevel(logging.INFO)
    lstm_model.fit(X_train_seq, y_train_lstm)
    logging.getLogger("LSTMModel").setLevel(logging.WARNING)

    # Evaluar LSTM
    lstm_preds = lstm_model.predict(X_test_seq)
    lstm_acc   = (lstm_preds == y_test_lstm).mean()
    print(f"\n  ✅ Precisión LSTM en conjunto de prueba: {BOLD}{lstm_acc:.2%}{RESET}")

    # ══ Paso 5: Entrenar Ensemble ══════════════════════════════════
    print_step(5, "Entrenamiento del modelo Ensemble (RF + LSTM + HMM)")
    ens_cfg = EnsembleConfig(
        rf_weight=0.40, lstm_weight=0.40, hmm_weight=0.20,
        lstm_config=lstm_cfg,
    )
    ensemble = GoldEnsemble(ens_cfg)
    ensemble.fit(df_feat, df_ind)

    # ══ Paso 6: Backtesting — Ensemble ═════════════════════
    print_step(6, "Backtesting (Ensemble)")
    results = ensemble.predict(df_feat, df_ind)

    # Precios de cierre para período de prueba
    test_prices_ens = df_ind.loc[results.signals.index, "Close"]

    bt_cfg         = BacktestConfig(initial_capital=10_000.0, commission_pct=0.001)
    bt_ensemble    = GoldBacktester(bt_cfg)
    bt_res_ens     = bt_ensemble.run(test_prices_ens, results.signals)

    # ══ Paso 7: Backtesting — RF solo (para comparación) ════════════
    print_step(7, "Backtesting (RF solo para comparación)")
    from feature_engineer import FEATURE_COLUMNS as FC, TARGET_COLUMN as TC

    feat_cols  = [c for c in FC if c in df_feat.columns]
    split      = int(len(df_feat) * 0.80)
    X_tr_rf    = df_feat[feat_cols].iloc[:split]
    y_tr_rf    = df_feat[TC].iloc[:split]
    X_te_rf    = df_feat[feat_cols].iloc[split:]
    test_idx_rf = X_te_rf.index

    rf_only = GoldPredictor(ModelConfig(n_estimators=200, max_depth=6,
                                        min_samples_leaf=10, n_cv_splits=5))
    rf_only.train(X_tr_rf, y_tr_rf, run_cv=False)
    rf_sigs, _ = rf_only.predict_with_confidence(X_te_rf, min_confidence=0.40)

    test_prices_rf = df_ind.loc[test_idx_rf, "Close"]
    rf_sigs_series = pd.Series(rf_sigs, index=test_idx_rf)

    bt_rf    = GoldBacktester(bt_cfg)
    bt_res_rf = bt_rf.run(test_prices_rf, rf_sigs_series)

    # ══ Paso 8: Guardar salidas ════════════════════════════════════
    print_step(8, "Guardando salidas")
    os.makedirs("output", exist_ok=True)

    # Guardar operaciones
    ens_trades = bt_ensemble.get_trades_df()
    if not ens_trades.empty:
        path = "output/trades_phase3_ensemble.csv"
        ens_trades.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Operaciones Ensemble: {path} ({len(ens_trades)} operaciones)")

    # Guardar curva de capital
    bt_res_ens.equity_curve.to_csv("output/equity_ensemble.csv", header=["Portfolio_Value"])
    print(f"  ✅ Curva de capital (Ensemble): output/equity_ensemble.csv")

    # Guardar señales con regímenes
    signal_df = pd.DataFrame({
        "Signal":     results.signals,
        "Confidence": results.confidence.round(4),
        "Regime":     results.regimes,
        "RF_Signal":  results.rf_signals.reindex(results.signals.index),
        "LSTM_Signal": results.lstm_signals.reindex(results.signals.index),
    })
    signal_df.to_csv("output/signals_phase3.csv", encoding="utf-8-sig")
    print(f"  ✅ Señales detalladas: output/signals_phase3.csv")

    # ══ Resumen final ════════════════════════════════════════════
    print(f"\n{GOLD}{BOLD}{SEP}")
    print(f"  🏆 Resumen de Fase 3 — Comparación de modelos")
    print(f"{SEP}{RESET}")

    print(f"\n  {'Métrica':<30}  {'RF solo':>14}  {'Ensemble':>14}")
    print(f"  {'─'*60}")

    def _fmt(val, pct=True):
        s = f"{val:+.2%}" if pct else f"{val:.3f}"
        c = GREEN if (val > 0 if pct else val > 1) else RED
        return f"{c}{BOLD}{s}{RESET}"

    print(f"  {'Precisión del modelo':<30}  {'—':>14}  {BOLD}{lstm_acc:.2%}{RESET:>14}")
    print(f"  {'Rendimiento estrategia':<30}  "
          f"{_fmt(bt_res_rf.total_return_pct):>14}  "
          f"{_fmt(bt_res_ens.total_return_pct):>14}")
    print(f"  {'Rendimiento Buy & Hold':<30}  "
          f"{_fmt(bt_res_rf.bnh_return_pct):>14}  "
          f"{_fmt(bt_res_ens.bnh_return_pct):>14}")
    print(f"  {'Alpha':<30}  "
          f"{_fmt(bt_res_rf.alpha):>14}  "
          f"{_fmt(bt_res_ens.alpha):>14}")
    print(f"  {'Sharpe Ratio':<30}  "
          f"{_fmt(bt_res_rf.sharpe_ratio, pct=False):>14}  "
          f"{_fmt(bt_res_ens.sharpe_ratio, pct=False):>14}")
    print(f"  {'Máx Drawdown':<30}  "
          f"{_fmt(bt_res_rf.max_drawdown):>14}  "
          f"{_fmt(bt_res_ens.max_drawdown):>14}")
    print(f"  {'Win Rate':<30}  "
          f"{BOLD}{bt_res_rf.win_rate:.2%}{RESET:>14}  "
          f"{BOLD}{bt_res_ens.win_rate:.2%}{RESET:>14}")
    print(f"\n{GOLD}{BOLD}{SEP}{RESET}\n")

    # Distribución de señales Ensemble
    sig_dist = results.signals.value_counts().sort_index()
    print(f"  Distribución de señales Ensemble:")
    print(f"    Venta  (-1): {sig_dist.get(-1, 0):>4}")
    print(f"    Neutral( 0): {sig_dist.get( 0, 0):>4}")
    print(f"    Compra (+1): {sig_dist.get( 1, 0):>4}")
    print()

    return ensemble, bt_res_ens


if __name__ == "__main__":
    main()
