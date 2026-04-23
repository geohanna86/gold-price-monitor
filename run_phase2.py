# ============================================================
# run_phase2.py — Corredor completo para Fase 2
# Gold Price Monitor
#
# Ejecución:
#   python run_phase2.py
# ============================================================

import logging
import os
import sys

import pandas as pd

from config import AppConfig
from data_fetcher import GoldDataFetcher
from indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer
from ml_model import GoldPredictor, ModelConfig
from backtester import GoldBacktester, BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Phase2")

GOLD, BLUE, GREEN, RESET, BOLD = (
    "\033[33m", "\033[94m", "\033[92m", "\033[0m", "\033[1m"
)
SEP = "=" * 65


def print_step(n: int, title: str):
    print(f"\n{BLUE}{BOLD}{SEP}")
    print(f"  Paso {n}: {title}")
    print(f"{SEP}{RESET}")


def main():
    print(f"\n{GOLD}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   🤖  Gold Price Monitor — Phase 2                       ║")
    print("  ║   Random Forest Predictor + Backtesting Engine           ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{RESET}")

    # ══ Paso 1: Obtener datos y calcular indicadores (de Fase 1) ══
    print_step(1, "Obteniendo datos y calculando indicadores")
    config  = AppConfig(mode="mock")
    df_raw  = GoldDataFetcher(config).get_data()
    df_ind  = (
        TechnicalIndicators(df_raw, config)
        .add_all()
        .get_dataframe()
    )
    print(f"  ✅ Datos listos | Filas: {len(df_ind)} | Columnas: {len(df_ind.columns)}")

    # ══ Paso 2: Ingeniería de características ══════════════════════════════════════
    print_step(2, "Ingeniería de características (Feature Engineering)")
    fe = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()

    X_train, X_test, y_train, y_test = fe.train_test_split(test_size=0.20)
    full_df = fe.get_full_data()

    print(f"  ✅ Características: {len(fe.get_feature_names())}")
    print(f"  ✅ Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")
    dist = y_train.value_counts().sort_index()
    print(f"  Distribución del objetivo (Entrenamiento): venta={dist.get(-1,0)} | neutral={dist.get(0,0)} | compra={dist.get(1,0)}")

    # ══ Paso 3: Entrenar modelo Random Forest + Cross-Validation ══════════════════════════════════════
    print_step(3, "Entrenamiento de modelo Random Forest + Cross-Validation")
    model_cfg = ModelConfig(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        n_cv_splits=5,
    )
    predictor = GoldPredictor(model_cfg)
    predictor.train(X_train, y_train, run_cv=True)

    # ══ Paso 4: Evaluación ════════════════════════════════════════════
    print_step(4, "Evaluación del modelo en datos de prueba")
    metrics = predictor.evaluate(X_test, y_test)
    predictor.print_report()

    # ══ Paso 5: Backtesting ════════════════════════════════════
    print_step(5, "Backtesting en conjunto de prueba")

    # Señales del modelo con umbral de confianza 45%
    signals, confidence = predictor.predict_with_confidence(X_test, min_confidence=0.40)

    # Precios del conjunto de prueba
    test_prices = full_df.loc[X_test.index, "Close"]

    bt_config  = BacktestConfig(initial_capital=10_000.0, commission_pct=0.001)
    backtester = GoldBacktester(bt_config)
    bt_results = backtester.run(test_prices, signals)
    backtester.print_report()

    # ══ Paso 6: Guardar salidas ════════════════════════════════════════
    print_step(6, "Guardando salidas")
    os.makedirs("output", exist_ok=True)

    # Guardar operaciones
    trades_df = backtester.get_trades_df()
    if not trades_df.empty:
        trades_path = "output/trades_phase2.csv"
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Registro de operaciones: {trades_path} ({len(trades_df)} operaciones)")

    # Guardar curva de capital
    equity_path = "output/equity_curve.csv"
    bt_results.equity_curve.to_csv(equity_path, header=["Portfolio_Value"])
    print(f"  ✅ Curva de capital: {equity_path}")

    # Guardar importancia de características
    importance_df = pd.DataFrame(
        list(metrics.feature_importance.items()),
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)
    importance_path = "output/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✅ Importancia de características: {importance_path}")

    # ══ Resumen final ═══════════════════════════════════════════════════
    print(f"\n{GOLD}{BOLD}{SEP}")
    print(f"  🏆 Resumen de Fase 2")
    print(f"{SEP}{RESET}")
    print(f"  Precisión del modelo: {BOLD}{metrics.accuracy:.2%}{RESET}")
    print(f"  CV Accuracy         : {BOLD}{metrics.cv_accuracy_mean:.2%} ± {metrics.cv_accuracy_std:.2%}{RESET}")
    print(f"  Rendimiento estrategia: {BOLD}{bt_results.total_return_pct:+.2%}{RESET}")
    print(f"  Rendimiento Buy & Hold : {BOLD}{bt_results.bnh_return_pct:+.2%}{RESET}")
    print(f"  Alpha               : {BOLD}{bt_results.alpha:+.2%}{RESET}")
    print(f"  Sharpe Ratio        : {BOLD}{bt_results.sharpe_ratio:.3f}{RESET}")
    print(f"  Win Rate            : {BOLD}{bt_results.win_rate:.2%}{RESET}")
    print(f"{GOLD}{BOLD}{SEP}{RESET}\n")

    return predictor, bt_results


if __name__ == "__main__":
    main()
