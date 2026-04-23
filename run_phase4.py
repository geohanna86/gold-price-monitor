# ============================================================
# run_phase4.py — Ejecución de Fase 4: Dashboard + Alertas
# Gold Price Monitor
#
# Opciones:
#   python run_phase4.py --mode dashboard   → Ejecuta Streamlit
#   python run_phase4.py --mode alerts      → Ejecuta sistema de alertas
#   python run_phase4.py --mode test        → Solo ejecuta pruebas
#   python run_phase4.py --mode demo        → Demo completo
# ============================================================

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase4")

GOLD, BLUE, GREEN, RED, RESET, BOLD = (
    "\033[33m", "\033[94m", "\033[92m", "\033[91m", "\033[0m", "\033[1m"
)
SEP = "─" * 60


def _header():
    print(f"\n{GOLD}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   🥇  Gold Price Monitor — Phase 4                       ║")
    print("  ║   Streamlit Dashboard + Telegram Alert System            ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{RESET}")


# ──────────────────────────────────────────────────────────────
# Modo Dashboard
# ──────────────────────────────────────────────────────────────
def run_dashboard():
    _header()
    print(f"{BLUE}⚡ Ejecutando Streamlit Dashboard...{RESET}\n")

    # Verificar Streamlit
    try:
        import streamlit
        print(f"  ✅ Streamlit v{streamlit.__version__} listo")
    except ImportError:
        print(f"  {RED}❌ Streamlit no está instalado.{RESET}")
        print(f"     Ejecuta: {BOLD}pip install streamlit plotly{RESET}")
        sys.exit(1)

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    print(f"  → Ejecutando: streamlit run {dashboard_path}")
    print(f"  → Abre en el navegador: {BOLD}http://localhost:8501{RESET}\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ])


# ──────────────────────────────────────────────────────────────
# Modo Alertas
# ──────────────────────────────────────────────────────────────
def run_alerts(mode: str = "mock", interval: int = 60, iterations: int = 3):
    _header()
    print(f"{BLUE}⚡ Ejecutando sistema de alertas...{RESET}\n")

    from alert_system import GoldAlertSystem, AlertConfig, GoldMonitorLoop

    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if token:
        print(f"  ✅ Token de Telegram encontrado — Se enviará a Telegram")
    else:
        print(f"  ⚠️  Sin token de Telegram — Las alertas se mostrarán en Consola")
        print(f"     Para configurar Telegram: export TELEGRAM_TOKEN=<token>")
        print(f"                               export TELEGRAM_CHAT_ID=<chat_id>\n")

    cfg    = AlertConfig(
        telegram_token=token,
        telegram_chat_id=chat_id,
        min_confidence=0.45,
        cooldown_seconds=interval // 2,
    )
    alert  = GoldAlertSystem(cfg)
    loop   = GoldMonitorLoop(alert, interval_seconds=interval, mode=mode)
    loop.setup(n_rows=400)

    print(f"\n  {GREEN}🔄 Iniciando monitoreo ({iterations} ciclos de prueba)...{RESET}")
    loop.start(max_iterations=iterations)

    # Mostrar registro
    df_hist = alert.get_alert_history_df()
    if not df_hist.empty:
        print(f"\n{GOLD}  📋 Registro de alertas:{RESET}")
        print(df_hist.to_string(index=False))


# ──────────────────────────────────────────────────────────────
# Modo Prueba
# ──────────────────────────────────────────────────────────────
def run_tests():
    _header()
    print(f"{BLUE}⚡ Ejecutando pruebas de Fase 4...{RESET}\n")

    # 1) Pruebas de alert_system
    from alert_system import _run_assertions as alert_assert
    alert_assert()

    # 2) Verificar imports del Dashboard
    print("  ← Verificando bibliotecas de Dashboard ...")
    try:
        import streamlit, plotly
        print(f"  ✅ streamlit={streamlit.__version__} | plotly={plotly.__version__}")
    except ImportError as e:
        print(f"  ⚠️  Biblioteca faltante: {e}")
        print(f"     Ejecuta: pip install streamlit plotly")

    # 3) Verificar Telegram (requests)
    print("  ← Verificando requests ...")
    try:
        import requests
        print(f"  ✅ requests={requests.__version__}")
    except ImportError:
        print(f"  ⚠️  Biblioteca requests faltante — Funcionará solo en modo Consola")
        print(f"     Ejecuta: pip install requests")

    print(f"\n{GREEN}{BOLD}  ✅ ¡Todas las pruebas de Fase 4 completadas!{RESET}\n")


# ──────────────────────────────────────────────────────────────
# Modo Demo Completo
# ──────────────────────────────────────────────────────────────
def run_demo():
    _header()
    print(f"{BLUE}⚡ Demo completo del sistema...{RESET}\n")

    print(f"  {SEP}")
    print(f"  Paso 1: Cargando modelo completo")
    print(f"  {SEP}")

    from config import AppConfig
    from data_fetcher import GoldDataFetcher
    from indicators import TechnicalIndicators
    from feature_engineer import FeatureEngineer
    from lstm_model import LSTMConfig
    from ensemble import GoldEnsemble, EnsembleConfig
    from backtester import GoldBacktester, BacktestConfig
    from alert_system import GoldAlertSystem, AlertConfig

    cfg    = AppConfig(mode="mock")
    cfg.mock.n_rows = 500

    df_raw = GoldDataFetcher(cfg).get_data()
    df_ind = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
    fe     = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    df_feat = fe.get_full_data()

    ens_cfg = EnsembleConfig(
        rf_weight=0.40, lstm_weight=0.40, hmm_weight=0.20,
        lstm_config=LSTMConfig(hidden_size=32, seq_length=15,
                               epochs=40, patience=10),
    )
    print(f"  ← Entrenando Ensemble...", end="", flush=True)
    ensemble = GoldEnsemble(ens_cfg)
    ensemble.fit(df_feat, df_ind)
    results = ensemble.predict(df_feat, df_ind)
    print(f" {GREEN}✅{RESET}")

    print(f"\n  {SEP}")
    print(f"  Paso 2: Resultados del modelo")
    print(f"  {SEP}")

    test_prices = df_ind.loc[results.signals.index, "Close"]
    bt = GoldBacktester(BacktestConfig())
    bt_res = bt.run(test_prices, results.signals)

    last_sig   = int(results.signals.iloc[-1])
    last_conf  = float(results.confidence.iloc[-1])
    last_regime = int(results.regimes.iloc[-1])
    last_price  = float(df_ind["Close"].iloc[-1])

    sig_label = {1: f"{GREEN}Compra 📈{RESET}", -1: f"{RED}Venta 📉{RESET}", 0: "Neutral ⏸"}
    reg_label = {0: f"{RED}Bajista 🔴{RESET}", 1: "Lateral ⚪", 2: f"{GREEN}Alcista 🟢{RESET}"}

    print(f"  💰 Precio: {BOLD}${last_price:,.2f}{RESET}")
    print(f"  📡 Señal: {BOLD}{sig_label.get(last_sig, '—')}{RESET}")
    print(f"  🎯 Confianza: {BOLD}{last_conf:.0%}{RESET}")
    print(f"  🌐 Régimen: {BOLD}{reg_label.get(last_regime, '—')}{RESET}")
    print(f"  📈 Rendimiento: {BOLD}{bt_res.total_return_pct:+.2%}{RESET}")
    print(f"  ⚡ Sharpe: {BOLD}{bt_res.sharpe_ratio:.2f}{RESET}")
    print(f"  🏆 Alpha: {BOLD}{bt_res.alpha:+.2%}{RESET}")

    print(f"\n  {SEP}")
    print(f"  Paso 3: Simulación del sistema de alertas")
    print(f"  {SEP}")

    alert_cfg = AlertConfig(cooldown_seconds=0)
    alert     = GoldAlertSystem(alert_cfg)

    # Simular cambios hipotéticos
    scenarios = [
        (0,  1, last_price * 0.995, 0.52),
        (1,  2, last_price * 1.003, 0.68),
        (-1, 0, last_price * 0.990, 0.61),
        (0,  0, last_price * 0.985, 0.44),   # confianza baja → sin alerta
    ]
    for sig, regime, price, conf in scenarios:
        alert.process_new_data(sig, regime, price, conf)

    df_hist = alert.get_alert_history_df()
    print(f"  Total de alertas enviadas: {BOLD}{len(df_hist)}{RESET}")

    print(f"\n  {SEP}")
    print(f"  Paso 4: Cómo ejecutar el Dashboard")
    print(f"  {SEP}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"  cd \"{script_dir}\"")
    print(f"  {BOLD}streamlit run dashboard.py{RESET}")
    print(f"  Luego abre: {BLUE}http://localhost:8501{RESET}\n")

    print(f"\n{GOLD}{BOLD}  🎉 ¡Sistema completamente listo! Todas las fases completadas.{RESET}\n")


# ──────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gold Price Monitor — Phase 4")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "alerts", "test", "demo"],
        default="demo",
        help="Modo de ejecución (predeterminado: demo)",
    )
    parser.add_argument(
        "--data-mode", default="mock", choices=["mock", "live"],
        help="Fuente de datos",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Período de actualización en segundos (para alertas)",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Número de ciclos (para alertas)",
    )
    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "alerts":
        run_alerts(args.data_mode, args.interval, args.iterations)
    elif args.mode == "test":
        run_tests()
    else:
        run_demo()


if __name__ == "__main__":
    main()
