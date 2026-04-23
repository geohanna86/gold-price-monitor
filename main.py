# ============================================================
# main.py — Punto de entrada principal para Fase 1
# Gold Price Monitor — Phase 1
#
# Ejecución:
#   python main.py              ← Modo Mock (datos de prueba)
#   python main.py --mode live  ← Modo Live (datos reales)
# ============================================================

import argparse
import logging
import os
import sys

import pandas as pd

from config import AppConfig, DEFAULT_CONFIG
from data_fetcher import GoldDataFetcher
from indicators import TechnicalIndicators

# ── Configuración del sistema de registros ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Main")

# ── Colores de terminal ──
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
GOLD   = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

SEPARATOR = "=" * 65


def print_header():
    print(f"\n{GOLD}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     🥇  Gold Price Monitor — Phase 1                     ║")
    print("  ║     Data Fetcher + Technical Indicators Engine           ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")


def print_section(title: str):
    print(f"\n{BLUE}{BOLD}{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}{RESET}")


def print_price_summary(fetcher: GoldDataFetcher):
    summary = fetcher.get_price_summary()
    print_section("📊 Resumen de datos de precios")
    for key, val in summary.items():
        unit = "" if "cantidad" in key else " $"
        print(f"  {BOLD}{key:<22}{RESET}: {GOLD}{val}{unit}{RESET}")


def print_signals(ti: TechnicalIndicators):
    print_section("🎯 Señales actuales (última vela)")
    try:
        summary_df = ti.get_indicator_summary()
        try:
            from tabulate import tabulate
            print(tabulate(
                summary_df,
                headers="keys",
                tablefmt="rounded_outline",
                showindex=False,
                colalign=("right", "center", "right"),
            ))
        except ImportError:
            # Impresión alternativa si tabulate no está instalado
            print(summary_df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error al imprimir señales: {e}")


def print_last_candles(df: pd.DataFrame, n: int = 5):
    print_section(f"🕯️  Últimas {n} velas")
    display_cols = ["Open", "High", "Low", "Close", "Volume",
                    "RSI", "MACD", "ATR", "Confluence_Score", "Signal_Label"]
    available   = [c for c in display_cols if c in df.columns]
    tail        = df[available].tail(n).round(2)
    try:
        from tabulate import tabulate
        print(tabulate(tail, headers="keys", tablefmt="rounded_outline"))
    except ImportError:
        print(tail.to_string())


def save_output(df: pd.DataFrame, config: AppConfig):
    """Guarda los resultados en un archivo CSV."""
    if not config.output.save_csv:
        return
    os.makedirs("output", exist_ok=True)
    path = config.output.csv_path
    df.round(4).to_csv(path)
    logger.info(f"✅ Resultados guardados: {path}")
    print(f"\n  {GREEN}✅ Resultados guardados en:{RESET} {BOLD}{path}{RESET}")


def run_pipeline(config: AppConfig) -> pd.DataFrame:
    """
    Tubería principal — devuelve un DataFrame completo con todos los indicadores.
    Puede ser importado y llamado desde otros archivos (como modelos predictivos en Fase 2).
    """
    print_header()
    mode_label = f"{GREEN}Live 🌐{RESET}" if config.mode == "live" else f"{YELLOW}Mock 🧪{RESET}"
    print(f"  Modo actual: {BOLD}{mode_label}")

    # ─ Paso 1: Obtener datos ─
    logger.info("Paso 1: Obteniendo datos...")
    fetcher = GoldDataFetcher(config)
    df_raw  = fetcher.get_data()
    print_price_summary(fetcher)

    # ─ Paso 2: Agregar indicadores ─
    logger.info("Paso 2: Calculando indicadores técnicos...")
    ti = (
        TechnicalIndicators(df_raw, config)
        .add_trend_indicators()
        .add_momentum_indicators()
        .add_volatility_indicators()
        .add_volume_indicators()
        .add_confluence_score()
    )
    df_final = ti.get_dataframe()

    # ─ Paso 3: Mostrar resultados ─
    print_signals(ti)
    print_last_candles(df_final, n=5)

    # ─ Paso 4: Guardar salidas ─
    save_output(df_final, config)

    logger.info(f"✅ Fase 1 completada | Columnas: {len(df_final.columns)} | Filas: {len(df_final)}")
    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Gold Price Monitor — Phase 1"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Modo de obtención de datos: mock (prueba) o live (real)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Marco de tiempo (1d, 1h, 15m, ...) — solo para modo live",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Período de tiempo (1y, 6mo, 3mo, ...) — solo para modo live",
    )
    args = parser.parse_args()

    config      = AppConfig(mode=args.mode)
    df_complete = run_pipeline(config)

    print(f"\n{GOLD}{BOLD}  ✅ ¡Fase 1 completada exitosamente!{RESET}")
    print(f"  Filas: {len(df_complete)} | Columnas: {len(df_complete.columns)}")
    print(f"  Columnas de indicadores: {[c for c in df_complete.columns if c not in ['Open','High','Low','Close','Volume']]}\n")

    return df_complete


if __name__ == "__main__":
    main()
