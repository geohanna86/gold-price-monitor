# ============================================================
# backtester.py — Motor de prueba inversa (Backtesting Engine)
# Gold Price Monitor — Phase 2
#
# Simula operaciones reales en datos de prueba con:
#  - Capital inicial ajustable
#  - Comisión realista por operación
#  - Métricas de rendimiento profesionales (Sharpe, Max Drawdown, Win Rate)
#  - Estrategia: Solo compra (compra en +1, salida en -1 o 0)
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("GoldBacktester")


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0  # Capital inicial (dólares)
    commission_pct:  float = 0.001     # Comisión del 0.1% por operación (ida y vuelta)
    risk_free_rate:  float = 0.02      # Tasa libre de riesgo (2% anual)
    trading_days:    int   = 252       # Días de negociación en un año


@dataclass
class TradeRecord:
    """Registro de una operación individual."""
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    shares:       float
    pnl:          float        # Ganancia/pérdida en dólares
    return_pct:   float        # Rendimiento en porcentaje
    signal:       int          # Señal que llevó a la entrada (+1)


@dataclass
class BacktestResults:
    """Resultados completos de la prueba inversa."""
    # ─ Rendimiento ─
    total_return_pct:   float = 0.0
    annual_return_pct:  float = 0.0
    sharpe_ratio:       float = 0.0
    max_drawdown_pct:   float = 0.0
    calmar_ratio:       float = 0.0   # rendimiento_anual / caída_máxima

    # ─ Operaciones ─
    total_trades:       int   = 0
    winning_trades:     int   = 0
    losing_trades:      int   = 0
    win_rate:           float = 0.0
    avg_win_pct:        float = 0.0
    avg_loss_pct:       float = 0.0
    profit_factor:      float = 0.0   # ganancias_totales / pérdidas_totales

    # ─ Capital ─
    initial_capital:    float = 0.0
    final_capital:      float = 0.0
    total_commission:   float = 0.0

    # ─ Comparación con Buy & Hold ─
    bnh_return_pct:     float = 0.0   # Rendimiento de Buy & Hold para comparar
    alpha:              float = 0.0   # Superación de Buy & Hold

    # ─ Series temporales ─
    equity_curve:       pd.Series = field(default_factory=pd.Series)
    daily_returns:      pd.Series = field(default_factory=pd.Series)
    trades:             List[TradeRecord] = field(default_factory=list)


class GoldBacktester:
    """
    Motor de prueba inversa — simula operaciones reales en Test Set.

    Estrategia:
      - Señal +1 → compra con todo el capital disponible
      - Señal -1 o 0 → venta (cierra la posición si existe)
      - Sin ventas en corto (solo compra larga)

    Ejemplo:
        results = GoldBacktester(config).run(prices, signals)
        backtester.print_report()
    """

    def __init__(self, config: BacktestConfig = None):
        self.config  = config or BacktestConfig()
        self.results: Optional[BacktestResults] = None

    # ─────────────────────────────────────────────────────────
    # Ejecución principal
    # ─────────────────────────────────────────────────────────
    def run(
        self,
        prices:  pd.Series,   # Serie de precios de cierre (solo Test Set)
        signals: np.ndarray,  # Señales del modelo {-1, 0, +1}
    ) -> BacktestResults:
        """
        Ejecuta la prueba inversa y devuelve los resultados completos.

        Advertencia: prices y signals deben tener la misma longitud.
        """
        assert len(prices) == len(signals), (
            f"Prices ({len(prices)}) y Signals ({len(signals)}) ¡tienen longitudes diferentes!"
        )

        cfg      = self.config
        capital  = cfg.initial_capital
        position = 0.0          # Cantidad de onzas poseídas
        cash     = capital
        equity   = []
        trades:  List[TradeRecord] = []
        total_commission = 0.0

        entry_price: Optional[float] = None
        entry_date:  Optional[pd.Timestamp] = None
        entry_shares: float = 0.0

        prices_arr  = prices.values
        dates_arr   = prices.index
        signals_arr = np.array(signals)

        for i in range(len(prices_arr)):
            price  = prices_arr[i]
            date   = dates_arr[i]
            signal = signals_arr[i]

            # ── Cerrar posición existente en señal de venta/neutral ──────────
            if position > 0 and signal != 1:
                # Calcular valor de venta después de comisión
                sell_value   = position * price * (1 - cfg.commission_pct)
                commission   = position * price * cfg.commission_pct
                total_commission += commission

                pnl        = sell_value - (position * entry_price)
                return_pct = (price / entry_price - 1) - cfg.commission_pct * 2

                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=price,
                    shares=position,
                    pnl=round(pnl, 2),
                    return_pct=round(return_pct, 6),
                    signal=1,
                ))

                cash     += sell_value
                position  = 0.0
                entry_price = entry_date = None

            # ── Abrir posición nueva en señal de compra ──────────────────
            if position == 0 and signal == 1:
                # Comprar con todo el efectivo disponible (después de comisión)
                commission   = cash * cfg.commission_pct
                invest_cash  = cash - commission
                shares       = invest_cash / price
                total_commission += commission

                position     = shares
                entry_price  = price
                entry_date   = date
                entry_shares = shares
                cash         = 0.0

            # ── Valor total de la cartera ─────────────────────────────
            portfolio_value = cash + position * price
            equity.append(portfolio_value)

        # ── Cerrar cualquier posición abierta al final del período ────────────────
        if position > 0:
            last_price  = prices_arr[-1]
            last_date   = dates_arr[-1]
            sell_value  = position * last_price * (1 - cfg.commission_pct)
            commission  = position * last_price * cfg.commission_pct
            pnl         = sell_value - (position * entry_price)
            total_commission += commission
            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=last_date,
                entry_price=entry_price,
                exit_price=last_price,
                shares=position,
                pnl=round(pnl, 2),
                return_pct=round(last_price / entry_price - 1, 6),
                signal=1,
            ))
            cash     += sell_value
            position  = 0.0

        # ── Construir curva de capital (Equity Curve) ───────────────────
        equity_series  = pd.Series(equity, index=prices.index)
        daily_returns  = equity_series.pct_change().fillna(0)
        final_capital  = equity[-1] if equity else capital

        # ── Calcular métricas ────────────────────────────────────────
        self.results = self._compute_metrics(
            equity_series, daily_returns, trades,
            capital, final_capital, total_commission,
            prices,
        )

        logger.info(
            f"✅ Prueba inversa completada | "
            f"Rendimiento: {self.results.total_return_pct:.2%} | "
            f"Sharpe: {self.results.sharpe_ratio:.3f} | "
            f"Caída máxima: {self.results.max_drawdown_pct:.2%} | "
            f"Operaciones: {self.results.total_trades}"
        )
        return self.results

    # ─────────────────────────────────────────────────────────
    # Calcular métricas
    # ─────────────────────────────────────────────────────────
    def _compute_metrics(
        self,
        equity:     pd.Series,
        ret:        pd.Series,
        trades:     List[TradeRecord],
        initial:    float,
        final:      float,
        commission: float,
        prices:     pd.Series,
    ) -> BacktestResults:
        cfg = self.config
        n   = len(equity)

        # ── Rendimiento total ─────────────────────────────────────
        total_return = (final / initial) - 1

        # ── Rendimiento anual ────────────────────────────────────
        years          = n / cfg.trading_days
        annual_return  = (1 + total_return) ** (1 / max(years, 1e-6)) - 1

        # ── Sharpe Ratio ─────────────────────────────────────
        rf_daily   = cfg.risk_free_rate / cfg.trading_days
        excess_ret = ret - rf_daily
        sharpe     = (
            excess_ret.mean() / excess_ret.std() * np.sqrt(cfg.trading_days)
            if excess_ret.std() > 1e-9 else 0.0
        )

        # ── Caída máxima ─────────────────────────────────────
        rolling_max    = equity.cummax()
        drawdown       = (equity - rolling_max) / rolling_max
        max_drawdown   = drawdown.min()  # siempre negativo

        # ── Calmar Ratio ─────────────────────────────────────
        calmar = (
            annual_return / abs(max_drawdown)
            if abs(max_drawdown) > 1e-9 else 0.0
        )

        # ── Estadísticas de operaciones ─────────────────────────────────
        winning = [t for t in trades if t.pnl > 0]
        losing  = [t for t in trades if t.pnl <= 0]

        win_rate  = len(winning) / len(trades) if trades else 0.0
        avg_win   = np.mean([t.return_pct for t in winning]) if winning else 0.0
        avg_loss  = np.mean([t.return_pct for t in losing])  if losing  else 0.0

        total_wins   = sum(t.pnl for t in winning) if winning else 0.0
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 1.0
        profit_factor = total_wins / total_losses if total_losses > 1e-9 else float("inf")

        # ── Comparación con Buy & Hold ────────────────────────────────
        bnh_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        alpha      = total_return - bnh_return

        return BacktestResults(
            total_return_pct=round(total_return, 6),
            annual_return_pct=round(annual_return, 6),
            sharpe_ratio=round(float(sharpe), 4),
            max_drawdown_pct=round(float(max_drawdown), 6),
            calmar_ratio=round(calmar, 4),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 6),
            avg_loss_pct=round(avg_loss, 6),
            profit_factor=round(profit_factor, 4),
            initial_capital=initial,
            final_capital=round(final, 2),
            total_commission=round(commission, 2),
            bnh_return_pct=round(bnh_return, 6),
            alpha=round(alpha, 6),
            equity_curve=equity,
            daily_returns=ret,
            trades=trades,
        )

    # ─────────────────────────────────────────────────────────
    # Mostrar
    # ─────────────────────────────────────────────────────────
    def print_report(self):
        """Imprime un informe profesional de los resultados de la prueba inversa."""
        if self.results is None:
            logger.warning("Llama a run() primero.")
            return

        r = self.results
        GREEN, RED, YELLOW, BLUE, GOLD, RESET, BOLD = (
            "\033[92m", "\033[91m", "\033[93m", "\033[94m",
            "\033[33m", "\033[0m", "\033[1m"
        )
        SEP = "=" * 55

        def fmt_pct(v, good_positive=True):
            color = GREEN if (v > 0) == good_positive else RED
            return f"{color}{v:+.2%}{RESET}"

        print(f"\n{GOLD}{BOLD}{SEP}")
        print(f"  📈 Informe de prueba inversa — XAU/USD")
        print(f"{SEP}{RESET}")

        print(f"\n{BLUE}{BOLD}  💰 Rendimiento financiero:{RESET}")
        print(f"  Capital inicial  : {BOLD}${r.initial_capital:,.2f}{RESET}")
        print(f"  Capital final    : {BOLD}${r.final_capital:,.2f}{RESET}")
        print(f"  Rendimiento total        : {fmt_pct(r.total_return_pct)}")
        print(f"  Rendimiento anual       : {fmt_pct(r.annual_return_pct)}")
        print(f"  Comisiones pagadas   : {RED}${r.total_commission:,.2f}{RESET}")

        print(f"\n{BLUE}{BOLD}  📊 Métricas de riesgo:{RESET}")
        sh_color = GREEN if r.sharpe_ratio > 1.0 else (YELLOW if r.sharpe_ratio > 0 else RED)
        print(f"  Sharpe Ratio        : {sh_color}{r.sharpe_ratio:.4f}{RESET}  {'(Excelente)' if r.sharpe_ratio > 2 else '(Bueno)' if r.sharpe_ratio > 1 else '(Aceptable)' if r.sharpe_ratio > 0 else '(Negativo)'}")
        print(f"  Max Drawdown        : {RED}{r.max_drawdown_pct:.2%}{RESET}")
        print(f"  Calmar Ratio        : {r.calmar_ratio:.4f}")

        print(f"\n{BLUE}{BOLD}  🔄 Estadísticas de operaciones:{RESET}")
        print(f"  Operaciones totales     : {BOLD}{r.total_trades}{RESET}")
        print(f"  Operaciones ganadoras        : {GREEN}{r.winning_trades}{RESET}")
        print(f"  Operaciones perdedoras        : {RED}{r.losing_trades}{RESET}")
        wr_color = GREEN if r.win_rate > 0.55 else (YELLOW if r.win_rate > 0.45 else RED)
        print(f"  Tasa de ganancias         : {wr_color}{r.win_rate:.2%}{RESET}")
        print(f"  Ganancia promedio/operación   : {fmt_pct(r.avg_win_pct)}")
        print(f"  Pérdida promedio/operación : {fmt_pct(r.avg_loss_pct, good_positive=False)}")
        pf_color = GREEN if r.profit_factor > 1.5 else (YELLOW if r.profit_factor > 1 else RED)
        print(f"  Profit Factor      : {pf_color}{r.profit_factor:.4f}{RESET}")

        print(f"\n{BLUE}{BOLD}  ⚖️  Comparación con Buy & Hold:{RESET}")
        print(f"  Rendimiento del modelo       : {fmt_pct(r.total_return_pct)}")
        print(f"  Rendimiento Buy & Hold    : {fmt_pct(r.bnh_return_pct)}")
        alpha_color = GREEN if r.alpha > 0 else RED
        print(f"  Alpha (Superación)     : {alpha_color}{r.alpha:+.2%}{RESET}")

        if r.trades:
            print(f"\n{BLUE}{BOLD}  📋 Últimas 3 operaciones:{RESET}")
            for t in r.trades[-3:]:
                sign = "+" if t.pnl > 0 else ""
                clr  = GREEN if t.pnl > 0 else RED
                print(
                    f"  {t.entry_date.date()} → {t.exit_date.date()} | "
                    f"Entrada: ${t.entry_price:.2f} | Salida: ${t.exit_price:.2f} | "
                    f"P&L: {clr}{sign}${t.pnl:.2f}{RESET}"
                )

        print(f"\n{GOLD}{BOLD}{SEP}{RESET}\n")

    def get_trades_df(self) -> pd.DataFrame:
        """Devuelve el registro de operaciones como un DataFrame."""
        if not self.results or not self.results.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "Fecha de entrada":  t.entry_date.date(),
                "Fecha de salida":  t.exit_date.date(),
                "Precio de entrada":   t.entry_price,
                "Precio de salida":   t.exit_price,
                "Número de onzas": round(t.shares, 4),
                "P&L ($)":      t.pnl,
                "Rendimiento (%)":   f"{t.return_pct:.2%}",
            }
            for t in self.results.trades
        ])
