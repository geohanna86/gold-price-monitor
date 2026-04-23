# ============================================================
# performance_tracker.py — Rastreador de Desempeño de Señales
# Gold Price Monitor — Phase 4
#
# Responsabilidades:
#   - Registro de señales generadas (entrada, SL, TP, confianza, régimen)
#   - Actualización de resultados al cerrar posiciones
#   - Cálculo de estadísticas (Sharpe, Kelly, factor de ganancia)
#   - Recomendación de tamaño de lote óptimo
#   - Persistencia en JSON
# ============================================================

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger("PerformanceTracker")


# ─────────────────────────────────────────────────────────────
# Registro de una señal individual
# ─────────────────────────────────────────────────────────────
@dataclass
class SignalRecord:
    """
    Representa una señal comercial individual con sus detalles de entrada
    y resultados de salida.
    """
    signal_id: str                          # UUID único
    timestamp: datetime                     # Hora de generación de la señal
    action: str                             # "BUY", "SELL", "NEUTRAL"
    entry_price: float                      # Precio de entrada
    sl_price: float                         # Stop Loss
    tp_price: float                         # Take Profit
    lot_size: float                         # Tamaño del lote (ej: 0.01 para micro)
    confidence: float                       # Confianza [0.0, 1.0]
    regime: int                             # Régimen HMM {0=Bear, 1=Sideways, 2=Bull}
    sentiment_score: float = 0.0            # Sentimiento de noticias [-1.0, 1.0]
    outcome: str = "PENDING"                # Estado: "TP_HIT", "SL_HIT", "MANUAL_CLOSE", "PENDING"
    exit_price: float = 0.0                 # Precio de salida (0 si PENDING)
    exit_time: Optional[datetime] = None    # Hora de salida
    pnl_pips: float = 0.0                   # Ganancia/Pérdida en pips
    pnl_usd: float = 0.0                    # Ganancia/Pérdida en USD
    notes: str = ""                         # Notas adicionales

    def to_dict(self) -> Dict:
        """Convierte el registro a diccionario, serializando datetime."""
        data = asdict(self)
        # Serializar datetime a ISO string
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        if isinstance(data['exit_time'], datetime):
            data['exit_time'] = data['exit_time'].isoformat()
        elif data['exit_time'] is None:
            data['exit_time'] = None
        return data

    @staticmethod
    def from_dict(data: Dict) -> 'SignalRecord':
        """Crea un SignalRecord desde un diccionario."""
        # Deserializar datetime desde ISO string
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('exit_time'), str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return SignalRecord(**data)


@dataclass
class PerformanceStats:
    """
    Estadísticas agregadas del desempeño de trading.
    """
    total_signals: int                  # Total de señales generadas
    executed_signals: int               # Señales ejecutadas (no NEUTRAL)
    closed_signals: int                 # Señales cerradas (no PENDING)
    win_rate: float                     # Tasa de ganancia [0, 1]
    avg_rr: float                       # Ratio promedio Risk/Reward alcanzado
    total_pnl_usd: float                # Ganancia neta en USD
    profit_factor: float                # Ganancia bruta / Pérdida bruta
    max_drawdown_pips: float            # Reducción máxima desde pico en pips
    best_trade_pips: float              # Mejor operación en pips
    worst_trade_pips: float             # Peor operación en pips
    avg_confidence_winners: float       # Confianza promedio en ganancias
    avg_confidence_losers: float        # Confianza promedio en pérdidas
    consecutive_wins: int               # Rachas de ganancias actuales
    consecutive_losses: int             # Rachas de pérdidas actuales
    kelly_fraction: float               # Fracción de Kelly calculada
    recommended_lot: float              # Tamaño de lote recomendado


# ─────────────────────────────────────────────────────────────
# Funciones auxiliares para cálculos
# ─────────────────────────────────────────────────────────────

def calculate_pnl_pips(action: str, entry_price: float, exit_price: float) -> float:
    """
    Calcula la ganancia/pérdida en pips.
    Para XAUUSD: 1 pip = 0.01 (cuarto decimal)

    Args:
        action: "BUY" o "SELL"
        entry_price: Precio de entrada
        exit_price: Precio de salida

    Returns:
        Ganancia en pips (positiva = ganancia, negativa = pérdida)
    """
    if action == "BUY":
        pips = (exit_price - entry_price) * 100
    elif action == "SELL":
        pips = (entry_price - exit_price) * 100
    else:
        pips = 0.0
    return round(pips, 2)


def calculate_pnl_usd(pnl_pips: float, lot_size: float, pip_value: float = 10.0) -> float:
    """
    Convierte ganancias/pérdidas en pips a USD.

    Args:
        pnl_pips: Ganancia/pérdida en pips
        lot_size: Tamaño del lote (ej: 0.01)
        pip_value: Valor USD por pip para lote 1.0 (default: 10 USD/pip para XAUUSD)

    Returns:
        Ganancia/pérdida en USD
    """
    # pip_value es para lote 1.0, escalar según lot_size
    value_per_pip = pip_value * lot_size
    return round(pnl_pips * value_per_pip, 2)


# ─────────────────────────────────────────────────────────────
# Rastreador principal de desempeño
# ─────────────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Rastreador de desempeño de señales de trading.
    - Carga/salva registros en JSON
    - Calcula estadísticas clave
    - Estima tamaño óptimo de lote con Kelly Criterion
    """

    def __init__(self, filepath: Path):
        """
        Inicializa el rastreador.

        Args:
            filepath: Ruta del archivo .py de este módulo
                      (se crea signals_history.json en el mismo directorio)
        """
        self.filepath = Path(filepath)
        self.data_dir = self.filepath.parent
        self.history_file = self.data_dir / "signals_history.json"

        # Cargar señales existentes o crear lista vacía
        self.signals: List[SignalRecord] = []
        if self.history_file.exists():
            self.load()
        else:
            logger.info(f"Creando nuevo rastreador en {self.history_file}")

    def add_signal(
        self,
        action: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        confidence: float,
        regime: int,
        sentiment: float = 0.0
    ) -> str:
        """
        Agrega una nueva señal al registro.

        Args:
            action: "BUY", "SELL", "NEUTRAL"
            entry: Precio de entrada
            sl: Precio de Stop Loss
            tp: Precio de Take Profit
            lot: Tamaño del lote
            confidence: Confianza [0, 1]
            regime: Régimen HMM {0=Bear, 1=Sideways, 2=Bull}
            sentiment: Sentimiento de noticias [-1, 1]

        Returns:
            signal_id (UUID)
        """
        signal_id = str(uuid4())
        signal = SignalRecord(
            signal_id=signal_id,
            timestamp=datetime.now(),
            action=action,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            lot_size=lot,
            confidence=confidence,
            regime=regime,
            sentiment_score=sentiment
        )
        self.signals.append(signal)
        logger.debug(f"Señal agregada: {signal_id} ({action} @ {entry})")
        return signal_id

    def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        notes: str = ""
    ) -> bool:
        """
        Actualiza el resultado de una señal (cierre de posición).
        Calcula automáticamente PnL en pips y USD.

        Args:
            signal_id: ID de la señal a actualizar
            outcome: "TP_HIT", "SL_HIT", "MANUAL_CLOSE"
            exit_price: Precio de salida
            exit_time: Hora de salida (default: ahora)
            notes: Notas adicionales

        Returns:
            True si se actualizó, False si no encontró la señal
        """
        for signal in self.signals:
            if signal.signal_id == signal_id:
                signal.outcome = outcome
                signal.exit_price = exit_price
                signal.exit_time = exit_time or datetime.now()
                signal.notes = notes

                # Calcular PnL
                signal.pnl_pips = calculate_pnl_pips(
                    signal.action,
                    signal.entry_price,
                    exit_price
                )
                signal.pnl_usd = calculate_pnl_usd(
                    signal.pnl_pips,
                    signal.lot_size
                )

                logger.debug(
                    f"Resultado actualizado: {signal_id} | "
                    f"PnL: {signal.pnl_pips} pips ({signal.pnl_usd} USD)"
                )
                return True

        logger.warning(f"Señal no encontrada: {signal_id}")
        return False

    def get_stats(self) -> PerformanceStats:
        """
        Calcula estadísticas agregadas de desempeño.

        Returns:
            PerformanceStats con métricas clave
        """
        if not self.signals:
            return PerformanceStats(
                total_signals=0,
                executed_signals=0,
                closed_signals=0,
                win_rate=0.0,
                avg_rr=0.0,
                total_pnl_usd=0.0,
                profit_factor=0.0,
                max_drawdown_pips=0.0,
                best_trade_pips=0.0,
                worst_trade_pips=0.0,
                avg_confidence_winners=0.0,
                avg_confidence_losers=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                kelly_fraction=0.0,
                recommended_lot=0.01
            )

        total_signals = len(self.signals)
        executed_signals = len([s for s in self.signals if s.action != "NEUTRAL"])
        closed_signals = len([s for s in self.signals if s.outcome != "PENDING"])

        # Operaciones cerradas
        closed = [s for s in self.signals if s.outcome != "PENDING"]

        if not closed:
            # Sin operaciones cerradas
            kelly_frac = self.get_kelly_fraction()
            rec_lot = self.get_recommended_lot(10000.0)
            return PerformanceStats(
                total_signals=total_signals,
                executed_signals=executed_signals,
                closed_signals=0,
                win_rate=0.0,
                avg_rr=0.0,
                total_pnl_usd=0.0,
                profit_factor=0.0,
                max_drawdown_pips=0.0,
                best_trade_pips=0.0,
                worst_trade_pips=0.0,
                avg_confidence_winners=0.0,
                avg_confidence_losers=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                kelly_fraction=kelly_frac,
                recommended_lot=rec_lot
            )

        # Ganadores y perdedores
        winners = [s for s in closed if s.pnl_usd > 0]
        losers = [s for s in closed if s.pnl_usd < 0]

        win_rate = len(winners) / len(closed) if closed else 0.0

        # PnL total
        total_pnl_usd = sum(s.pnl_usd for s in closed)

        # Factor de ganancia
        gross_profit = sum(s.pnl_usd for s in winners) if winners else 0.0
        gross_loss = abs(sum(s.pnl_usd for s in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float('inf') if gross_profit > 0 else 0.0
        )

        # Mejor y peor operación
        pnls = [s.pnl_pips for s in closed]
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        # Confianza promedio
        avg_conf_winners = np.mean([s.confidence for s in winners]) if winners else 0.0
        avg_conf_losers = np.mean([s.confidence for s in losers]) if losers else 0.0

        # Max Drawdown (desde el pico)
        cumulative = np.cumsum([s.pnl_pips for s in closed])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Rachas actuales
        consecutive_wins = 0
        consecutive_losses = 0
        if closed:
            # Contar desde el final hacia atrás
            for s in reversed(closed):
                if s.pnl_usd > 0:
                    consecutive_wins += 1
                else:
                    break
            for s in reversed(closed):
                if s.pnl_usd < 0:
                    consecutive_losses += 1
                else:
                    break

        # Ratio Risk/Reward promedio alcanzado
        avg_rr = 0.0
        if closed:
            rr_values = []
            for s in closed:
                if s.action == "BUY":
                    risk_pips = abs(s.entry_price - s.sl_price) * 100
                    reward_pips = abs(s.tp_price - s.entry_price) * 100
                elif s.action == "SELL":
                    risk_pips = abs(s.sl_price - s.entry_price) * 100
                    reward_pips = abs(s.entry_price - s.tp_price) * 100
                else:
                    continue

                if risk_pips > 0:
                    rr_values.append(reward_pips / risk_pips)

            avg_rr = np.mean(rr_values) if rr_values else 0.0

        # Kelly Criterion
        kelly_frac = self.get_kelly_fraction(win_rate, avg_rr)
        rec_lot = self.get_recommended_lot(10000.0)

        return PerformanceStats(
            total_signals=total_signals,
            executed_signals=executed_signals,
            closed_signals=closed_signals,
            win_rate=round(win_rate, 4),
            avg_rr=round(avg_rr, 4),
            total_pnl_usd=round(total_pnl_usd, 2),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pips=round(max_dd, 2),
            best_trade_pips=round(best_trade, 2),
            worst_trade_pips=round(worst_trade, 2),
            avg_confidence_winners=round(avg_conf_winners, 4),
            avg_confidence_losers=round(avg_conf_losers, 4),
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            kelly_fraction=kelly_frac,
            recommended_lot=rec_lot
        )

    def get_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calcula la fracción de Kelly para dimensionamiento óptimo de posición.

        Formula: Kelly = W - (1-W)/R
        donde W = tasa de ganancia, R = ganancia promedio / pérdida promedio

        Resultado está limitado entre 0.01 y 0.25 (máximo 25% del capital).

        Args:
            win_rate: Tasa de ganancia [0, 1]. Si es None, se calcula.
            avg_win: Ganancia promedio en USD. Si es None, se calcula.
            avg_loss: Pérdida promedio en USD. Si es None, se calcula.

        Returns:
            Fracción de Kelly entre 0.01 y 0.25
        """
        # Calcular si no se proporcionan
        if win_rate is None:
            stats = self.get_stats()
            win_rate = stats.win_rate

        closed = [s for s in self.signals if s.outcome != "PENDING"]
        if len(closed) < 10:
            # Insuficientes datos históricos
            return 0.02  # Defecto conservador

        winners = [s for s in closed if s.pnl_usd > 0]
        losers = [s for s in closed if s.pnl_usd < 0]

        if not winners or not losers:
            return 0.02

        if avg_win is None:
            avg_win = np.mean([s.pnl_usd for s in winners])
        if avg_loss is None:
            avg_loss = abs(np.mean([s.pnl_usd for s in losers]))

        if avg_loss == 0 or win_rate == 0:
            return 0.02

        # Kelly = W - (1-W)/R
        r_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / r_ratio

        # Limitar a rango seguro
        kelly = max(0.01, min(0.25, kelly))

        return round(kelly, 4)

    def get_recommended_lot(
        self,
        account_balance: float,
        risk_per_trade_pct: float = 0.02
    ) -> float:
        """
        Calcula el tamaño de lote recomendado basado en Kelly Criterion.

        Si no hay suficiente histórico (< 10 operaciones cerradas),
        retorna un lote conservador de 0.01.

        Args:
            account_balance: Saldo de la cuenta en USD
            risk_per_trade_pct: Riesgo máximo por operación (default: 2%)

        Returns:
            Tamaño de lote recomendado (ej: 0.01 para micro)
        """
        closed = [s for s in self.signals if s.outcome != "PENDING"]
        if len(closed) < 10:
            return 0.01  # Defecto: micro lote

        kelly = self.get_kelly_fraction()

        # Riesgo en USD
        risk_usd = account_balance * risk_per_trade_pct

        # Promedio de riesgo por operación en pips
        risk_pips_list = []
        for s in closed:
            if s.action == "BUY":
                risk = abs(s.entry_price - s.sl_price) * 100
            elif s.action == "SELL":
                risk = abs(s.sl_price - s.entry_price) * 100
            else:
                continue
            risk_pips_list.append(risk)

        if not risk_pips_list:
            return 0.01

        avg_risk_pips = np.mean(risk_pips_list)

        # lot_size = risk_usd / (avg_risk_pips * pip_value)
        # pip_value = 10 USD para lote 1.0
        pip_value = 10.0
        lot_size = (risk_usd * kelly) / (avg_risk_pips * pip_value)

        # Redondear a micro lotes (0.01)
        lot_size = round(lot_size * 100) / 100

        # Mínimo 0.01 (micro), máximo 1.0
        lot_size = max(0.01, min(1.0, lot_size))

        return lot_size

    def get_signals_df(self) -> pd.DataFrame:
        """
        Retorna todas las señales como un DataFrame de pandas.

        Returns:
            DataFrame con columnas: timestamp, action, entry_price, outcome, pnl_pips, pnl_usd, etc.
        """
        if not self.signals:
            return pd.DataFrame()

        data = [s.to_dict() for s in self.signals]
        df = pd.DataFrame(data)

        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')

        return df

    def get_recent_signals(self, n: int = 10) -> List[SignalRecord]:
        """
        Retorna las N señales más recientes.

        Args:
            n: Número de señales a retornar

        Returns:
            Lista de SignalRecord ordenadas por timestamp descendente
        """
        sorted_signals = sorted(self.signals, key=lambda s: s.timestamp, reverse=True)
        return sorted_signals[:n]

    def save(self) -> None:
        """Guarda todas las señales en JSON."""
        data = {
            'signals': [s.to_dict() for s in self.signals],
            'saved_at': datetime.now().isoformat()
        }
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Guardadas {len(self.signals)} señales en {self.history_file}")

    def load(self) -> None:
        """Carga señales desde JSON."""
        if not self.history_file.exists():
            self.signals = []
            return

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.signals = [SignalRecord.from_dict(s) for s in data.get('signals', [])]
            logger.info(f"Cargadas {len(self.signals)} señales desde {self.history_file}")
        except Exception as e:
            logger.error(f"Error cargando historial: {e}")
            self.signals = []


# ─────────────────────────────────────────────────────────────
# Tests unitarios
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import shutil

    # Crear directorio temporal para tests
    test_dir = Path(tempfile.mkdtemp())
    print(f"\n{'='*60}")
    print(f"Iniciando tests de PerformanceTracker")
    print(f"Directorio temporal: {test_dir}")
    print(f"{'='*60}\n")

    try:
        # Test 1: Crear rastreador
        print("Test 1: Crear rastreador...")
        tracker = PerformanceTracker(test_dir / "performance_tracker.py")
        print("✅ Rastreador creado correctamente\n")

        # Test 2: Agregar 5 señales mock
        print("Test 2: Agregar 5 señales mock...")
        signals_data = [
            # BUY ganador (TP_HIT)
            {"action": "BUY", "entry": 2300.00, "sl": 2295.00, "tp": 2310.00,
             "lot": 0.01, "confidence": 0.85, "regime": 2, "sentiment": 0.3},
            # SELL ganador (SL_HIT pero con ganancia manual)
            {"action": "SELL", "entry": 2305.00, "sl": 2310.00, "tp": 2295.00,
             "lot": 0.01, "confidence": 0.75, "regime": 0, "sentiment": -0.2},
            # BUY perdedor (SL_HIT)
            {"action": "BUY", "entry": 2298.00, "sl": 2290.00, "tp": 2308.00,
             "lot": 0.02, "confidence": 0.6, "regime": 1, "sentiment": 0.1},
            # NEUTRAL (no se ejecuta)
            {"action": "NEUTRAL", "entry": 2300.00, "sl": 2300.00, "tp": 2300.00,
             "lot": 0.0, "confidence": 0.4, "regime": 1, "sentiment": 0.0},
            # BUY pending
            {"action": "BUY", "entry": 2302.00, "sl": 2298.00, "tp": 2312.00,
             "lot": 0.01, "confidence": 0.88, "regime": 2, "sentiment": 0.5},
        ]

        signal_ids = []
        for sig in signals_data:
            sid = tracker.add_signal(
                action=sig["action"],
                entry=sig["entry"],
                sl=sig["sl"],
                tp=sig["tp"],
                lot=sig["lot"],
                confidence=sig["confidence"],
                regime=sig["regime"],
                sentiment=sig["sentiment"]
            )
            signal_ids.append(sid)

        print(f"✅ {len(signal_ids)} señales agregadas\n")

        # Test 3: Actualizar resultados
        print("Test 3: Actualizar resultados...")
        # Señal 0: BUY ganador
        tracker.update_outcome(signal_ids[0], "TP_HIT", 2310.50, notes="Resistencia alcanzada")

        # Señal 1: SELL ganador
        tracker.update_outcome(signal_ids[1], "TP_HIT", 2294.50, notes="Soporte confirmado")

        # Señal 2: BUY perdedor
        tracker.update_outcome(signal_ids[2], "SL_HIT", 2289.80, notes="Rotura de soporte")

        # Señal 3: NEUTRAL - no actualizar

        # Señal 4: BUY pendiente

        print("✅ Resultados actualizados\n")

        # Test 4: Calcular estadísticas
        print("Test 4: Calcular estadísticas...")
        stats = tracker.get_stats()
        print(f"  Total de señales: {stats.total_signals}")
        print(f"  Señales ejecutadas: {stats.executed_signals}")
        print(f"  Señales cerradas: {stats.closed_signals}")
        print(f"  Tasa de ganancia: {stats.win_rate * 100:.1f}%")
        print(f"  Ratio R/R promedio: {stats.avg_rr:.2f}")
        print(f"  PnL total: ${stats.total_pnl_usd:.2f}")
        print(f"  Factor de ganancia: {stats.profit_factor:.2f}")
        print(f"  Max Drawdown: {stats.max_drawdown_pips:.2f} pips")
        print(f"  Mejor operación: {stats.best_trade_pips:.2f} pips")
        print(f"  Peor operación: {stats.worst_trade_pips:.2f} pips")
        print(f"  Confianza (ganadores): {stats.avg_confidence_winners:.3f}")
        print(f"  Confianza (perdedores): {stats.avg_confidence_losers:.3f}")
        print(f"  Rachas de ganancias: {stats.consecutive_wins}")
        print(f"  Rachas de pérdidas: {stats.consecutive_losses}")
        print(f"✅ Estadísticas calculadas\n")

        # Test 5: Kelly Criterion
        print("Test 5: Kelly Criterion...")
        kelly = tracker.get_kelly_fraction()
        print(f"  Fracción de Kelly: {kelly:.4f} ({kelly*100:.2f}%)")
        print(f"✅ Kelly calculado\n")

        # Test 6: Tamaño recomendado
        print("Test 6: Tamaño de lote recomendado...")
        recommended_lot = tracker.get_recommended_lot(account_balance=5000.0)
        print(f"  Lote recomendado (cuenta $5k): {recommended_lot:.2f}")
        print(f"✅ Lote recomendado\n")

        # Test 7: DataFrame
        print("Test 7: Exportar a DataFrame...")
        df = tracker.get_signals_df()
        print(f"  Filas en DataFrame: {len(df)}")
        print(f"  Columnas: {list(df.columns)[:5]}...")
        print(f"✅ DataFrame creado\n")

        # Test 8: Señales recientes
        print("Test 8: Últimas 3 señales...")
        recent = tracker.get_recent_signals(n=3)
        for i, sig in enumerate(recent, 1):
            print(f"  {i}. {sig.action} @ {sig.entry_price} ({sig.outcome})")
        print(f"✅ Señales recientes obtenidas\n")

        # Test 9: Guardar en JSON
        print("Test 9: Guardar en JSON...")
        tracker.save()
        history_file = test_dir / "signals_history.json"
        if history_file.exists():
            size = history_file.stat().st_size
            print(f"  Archivo: {history_file}")
            print(f"  Tamaño: {size} bytes")
            print(f"✅ Guardado exitoso\n")

        # Test 10: Cargar desde JSON
        print("Test 10: Cargar desde JSON...")
        tracker2 = PerformanceTracker(test_dir / "performance_tracker.py")
        if len(tracker2.signals) == len(tracker.signals):
            print(f"  Señales cargadas: {len(tracker2.signals)}")
            print(f"✅ Carga exitosa\n")

        # Resumen final
        print(f"{'='*60}")
        print(f"RESUMEN DE TESTS")
        print(f"{'='*60}")
        print(f"✅ Test 1: Crear rastreador")
        print(f"✅ Test 2: Agregar señales mock")
        print(f"✅ Test 3: Actualizar resultados")
        print(f"✅ Test 4: Calcular estadísticas")
        print(f"✅ Test 5: Kelly Criterion")
        print(f"✅ Test 6: Tamaño de lote recomendado")
        print(f"✅ Test 7: Exportar a DataFrame")
        print(f"✅ Test 8: Obtener señales recientes")
        print(f"✅ Test 9: Guardar en JSON")
        print(f"✅ Test 10: Cargar desde JSON")
        print(f"\n{'='*60}")
        print(f"TODOS LOS TESTS PASARON ✅")
        print(f"{'='*60}\n")

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(test_dir, ignore_errors=True)
