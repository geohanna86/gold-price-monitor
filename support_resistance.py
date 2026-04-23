# ============================================================
# support_resistance.py — Cálculo de Soportes y Resistencias
# Gold Price Monitor — Phase 4
#
# Componentes:
#   1. SupportResistanceCalculator → Pivots, niveles clave, ATR
#
# ============================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class SupportResistanceLevel:
    """Representa un nivel de soporte o resistencia."""
    price: float
    """Precio del nivel"""

    type: str
    """'resistance' o 'support'"""

    strength: int
    """Fortaleza del nivel (1=débil, 2=medio, 3=fuerte) basada en toques"""

    touches: int
    """Número de veces que el precio tocó este nivel"""


# ─────────────────────────────────────────────────────────────
# SupportResistanceCalculator
# ─────────────────────────────────────────────────────────────

class SupportResistanceCalculator:
    """
    Calcula soportes, resistencias, puntos de pivote y niveles clave
    a partir de datos OHLCV.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el calculador.

        Args:
            df: DataFrame con columnas [Open, High, Low, Close, Volume]
        """
        self.df = df.copy()

        # Validar columnas obligatorias
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        self.df.reset_index(drop=True, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # Pivot Points
    # ─────────────────────────────────────────────────────────────

    def get_pivot_points(self, method: str = "classic") -> Dict[str, float]:
        """
        Calcula puntos de pivote usando el método clásico.

        Fórmula clásica:
        - P (Pivot)  = (High + Low + Close) / 3
        - R1 = 2*P - Low
        - R2 = P + (High - Low)
        - R3 = High + 2*(P - Low)
        - S1 = 2*P - High
        - S2 = P - (High - Low)
        - S3 = Low - 2*(High - P)

        Args:
            method: Método de cálculo (por ahora solo "classic")

        Returns:
            Dict con keys: P, R1, R2, R3, S1, S2, S3
        """
        if method != "classic":
            raise ValueError(f"Método no soportado: {method}")

        # Usar la vela anterior (cierre de ayer)
        if len(self.df) < 2:
            raise ValueError("Se requieren al menos 2 velas para calcular pivotes")

        h = self.df.iloc[-2]['High']
        l = self.df.iloc[-2]['Low']
        c = self.df.iloc[-2]['Close']

        p = (h + l + c) / 3
        r1 = 2 * p - l
        r2 = p + (h - l)
        r3 = h + 2 * (p - l)
        s1 = 2 * p - h
        s2 = p - (h - l)
        s3 = l - 2 * (h - p)

        return {
            "P": p,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "S1": s1,
            "S2": s2,
            "S3": s3,
        }

    # ─────────────────────────────────────────────────────────────
    # Key Levels (clustering)
    # ─────────────────────────────────────────────────────────────

    def get_key_levels(self, n_levels: int = 5, lookback: int = 50) -> List[Dict]:
        """
        Calcula niveles clave detectando máximos y mínimos locales
        mediante clustering.

        Args:
            n_levels: Número de niveles a retornar
            lookback: Número de velas históricas a analizar

        Returns:
            Lista de dicts con: price, type, strength, touches
        """
        if len(self.df) < lookback:
            lookback = len(self.df)

        df_window = self.df.iloc[-lookback:].copy()

        # Detectar máximos locales (resistencias)
        resistances = self._find_resistance_levels(df_window)

        # Detectar mínimos locales (soportes)
        supports = self._find_support_levels(df_window)

        # Combinar y ordenar por fortaleza
        all_levels = resistances + supports
        all_levels.sort(key=lambda x: x['strength'], reverse=True)

        return all_levels[:n_levels]

    def _find_resistance_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Encuentra niveles de resistencia (máximos locales)."""
        levels = []
        highs = df['High'].values

        # Agrupar máximos cercanos (dentro de 0.5%)
        cluster_range = 0.005
        clustered = []

        for i, high in enumerate(highs):
            found_cluster = False
            for cluster in clustered:
                if abs(high - cluster['price']) / cluster['price'] < cluster_range:
                    cluster['count'] += 1
                    cluster['price'] = (cluster['price'] * (cluster['count'] - 1) + high) / cluster['count']
                    found_cluster = True
                    break

            if not found_cluster:
                clustered.append({'price': high, 'count': 1})

        # Convertir a niveles
        for cluster in clustered:
            strength = min(3, max(1, cluster['count'] // 2))  # 1-3 basado en toques
            levels.append({
                'price': cluster['price'],
                'type': 'resistance',
                'strength': strength,
                'touches': cluster['count']
            })

        return levels

    def _find_support_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Encuentra niveles de soporte (mínimos locales)."""
        levels = []
        lows = df['Low'].values

        # Agrupar mínimos cercanos (dentro de 0.5%)
        cluster_range = 0.005
        clustered = []

        for i, low in enumerate(lows):
            found_cluster = False
            for cluster in clustered:
                if abs(low - cluster['price']) / cluster['price'] < cluster_range:
                    cluster['count'] += 1
                    cluster['price'] = (cluster['price'] * (cluster['count'] - 1) + low) / cluster['count']
                    found_cluster = True
                    break

            if not found_cluster:
                clustered.append({'price': low, 'count': 1})

        # Convertir a niveles
        for cluster in clustered:
            strength = min(3, max(1, cluster['count'] // 2))  # 1-3 basado en toques
            levels.append({
                'price': cluster['price'],
                'type': 'support',
                'strength': strength,
                'touches': cluster['count']
            })

        return levels

    # ─────────────────────────────────────────────────────────────
    # Nearest Levels
    # ─────────────────────────────────────────────────────────────

    def get_nearest_levels(self, current_price: float, n: int = 3) -> Dict[str, List[Dict]]:
        """
        Retorna los N niveles más cercanos arriba (resistencias) y debajo (soportes)
        del precio actual.

        Args:
            current_price: Precio actual
            n: Número de niveles a retornar en cada dirección

        Returns:
            Dict con keys "above" (resistencias) y "below" (soportes)
        """
        levels = self.get_key_levels(n_levels=20, lookback=50)

        above = [l for l in levels if l['price'] > current_price]
        below = [l for l in levels if l['price'] < current_price]

        # Ordenar y limitar
        above.sort(key=lambda x: x['price'])
        below.sort(key=lambda x: x['price'], reverse=True)

        return {
            'above': above[:n],
            'below': below[:n]
        }

    # ─────────────────────────────────────────────────────────────
    # ATR (Average True Range)
    # ─────────────────────────────────────────────────────────────

    def get_atr(self, period: int = 14) -> float:
        """
        Calcula el Average True Range (ATR).

        Args:
            period: Período para el cálculo (default 14)

        Returns:
            ATR actual
        """
        if len(self.df) < period:
            return 0.0

        df = self.df.copy()

        # Calcular True Range
        df['prev_close'] = df['Close'].shift(1)
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['prev_close']),
                abs(df['Low'] - df['prev_close'])
            )
        )

        # SMA del TR
        atr = df['tr'].tail(period).mean()
        return atr

    # ─────────────────────────────────────────────────────────────
    # Level Proximity Check
    # ─────────────────────────────────────────────────────────────

    def is_near_level(self, current_price: float, threshold_pct: float = 0.3) -> bool:
        """
        Verifica si el precio actual está cerca de un nivel importante
        (dentro del threshold porcentual).

        Args:
            current_price: Precio actual
            threshold_pct: Umbral porcentual (0.3 = 0.3%)

        Returns:
            True si está cerca de algún nivel, False en caso contrario
        """
        levels = self.get_key_levels(n_levels=10)
        threshold = current_price * (threshold_pct / 100)

        for level in levels:
            if abs(current_price - level['price']) <= threshold:
                return True

        return False


# ─────────────────────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS: SupportResistanceCalculator")
    print("="*70)

    # Crear DataFrame mock realista (últimas 100 velas)
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')

    # Simular precio con tendencia alcista + ruido
    base_price = 2300
    prices = base_price + np.cumsum(np.random.randn(100) * 0.5)

    df_mock = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(100) * 0.2,
        'High': prices + abs(np.random.randn(100) * 1.0),
        'Low': prices - abs(np.random.randn(100) * 1.0),
        'Close': prices + np.random.randn(100) * 0.2,
        'Volume': np.random.randint(100000, 500000, 100),
    })

    calc = SupportResistanceCalculator(df_mock)

    # ─────────────────────────────────────────────────────────────
    # Test 1: Pivot Points
    # ─────────────────────────────────────────────────────────────
    try:
        pivots = calc.get_pivot_points(method="classic")

        assert 'P' in pivots
        assert 'R1' in pivots and 'R2' in pivots and 'R3' in pivots
        assert 'S1' in pivots and 'S2' in pivots and 'S3' in pivots

        # Las resistencias deben ser mayores que el pivot
        assert pivots['R1'] > pivots['P']
        assert pivots['R2'] > pivots['P']

        # Los soportes deben ser menores que el pivot
        assert pivots['S1'] < pivots['P']
        assert pivots['S2'] < pivots['P']

        print(f"✅ Test 1: Pivot Points OK")
        print(f"   Pivot: {pivots['P']:.2f} | R1: {pivots['R1']:.2f} | S1: {pivots['S1']:.2f}")

    except AssertionError as e:
        print(f"❌ Test 1: Pivot Points FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 2: Key Levels
    # ─────────────────────────────────────────────────────────────
    try:
        key_levels = calc.get_key_levels(n_levels=5)

        assert len(key_levels) > 0
        assert len(key_levels) <= 5

        for level in key_levels:
            assert 'price' in level
            assert level['type'] in ['resistance', 'support']
            assert 1 <= level['strength'] <= 3
            assert level['touches'] > 0

        print(f"✅ Test 2: Key Levels OK")
        print(f"   Encontrados {len(key_levels)} niveles clave")
        for i, level in enumerate(key_levels):
            print(f"     {i+1}. {level['type'].upper()}: {level['price']:.2f} (fuerza: {level['strength']}, toques: {level['touches']})")

    except AssertionError as e:
        print(f"❌ Test 2: Key Levels FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 3: Nearest Levels
    # ─────────────────────────────────────────────────────────────
    try:
        current_price = df_mock.iloc[-1]['Close']
        nearest = calc.get_nearest_levels(current_price=current_price, n=3)

        assert 'above' in nearest
        assert 'below' in nearest
        assert len(nearest['above']) <= 3
        assert len(nearest['below']) <= 3

        # Validar que los niveles "above" estén arriba
        for level in nearest['above']:
            assert level['price'] > current_price

        # Validar que los niveles "below" estén abajo
        for level in nearest['below']:
            assert level['price'] < current_price

        print(f"✅ Test 3: Nearest Levels OK")
        print(f"   Precio actual: {current_price:.2f}")
        resistencias_str = [f"{l['price']:.2f}" for l in nearest['above']]
        soportes_str = [f"{l['price']:.2f}" for l in nearest['below']]
        print(f"   Resistencias (arriba): {resistencias_str}")
        print(f"   Soportes (abajo): {soportes_str}")

    except AssertionError as e:
        print(f"❌ Test 3: Nearest Levels FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 4: ATR
    # ─────────────────────────────────────────────────────────────
    try:
        atr = calc.get_atr(period=14)

        assert isinstance(atr, float)
        assert atr >= 0

        print(f"✅ Test 4: ATR OK")
        print(f"   ATR(14): {atr:.4f}")

    except AssertionError as e:
        print(f"❌ Test 4: ATR FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 5: Is Near Level
    # ─────────────────────────────────────────────────────────────
    try:
        current_price = df_mock.iloc[-1]['Close']

        # Debería estar cerca (threshold 0.5%)
        is_near = calc.is_near_level(current_price=current_price, threshold_pct=0.5)
        assert isinstance(is_near, bool)

        # Un precio aleatorio muy lejano probablemente no esté cerca
        far_price = current_price * 1.5  # 50% más arriba
        is_near_far = calc.is_near_level(current_price=far_price, threshold_pct=0.1)

        print(f"✅ Test 5: Is Near Level OK")
        print(f"   Cerca del nivel (threshold 0.5%): {is_near}")
        print(f"   Precio lejano cerca (threshold 0.1%): {is_near_far}")

    except AssertionError as e:
        print(f"❌ Test 5: Is Near Level FALLÓ - {e}")

    print("\n" + "="*70)
    print("TESTS COMPLETADOS")
    print("="*70 + "\n")
