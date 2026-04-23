# ============================================================
# economic_calendar.py — Calendario Económico
# Gold Price Monitor — Phase 4
#
# Componentes:
#   1. EconomicEvent (dataclass) → Evento económico
#   2. EconomicCalendar → Gestión de eventos y advertencias
#
# ============================================================

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("EconomicCalendar")


# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class EconomicEvent:
    """Representa un evento económico."""

    name: str
    """Nombre del evento (ej: 'FOMC', 'NFP', 'CPI')"""

    date: datetime
    """Fecha y hora del evento"""

    impact: str
    """Nivel de impacto: 'HIGH', 'MEDIUM', 'LOW'"""

    currency: str
    """Moneda afectada (ej: 'USD', 'EUR', 'GBP')"""

    actual: str
    """Valor actual reportado"""

    forecast: str
    """Valor proyectado"""

    previous: str
    """Valor anterior"""

    description: str
    """Descripción adicional del evento"""

    def __repr__(self) -> str:
        return (f"EconomicEvent({self.name} | {self.date.strftime('%Y-%m-%d %H:%M')} | "
                f"Impact: {self.impact} | {self.currency})")


# ─────────────────────────────────────────────────────────────
# EconomicCalendar
# ─────────────────────────────────────────────────────────────

class EconomicCalendar:
    """
    Gestiona eventos económicos: recuperación, filtrado y generación
    de advertencias de riesgo.
    """

    # Patrones de eventos recurrentes (aproximación para 2026)
    RECURRING_EVENTS = {
        'FOMC': {
            'impact': 'HIGH',
            'currency': 'USD',
            'frequency': 'cada 6 semanas (miércoles)',
            'dates_2026': [
                datetime(2026, 1, 28),
                datetime(2026, 3, 17),
                datetime(2026, 5, 7),
                datetime(2026, 6, 17),
                datetime(2026, 7, 28),
                datetime(2026, 9, 16),
                datetime(2026, 11, 1),
                datetime(2026, 12, 16),
            ],
            'description': 'Federal Open Market Committee - Decisión de tasa de interés'
        },
        'NFP': {
            'impact': 'HIGH',
            'currency': 'USD',
            'frequency': 'primer viernes de cada mes (8:30 AM ET)',
            'description': 'Non-Farm Payroll - Empleo no agrícola'
        },
        'CPI': {
            'impact': 'HIGH',
            'currency': 'USD',
            'frequency': 'segundo miércoles de cada mes',
            'description': 'Consumer Price Index - Inflación al consumidor'
        },
        'PPI': {
            'impact': 'MEDIUM',
            'currency': 'USD',
            'frequency': 'segundo jueves de cada mes',
            'description': 'Producer Price Index - Inflación productiva'
        },
        'GDP': {
            'impact': 'HIGH',
            'currency': 'USD',
            'frequency': 'fin de mes',
            'description': 'Gross Domestic Product - Crecimiento económico'
        },
        'RETAIL_SALES': {
            'impact': 'MEDIUM',
            'currency': 'USD',
            'frequency': 'tercer miércoles de cada mes',
            'description': 'Retail Sales - Ventas minoristas'
        },
    }

    def __init__(self):
        """Inicializa el calendario económico."""
        self._cache = None
        self._cache_time = None

    # ─────────────────────────────────────────────────────────────
    # Getters principales
    # ─────────────────────────────────────────────────────────────

    def get_upcoming_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """
        Retorna eventos económicos próximos.

        Intenta usar datos de APIs (investing.com o finnhub.io).
        Fallback: base de datos estática de eventos recurrentes.

        Args:
            days_ahead: Días hacia el futuro a buscar (default 7)

        Returns:
            Lista de eventos económicos ordenados por fecha
        """
        now = datetime.now()
        future_limit = now + timedelta(days=days_ahead)

        # Intentar cargar de API (placeholder para expansión futura)
        events = self._fetch_from_api(now, future_limit)

        # Si falla o está vacío, usar base de datos estática
        if not events:
            events = self._get_static_events(now, future_limit)

        return sorted(events, key=lambda e: e.date)

    def get_today_events(self) -> List[EconomicEvent]:
        """
        Retorna eventos económicos de hoy solamente.

        Returns:
            Lista de eventos de hoy
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        return [e for e in self.get_upcoming_events(days_ahead=1)
                if today_start <= e.date <= today_end]

    def get_next_high_impact(self) -> Optional[EconomicEvent]:
        """
        Retorna el próximo evento de alto impacto (HIGH).

        Returns:
            EconomicEvent o None si no hay HIGH impact próximo
        """
        now = datetime.now()
        upcoming = self.get_upcoming_events(days_ahead=30)
        high_impact = [e for e in upcoming if e.impact == 'HIGH']

        return high_impact[0] if high_impact else None

    def is_high_impact_day(self) -> Tuple[bool, str]:
        """
        Verifica si hoy hay eventos de alto impacto.

        Returns:
            Tupla (bool, mensaje) — True si hay HIGH impact hoy
        """
        today_events = self.get_today_events()
        high_impact_events = [e for e in today_events if e.impact == 'HIGH']

        if high_impact_events:
            event_names = ', '.join([e.name for e in high_impact_events])
            return True, f"⚠️ Evento de alto impacto HOY: {event_names}"

        return False, ""

    def get_warning_level(self) -> str:
        """
        Determina el nivel de advertencia basado en eventos próximos.

        Niveles:
        - 'DANGER': Evento HIGH impact en las próximas 2 horas
        - 'CAUTION': Evento HIGH impact hoy O HIGH impact mañana
        - 'CLEAR': Sin peligro inmediato

        Returns:
            'DANGER', 'CAUTION', o 'CLEAR'
        """
        now = datetime.now()
        upcoming = self.get_upcoming_events(days_ahead=2)

        # ─ DANGER: HIGH impact en las próximas 2 horas
        danger_limit = now + timedelta(hours=2)
        for event in upcoming:
            if event.impact == 'HIGH' and now <= event.date <= danger_limit:
                logger.warning(f"🚨 DANGER: {event.name} en las próximas 2 horas")
                return 'DANGER'

        # ─ CAUTION: HIGH impact hoy O mañana
        tomorrow = now + timedelta(days=1)
        for event in upcoming:
            if event.impact == 'HIGH':
                if now.date() == event.date.date() or \
                   (tomorrow.date() == event.date.date()):
                    logger.warning(f"⚠️ CAUTION: {event.name} próximamente")
                    return 'CAUTION'

        # ─ CLEAR: Sin peligro inmediato
        return 'CLEAR'

    # ─────────────────────────────────────────────────────────────
    # Métodos privados
    # ─────────────────────────────────────────────────────────────

    def _fetch_from_api(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        Intenta recuperar eventos de APIs públicas.

        Soporta (sin API key requerida):
        - investing.com RSS
        - finnhub.io calendar

        Args:
            start: Fecha de inicio
            end: Fecha de fin

        Returns:
            Lista de eventos (vacía si falla)
        """
        # Placeholder para integración real con APIs
        # Por ahora retorna lista vacía para usar fallback

        # TODO: Implementar llamadas reales a investing.com o finnhub.io
        # Ejemplo (requeriría manejo de conexión):
        #   try:
        #       response = urllib.request.urlopen('https://...')
        #       data = json.loads(response.read())
        #       return [EconomicEvent(...) for item in data]
        #   except:
        #       return []

        return []

    def _get_static_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        Retorna eventos de la base de datos estática.

        Args:
            start: Fecha de inicio
            end: Fecha de fin

        Returns:
            Lista de eventos estáticos en el rango
        """
        events = []

        # ─ FOMC (fechas específicas 2026)
        fomc_event = self.RECURRING_EVENTS['FOMC']
        for date in fomc_event['dates_2026']:
            # Fijar hora típica (2:00 PM ET)
            date_with_time = date.replace(hour=14, minute=0, second=0)
            if start <= date_with_time <= end:
                events.append(EconomicEvent(
                    name='FOMC',
                    date=date_with_time,
                    impact=fomc_event['impact'],
                    currency=fomc_event['currency'],
                    actual='',
                    forecast='',
                    previous='',
                    description=fomc_event['description']
                ))

        # ─ NFP (primer viernes de cada mes)
        for month in range(start.month, end.month + 1):
            year = start.year if month == start.month else end.year
            first_day = datetime(year, month if month <= 12 else 1, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)

            # Si el 1º es viernes, ese es el primer viernes
            if first_day.weekday() == 4:
                first_friday = first_day
            else:
                first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)

            first_friday = first_friday.replace(hour=8, minute=30, second=0)

            if start <= first_friday <= end:
                events.append(EconomicEvent(
                    name='NFP',
                    date=first_friday,
                    impact='HIGH',
                    currency='USD',
                    actual='',
                    forecast='',
                    previous='',
                    description='Non-Farm Payroll - Empleo no agrícola'
                ))

        # ─ CPI (segundo miércoles)
        for month in range(start.month, end.month + 1):
            year = start.year if month == start.month else end.year
            # Segundo miércoles = día 8-14, Wednesday=2
            for day in range(8, 15):
                if day > 31:
                    break
                try:
                    d = datetime(year, month if month <= 12 else 1, day)
                    if d.weekday() == 2:  # Miércoles
                        d = d.replace(hour=13, minute=30, second=0)
                        if start <= d <= end:
                            events.append(EconomicEvent(
                                name='CPI',
                                date=d,
                                impact='HIGH',
                                currency='USD',
                                actual='',
                                forecast='',
                                previous='',
                                description='Consumer Price Index - Inflación'
                            ))
                        break
                except ValueError:
                    break

        # ─ PPI (segundo jueves)
        for month in range(start.month, end.month + 1):
            year = start.year if month == start.month else end.year
            # Segundo jueves = día 8-14, Thursday=3
            for day in range(8, 15):
                if day > 31:
                    break
                try:
                    d = datetime(year, month if month <= 12 else 1, day)
                    if d.weekday() == 3:  # Jueves
                        d = d.replace(hour=13, minute=30, second=0)
                        if start <= d <= end:
                            events.append(EconomicEvent(
                                name='PPI',
                                date=d,
                                impact='MEDIUM',
                                currency='USD',
                                actual='',
                                forecast='',
                                previous='',
                                description='Producer Price Index - Inflación productiva'
                            ))
                        break
                except ValueError:
                    break

        # ─ GDP (fin de mes)
        for month in range(start.month, end.month + 1):
            year = start.year if month == start.month else end.year
            try:
                # Último día del mes (típicamente últimas 2 semanas)
                last_day = datetime(year, month if month <= 12 else 1, 28)
                while True:
                    try:
                        last_day += timedelta(days=1)
                    except OverflowError:
                        break
                    if last_day.month != month:
                        last_day -= timedelta(days=1)
                        break

                last_day = last_day.replace(hour=12, minute=30, second=0)
                if start <= last_day <= end and last_day.day >= 25:
                    events.append(EconomicEvent(
                        name='GDP',
                        date=last_day,
                        impact='HIGH',
                        currency='USD',
                        actual='',
                        forecast='',
                        previous='',
                        description='Gross Domestic Product - Crecimiento económico'
                    ))
            except (ValueError, OverflowError):
                pass

        # ─ Retail Sales (tercer miércoles)
        for month in range(start.month, end.month + 1):
            year = start.year if month == start.month else end.year
            # Tercer miércoles = día 15-21, Wednesday=2
            for day in range(15, 22):
                if day > 31:
                    break
                try:
                    d = datetime(year, month if month <= 12 else 1, day)
                    if d.weekday() == 2:  # Miércoles
                        d = d.replace(hour=13, minute=30, second=0)
                        if start <= d <= end:
                            events.append(EconomicEvent(
                                name='RETAIL_SALES',
                                date=d,
                                impact='MEDIUM',
                                currency='USD',
                                actual='',
                                forecast='',
                                previous='',
                                description='Retail Sales - Ventas minoristas'
                            ))
                        break
                except ValueError:
                    break

        return events


# ─────────────────────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS: EconomicCalendar")
    print("="*70)

    cal = EconomicCalendar()

    # ─────────────────────────────────────────────────────────────
    # Test 1: Get Upcoming Events
    # ─────────────────────────────────────────────────────────────
    try:
        events = cal.get_upcoming_events(days_ahead=14)

        assert isinstance(events, list)
        assert len(events) > 0
        assert all(isinstance(e, EconomicEvent) for e in events)

        # Verificar que están ordenados por fecha
        for i in range(1, len(events)):
            assert events[i].date >= events[i-1].date

        print(f"✅ Test 1: Get Upcoming Events OK")
        print(f"   Encontrados {len(events)} eventos en los próximos 14 días")
        for i, e in enumerate(events[:3]):
            print(f"     {i+1}. {e.name} | {e.date.strftime('%Y-%m-%d %H:%M')} | {e.impact}")

    except AssertionError as e:
        print(f"❌ Test 1: Get Upcoming Events FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 2: Get Today Events
    # ─────────────────────────────────────────────────────────────
    try:
        today_events = cal.get_today_events()

        assert isinstance(today_events, list)
        # Los eventos de hoy deben estar hoy
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        for event in today_events:
            assert today_start <= event.date <= today_end

        print(f"✅ Test 2: Get Today Events OK")
        print(f"   {len(today_events)} eventos hoy")
        for e in today_events:
            print(f"     - {e.name} | {e.date.strftime('%H:%M')} | {e.impact}")

    except AssertionError as e:
        print(f"❌ Test 2: Get Today Events FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 3: Get Next High Impact
    # ─────────────────────────────────────────────────────────────
    try:
        next_high = cal.get_next_high_impact()

        if next_high:
            assert isinstance(next_high, EconomicEvent)
            assert next_high.impact == 'HIGH'
            assert next_high.date >= datetime.now()

            print(f"✅ Test 3: Get Next High Impact OK")
            print(f"   Próximo HIGH: {next_high.name} | {next_high.date.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"✅ Test 3: Get Next High Impact OK (sin HIGH impact próximo)")

    except AssertionError as e:
        print(f"❌ Test 3: Get Next High Impact FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 4: Is High Impact Day
    # ─────────────────────────────────────────────────────────────
    try:
        is_high, msg = cal.is_high_impact_day()

        assert isinstance(is_high, bool)
        assert isinstance(msg, str)

        if is_high:
            assert "⚠️" in msg or msg != ""

        print(f"✅ Test 4: Is High Impact Day OK")
        print(f"   Hoy es día de alto impacto: {is_high}")
        if msg:
            print(f"   {msg}")

    except AssertionError as e:
        print(f"❌ Test 4: Is High Impact Day FALLÓ - {e}")

    # ─────────────────────────────────────────────────────────────
    # Test 5: Get Warning Level
    # ─────────────────────────────────────────────────────────────
    try:
        warning = cal.get_warning_level()

        assert warning in ['DANGER', 'CAUTION', 'CLEAR']

        print(f"✅ Test 5: Get Warning Level OK")
        print(f"   Nivel de advertencia: {warning}")

        # Descripción del nivel
        warnings_desc = {
            'DANGER': '🚨 Evento HIGH impact en las próximas 2 horas',
            'CAUTION': '⚠️ Evento HIGH impact hoy o mañana',
            'CLEAR': '✅ Sin peligro económico inmediato'
        }
        print(f"   ({warnings_desc.get(warning, '')})")

    except AssertionError as e:
        print(f"❌ Test 5: Get Warning Level FALLÓ - {e}")

    print("\n" + "="*70)
    print("TESTS COMPLETADOS")
    print("="*70 + "\n")
