# ============================================================
# news_filter.py — Filtro de Noticias y Sentiment para Oro
# Gold Price Monitor — Phase 5
#
# Fuentes de datos soportadas:
#   1) RSS gratuito (Yahoo Finance, Google News) — sin API key
#   2) NewsAPI (newsapi.org)     — API key gratuita
#   3) Alpha Vantage News       — API key gratuita
#
# Lógica:
#   Descarga titulares → Score de sentiment por keywords →
#   Multiplica la confianza del Ensemble → Filtra señales
#
# Configuración:
#   export NEWS_API_KEY="tu_clave_de_newsapi"
#   export ALPHAVANTAGE_KEY="tu_clave_de_alphavantage"
# ============================================================

import os
import json
import logging
from pathlib import Path


def _load_env_file():
    """Carga variables desde config.env si existe."""
    env_path = Path(__file__).parent / "config.env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_env_file()
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger("NewsFilter")


# ─────────────────────────────────────────────────────────────
# Diccionarios de keywords — efecto sobre el oro
# ─────────────────────────────────────────────────────────────

# Palabras que SUBEN el oro (sentiment alcista)
GOLD_BULLISH_KEYWORDS: Dict[str, float] = {
    # Trump / Geopolítica
    "tariff":           +0.85,
    "tariffs":          +0.85,
    "trade war":        +0.90,
    "trade conflict":   +0.80,
    "trump":            +0.40,   # neutro-positivo por defecto
    "sanction":         +0.75,
    "sanctions":        +0.75,
    "embargo":          +0.70,

    # Guerra / Conflicto
    "war":              +0.90,
    "attack":           +0.75,
    "missile":          +0.85,
    "nuclear":          +0.95,
    "invasion":         +0.90,
    "conflict":         +0.70,
    "escalation":       +0.80,
    "geopolitic":       +0.65,
    "tension":          +0.60,
    "crisis":           +0.70,
    "threat":           +0.60,

    # Política monetaria / Inflación
    "rate cut":         +0.80,
    "rate cuts":        +0.80,
    "interest rate cut":+0.85,
    "fed cut":          +0.80,
    "dovish":           +0.75,
    "inflation":        +0.70,
    "hyperinflation":   +0.95,
    "stagflation":      +0.85,
    "quantitative easing": +0.80,
    "money printing":   +0.85,
    "debt ceiling":     +0.70,
    "default":          +0.80,
    "recession":        +0.65,
    "dollar weak":      +0.75,
    "dollar falls":     +0.70,
    "usd decline":      +0.70,

    # Safe Haven / Refugio
    "safe haven":       +0.85,
    "flight to safety": +0.90,
    "uncertainty":      +0.55,
    "risk off":         +0.75,
    "market crash":     +0.90,
    "panic":            +0.80,
    "volatility spike": +0.70,

    # Demanda de oro
    "gold demand":      +0.70,
    "gold reserve":     +0.65,
    "central bank buy": +0.85,
    "gold buying":      +0.70,
    "gold rally":       +0.75,
    "gold surge":       +0.80,
    "gold soars":       +0.80,
    "gold hits record": +0.90,
    "all-time high":    +0.75,
}

# Palabras que BAJAN el oro (sentiment bajista)
GOLD_BEARISH_KEYWORDS: Dict[str, float] = {
    # Paz / Acuerdos
    "ceasefire":        -0.80,
    "peace deal":       -0.85,
    "peace agreement":  -0.85,
    "trade deal":       -0.80,
    "trade agreement":  -0.75,
    "tariff removed":   -0.85,
    "tariff cut":       -0.80,
    "de-escalation":    -0.75,
    "diplomacy":        -0.55,

    # Política monetaria restrictiva
    "rate hike":        -0.80,
    "rate hikes":       -0.80,
    "interest rate hike": -0.85,
    "hawkish":          -0.75,
    "fed hike":         -0.80,
    "tightening":       -0.65,
    "quantitative tightening": -0.75,

    # Fortaleza del dólar
    "dollar strong":    -0.75,
    "dollar rises":     -0.70,
    "usd rally":        -0.70,
    "dollar surge":     -0.75,
    "dxy up":           -0.65,

    # Risk On / Mercados al alza
    "risk on":          -0.70,
    "stock rally":      -0.60,
    "market rally":     -0.55,
    "bull market":      -0.50,
    "economic growth":  -0.50,
    "gdp growth":       -0.55,
    "recovery":         -0.50,
    "job growth":       -0.45,
    "strong economy":   -0.55,

    # Caída del oro
    "gold falls":       -0.75,
    "gold drops":       -0.75,
    "gold decline":     -0.70,
    "gold slumps":      -0.80,
    "gold selling":     -0.65,
    "gold outflows":    -0.70,
}

# Keywords especiales de Trump (contexto específico)
TRUMP_CONTEXT_KEYWORDS: Dict[str, float] = {
    "trump tariff":     +0.95,
    "trump trade":      +0.70,
    "trump sanction":   +0.85,
    "trump threatens":  +0.80,
    "trump warning":    +0.75,
    "trump deal":       -0.60,
    "trump agreement":  -0.65,
    "trump signs":      -0.40,
    "trump fed":        +0.70,   # presión sobre la Fed → oro sube
    "trump dollar":     +0.55,
    "trump china":      +0.75,
    "trump iran":       +0.85,
    "trump russia":     +0.80,
    "trump ukraine":    +0.75,
    "trump nato":       +0.65,
    "trump withdraw":   +0.60,
}


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
@dataclass
class NewsConfig:
    # API keys — opcionales (usar variables de entorno)
    newsapi_key:       str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    alphavantage_key:  str = field(default_factory=lambda: os.getenv("ALPHAVANTAGE_KEY", ""))

    # Consultas de búsqueda
    search_queries:    List[str] = field(default_factory=lambda: [
        "gold price",
        "XAU USD",
        "gold tariff",
        "Trump gold",
        "Federal Reserve gold",
        "gold safe haven",
        "geopolitical gold",
    ])

    # Filtros
    max_articles:      int   = 30        # máximo artículos a analizar
    cache_minutes:     int   = 30        # tiempo de caché en minutos
    min_relevance:     float = 0.10      # score mínimo para considerar

    # Pesos del efecto sobre el score final
    sentiment_cap:     float = 0.40      # máximo ajuste de confianza ±40%
    trump_boost:       float = 1.25      # multiplicador extra para noticias de Trump

    # Timeout para requests HTTP
    request_timeout:   int   = 8


# ─────────────────────────────────────────────────────────────
# Estructura de un artículo
# ─────────────────────────────────────────────────────────────
@dataclass
class NewsArticle:
    title:       str
    description: str
    source:      str
    published:   datetime
    url:         str
    raw_score:   float = 0.0    # score de sentiment crudo (-1 a +1)
    relevance:   float = 0.0    # qué tan relevante es para el oro (0 a 1)
    is_trump:    bool  = False   # ¿menciona a Trump?
    keywords_hit: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title} {self.description}".lower()

    @property
    def weighted_score(self) -> float:
        """Score ponderado por relevancia y factor Trump."""
        base = self.raw_score * self.relevance
        if self.is_trump:
            base *= 1.25
        return np.clip(base, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Motor de Scoring por Keywords
# ─────────────────────────────────────────────────────────────
class KeywordSentimentScorer:
    """
    Análisis de sentiment basado en keywords gold-específicas.
    No requiere GPU ni modelos pesados — funciona offline.
    """

    def __init__(self):
        # Combinar todos los diccionarios
        self._bullish = GOLD_BULLISH_KEYWORDS
        self._bearish = GOLD_BEARISH_KEYWORDS
        self._trump   = TRUMP_CONTEXT_KEYWORDS

    def score(self, article: NewsArticle) -> NewsArticle:
        """Calcula el score de sentiment para un artículo."""
        text  = article.text
        score = 0.0
        hits  = []
        relevance_hits = 0

        # Detectar mención de Trump (cualquier ocurrencia)
        if "trump" in text:
            article.is_trump = True

        # Verificar keywords Trump primero (mayor especificidad)
        for kw, weight in self._trump.items():
            if kw in text:
                score += weight
                hits.append(f"trump:{kw}")
                relevance_hits += 2
                article.is_trump = True

        # Keywords alcistas
        for kw, weight in self._bullish.items():
            if kw in text:
                score += weight
                hits.append(f"bull:{kw}")
                relevance_hits += 1

        # Keywords bajistas
        for kw, weight in self._bearish.items():
            if kw in text:
                score += weight
                hits.append(f"bear:{kw}")
                relevance_hits += 1

        # Normalizar score a [-1, +1]
        if hits:
            score = score / (len(hits) + 1)   # evitar outliers
            score = float(np.clip(score, -1.0, 1.0))

        # Relevancia: proporción de keywords coincidentes
        total_keywords = len(self._bullish) + len(self._bearish) + len(self._trump)
        relevance = min(1.0, relevance_hits / 5.0)   # normalizado

        # Boost de relevancia si menciona "gold" o "xau"
        if "gold" in text or "xau" in text or "bullion" in text:
            relevance = min(1.0, relevance + 0.30)

        article.raw_score     = score
        article.relevance     = relevance
        article.keywords_hit  = hits[:10]   # top 10 keywords
        return article


# ─────────────────────────────────────────────────────────────
# Fetchers (fuentes de noticias)
# ─────────────────────────────────────────────────────────────
class RSSFetcher:
    """
    Obtiene noticias gratis vía RSS — sin API key.
    Fuentes: Yahoo Finance, Google News, Reuters RSS.
    """

    RSS_FEEDS = [
        # Yahoo Finance — Oro (Futures)
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
        # Google News — Gold price
        "https://news.google.com/rss/search?q=gold+price+market&hl=en-US&gl=US&ceid=US:en",
        # Google News — Trump economy
        "https://news.google.com/rss/search?q=trump+tariff+economy+gold&hl=en-US&gl=US&ceid=US:en",
        # Google News — Federal Reserve
        "https://news.google.com/rss/search?q=federal+reserve+interest+rate+gold&hl=en-US&gl=US&ceid=US:en",
        # Google News — Geopolitics
        "https://news.google.com/rss/search?q=geopolitical+war+conflict+gold&hl=en-US&gl=US&ceid=US:en",
    ]

    def __init__(self, timeout: int = 8):
        self.timeout = timeout

    def fetch(self, max_articles: int = 30) -> List[NewsArticle]:
        articles = []
        for feed_url in self.RSS_FEEDS:
            try:
                arts = self._parse_feed(feed_url)
                articles.extend(arts)
                if len(articles) >= max_articles:
                    break
            except Exception as e:
                logger.debug(f"RSS feed falló ({feed_url[:50]}...): {e}")
        return articles[:max_articles]

    def _parse_feed(self, url: str) -> List[NewsArticle]:
        req  = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=self.timeout)
        xml  = resp.read().decode("utf-8", errors="replace")

        root  = ET.fromstring(xml)
        items = root.findall(".//item")
        arts  = []

        for item in items:
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            link  = (item.findtext("link") or "").strip()
            pub   = item.findtext("pubDate") or ""

            # Limpiar HTML del description
            desc = re.sub(r"<[^>]+>", " ", desc)
            desc = re.sub(r"\s+", " ", desc).strip()

            # Parsear fecha
            try:
                pub_dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M")
            except Exception:
                pub_dt = datetime.now()

            arts.append(NewsArticle(
                title=title, description=desc,
                source="RSS", published=pub_dt, url=link,
            ))
        return arts


class NewsAPIFetcher:
    """
    Obtiene noticias de newsapi.org (hasta 100 req/día gratis).
    Regístrate en https://newsapi.org para obtener API key.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, timeout: int = 8):
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, queries: List[str], max_articles: int = 30) -> List[NewsArticle]:
        if not self.api_key:
            return []

        articles = []
        query = " OR ".join(f'"{q}"' for q in queries[:3])
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": min(max_articles, 100),
            "apiKey":   self.api_key,
        }

        try:
            url  = f"{self.BASE_URL}?{urlencode(params)}"
            req  = Request(url, headers={"User-Agent": "GoldMonitor/1.0"})
            resp = urlopen(req, timeout=self.timeout)
            data = json.loads(resp.read().decode())

            for art in data.get("articles", []):
                try:
                    pub_dt = datetime.fromisoformat(
                        art.get("publishedAt", "")[:19]
                    )
                except Exception:
                    pub_dt = datetime.now()

                articles.append(NewsArticle(
                    title=art.get("title") or "",
                    description=art.get("description") or "",
                    source=art.get("source", {}).get("name", "NewsAPI"),
                    published=pub_dt,
                    url=art.get("url", ""),
                ))

        except Exception as e:
            logger.warning(f"NewsAPI falló: {e}")

        return articles[:max_articles]


class AlphaVantageFetcher:
    """
    Obtiene sentiment de Alpha Vantage News API (500 req/día gratis).
    Regístrate en https://www.alphavantage.co para obtener API key.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: int = 8):
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, max_articles: int = 20) -> List[NewsArticle]:
        if not self.api_key:
            return []

        params = {
            "function":    "NEWS_SENTIMENT",
            "tickers":     "FOREX:XAUUSD",
            "topics":      "economy_macro,financial_markets,gold",
            "sort":        "LATEST",
            "limit":       min(max_articles, 50),
            "apikey":      self.api_key,
        }

        articles = []
        try:
            url  = f"{self.BASE_URL}?{urlencode(params)}"
            req  = Request(url, headers={"User-Agent": "GoldMonitor/1.0"})
            resp = urlopen(req, timeout=self.timeout)
            data = json.loads(resp.read().decode())

            for art in data.get("feed", []):
                try:
                    pub_dt = datetime.strptime(
                        art.get("time_published", "")[:15], "%Y%m%dT%H%M%S"
                    )
                except Exception:
                    pub_dt = datetime.now()

                # Alpha Vantage ya provee un score de sentiment
                av_score = float(art.get("overall_sentiment_score", 0.0))

                # Buscar score específico para XAUUSD
                for ticker_data in art.get("ticker_sentiment", []):
                    if "XAU" in ticker_data.get("ticker", ""):
                        av_score = float(
                            ticker_data.get("ticker_sentiment_score", av_score)
                        )
                        break

                article = NewsArticle(
                    title=art.get("title", ""),
                    description=art.get("summary", ""),
                    source=art.get("source", "AlphaVantage"),
                    published=pub_dt,
                    url=art.get("url", ""),
                    raw_score=av_score,
                    relevance=0.85,  # Alpha Vantage ya filtra por relevancia
                )
                articles.append(article)

        except Exception as e:
            logger.warning(f"Alpha Vantage falló: {e}")

        return articles[:max_articles]


# ─────────────────────────────────────────────────────────────
# Sistema Central de Filtrado
# ─────────────────────────────────────────────────────────────
class GoldNewsSentimentFilter:
    """
    Filtra y ajusta señales de trading según el sentiment de noticias.

    Flujo:
        1. Descarga titulares (RSS gratis + NewsAPI + Alpha Vantage)
        2. Calcula score de sentiment por keywords
        3. Devuelve un SentimentResult que ajusta la confianza del Ensemble

    Uso:
        filter = GoldNewsSentimentFilter(NewsConfig())
        result = filter.analyze()
        adjusted_conf = result.adjust_confidence(ensemble_confidence)
    """

    def __init__(self, config: NewsConfig = None):
        self.cfg     = config or NewsConfig()
        self.scorer  = KeywordSentimentScorer()
        self._cache: Optional["SentimentResult"] = None
        self._cache_time: Optional[datetime] = None

        # Inicializar fetchers
        self._rss_fetcher = RSSFetcher(timeout=self.cfg.request_timeout)
        self._newsapi     = NewsAPIFetcher(
            api_key=self.cfg.newsapi_key,
            timeout=self.cfg.request_timeout,
        )
        self._alphavantage = AlphaVantageFetcher(
            api_key=self.cfg.alphavantage_key,
            timeout=self.cfg.request_timeout,
        )

    def analyze(self, force_refresh: bool = False) -> "SentimentResult":
        """
        Analiza el sentiment actual. Usa caché para no sobrecargar las APIs.
        """
        # Verificar caché
        if not force_refresh and self._cache is not None:
            elapsed = (datetime.now() - self._cache_time).total_seconds() / 60
            if elapsed < self.cfg.cache_minutes:
                logger.info(f"Usando caché de noticias ({elapsed:.0f} min)")
                return self._cache

        articles = self._fetch_all_articles()

        if not articles:
            logger.warning("No se pudieron obtener noticias — retornando resultado neutro")
            return SentimentResult.neutral()

        # Scoring
        scored = [self.scorer.score(a) for a in articles]

        # Filtrar por relevancia mínima
        relevant = [a for a in scored if a.relevance >= self.cfg.min_relevance]

        if not relevant:
            relevant = scored   # si nada es relevante, usar todos

        result = self._build_result(relevant)
        self._cache      = result
        self._cache_time = datetime.now()

        logger.info(
            f"Análisis de noticias: score={result.aggregate_score:+.3f} | "
            f"label={result.sentiment_label} | "
            f"artículos={len(relevant)} | "
            f"noticias_trump={result.trump_news_count}"
        )
        return result

    def _fetch_all_articles(self) -> List[NewsArticle]:
        """Recopila artículos de TODAS las fuentes en paralelo (APIs + RSS siempre)."""
        articles: List[NewsArticle] = []
        source_counts: Dict[str, int] = {}

        # 1. Alpha Vantage (más preciso para finanzas) — si hay API key
        if self.cfg.alphavantage_key:
            try:
                av_arts = self._alphavantage.fetch(max_articles=20)
                articles.extend(av_arts)
                source_counts["AlphaVantage"] = len(av_arts)
                logger.info(f"Alpha Vantage: {len(av_arts)} artículos")
            except Exception as e:
                logger.warning(f"Alpha Vantage falló: {e}")
                source_counts["AlphaVantage"] = 0

        # 2. NewsAPI — si hay API key
        if self.cfg.newsapi_key:
            try:
                na_arts = self._newsapi.fetch(
                    queries=self.cfg.search_queries,
                    max_articles=20,
                )
                articles.extend(na_arts)
                source_counts["NewsAPI"] = len(na_arts)
                logger.info(f"NewsAPI: {len(na_arts)} artículos")
            except Exception as e:
                logger.warning(f"NewsAPI falló: {e}")
                source_counts["NewsAPI"] = 0

        # 3. RSS gratuito — SIEMPRE se obtiene (no solo como fallback)
        #    Esto garantiza cobertura amplia independientemente de las APIs
        try:
            rss_arts = self._rss_fetcher.fetch(max_articles=30)
            articles.extend(rss_arts)
            source_counts["RSS"] = len(rss_arts)
            logger.info(f"RSS: {len(rss_arts)} artículos")
        except Exception as e:
            logger.warning(f"RSS falló: {e}")
            source_counts["RSS"] = 0

        # Log resumen de fuentes
        total_raw = sum(source_counts.values())
        logger.info(f"Total artículos brutos: {total_raw} | Fuentes: {source_counts}")

        # Deduplicar por título (priorizar APIs sobre RSS si hay duplicados)
        seen_titles = set()
        unique = []
        for a in articles:
            key = a.title[:60].lower().strip()
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique.append(a)

        logger.info(f"Artículos únicos después de deduplicar: {len(unique)}")
        return unique[:self.cfg.max_articles]

    def _build_result(self, articles: List[NewsArticle]) -> "SentimentResult":
        """Construye el SentimentResult agregado desde los artículos."""
        scores     = np.array([a.weighted_score for a in articles])
        relevances = np.array([a.relevance for a in articles])

        # Score ponderado por relevancia
        if relevances.sum() > 0:
            aggregate = float(np.average(scores, weights=relevances))
        else:
            aggregate = float(scores.mean())

        aggregate = float(np.clip(aggregate, -1.0, 1.0))

        # Contar noticias Trump
        trump_count    = sum(1 for a in articles if a.is_trump)
        bullish_count  = sum(1 for a in articles if a.weighted_score > 0.1)
        bearish_count  = sum(1 for a in articles if a.weighted_score < -0.1)

        # Top headlines — todos los artículos ordenados por impacto
        top = sorted(articles, key=lambda x: abs(x.weighted_score), reverse=True)

        # Label
        if aggregate >= 0.25:
            label = "Muy Alcista 🟢🟢"
        elif aggregate >= 0.10:
            label = "Alcista 🟢"
        elif aggregate <= -0.25:
            label = "Muy Bajista 🔴🔴"
        elif aggregate <= -0.10:
            label = "Bajista 🔴"
        else:
            label = "Neutral ⚪"

        # Advertencia de alta volatilidad
        high_volatility = (
            trump_count >= 3 or
            abs(aggregate) >= 0.50 or
            any("nuclear" in a.text or "war" in a.text for a in articles[:10])
        )

        return SentimentResult(
            aggregate_score   = aggregate,
            sentiment_label   = label,
            total_articles    = len(articles),
            bullish_count     = bullish_count,
            bearish_count     = bearish_count,
            trump_news_count  = trump_count,
            top_articles      = top,
            all_articles      = articles,
            high_volatility   = high_volatility,
            timestamp         = datetime.now(),
            sources_used      = list({a.source for a in articles}),
        )


# ─────────────────────────────────────────────────────────────
# Resultado del análisis
# ─────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    aggregate_score:  float         # -1.0 a +1.0
    sentiment_label:  str           # Muy Alcista / Alcista / Neutral / Bajista / Muy Bajista
    total_articles:   int
    bullish_count:    int
    bearish_count:    int
    trump_news_count: int
    top_articles:     List[NewsArticle]
    all_articles:     List[NewsArticle]
    high_volatility:  bool          # True = mercado muy nervioso hoy
    timestamp:        datetime
    sources_used:     List[str]

    @classmethod
    def neutral(cls) -> "SentimentResult":
        """Resultado neutro cuando no hay datos."""
        return cls(
            aggregate_score=0.0, sentiment_label="Neutral ⚪",
            total_articles=0, bullish_count=0, bearish_count=0,
            trump_news_count=0, top_articles=[], all_articles=[],
            high_volatility=False, timestamp=datetime.now(),
            sources_used=["sin datos"],
        )

    def adjust_confidence(
        self,
        confidence: float,
        signal: int,
        max_adjustment: float = 0.35,
    ) -> float:
        """
        Ajusta la confianza del Ensemble según el sentiment.

        Reglas:
          - Signal BUY  + sentiment alcista  → boost de confianza
          - Signal BUY  + sentiment bajista  → reducción de confianza
          - Signal SELL + sentiment bajista  → boost de confianza
          - Signal SELL + sentiment alcista  → reducción de confianza
          - Alta volatilidad → reducir confianza siempre (precaución)
        """
        adj = self.aggregate_score * max_adjustment

        if signal == 1:    # BUY — sentiment alcista ayuda
            new_conf = confidence + adj
        elif signal == -1: # SELL — sentiment bajista ayuda
            new_conf = confidence - adj
        else:
            new_conf = confidence

        # Penalización por alta volatilidad (mercado impredecible)
        if self.high_volatility:
            new_conf *= 0.90

        return float(np.clip(new_conf, 0.0, 1.0))

    def should_freeze_trading(self) -> bool:
        """
        True = NO operar ahora (evento extremo inminente).
        Ejemplo: score muy extremo + muchas noticias Trump en 1 hora.
        """
        return (
            abs(self.aggregate_score) >= 0.85 and
            self.trump_news_count >= 5
        )

    def get_signal_modifier(self) -> int:
        """
        Modifica la señal en casos extremos:
         +1 si es muy alcista (forzar BUY aunque ensemble diga neutro)
         -1 si es muy bajista
          0 si es neutro
        """
        if self.aggregate_score >= 0.70:
            return +1
        elif self.aggregate_score <= -0.70:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {
            "aggregate_score":  round(self.aggregate_score, 4),
            "sentiment_label":  self.sentiment_label,
            "total_articles":   self.total_articles,
            "bullish_count":    self.bullish_count,
            "bearish_count":    self.bearish_count,
            "trump_news_count": self.trump_news_count,
            "high_volatility":  self.high_volatility,
            "should_freeze":    self.should_freeze_trading(),
            "signal_modifier":  self.get_signal_modifier(),
            "sources":          self.sources_used,
            "timestamp":        self.timestamp.isoformat(),
        }

    def get_headlines_df(self) -> pd.DataFrame:
        """DataFrame con los titulares principales (incluye URL para links)."""
        if not self.top_articles:
            return pd.DataFrame()
        rows = []
        for a in self.top_articles:
            rows.append({
                "Fecha":     a.published.strftime("%Y-%m-%d %H:%M"),
                "Titular":   a.title[:90] + ("..." if len(a.title) > 90 else ""),
                "Fuente":    a.source,
                "Score":     f"{a.weighted_score:+.2f}",
                "Trump":     "🎯" if a.is_trump else "",
                "Keywords":  ", ".join(a.keywords_hit[:3]),
                "URL":       a.url if a.url else "",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Calendario Económico (eventos de alto impacto)
# ─────────────────────────────────────────────────────────────
class EconomicEventFilter:
    """
    Detecta si hoy hay eventos de alto impacto que ameriten
    congelar operaciones (FOMC, NFP, CPI, etc.)
    """

    # Días típicos del mes para eventos recurrentes (aproximación)
    HIGH_IMPACT_PATTERNS = {
        "FOMC":       {"weeks": [1, 3], "day": 3},  # miércoles semanas 1 y 3
        "NFP":        {"week": 1, "day": 4},         # primer viernes del mes
        "CPI":        {"week": 2, "day": 2},         # segundo miércoles
    }

    def is_high_impact_day(self) -> Tuple[bool, str]:
        """
        Verifica si hoy es un día de evento de alto impacto.
        Retorna (True/False, nombre_del_evento).
        """
        today = datetime.now()
        dow   = today.weekday()  # 0=lunes, 4=viernes

        # FOMC — miércoles de semanas 1 y 3 del mes
        if dow == 2:
            week_of_month = (today.day - 1) // 7 + 1
            if week_of_month in [1, 3]:
                return True, "⚠️ Posible FOMC hoy — alta volatilidad esperada"

        # NFP — primer viernes del mes
        if dow == 4 and today.day <= 7:
            return True, "⚠️ Posible NFP hoy — mercados volátiles"

        # CPI — segundo miércoles del mes (aproximado)
        if dow == 2 and 8 <= today.day <= 14:
            return True, "⚠️ Posible CPI hoy — esperar datos"

        return False, ""


# ─────────────────────────────────────────────────────────────
# Assertions / Unit Tests
# ─────────────────────────────────────────────────────────────
def _run_assertions():
    print("  ← Ejecutando Assertions para NewsFilter ...")

    scorer = KeywordSentimentScorer()

    # 1) Artículo muy alcista (guerra + aranceles Trump)
    art1 = NewsArticle(
        title="Trump announces 200% tariffs on China amid escalating trade war",
        description="Gold surges to record high as safe haven demand spikes",
        source="Test", published=datetime.now(), url="",
    )
    art1 = scorer.score(art1)
    assert art1.raw_score > 0, f"Artículo bélico debe tener score positivo, got {art1.raw_score}"
    assert art1.is_trump, "Debe detectar mención de Trump"
    assert art1.relevance > 0, "Debe tener relevancia > 0"

    # 2) Artículo bajista (acuerdo de paz + subida de tasas)
    art2 = NewsArticle(
        title="US China reach historic trade agreement, Fed signals rate hike",
        description="Dollar surges as risk-on sentiment returns, gold drops",
        source="Test", published=datetime.now(), url="",
    )
    art2 = scorer.score(art2)
    assert art2.raw_score < 0, f"Acuerdo paz + subida tasas debe ser bajista, got {art2.raw_score}"

    # 3) Resultado neutro de fallback
    result_neutral = SentimentResult.neutral()
    assert result_neutral.aggregate_score == 0.0
    assert result_neutral.adjust_confidence(0.70, 1) == 0.70  # sin ajuste

    # 4) Ajuste de confianza BUY con sentiment alcista
    result_bull = SentimentResult(
        aggregate_score=0.60, sentiment_label="Alcista 🟢",
        total_articles=5, bullish_count=4, bearish_count=1,
        trump_news_count=2, top_articles=[], all_articles=[],
        high_volatility=False, timestamp=datetime.now(),
        sources_used=["Test"],
    )
    adj_conf = result_bull.adjust_confidence(0.65, signal=1)
    assert adj_conf > 0.65, "BUY + sentiment alcista debe aumentar confianza"
    assert adj_conf <= 1.0, "Confianza no puede superar 1.0"

    # 5) Ajuste de confianza BUY con sentiment bajista
    result_bear = SentimentResult(
        aggregate_score=-0.50, sentiment_label="Bajista 🔴",
        total_articles=5, bullish_count=1, bearish_count=4,
        trump_news_count=0, top_articles=[], all_articles=[],
        high_volatility=False, timestamp=datetime.now(),
        sources_used=["Test"],
    )
    adj_conf2 = result_bear.adjust_confidence(0.65, signal=1)
    assert adj_conf2 < 0.65, "BUY + sentiment bajista debe reducir confianza"

    # 6) Congelación en eventos extremos
    result_extreme = SentimentResult(
        aggregate_score=0.90, sentiment_label="Muy Alcista 🟢🟢",
        total_articles=10, bullish_count=9, bearish_count=1,
        trump_news_count=6, top_articles=[], all_articles=[],
        high_volatility=True, timestamp=datetime.now(),
        sources_used=["Test"],
    )
    assert result_extreme.should_freeze_trading(), "Evento extremo debe congelar trading"

    # 7) get_headlines_df con artículos
    result_bull.top_articles = [art1, art2]
    df = result_bull.get_headlines_df()
    assert len(df) == 2, "DataFrame debe tener 2 filas"
    assert "Titular" in df.columns

    # 8) Calendario económico
    cal = EconomicEventFilter()
    is_high, msg = cal.is_high_impact_day()
    assert isinstance(is_high, bool)

    # 9) to_dict
    d = result_bull.to_dict()
    assert "aggregate_score" in d
    assert "sentiment_label" in d

    print("  ✅ Todos los Assertions de NewsFilter superados exitosamente!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_assertions()

    # Demo con RSS (no requiere API key)
    print("\n🔍 Probando conexión RSS...")
    cfg    = NewsConfig()
    nf     = GoldNewsSentimentFilter(cfg)
    result = nf.analyze()

    print(f"\n{'='*55}")
    print(f"  📰 Resumen de Sentiment — Gold Price Monitor")
    print(f"{'='*55}")
    print(f"  Score agregado : {result.aggregate_score:+.3f}")
    print(f"  Etiqueta       : {result.sentiment_label}")
    print(f"  Artículos      : {result.total_articles}")
    print(f"  Alcistas       : {result.bullish_count}")
    print(f"  Bajistas       : {result.bearish_count}")
    print(f"  Noticias Trump : {result.trump_news_count}")
    print(f"  Alta volatilidad: {result.high_volatility}")
    print(f"  Fuentes        : {', '.join(result.sources_used)}")
    print(f"\n  Top titulares:")
    for a in result.top_articles[:3]:
        print(f"  [{a.weighted_score:+.2f}] {a.title[:70]}")
    print(f"{'='*55}\n")
