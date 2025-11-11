# app/tools/stock_data_api.py
"""
Données boursières via yfinance.
Commandes:
  - pe <TICKER>              -> renvoie le P/E (TTM si dispo)
  - close <TICKER> [period] [interval] -> dernier cours de clôture
Exemples:
  pe AAPL
  close AAPL 1mo 1d
"""
import re
from typing import Optional
import yfinance as yf
from langchain.tools import Tool

def _sanitize_cmd(s: str) -> str:
    s = re.sub(r'[\"\'“”’]', "", s)     # enlève guillemets
    s = re.sub(r"\s+", " ", s).strip()  # espaces multiples
    return s.rstrip(".:;")              # ponctuation finale

def _safe_pe(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        # Essai 1: info (traillingPE)
        info = tk.info or {}
        pe = info.get("trailingPE", None)
        if pe is not None:
            return float(pe)
        # Essai 2: fundamentals TTM (si dispo dans certaines versions)
        try:
            fin = tk.get_income_stmt(trailing=True)  # yfinance>=0.2.x
            # fallback naïf si besoin: non déterministe -> pas garanti
        except Exception:
            pass
        return None
    except Exception:
        return None

def _cmd_pe(parts):
    if len(parts) < 2:
        return "Usage: pe <TICKER> (ex: pe AAPL)"
    ticker = parts[1].upper()
    pe = _safe_pe(ticker)
    if pe is None:
        return f"P/E indisponible pour {ticker}."
    return f"P/E (TTM) {ticker} ≈ {pe:.2f}"

def _cmd_close(parts):
    if len(parts) < 2:
        return "Usage: close <TICKER> [period] [interval] (ex: close AAPL 1mo 1d)"
    ticker = parts[1].upper()
    period = parts[2] if len(parts) >= 3 else "1mo"
    interval = parts[3] if len(parts) >= 4 else "1d"
    try:
        hist = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if hist is None or hist.empty:
            return f"Aucune donnée pour {ticker} (period={period}, interval={interval})."
        last_close = float(hist["Close"].dropna().iloc[-1])
        return f"Close {ticker} ({period}/{interval}) = {last_close:.2f}"
    except Exception as e:
        return f"Erreur récupération cours pour {ticker}: {e}"

def _stock_api_fn(query: str) -> str:
    q = _sanitize_cmd(query)
    parts = q.replace(",", " ").split()
    if not parts:
        return "Commande vide. Ex: 'pe AAPL' ou 'close AAPL 1mo 1d'"
    cmd = parts[0].lower()
    if cmd == "pe":
        return _cmd_pe(parts)
    if cmd == "close":
        return _cmd_close(parts)
    return f"Commande inconnue: '{cmd}'. Commandes valides: 'pe', 'close'."

get_stock_data = Tool.from_function(
    func=_stock_api_fn,
    name="stock_data_api",
    description="Infos boursières. 'pe <TICKER>' ou 'close <TICKER> [period] [interval]'."
)
