"""
자산배분 포트폴리오 대시보드
- 현금보유 시: 주식25% / 현금25% / 금25% / 국채10년25%
- 주식투자 시: 주식50% / 현금0%  / 금25% / 국채10년25%
- 신호: 끝자리 전략6 (수익확률 >= 60% 끝자리만 투자)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import io
import re
import warnings
import pathlib
warnings.filterwarnings("ignore")

try:
    import pandas_datareader.data as pdr
    _PDR_OK = True
except ImportError:
    _PDR_OK = False

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ═══════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="자산배분 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════
# 다크모드 CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0a0e1a;
}
.main .block-container { padding: 1.2rem 2rem; max-width: 1400px; }
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #9ca3af; }
.stTabs [aria-selected="true"] { color: #f1f5f9 !important; }
[data-testid="stDataFrame"] { background: #111827; }

.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 14px;
    padding: 20px 28px; margin-bottom: 20px;
}
.section-title {
    color: #5b9bd5; font-size: 14px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 8px; margin: 28px 0 16px 0;
}
.invest-card {
    background: linear-gradient(160deg, #0d2b1a 0%, #071a10 100%);
    border: 1.5px solid #16a34a; border-radius: 14px; padding: 20px;
}
.cash-card {
    background: linear-gradient(160deg, #1c1208 0%, #120d05 100%);
    border: 1.5px solid #92400e; border-radius: 14px; padding: 20px;
}
.asset-box {
    background: #111827; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 12px 16px; text-align: center; margin-bottom: 8px;
}
.footer-txt {
    color: #374151; font-size: 11px; text-align: center; margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════
THRESHOLD   = 0.60
DURATION_US = 8.5    # 미국 10년채 Modified Duration 근사

# ═══════════════════════════════════════════
# 데이터 로딩 (1시간 캐시)
# ═══════════════════════════════════════════
# 오프라인 폴백: 최신 백업 CSV 경로 탐색
# ═══════════════════════════════════════════
def _latest_backup_dir():
    """backup/ 폴더에서 가장 최근 날짜 폴더를 반환. 없으면 None."""
    bd = pathlib.Path(__file__).parent / "backup"
    if not bd.exists():
        return None
    dirs = sorted(
        [d for d in bd.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8],
        reverse=True,
    )
    return dirs[0] if dirs else None

def _load_csv_fallback(name: str) -> pd.Series:
    """최신 백업 CSV에서 Close 시리즈를 읽어 반환."""
    latest = _latest_backup_dir()
    if latest is None:
        return pd.Series(dtype=float)
    csv_path = latest / "data" / f"{name}.csv"
    if not csv_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def load_raw():
    # 티커 → (yfinance 티커, 시작일, 백업 CSV 파일명)
    _ticker_map = {
        "kosdaq": ("^KQ11",     "1997-01-01", "KOSDAQ"),
        "krbond": ("114820.KS", "2009-01-01", "KRBOND"),
    }

    def dl(ticker, start, csv_name):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                             multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns or df.empty:
                raise ValueError("no data")
            return df["Close"].dropna(), False
        except Exception:
            return _load_csv_fallback(csv_name), True

    def load_kospi():
        """kospi_history.csv(1980~) + yfinance 최신 데이터 병합"""
        hist_path = pathlib.Path(__file__).parent / "kospi_history.csv"
        try:
            hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            s_hist = hist["Close"].dropna().sort_index()
            # yfinance로 최신 데이터 보완
            last_date = s_hist.index[-1]
            yf_start  = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                yf_df = yf.download("^KS11", start=yf_start, auto_adjust=True,
                                    progress=False, multi_level_index=False)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                if not yf_df.empty and "Close" in yf_df.columns:
                    s_yf = yf_df["Close"].dropna()
                    s_yf.index = pd.to_datetime(s_yf.index).tz_localize(None)
                    combined = pd.concat([s_hist, s_yf])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    return combined, False
            except Exception:
                pass
            return s_hist, False
        except Exception:
            # kospi_history.csv 없으면 yfinance 폴백
            s, offline = dl("^KS11", "1985-01-01", "KOSPI")
            return s, offline

    def load_from_csv(csv_filename, yf_ticker, start_date="1970-01-01", fallback_csv=""):
        """로컬 CSV(1970~) + yfinance 최신 보완 공통 헬퍼"""
        CSV_PATH = pathlib.Path(__file__).parent / csv_filename
        try:
            hist = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            s_hist = hist["Close"].dropna().sort_index()
            s_hist = s_hist[s_hist.index >= start_date]
            last_date = s_hist.index[-1]
            yf_start  = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                yf_df = yf.download(yf_ticker, start=yf_start, auto_adjust=True,
                                    progress=False, multi_level_index=False)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                if not yf_df.empty and "Close" in yf_df.columns:
                    s_yf = yf_df["Close"].dropna()
                    s_yf.index = pd.to_datetime(s_yf.index).tz_localize(None)
                    combined = pd.concat([s_hist, s_yf])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    return combined, False
            except Exception:
                pass
            return s_hist, False
        except Exception:
            return dl(yf_ticker, "1985-01-01", fallback_csv or csv_filename.upper())

    def load_us30y():
        """FRED DGS10(1962~) + yfinance ^TNX 최신 보완 (10년물)"""
        if _PDR_OK:
            try:
                s = pdr.DataReader("DGS10", "fred",
                                   start=date(1962, 1, 1))["DGS10"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s.sort_index()
                # yfinance로 FRED 지연분 보완
                last_date = s.index[-1]
                yf_start  = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    yf_df = yf.download("^TNX", start=yf_start, auto_adjust=True,
                                        progress=False, multi_level_index=False)
                    if isinstance(yf_df.columns, pd.MultiIndex):
                        yf_df.columns = yf_df.columns.get_level_values(0)
                    if not yf_df.empty and "Close" in yf_df.columns:
                        s_yf = yf_df["Close"].dropna()
                        s_yf.index = pd.to_datetime(s_yf.index).tz_localize(None)
                        combined = pd.concat([s, s_yf])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        return combined, False
                except Exception:
                    pass
                return s, False
            except Exception:
                pass
        # FRED 실패 시 yfinance ^TNX 폴백 (10년물, 1985~)
        return dl("^TNX", "1985-01-01", "US30Y")

    results, offline_flags = {}, {}
    for key, (ticker, start, csv_name) in _ticker_map.items():
        results[key], offline_flags[key] = dl(ticker, start, csv_name)
    results["kospi"],  offline_flags["kospi"]  = load_kospi()
    results["sp500"],  offline_flags["sp500"]  = load_from_csv("sp500_history.csv",  "^GSPC",  "1970-01-01", "SP500")
    results["gold"],   offline_flags["gold"]   = load_from_csv("gold_history.csv",   "GC=F",   "1970-01-01", "GOLD")
    results["nasdaq"], offline_flags["nasdaq"] = load_from_csv("nasdaq_history.csv", "^IXIC",  "1970-01-01", "NASDAQ")
    results["dow"],    offline_flags["dow"]    = load_from_csv("dow_history.csv",    "^DJI",   "1970-01-01", "DOW")
    results["csi300"], offline_flags["csi300"] = load_from_csv("csi300_history.csv", "000300.SS", "2005-01-01", "CSI300")
    results["us30y"],  offline_flags["us30y"]  = load_us30y()

    # 오프라인 여부를 session_state에 기록 (UI 배너용)
    is_offline = any(offline_flags.values())
    latest     = _latest_backup_dir()
    st.session_state["_offline_mode"]   = is_offline
    st.session_state["_offline_date"]   = latest.name if latest else "알 수 없음"
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def load_vix_history():
    try:
        df = yf.download("^VIX", start="1990-01-01", auto_adjust=True,
                         progress=False, multi_level_index=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_vix_now():
    try:
        df = yf.download("^VIX", period="5d", auto_adjust=True, progress=False,
                         multi_level_index=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return round(float(df["Close"].dropna().iloc[-1]), 1)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fear_greed():
    if not _REQUESTS_OK:
        return None
    try:
        r = _requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        d = r.json()
        return round(float(d["fear_and_greed"]["score"]), 1)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_multpl(url):
    if not _REQUESTS_OK:
        return None
    try:
        r = _requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = re.search(r'id=["\']current-value["\'][^>]*>\s*([\d.]+)', r.text)
        if m:
            return float(m.group(1))
        m2 = re.search(r'<td[^>]*class=["\'][^"\']*current[^"\']*["\'][^>]*>([\d.]+)', r.text)
        if m2:
            return float(m2.group(1))
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_kodex_etfs():
    """KODEX/ACE ETF 일별 종가 로드 (2021~)"""
    TICKERS = {
        "sp500":  "379800.KS",   # KODEX 미국S&P500
        "nasdaq": "379810.KS",   # KODEX 미국나스닥100
        "kospi":  "069500.KS",   # KODEX 200
        "kosdaq": "229200.KS",   # KODEX 코스닥150
        "csi300": "168580.KS",   # KODEX 차이나CSI300
        "gold":   "411060.KS",   # ACE KRX금현물
        "krbond": "114820.KS",   # KODEX 국고채3년
        "us10y":  "304660.KS",   # KODEX 미국10년국채선물
    }
    START = "2021-01-01"
    out = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                out[key] = pd.Series(dtype=float)
            else:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                out[key] = s
        except Exception:
            out[key] = pd.Series(dtype=float)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_ohlc():
    result = {}

    # CSV(1970~) + yfinance 최신 보완 공통 함수
    def from_csv(csv_filename, yf_ticker, fallback_start="1985-01-01"):
        CSV_PATH = pathlib.Path(__file__).parent / csv_filename
        try:
            hist = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            s = hist["Close"].dropna().sort_index()
            s = s[s.index >= "1970-01-01"]
            last = s.index[-1]
            try:
                yf_df = yf.download(yf_ticker,
                                    start=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                    auto_adjust=True, progress=False, multi_level_index=False)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                if not yf_df.empty and "Close" in yf_df.columns:
                    s_yf = yf_df["Close"].dropna()
                    s_yf.index = pd.to_datetime(s_yf.index).tz_localize(None)
                    s = pd.concat([s, s_yf])
                    s = s[~s.index.duplicated(keep="last")].sort_index()
            except Exception:
                pass
            return s
        except Exception:
            try:
                yf_df = yf.download(yf_ticker, start=fallback_start, auto_adjust=True,
                                    progress=False, multi_level_index=False)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                s = yf_df["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s
            except Exception:
                return pd.Series(dtype=float)

    # CSV 기반 1970~ 로드
    csv_sources = {
        "sp500":  ("sp500_history.csv",  "^GSPC"),
        "nasdaq": ("nasdaq_history.csv", "^IXIC"),
        "dow":    ("dow_history.csv",    "^DJI"),
        "csi300": ("csi300_history.csv", "000300.SS"),
    }
    for key, (csv_file, ticker) in csv_sources.items():
        s = from_csv(csv_file, ticker)
        df = s.to_frame("Close").dropna()
        df["prev_close"] = df["Close"].shift(1)
        df["daily_ret"]  = df["Close"].pct_change()
        result[key] = df.dropna()

    # yfinance 전용 (KOSPI/KOSDAQ)
    yf_sources = {
        "kospi":  ("^KS11", "1990-01-01", "KOSPI"),
        "kosdaq": ("^KQ11", "1997-01-01", "KOSDAQ"),
    }
    for key, (ticker, start, csv_name) in yf_sources.items():
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                             multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns or df.empty:
                raise ValueError("no data")
        except Exception:
            s = _load_csv_fallback(csv_name)
            df = s.to_frame("Close") if not s.empty else pd.DataFrame(columns=["Close"])
        if "Close" not in df.columns:
            result[key] = pd.DataFrame(columns=["Close", "prev_close", "daily_ret"])
            continue
        df = df[["Close"]].dropna()
        df["prev_close"] = df["Close"].shift(1)
        df["daily_ret"]  = df["Close"].pct_change()
        result[key] = df.dropna()

    return result


# ═══════════════════════════════════════════
# 전략 계산 (1시간 캐시)
# ═══════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def compute_strategy():
    raw = load_raw()

    # ── 월별 수익률 ──
    def mr(price):
        return price.resample("ME").last().pct_change().dropna()

    def mr_bond(yield_pct):
        m = yield_pct.resample("ME").last()
        return (-DURATION_US * m.diff() / 100).dropna()

    monthly = {
        "sp500":  mr(raw["sp500"]),
        "nasdaq": mr(raw["nasdaq"]),
        "kospi":  mr(raw["kospi"]),
        "dow":    mr(raw["dow"]),
        "kosdaq": mr(raw["kosdaq"]),
        "csi300": mr(raw["csi300"]),
        "gold":   mr(raw["gold"]),
        "us30y":  mr_bond(raw["us30y"]),
        "krbond": mr(raw["krbond"]),
    }

    # ── 반기 수익률 ──
    def halfyear(series, s, e):
        rows = []
        for y in sorted(series.index.year.unique()):
            if s > e:
                m1 = (series.index.year == y)   & (series.index.month >= s)
                m2 = (series.index.year == y+1) & (series.index.month <= e)
                rets = series[m1 | m2]
            else:
                mask = (series.index.year == y) & \
                       (series.index.month >= s) & (series.index.month <= e)
                rets = series[mask]
            if len(rets) < 4:
                continue
            rows.append({"year": y, "ret": (1 + rets).prod() - 1})
        if not rows:
            return pd.DataFrame(columns=["ret"])
        return pd.DataFrame(rows).set_index("year")

    def seasonal(key):
        return halfyear(monthly[key], 11, 4), halfyear(monthly[key], 5, 10)

    seas = {k: seasonal(k) for k in monthly}

    # ── 끝자리 전략 ──
    def digit_strategy(na, mo):
        def prob_map(df):
            df = df.copy()
            df["d"] = df.index % 10
            return {
                d: float((sub := df[df["d"] == d]["ret"]).pipe(
                    lambda s: (s > 0).mean() if len(s) > 0 else 0))
                for d in range(10)
            }

        pna, pmo = prob_map(na), prob_map(mo)
        return (
            set(d for d, p in pna.items() if p >= THRESHOLD),
            set(d for d, p in pmo.items() if p >= THRESHOLD),
            pna, pmo,
        )

    strategies = {}
    for key in ["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"]:
        na, mo = seas[key]
        inv_na, inv_mo, pna, pmo = digit_strategy(na, mo)
        strategies[key] = {
            "na": na, "mo": mo,
            "inv_na": inv_na, "inv_mo": inv_mo,
            "pna": pna, "pmo": pmo,
        }

    # ── 포트폴리오 시뮬레이션 ──
    def simulate(stock_key):
        info = strategies[stock_key]
        bond_key = "krbond" if stock_key in ("kospi", "kosdaq") else "us30y"
        # csi300: 중국 채권 데이터 없으므로 미국채10년 대용 (bond_key = "us30y" 유지)
        bond_na, bond_mo = seas[bond_key]
        gold_na, gold_mo = seas["gold"]
        na, mo = info["na"], info["mo"]
        inv_na, inv_mo = info["inv_na"], info["inv_mo"]

        records = []
        v_s6 = v_bhmin = v_bhmax = v_stk = 100.0

        for y in sorted(set(na.index) | set(mo.index)):
            digit = y % 10
            for season, stk_df, bnd_df, inv_set in [
                ("Nov-Apr", na, bond_na, inv_na),
                ("May-Oct", mo, bond_mo, inv_mo),
            ]:
                if y not in stk_df.index:
                    continue
                g_df = gold_na if season == "Nov-Apr" else gold_mo
                if y not in g_df.index or y not in bnd_df.index:
                    continue

                rs = float(stk_df.loc[y, "ret"])
                rg = float(g_df.loc[y, "ret"])
                rb = float(bnd_df.loc[y, "ret"])
                inv = digit in inv_set

                r_s6    = (0.50*rs + 0.25*rg + 0.25*rb) if inv \
                          else (0.25*rs + 0.25*rg + 0.25*rb)
                r_bhmin = 0.25*rs + 0.25*rg + 0.25*rb
                r_bhmax = 0.50*rs + 0.25*rg + 0.25*rb
                r_stk   = rs  # 주식단독: 항상 100% 주식 BH

                v_s6    *= (1 + r_s6)
                v_bhmin *= (1 + r_bhmin)
                v_bhmax *= (1 + r_bhmax)
                v_stk   *= (1 + r_stk)

                records.append({
                    "연도": y, "시즌": season, "끝자리": digit, "투자": inv,
                    "주식수익": rs, "금수익": rg, "채권수익": rb, "포트수익": r_s6,
                    "전략6":   round(v_s6, 2),
                    "BH_min":  round(v_bhmin, 2),
                    "BH_max":  round(v_bhmax, 2),
                    "주식단독": round(v_stk, 2),
                })

        _cols = ["연도","시즌","끝자리","투자","주식수익","금수익","채권수익",
                 "포트수익","전략6","BH_min","BH_max","주식단독"]
        if not records:
            return pd.DataFrame(columns=_cols)
        return pd.DataFrame(records)

    sims = {k: simulate(k) for k in ["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"]}

    # ── 전략8: SP500 + KOSPI + CSI300 3지수 동적 비중 ──
    def simulate_s8():
        sp_info = strategies["sp500"]
        ko_info = strategies["kospi"]
        cs_info = strategies["csi300"]

        sp_na, sp_mo = seas["sp500"]
        nq_na, nq_mo = seas["nasdaq"]   # 나스닥 (SP500 투자시즌 연동)
        ko_na, ko_mo = seas["kospi"]
        kq_na, kq_mo = seas["kosdaq"]   # 코스닥 (KOSPI 투자시즌 연동)
        cs_na, cs_mo = seas["csi300"]
        kr_na, kr_mo = seas["krbond"]
        us_na, us_mo = seas["us30y"]
        g_na,  g_mo  = seas["gold"]

        records = []
        v_s8 = v_bhmax = v_bhmin = 100.0

        for y in sorted(set(sp_na.index) | set(sp_mo.index)):
            digit = y % 10
            for season, sp_df, nq_df, ko_df, kq_df, cs_df, kr_df, us_df, gdf, \
                    sp_inv_set, ko_inv_set, cs_inv_set in [
                ("Nov-Apr", sp_na, nq_na, ko_na, kq_na, cs_na, kr_na, us_na, g_na,
                 sp_info["inv_na"], ko_info["inv_na"], cs_info["inv_na"]),
                ("May-Oct", sp_mo, nq_mo, ko_mo, kq_mo, cs_mo, kr_mo, us_mo, g_mo,
                 sp_info["inv_mo"], ko_info["inv_mo"], cs_info["inv_mo"]),
            ]:
                if y not in sp_df.index: continue
                if y not in gdf.index:   continue
                if y not in us_df.index: continue

                r_sp = float(sp_df.loc[y, "ret"])
                r_nq = float(nq_df.loc[y, "ret"]) if y in nq_df.index else r_sp
                r_ko = float(ko_df.loc[y, "ret"]) if y in ko_df.index else r_sp
                r_kq = float(kq_df.loc[y, "ret"]) if y in kq_df.index else r_ko
                r_cs = float(cs_df.loc[y, "ret"]) if y in cs_df.index else r_sp
                r_g  = float(gdf.loc[y, "ret"])
                r_kr = float(kr_df.loc[y, "ret"]) if y in kr_df.index else 0.0
                r_us = float(us_df.loc[y, "ret"])

                sp_inv = digit in sp_inv_set
                ko_inv = digit in ko_inv_set
                cs_inv = digit in cs_inv_set
                n_inv  = sum([sp_inv, ko_inv, cs_inv])

                # SP500 투자시즌: SP500 10% + 나스닥 10% / 비투자: SP500 10%만
                w_sp = 0.10
                w_nq = 0.10 if sp_inv else 0.0
                # KOSPI 투자시즌: KOSPI 10% + 코스닥 10% / 비투자: KOSPI 10%만
                w_ko = 0.10
                w_kq = 0.10 if ko_inv else 0.0
                # CSI300: 투자시즌 20% / 비투자 10%
                w_cs = 0.20 if cs_inv else 0.10

                # 금20% + 한국채10% + 미국채10% 고정, 현금 → KODEX 국고채3년(한국채) 운용
                w_cash  = round(1.0 - w_sp - w_nq - w_ko - w_kq - w_cs - 0.20 - 0.10 - 0.10, 10)
                r_s8    = (w_sp*r_sp + w_nq*r_nq + w_ko*r_ko + w_kq*r_kq
                           + w_cs*r_cs + 0.20*r_g + 0.10*r_kr + 0.10*r_us
                           + w_cash*r_kr)   # 현금 → 국고채3년 운용
                r_bhmax = 0.50*r_sp + 0.25*r_g + 0.25*r_us
                r_bhmin = 0.25*r_sp + 0.25*r_g + 0.25*r_us

                v_s8    *= (1 + r_s8)
                v_bhmax *= (1 + r_bhmax)
                v_bhmin *= (1 + r_bhmin)

                total_stk = w_sp + w_nq + w_ko + w_kq + w_cs
                records.append({
                    "연도": y, "시즌": season, "끝자리": digit,
                    "SP투자": "✅" if sp_inv else "💤",
                    "KO투자": "✅" if ko_inv else "💤",
                    "CS투자": "✅" if cs_inv else "💤",
                    "투자지수수": n_inv,
                    "주식비중(%)": round(total_stk*100),
                    "SP500(%)":  round(r_sp*100, 2),
                    "나스닥(%)": round(r_nq*100, 2),
                    "KOSPI(%)":  round(r_ko*100, 2),
                    "코스닥(%)": round(r_kq*100, 2),
                    "CSI300(%)": round(r_cs*100, 2),
                    "금(%)":    round(r_g*100, 2),
                    "한국채(%)": round(r_kr*100, 2),
                    "미국채(%)": round(r_us*100, 2),
                    "전략8":  round(v_s8, 2),
                    "BH_max": round(v_bhmax, 2),
                    "BH_min": round(v_bhmin, 2),
                })

        return pd.DataFrame(records) if records else pd.DataFrame()

    sims["s8"] = simulate_s8()
    return raw, monthly, seas, strategies, sims


# ═══════════════════════════════════════════
# 현재 시즌 정보
# ═══════════════════════════════════════════
def current_season_info():
    today = date.today()
    m, y = today.month, today.year
    if m in [11, 12]:
        season, sig_year, start = "Nov-Apr", y,   date(y, 11, 1)
    elif m in [1, 2, 3, 4]:
        season, sig_year, start = "Nov-Apr", y-1, date(y-1, 11, 1)
    else:
        season, sig_year, start = "May-Oct", y,   date(y, 5, 1)
    return season, sig_year, sig_year % 10, start


def season_return(price_series, start_date):
    s = pd.Timestamp(start_date)
    sub = price_series[price_series.index >= s]
    if len(sub) < 2:
        return None
    return float(sub.iloc[-1] / sub.iloc[0] - 1)


def season_return_bond(yield_series, start_date):
    s = pd.Timestamp(start_date)
    sub = yield_series[yield_series.index >= s]
    if len(sub) < 2:
        return None
    daily_ret = -DURATION_US * sub.diff().dropna() / 100
    return float((1 + daily_ret).prod() - 1)


# ═══════════════════════════════════════════
# 포맷 헬퍼
# ═══════════════════════════════════════════
def fp(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:+.{d}f}%"

def clr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#9ca3af"
    return "#34d399" if v >= 0 else "#f87171"


# ═══════════════════════════════════════════
# 하락 사유 매핑
# ═══════════════════════════════════════════
def get_crash_reason(key, dt):
    y, m = dt.year, dt.month
    ds = dt.strftime("%Y-%m-%d")

    specific = {
        # ── 블랙먼데이 ──
        ("sp500",  "1987-10-19"): "🏦 금융위기 | 블랙먼데이 — 프로그램 매매 폭주로 역대 최대 단일일 폭락",
        ("nasdaq", "1987-10-19"): "🏦 금융위기 | 블랙먼데이 — 프로그램 매매 폭주로 역대 최대 단일일 폭락",
        ("kospi",  "1987-10-20"): "🏦 금융위기 | 블랙먼데이 글로벌 여파",
        # ── 9/11 ──
        ("sp500",  "2001-09-17"): "⚔️ 지정학 | 9/11 테러 — 미국 시장 4일 휴장 후 재개, 항공·보험주 폭락",
        ("nasdaq", "2001-09-17"): "⚔️ 지정학 | 9/11 테러 — 미국 시장 4일 휴장 후 재개",
        ("kospi",  "2001-09-12"): "⚔️ 지정학 | 9/11 테러 글로벌 여파",
        # ── 2008 금융위기 ──
        ("sp500",  "2008-09-29"): "🏦 금융위기 | 리먼 파산 후폭풍 — 하원 TARP 구제금융 법안 부결",
        ("sp500",  "2008-10-15"): "🏦 금융위기 | 글로벌 신용경색 — 리먼 파산 여파 전세계 확산",
        ("nasdaq", "2008-10-15"): "🏦 금융위기 | 글로벌 신용경색 — 리먼 파산 여파 전세계 확산",
        ("kospi",  "2008-10-24"): "🏦 금융위기 | 글로벌 금융위기 — 원/달러 1,500원 돌파, 외국인 대규모 이탈",
        # ── COVID ──
        ("sp500",  "2020-03-16"): "🦠 팬데믹 | COVID-19 — 미국 전국 비상사태 선포, 이동 제한령",
        ("nasdaq", "2020-03-16"): "🦠 팬데믹 | COVID-19 — 미국 전국 비상사태 선포, 이동 제한령",
        ("sp500",  "2020-03-12"): "🦠 팬데믹 | COVID-19 — 트럼프 유럽 입국 금지 발표, 공포 극대화",
        ("nasdaq", "2020-03-12"): "🦠 팬데믹 | COVID-19 — 트럼프 유럽 입국 금지 발표, 공포 극대화",
        ("kospi",  "2020-03-19"): "🦠 팬데믹 | COVID-19 — 서킷브레이커 발동, 외국인 역대급 순매도",
        # ── IMF ──
        ("kospi",  "1997-11-24"): "💱 외환위기 | IMF 외환위기 — 한국 구제금융 신청, 원화 폭락",
        ("kospi",  "1997-12-12"): "💱 외환위기 | IMF 외환위기 — 원/달러 2,000원 육박, 기업 연쇄 부도",
        # ── 닷컴 ──
        ("nasdaq", "2000-04-14"): "📉 버블붕괴 | 닷컴 버블 — 나스닥 역사상 최대 주간 낙폭 (-25%)",
        ("nasdaq", "2000-04-03"): "📉 버블붕괴 | 닷컴 버블 — 마이크로소프트 반독점 판결 충격",
        # ── 인플레이션 CPI ──
        ("sp500",  "2022-09-13"): "📊 정책/긴축 | 예상 상회한 CPI 발표 — 연준 자이언트스텝 공포",
        ("nasdaq", "2022-09-13"): "📊 정책/긴축 | 예상 상회한 CPI 발표 — 연준 자이언트스텝 공포",
        # ── 걸프전 ──
        ("sp500",  "1990-08-06"): "⚔️ 지정학 | 걸프전 — 이라크 쿠웨이트 침공, 유가 급등",
        ("nasdaq", "1990-08-06"): "⚔️ 지정학 | 걸프전 — 이라크 쿠웨이트 침공, 유가 급등",
        # ── 러-우 전쟁 ──
        ("sp500",  "2022-02-24"): "⚔️ 지정학 | 러시아-우크라이나 전쟁 — 러시아 전면 침공 개시",
        ("nasdaq", "2022-02-24"): "⚔️ 지정학 | 러시아-우크라이나 전쟁 — 러시아 전면 침공 개시",
        ("kospi",  "2022-02-24"): "⚔️ 지정학 | 러시아-우크라이나 전쟁 — 러시아 전면 침공 개시",
    }

    k = (key, ds)
    if k in specific:
        return specific[k]

    # ── 기간별 분류 ──
    if y == 1987 and m in [10, 11]:
        return "🏦 금융위기 | 블랙먼데이 여파 — 프로그램 매매 충격 지속"
    if y == 1989 and m == 10:
        return "🏦 금융위기 | 블랙 프라이데이 — 차입 매수(LBO) 붕괴 우려"
    if y == 1990 and m in [8, 9, 10]:
        return "⚔️ 지정학 | 걸프전 — 이라크·쿠웨이트 전쟁, 유가 급등·경기침체 우려"
    if key in ("kospi", "kosdaq") and ((y == 1997 and m >= 7) or (y == 1998 and m <= 8)):
        return "💱 외환위기 | IMF 외환위기 — 아시아 외환위기, 한국 구제금융"
    if y in [1997, 1998] and m in [7, 8, 9, 10, 11] and key not in ("kospi", "kosdaq"):
        return "💱 외환위기 | 아시아 외환위기 — 태국 바트화 붕괴, 신흥국 전이"
    if y == 1998 and m in [8, 9]:
        return "🏦 금융위기 | 러시아 국채 디폴트 / LTCM 헤지펀드 붕괴"
    if y == 2000 and key in ["sp500", "nasdaq"]:
        return "📉 버블붕괴 | 닷컴 버블 붕괴 — IT 기업 실적 쇼크, 과열 밸류에이션 붕괴"
    if y == 2001 and m == 9:
        return "⚔️ 지정학 | 9/11 테러 여파 — 항공·여행·보험 섹터 급락"
    if y == 2001 and key in ["sp500", "nasdaq"]:
        return "📉 버블붕괴 | 닷컴 버블 / 경기침체 — IT 투자 급감, 기업 실적 악화"
    if y == 2002 and key in ["sp500", "nasdaq"]:
        return "📉 버블붕괴 | 닷컴 버블 / 회계 스캔들 — 엔론·월드컴 분식회계 충격"
    if y == 2003 and m <= 3:
        return "⚔️ 지정학 | 이라크 전쟁 — 미국 이라크 침공, 지정학 불안"
    if y == 2007:
        return "🏦 금융위기 | 서브프라임 모기지 위기 시작 — 주택담보대출 부실 수면 위로"
    if y == 2008:
        return "🏦 금융위기 | 글로벌 금융위기 — 리먼 브라더스 파산, 금융 시스템 붕괴 위기"
    if y == 2009 and m <= 3:
        return "🏦 금융위기 | 글로벌 금융위기 여파 — 실물경제 침체, 실업률 급등"
    if y == 2010 and m == 5:
        return "⚡ 기술충격 | 플래시 크래시 — 알고리즘 매매 오작동으로 순간 -9% 폭락"
    if y == 2011 and m in [7, 8, 9]:
        return "🏦 금융위기 | 미국 신용등급 강등 (S&P) / 유럽 재정위기 — 그리스·이탈리아 국채 위기"
    if y == 2014 and m in [9, 10]:
        return "🌏 글로벌경기 | 에볼라 공포 / 글로벌 성장 둔화 우려"
    if y == 2015 and m in [8, 9]:
        return "🌏 글로벌경기 | 중국 위안화 평가절하 — 중국 경기 급랭 우려, 신흥국 자금 이탈"
    if y == 2016 and m == 6:
        return "⚔️ 지정학 | 브렉시트 — 영국 EU 탈퇴 국민투표 가결 충격"
    if y == 2018 and m == 2:
        return "⚡ 기술충격 | 볼마게돈 — VIX 역매수 ETF 붕괴, 변동성 폭발"
    if y == 2018 and m in [10, 11, 12]:
        return "📊 정책/긴축 | 미중 무역전쟁 / 연준 금리 인상 — 글로벌 경기 둔화 우려"
    if y == 2019 and m == 8:
        return "⚔️ 지정학 | 미중 무역전쟁 격화 — 추가 관세 부과, 위안화 7위안 돌파"
    if y == 2020 and m in [2, 3]:
        return "🦠 팬데믹 | COVID-19 — 전세계 봉쇄령, 경제 활동 전면 중단"
    if y == 2022 and m == 2:
        return "⚔️ 지정학 | 러시아-우크라이나 전쟁 — 에너지·식량 공급망 충격"
    if y == 2022:
        return "📊 정책/긴축 | 인플레이션 / 연준 자이언트스텝 — 40년 만의 최고 물가, 급격한 금리 인상"
    if y == 2023 and m == 3:
        return "🏦 금융위기 | 미국 지역은행 위기 — SVB·시그니처뱅크 파산, 뱅크런 공포"
    if y == 2023 and m == 10:
        return "⚔️ 지정학 | 이스라엘-하마스 전쟁 — 중동 확전 우려, 유가 급등"
    if y == 2024 and m == 8:
        return "🏦 금융위기 | 엔 캐리 트레이드 청산 — 일본 금리 인상 쇼크, 글로벌 레버리지 축소"
    if y == 2025:
        return "⚔️ 지정학 | 미중 관세 전쟁 — 트럼프 상호관세 부과, 글로벌 공급망 재편 충격"

    return "🌏 글로벌경기 | 글로벌 경기 불안 / 시장 조정"


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════
def main():
    # ── 데이터 로딩 ──
    with st.spinner("📡 데이터 로딩 중... (최초 1회 약 10~20초)"):
        raw, monthly, seas, strategies, sims = compute_strategy()
        ohlc = load_daily_ohlc()

    today         = date.today()
    season, sig_year, digit, season_start = current_season_info()
    season_kr     = "11월~4월 시즌" if season == "Nov-Apr" else "5월~10월 시즌"

    # ── 현재 신호 상태 (전략7) ──
    def _cur_sig_state(key):
        """현재 전략7 매도 신호 상태: (sell_active, current_count)"""
        try:
            px = raw[key].copy()
            px.index = pd.to_datetime(px.index).tz_localize(None)
            dd_s  = (px - px.cummax()) / px.cummax() * 100
            dlt   = px.diff()
            gain  = dlt.clip(lower=0).rolling(14).mean()
            loss  = (-dlt.clip(upper=0)).rolling(14).mean()
            rsi_s = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            ma200 = px.rolling(200).mean()
            ma_g  = (px - ma200) / ma200 * 100
            ma20  = px.rolling(20).mean(); std20 = px.rolling(20).std()
            bb_rng = (ma20 + 2*std20) - (ma20 - 2*std20)
            bb_p  = (px - (ma20 - 2*std20)) / bb_rng.replace(0, np.nan) * 100
            if key in ("kospi", "kosdaq", "csi300"):
                s_vol = (px.pct_change().rolling(20).std() * (252**0.5) * 100 >= 35).astype(int)
            else:
                vix_h = load_vix_history()
                vix_h.index = pd.to_datetime(vix_h.index).tz_localize(None)
                s_vol = (vix_h.reindex(px.index, method="ffill").fillna(20) >= 40).astype(int)
            sig = ((dd_s<=-30).astype(int)+(rsi_s<=30).astype(int)+
                   (ma_g<=-15).astype(int)+(bb_p<=0).astype(int)+s_vol).fillna(0)
            # 마지막 sell/normal 이벤트 비교로 현재 상태 결정
            s_ge2 = sig[sig >= 2]; s_eq0 = sig[sig == 0]
            last_sell = s_ge2.index[-1] if not s_ge2.empty else pd.Timestamp.min
            last_norm = s_eq0.index[-1] if not s_eq0.empty else pd.Timestamp.min
            return bool(last_sell > last_norm), int(sig.iloc[-1])
        except Exception:
            return False, 0

    # ── 현재 시즌 수익률 ──
    cur = {}
    for key in ["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"]:
        rs = season_return(raw[key],   season_start)
        rg = season_return(raw["gold"], season_start)
        rb = season_return(raw["krbond"], season_start) if key in ("kospi", "kosdaq") \
             else season_return_bond(raw["us30y"], season_start)
        inv  = digit in (strategies[key]["inv_na"] if season == "Nov-Apr"
                         else strategies[key]["inv_mo"])
        sell, sig_cnt = _cur_sig_state(key)
        # 전략7 주식 비중
        w7 = 0.0 if sell else (0.50 if inv else 0.25)
        rp = (w7*rs + 0.25*rg + 0.25*rb) if all(v is not None for v in [rs, rg, rb]) else None
        cur[key] = {"stock": rs, "gold": rg, "bond": rb, "port": rp,
                    "invest": inv, "sell": sell, "sig_cnt": sig_cnt, "w7": w7}

    meta = {
        "sp500":  {"name": "S&P500", "emoji": "🇺🇸", "bond": "미국채10년", "color": "#5b9bd5"},
        "nasdaq": {"name": "NASDAQ", "emoji": "💻",  "bond": "미국채10년", "color": "#ef4444"},
        "kospi":  {"name": "KOSPI",  "emoji": "🇰🇷", "bond": "한국채30년", "color": "#22c55e"},
        "dow":    {"name": "DOW",    "emoji": "🏛️",  "bond": "미국채10년", "color": "#f59e0b"},
        "kosdaq": {"name": "KOSDAQ", "emoji": "📱",  "bond": "한국채30년", "color": "#06b6d4"},
        "csi300": {"name": "CSI300", "emoji": "🇨🇳", "bond": "미국채10년", "color": "#f97316"},
    }

    # ════════════════════════════════════════
    # 최상위 탭
    # ════════════════════════════════════════
    main_tab0, main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7, main_tab8 = st.tabs(["💡 실사용 전략", "📊 자산배분", "📉 역대 폭락일", "🔍 폭락 후 전략", "📈 시장 사이클", "📡 저점 레이더", "📅 연간 수익률", "⚡ 급락 패턴", "📝 메모장"])

    # ════════════════════════════════════════
    # TAB 1: 자산배분
    # ════════════════════════════════════════
    with main_tab1:

        # ── 1. 헤더 배너 ──
        st.markdown(f"""
        <div class="header-banner">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
              <div style="color:#5b9bd5;font-size:12px;font-weight:700;letter-spacing:2px;">
                ASSET ALLOCATION DASHBOARD
              </div>
              <div style="color:#f1f5f9;font-size:26px;font-weight:800;margin-top:4px;">
                📊 자산배분 포트폴리오
              </div>
            </div>
            <div style="text-align:right;">
              <div style="color:#fbbf24;font-size:20px;font-weight:700;">{season_kr}</div>
              <div style="color:#94a3b8;font-size:13px;margin-top:4px;">
                {sig_year}년 &nbsp;|&nbsp; 끝자리
                <span style="color:#fbbf24;font-weight:800;font-size:18px;"> {digit} </span>
                &nbsp;|&nbsp; 시즌 시작 {season_start.strftime('%Y.%m.%d')}
              </div>
              <div style="color:#4b5563;font-size:11px;margin-top:2px;">
                업데이트: {today.strftime('%Y년 %m월 %d일')}
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 오프라인 모드 경고 배너 ──
        if st.session_state.get("_offline_mode"):
            _odate = st.session_state.get("_offline_date", "알 수 없음")
            st.warning(
                f"📴 **오프라인 모드** — 인터넷 연결 없음. "
                f"백업 데이터({_odate}) 기준으로 표시됩니다.",
                icon="📴",
            )

        # ── 2. 현재 권장 자산배분 카드 ──
        st.markdown('<div class="section-title">🎯 현재 권장 자산배분 (전략7)</div>', unsafe_allow_html=True)

        cols5 = st.columns(6)
        for key, col in zip(["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"], cols5):
            with col:
                m       = meta[key]
                inv     = cur[key]["invest"]
                sell    = cur[key]["sell"]
                sig_cnt = cur[key]["sig_cnt"]
                w7      = cur[key]["w7"]
                rp      = cur[key]["port"]
                stk_pct = int(w7 * 100)

                # 상태별 스타일
                if sell:
                    card_bg     = "linear-gradient(160deg,#1f0808 0%,#120404 100%)"
                    card_border = "#dc2626"
                    icon        = "🔴"
                    status      = f"매도 신호 ({sig_cnt}/5)"
                    sc          = "#f87171"
                elif inv:
                    card_bg     = "linear-gradient(160deg,#0d2b1a 0%,#071a10 100%)"
                    card_border = "#16a34a"
                    icon        = "✅"
                    status      = "투자 시즌"
                    sc          = "#34d399"
                else:
                    card_bg     = "linear-gradient(160deg,#1c1208 0%,#120d05 100%)"
                    card_border = "#92400e"
                    icon        = "💤"
                    status      = "현금 시즌"
                    sc          = "#fb923c"

                # 비중 뱃지
                stk_bg  = "#7f1d1d" if sell else ("#065f46" if inv else "#374151")
                stk_clr = "#fca5a5" if sell else ("#34d399" if inv else "#9ca3af")
                badge_stk  = f'<span style="background:{stk_bg};color:{stk_clr};border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">주식&nbsp;{stk_pct}%</span>'
                cash_pct   = 100 - stk_pct - 50  # 금25+채권25=50
                badge_cash = f'<span style="background:#374151;color:#9ca3af;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">현금&nbsp;{cash_pct}%</span>' \
                             if cash_pct > 0 else ''
                badge_gold = f'<span style="background:#78350f;color:#fbbf24;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">금&nbsp;25%</span>'
                badge_bond = f'<span style="background:#1e3a5f;color:#93c5fd;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">{m["bond"]}&nbsp;25%</span>'
                badges     = f'{badge_stk} {badge_cash} {badge_gold} {badge_bond}'

                # 매도 신호 시 신호 상세 표시
                sell_note = (
                    f'<div style="color:#f87171;font-size:10px;margin-bottom:10px;'
                    f'background:#2d0a0a;border-radius:6px;padding:5px 8px;">'
                    f'⚡ 전략7 매도 신호 활성 — 주식 비중 0%</div>'
                ) if sell else ''

                port_color = clr(rp)
                port_val   = fp(rp)
                st.markdown(
                    f'<div style="background:{card_bg};border:1.5px solid {card_border};border-radius:14px;padding:20px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
                    f'<div><div style="color:#cbd5e1;font-size:16px;font-weight:700;letter-spacing:1px;">{m["emoji"]} {m["name"]}</div>'
                    f'<div style="color:#f1f5f9;font-size:24px;font-weight:800;margin-top:2px;">주식 {stk_pct}%</div></div>'
                    f'<div style="text-align:right;"><div style="color:{sc};font-size:13px;font-weight:700;">{icon} {status}</div>'
                    f'<div style="color:#6b7280;font-size:11px;margin-top:2px;">끝자리 {digit}</div></div></div>'
                    f'{sell_note}'
                    f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:14px;">{badges}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 3. 이번 시즌 자산별 수익률 ──
        st.markdown('<div class="section-title">📈 이번 시즌 자산별 수익률</div>', unsafe_allow_html=True)

        for key in ["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"]:
            m = meta[key]
            c1, c2, c3, c4 = st.columns(4)
            items = [
                ("주식",           cur[key]["stock"]),
                ("금",             cur[key]["gold"]),
                (m["bond"],        cur[key]["bond"]),
                ("포트폴리오(전략7)", cur[key]["port"]),
            ]
            for col, (label, val) in zip([c1, c2, c3, c4], items):
                is_port = label == "포트폴리오(전략7)"
                with col:
                    bc  = "#1e3a2a" if is_port else "#1e2a3a"
                    vc  = clr(val)
                    fs  = "21px" if is_port else "18px"
                    fw  = "800"  if is_port else "600"
                    ico = f"{m['emoji']} " if label == "주식" else ""
                    st.markdown(
                        f'<div class="asset-box" style="border-color:{bc};">'
                        f'<div style="color:#6b7280;font-size:11px;margin-bottom:4px;">{ico}{label}</div>'
                        f'<div style="color:{vc};font-size:{fs};font-weight:{fw};">{fp(val)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f"<div style='color:#4b5563;font-size:10px;margin-bottom:12px;'>"
                f"  {m['emoji']} {m['name']} · 시즌시작({season_start}) 대비</div>",
                unsafe_allow_html=True,
            )

        # ── 4. 장기 백테스트 성과 ──
        st.markdown('<div class="section-title">📊 장기 백테스트 성과</div>', unsafe_allow_html=True)

        # ── 수수료 입력 ──
        if "applied_fee" not in st.session_state:
            st.session_state.applied_fee = 0.0

        _fc1, _fc2, _fc3 = st.columns([1.2, 0.6, 4])
        with _fc1:
            _fee_input = st.number_input(
                "왕복 수수료 (%)", min_value=0.0, max_value=2.0,
                value=st.session_state.applied_fee, step=0.05, format="%.2f",
                help="매도+매수 1사이클 합산 비용 (예: 0.30 = 왕복 0.3%)",
                key="fee_input_widget",
            )
        with _fc2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("✅ 적용", key="fee_apply_btn"):
                st.session_state.applied_fee = _fee_input
                st.rerun()
        with _fc3:
            _af = st.session_state.applied_fee
            if _af > 0:
                st.markdown(
                    f"<div style='margin-top:32px;color:#fbbf24;font-size:12px;'>"
                    f"💡 현재 적용 수수료: 왕복 <b>{_af:.2f}%</b> — "
                    f"매도신호 1회당 포트폴리오의 <b>{_af/2:.2f}%</b>씩 2회 차감</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='margin-top:32px;color:#4b5563;font-size:12px;'>"
                    "수수료 0% 적용 중 (거래비용 미반영)</div>",
                    unsafe_allow_html=True,
                )
        _applied_fee = st.session_state.applied_fee

        def _daily_pf_series(key, fee_pct=0.0):
            """일별 포트폴리오 가치 시리즈 반환 — 전략6/전략7/BH/주식단독
            fee_pct: 왕복 수수료(%) — 매도/매수 각 leg마다 fee_pct/2 차감
            """
            sim = sims.get(key, pd.DataFrame())
            if sim.empty:
                return {}
            bond_key = "krbond" if key in ("kospi", "kosdaq") else "us30y"

            # 일별 수익률
            d_stk  = raw[key].pct_change().dropna()
            d_gold = raw["gold"].pct_change().dropna()
            d_bond = (-DURATION_US * raw["us30y"].diff() / 100).dropna() \
                     if bond_key == "us30y" else raw["krbond"].pct_change().dropna()

            for s in [d_stk, d_gold, d_bond]:
                s.index = pd.to_datetime(s.index).tz_localize(None)

            # ── 전략7용 신호 계산 ──
            px = raw[key].copy()
            px.index = pd.to_datetime(px.index).tz_localize(None)

            dd_s   = (px - px.cummax()) / px.cummax() * 100
            dlt    = px.diff()
            gain   = dlt.clip(lower=0).rolling(14).mean()
            loss   = (-dlt.clip(upper=0)).rolling(14).mean()
            rsi_s  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            ma200  = px.rolling(200).mean()
            ma_g   = (px - ma200) / ma200 * 100
            ma20   = px.rolling(20).mean()
            std20  = px.rolling(20).std()
            bb_rng = (ma20 + 2*std20) - (ma20 - 2*std20)
            bb_p   = (px - (ma20 - 2*std20)) / bb_rng.replace(0, np.nan) * 100

            if key in ("kospi", "kosdaq", "csi300"):
                s_vol = (px.pct_change().rolling(20).std() * (252**0.5) * 100 >= 35).astype(int)
            else:
                vix_h = load_vix_history()
                vix_h.index = pd.to_datetime(vix_h.index).tz_localize(None)
                vix_r = vix_h.reindex(px.index, method="ffill").fillna(20)
                s_vol = (vix_r >= 40).astype(int)

            sig = ((dd_s <= -30).astype(int) + (rsi_s <= 30).astype(int) +
                   (ma_g  <= -15).astype(int) + (bb_p  <=   0).astype(int) +
                   s_vol).fillna(0)
            sig_dict = sig.to_dict()

            # 시즌별 투자 여부 맵
            invest_map = {(int(r["연도"]), r["시즌"]): bool(r["투자"]) for _, r in sim.iterrows()}

            cols  = ["전략6", "전략7", "BH_max", "BH_min", "주식단독"]
            vals  = {c: [] for c in cols}
            v     = {c: 100.0 for c in cols}
            dates = []
            gold_idx = set(d_gold.index)
            bond_idx = set(d_bond.index)
            s7_sell  = False  # 전략7 매도 상태
            fee_leg  = fee_pct / 100 / 2   # 매 leg(매도 or 매수)당 비용

            # ── BH_max / BH_min 자산별 개별 추적 (6개월 리밸런싱용) ──
            # BH_max : 주식50 / 금25 / 채권25 / 현금0
            # BH_min : 주식25 / 금25 / 채권25 / 현금25
            bm_stk, bm_gld, bm_bnd          = 50.0, 25.0, 25.0        # BH_max 컴포넌트
            bl_stk, bl_gld, bl_bnd, bl_csh  = 25.0, 25.0, 25.0, 25.0  # BH_min 컴포넌트
            prev_season_key = None  # 시즌 전환 감지용

            for dt in d_stk.index:
                mo = dt.month
                if mo >= 11:   season, sy = "Nov-Apr", dt.year
                elif mo <= 4:  season, sy = "Nov-Apr", dt.year - 1
                else:          season, sy = "May-Oct", dt.year
                season_key = (sy, season)

                invest = invest_map.get((sy, season), False)
                rs = float(d_stk[dt])
                rg = float(d_gold[dt]) if dt in gold_idx else 0.0
                rb = float(d_bond[dt]) if dt in bond_idx else 0.0

                # ── 시즌 전환 시 BH 리밸런싱 ──
                if season_key != prev_season_key and prev_season_key is not None:
                    # BH_max → 50:25:25 복원
                    tot_max = bm_stk + bm_gld + bm_bnd
                    bm_stk, bm_gld, bm_bnd = tot_max*0.50, tot_max*0.25, tot_max*0.25
                    # BH_min → 25:25:25:25 복원
                    tot_min = bl_stk + bl_gld + bl_bnd + bl_csh
                    bl_stk = bl_gld = bl_bnd = bl_csh = tot_min * 0.25
                prev_season_key = season_key

                # 전략7 매도 상태 갱신 + 수수료 차감
                s = float(sig_dict.get(dt, 0))
                prev_sell = s7_sell
                if   s >= 2: s7_sell = True
                elif s == 0: s7_sell = False
                if s7_sell != prev_sell:
                    v["전략7"] *= (1 - fee_leg)

                w6 = 0.50 if invest else 0.25
                w7 = 0.0  if s7_sell else w6

                # 전략6/7: 기존 방식 (시즌 시작 시 비중 결정 후 일별 적용)
                v["전략6"]   *= (1 + w6*rs  + 0.25*rg + 0.25*rb)
                v["전략7"]   *= (1 + w7*rs  + 0.25*rg + 0.25*rb)
                v["주식단독"] *= (1 + rs)

                # BH_max / BH_min: 자산별 성장 후 합산
                bm_stk *= (1 + rs); bm_gld *= (1 + rg); bm_bnd *= (1 + rb)
                bl_stk *= (1 + rs); bl_gld *= (1 + rg); bl_bnd *= (1 + rb)
                # bl_csh: 무이자 현금 → 변화 없음
                v["BH_max"] = bm_stk + bm_gld + bm_bnd
                v["BH_min"] = bl_stk + bl_gld + bl_bnd + bl_csh

                for c in cols: vals[c].append(v[c])
                dates.append(dt)

            return {c: pd.Series(vals[c], index=dates) for c in cols}

        def _daily_pf_series_s8(fee_pct=0.0):
            """전략8 일별 포트폴리오 시리즈 — 5지수 동적 비중 (SP500+나스닥/KOSPI+코스닥/CSI300)"""
            sim_s8 = sims.get("s8", pd.DataFrame())
            if sim_s8.empty:
                return {}

            d_sp = raw["sp500"].pct_change().dropna()
            d_nq = raw["nasdaq"].pct_change().dropna()
            d_ko = raw["kospi"].pct_change().dropna()
            d_kq = raw["kosdaq"].pct_change().dropna()
            d_cs = raw["csi300"].pct_change().dropna()
            d_g  = raw["gold"].pct_change().dropna()
            d_kr = raw["krbond"].pct_change().dropna()
            d_us = (-DURATION_US * raw["us30y"].diff() / 100).dropna()

            for s in [d_sp, d_nq, d_ko, d_kq, d_cs, d_g, d_kr, d_us]:
                s.index = pd.to_datetime(s.index).tz_localize(None)

            nq_idx = set(d_nq.index)
            ko_idx = set(d_ko.index)
            kq_idx = set(d_kq.index)
            cs_idx = set(d_cs.index)
            g_idx  = set(d_g.index)
            kr_idx = set(d_kr.index)
            us_idx = set(d_us.index)

            # 3개 메인 지수 투자 맵
            sp_sim = sims.get("sp500",  pd.DataFrame())
            ko_sim = sims.get("kospi",  pd.DataFrame())
            cs_sim = sims.get("csi300", pd.DataFrame())

            sp_inv_map = {(int(r["연도"]), r["시즌"]): bool(r["투자"]) for _, r in sp_sim.iterrows()} if not sp_sim.empty else {}
            ko_inv_map = {(int(r["연도"]), r["시즌"]): bool(r["투자"]) for _, r in ko_sim.iterrows()} if not ko_sim.empty else {}
            cs_inv_map = {(int(r["연도"]), r["시즌"]): bool(r["투자"]) for _, r in cs_sim.iterrows()} if not cs_sim.empty else {}

            cols = ["전략8", "BH_max", "BH_min"]
            v    = {"전략8": 100.0, "BH_max": 100.0, "BH_min": 100.0}
            vals = {c: [] for c in cols}
            dates = []

            bm_stk, bm_gld, bm_bnd         = 50.0, 25.0, 25.0
            bl_stk, bl_gld, bl_bnd, bl_csh = 25.0, 25.0, 25.0, 25.0
            prev_season_key = None

            for dt in d_sp.index:
                mo = dt.month
                if mo >= 11:   season, sy = "Nov-Apr", dt.year
                elif mo <= 4:  season, sy = "Nov-Apr", dt.year - 1
                else:          season, sy = "May-Oct", dt.year
                season_key = (sy, season)

                r_sp = float(d_sp[dt])
                r_nq = float(d_nq[dt]) if dt in nq_idx else r_sp
                r_ko = float(d_ko[dt]) if dt in ko_idx else r_sp
                r_kq = float(d_kq[dt]) if dt in kq_idx else r_ko
                r_cs = float(d_cs[dt]) if dt in cs_idx else r_sp
                r_g  = float(d_g[dt])  if dt in g_idx  else 0.0
                r_kr = float(d_kr[dt]) if dt in kr_idx else 0.0
                r_us = float(d_us[dt]) if dt in us_idx else 0.0

                sp_inv = sp_inv_map.get((sy, season), False)
                ko_inv = ko_inv_map.get((sy, season), sp_inv)
                cs_inv = cs_inv_map.get((sy, season), sp_inv)

                # SP500 투자시즌: SP500 10% + 나스닥 10% / 비투자: SP500 10%만
                w_sp = 0.10
                w_nq = 0.10 if sp_inv else 0.0
                # KOSPI 투자시즌: KOSPI 10% + 코스닥 10% / 비투자: KOSPI 10%만
                w_ko = 0.10
                w_kq = 0.10 if ko_inv else 0.0
                # CSI300: 투자시즌 20% / 비투자 10%
                w_cs = 0.20 if cs_inv else 0.10

                # 시즌 전환 시 BH 리밸런싱
                if season_key != prev_season_key and prev_season_key is not None:
                    tot_max = bm_stk + bm_gld + bm_bnd
                    bm_stk, bm_gld, bm_bnd = tot_max*0.50, tot_max*0.25, tot_max*0.25
                    tot_min = bl_stk + bl_gld + bl_bnd + bl_csh
                    bl_stk = bl_gld = bl_bnd = bl_csh = tot_min * 0.25
                prev_season_key = season_key

                w_cash = round(1.0 - w_sp - w_nq - w_ko - w_kq - w_cs - 0.20 - 0.10 - 0.10, 10)
                r_s8 = (w_sp*r_sp + w_nq*r_nq + w_ko*r_ko + w_kq*r_kq
                        + w_cs*r_cs + 0.20*r_g + 0.10*r_kr + 0.10*r_us
                        + w_cash*r_kr)   # 현금 → KODEX 국고채3년 운용
                v["전략8"] *= (1 + r_s8)

                bm_stk *= (1+r_sp); bm_gld *= (1+r_g); bm_bnd *= (1+r_us)
                bl_stk *= (1+r_sp); bl_gld *= (1+r_g); bl_bnd *= (1+r_us)
                v["BH_max"] = bm_stk + bm_gld + bm_bnd
                v["BH_min"] = bl_stk + bl_gld + bl_bnd + bl_csh

                for c in cols: vals[c].append(v[c])
                dates.append(dt)

            return {c: pd.Series(vals[c], index=dates) for c in cols}

        def _stats(s):
            if s is None or s.empty: return 0.0, 0.0, 0.0
            cum  = s.iloc[-1] - 100.0
            n_yr = (s.index[-1] - s.index[0]).days / 365.25
            cagr = ((s.iloc[-1]/100.0)**(1/n_yr) - 1)*100 if n_yr > 0 else 0.0
            mdd  = ((s - s.cummax()) / s.cummax() * 100).min()
            return cum, cagr, mdd

        def _card_html(title, bg, border, cum, cagr, mdd, large=False):
            cs = "+" if cum  >= 0 else ""
            gs = "+" if cagr >= 0 else ""
            cc = "#34d399" if cum  >= 0 else "#f87171"
            gc = "#34d399" if cagr >= 0 else "#f87171"
            cum_fs  = "32px" if large else "26px"
            sub_fs  = "19px" if large else "16px"
            pad     = "24px 28px" if large else "18px 20px"
            return (
                f'<div style="background:{bg};border:1.5px solid {border};'
                f'border-radius:14px;padding:{pad};height:100%;">'
                f'<div style="color:{border};font-size:{"13px" if large else "11px"};'
                f'font-weight:800;letter-spacing:0.8px;margin-bottom:{"20px" if large else "14px"};">'
                f'{title}</div>'
                f'<div style="margin-bottom:{"16px" if large else "12px"};">'
                f'<div style="color:#4b5563;font-size:10px;letter-spacing:0.5px;margin-bottom:4px;">누적수익률</div>'
                f'<div style="color:{cc};font-size:{cum_fs};font-weight:900;line-height:1;letter-spacing:-0.5px;">'
                f'{cs}{cum:.0f}%</div></div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;'
                f'border-top:1px solid #1e2a3a;padding-top:{"16px" if large else "12px"};">'
                f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">CAGR</div>'
                f'<div style="color:{gc};font-size:{sub_fs};font-weight:700;">{gs}{cagr:.1f}%</div></div>'
                f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">MDD</div>'
                f'<div style="color:#f87171;font-size:{sub_fs};font-weight:700;">{mdd:.1f}%</div></div>'
                f'</div></div>'
            )

        def _render_perf_tab(key, fee_pct=0.0):
            pf = _daily_pf_series(key, fee_pct=fee_pct)
            if not pf:
                st.warning("데이터 없음")
                return

            s7_cum, s7_cagr, s7_mdd = _stats(pf["전략7"])
            s6_cum, s6_cagr, s6_mdd = _stats(pf["전략6"])
            b5_cum, b5_cagr, b5_mdd = _stats(pf["BH_max"])
            b2_cum, b2_cagr, b2_mdd = _stats(pf["BH_min"])
            sk_cum, sk_cagr, sk_mdd = _stats(pf["주식단독"])

            y0 = pf["전략7"].index[0].year
            y1 = pf["전략7"].index[-1].year

            # ── 수수료 배지 ──
            fee_badge = (
                f" &nbsp;<span style='font-size:10px;color:#fbbf24;background:#1a1305;"
                f"border:1px solid #92400e;border-radius:4px;padding:1px 6px;'>"
                f"수수료 {fee_pct:.2f}% 반영</span>"
                if fee_pct > 0 else ""
            )
            # ── 전략7 히어로 카드 ──
            hero = _card_html(f"⚡ 전략7  (전략6 + 매도신호≥2 시 주식0%){fee_badge}",
                               "#0d0f1a", "#a78bfa", s7_cum, s7_cagr, s7_mdd, large=True)
            # ── 비교 카드 4개 ──
            cmp = (
                _card_html("⚙️ 전략6",      "#071a10", "#16a34a", s6_cum, s6_cagr, s6_mdd)
              + _card_html("📈 주식50% BH", "#0d0d20", "#6366f1", b5_cum, b5_cagr, b5_mdd)
              + _card_html("📊 주식25% BH", "#111827", "#6b7280", b2_cum, b2_cagr, b2_mdd)
              + _card_html("💹 주식단독",   "#1a1305", "#f59e0b", sk_cum, sk_cagr, sk_mdd)
            )
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 2fr;gap:14px;margin-bottom:4px;">'
                f'  {hero}'
                f'  <div style="display:grid;grid-template-columns:repeat(2,1fr);'
                f'       grid-template-rows:repeat(2,1fr);gap:10px;">{cmp}</div>'
                f'</div>'
                f'<div style="color:#374151;font-size:10px;margin-bottom:10px;">'
                f'📌 백테스트 기간 {y0}~{y1}년 · 일별 포트폴리오 기준 · '
                f'전략7 조건: 5개 신호 중 2개 이상 동시 점등 시 주식 0%, 신호 0개 시 전략6으로 복귀</div>',
                unsafe_allow_html=True,
            )

            # ── 누적 성과 비교 차트 ──
            fee_title = f"  (수수료 {fee_pct:.2f}% 반영)" if fee_pct > 0 else ""
            fig_c = go.Figure()
            chart_cfg = [
                ("전략7",    "⚡ 전략7",       "#a78bfa", 2.5),
                ("전략6",    "⚙️ 전략6",       "#34d399", 1.5),
                ("BH_max",  "📈 주식50%BH",   "#6366f1", 1.0),
                ("BH_min",  "📊 주식25%BH",   "#6b7280", 1.0),
                ("주식단독", "💹 주식단독BH",   "#fbbf24", 1.0),
            ]
            for col, name, color, width in chart_cfg:
                s = pf[col].resample("W").last()
                fig_c.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=name, mode="lines",
                    line=dict(color=color, width=width),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
                ))
            fig_c.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                height=400, margin=dict(l=0, r=0, t=44, b=0),
                title=dict(text=f"누적 성과 비교 (시작=100){fee_title}", font=dict(size=13, color="#f1f5f9")),
                legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10, color="#9ca3af")),
                yaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_c, use_container_width=True)

        ptab_s, ptab_n, ptab_k, ptab_d, ptab_kq, ptab_cn, ptab_s8 = st.tabs(
            ["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300", "🌏 전략8"]
        )
        for ptab, pkey in [
            (ptab_s,"sp500"),(ptab_n,"nasdaq"),(ptab_k,"kospi"),
            (ptab_d,"dow"),(ptab_kq,"kosdaq"),(ptab_cn,"csi300")
        ]:
            with ptab:
                _render_perf_tab(pkey, fee_pct=_applied_fee)

        with ptab_s8:
            pf8 = _daily_pf_series_s8(fee_pct=_applied_fee)
            if not pf8:
                st.warning("데이터 없음")
            else:
                s8_cum,  s8_cagr,  s8_mdd  = _stats(pf8["전략8"])
                bx_cum,  bx_cagr,  bx_mdd  = _stats(pf8["BH_max"])
                bn_cum,  bn_cagr,  bn_mdd  = _stats(pf8["BH_min"])
                y0 = pf8["전략8"].index[0].year
                y1 = pf8["전략8"].index[-1].year

                fee_badge = (
                    f" &nbsp;<span style='font-size:10px;color:#fbbf24;background:#1a1305;"
                    f"border:1px solid #92400e;border-radius:4px;padding:1px 6px;'>"
                    f"수수료 {_applied_fee:.2f}% 반영</span>"
                    if _applied_fee > 0 else ""
                )
                hero8 = _card_html(
                    f"🌏 전략8  (SP500+KOSPI+CSI300 3지수 동적배분){fee_badge}",
                    "#0a1020", "#38bdf8", s8_cum, s8_cagr, s8_mdd, large=True
                )
                cmp8 = (
                    _card_html("📈 주식50% BH", "#0d0d20", "#6366f1", bx_cum, bx_cagr, bx_mdd)
                  + _card_html("📊 주식25% BH", "#111827", "#6b7280", bn_cum, bn_cagr, bn_mdd)
                )
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:4px;">'
                    f'  {hero8}'
                    f'  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">{cmp8}</div>'
                    f'</div>'
                    f'<div style="color:#374151;font-size:10px;margin-bottom:10px;">'
                    f'📌 백테스트 기간 {y0}~{y1}년 · 일별 포트폴리오 기준 · '
                    f'금20% + 한국채10% + 미국채10% 고정 · 3지수 투자시즌 합의에 따라 주식 30~60% 동적조절</div>',
                    unsafe_allow_html=True,
                )

                # 누적 성과 차트
                fee_title = f"  (수수료 {_applied_fee:.2f}% 반영)" if _applied_fee > 0 else ""
                fig_s8 = go.Figure()
                for col, name, color, width in [
                    ("전략8",  "🌏 전략8",      "#38bdf8", 2.5),
                    ("BH_max", "📈 주식50%BH",  "#6366f1", 1.0),
                    ("BH_min", "📊 주식25%BH",  "#6b7280", 1.0),
                ]:
                    s = pf8[col].resample("W").last()
                    fig_s8.add_trace(go.Scatter(
                        x=s.index, y=s.values, name=name, mode="lines",
                        line=dict(color=color, width=width),
                        hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
                    ))
                fig_s8.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    height=400, margin=dict(l=0, r=0, t=44, b=0),
                    title=dict(text=f"누적 성과 비교 (시작=100){fee_title}", font=dict(size=13, color="#f1f5f9")),
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                    xaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10, color="#9ca3af")),
                    yaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_s8, use_container_width=True)

                # 시즌별 상세 테이블
                with st.expander("📋 시즌별 상세 데이터"):
                    sim8 = sims.get("s8", pd.DataFrame())
                    if not sim8.empty:
                        disp8 = sim8[["연도","시즌","끝자리","SP투자","KO투자","CS투자",
                                      "투자지수수","주식비중(%)","SP500(%)","KOSPI(%)","CSI300(%)",
                                      "금(%)","한국채(%)","미국채(%)","전략8","BH_max","BH_min"]].copy()
                        disp8 = disp8.sort_values("연도", ascending=False).reset_index(drop=True)
                        st.dataframe(disp8, use_container_width=True, height=400)

        # ── 6. 과거 데이터 테이블 ──
        st.markdown('<div class="section-title">🗂️ 과거 데이터 테이블</div>', unsafe_allow_html=True)

        hist_tab1, hist_tab2 = st.tabs(["📋 반기별 수익률", "📊 끝자리 통계"])

        with hist_tab1:
            def fmt_sim(sim_df):
                df = sim_df.copy().sort_values("연도", ascending=False).reset_index(drop=True)
                df["투자여부"] = df["투자"].map({True: "✅ 투자", False: "💤 현금"})
                for c in ["주식수익", "금수익", "채권수익", "포트수익"]:
                    df[c] = df[c].apply(lambda v: f"{v*100:+.1f}%")
                return df[["연도","시즌","끝자리","투자여부",
                            "주식수익","금수익","채권수익","포트수익",
                            "전략6","BH_min","BH_max","주식단독"]]

            def style_sim(df):
                def _c(val):
                    s = str(val)
                    if "%" in s:
                        try:
                            v = float(s.replace("%","").replace("+",""))
                            if v > 0: return "color:#34d399;font-weight:600"
                            if v < 0: return "color:#f87171;font-weight:600"
                        except: pass
                    if "✅" in s: return "color:#34d399"
                    if "💤" in s: return "color:#fb923c"
                    return "color:#d1d5db"
                return df.style.map(_c)

            s_tab, n_tab, k_tab, d_tab, kq_tab, cn_tab = st.tabs(["S&P500", "NASDAQ", "KOSPI", "DOW", "KOSDAQ", "CSI300"])
            for tab, key in [(s_tab,"sp500"), (n_tab,"nasdaq"), (k_tab,"kospi"), (d_tab,"dow"), (kq_tab,"kosdaq"), (cn_tab,"csi300")]:
                with tab:
                    st.dataframe(style_sim(fmt_sim(sims[key])),
                                 use_container_width=True, height=480, hide_index=True)

        with hist_tab2:
            def build_digit_tbl(key, season_key):
                info    = strategies[key]
                df_src  = info["na"] if season_key == "Nov-Apr" else info["mo"]
                prob_src= info["pna"] if season_key == "Nov-Apr" else info["pmo"]
                inv_set = info["inv_na"] if season_key == "Nov-Apr" else info["inv_mo"]

                df_src = df_src.copy()
                df_src["d"] = df_src.index % 10
                rows = []
                for d in range(10):
                    sub = df_src[df_src["d"] == d]["ret"].dropna()
                    if len(sub) == 0: continue
                    p = prob_src.get(d, 0)
                    yrs = sorted(df_src[df_src["d"] == d].index.tolist())
                    rows.append({
                        "끝자리":   d,
                        "해당연도": ", ".join(str(y) for y in yrs),
                        "횟수":     len(sub),
                        "수익확률": f"{p*100:.0f}%",
                        "평균":     f"{sub.mean()*100:+.1f}%",
                        "최고":     f"{sub.max()*100:+.1f}%",
                        "최악":     f"{sub.min()*100:+.1f}%",
                        "신호":     "✅ 투자" if d in inv_set else "💤 현금",
                    })
                return pd.DataFrame(rows)

            def style_digit(df):
                def _c(val):
                    s = str(val)
                    if "%" in s:
                        try:
                            v = float(s.replace("%","").replace("+",""))
                            if v > 0: return "color:#34d399;font-weight:600"
                            if v < 0: return "color:#f87171;font-weight:600"
                        except: pass
                    if "✅" in s: return "color:#34d399;font-weight:700"
                    if "💤" in s: return "color:#fb923c"
                    return "color:#d1d5db"
                return df.style.map(_c)

            ds_tab, dn_tab, dk_tab, dd_tab, dkq_tab, dcn_tab = st.tabs(["S&P500", "NASDAQ", "KOSPI", "DOW", "KOSDAQ", "CSI300"])
            for tab, key in [(ds_tab,"sp500"), (dn_tab,"nasdaq"), (dk_tab,"kospi"), (dd_tab,"dow"), (dkq_tab,"kosdaq"), (dcn_tab,"csi300")]:
                with tab:
                    na_t, mo_t = st.tabs(["Nov-Apr (11~4월)", "May-Oct (5~10월)"])
                    with na_t:
                        st.dataframe(style_digit(build_digit_tbl(key, "Nov-Apr")),
                                     use_container_width=True, hide_index=True)
                    with mo_t:
                        st.dataframe(style_digit(build_digit_tbl(key, "May-Oct")),
                                     use_container_width=True, hide_index=True)

        # ── 6. 끝자리 수익확률 히트맵 ──
        st.markdown('<div class="section-title">🗺️ 끝자리 수익확률 히트맵</div>', unsafe_allow_html=True)
        st.caption("초록 = 수익확률 ≥ 60% (투자 시즌) · 빨강 = 현금 보유 · 밝을수록 확률 높음")

        def make_heatmap(key):
            info = strategies[key]
            z, texts, hovers = [], [], []
            for sk, prob_src, inv_set in [
                ("Nov-Apr", info["pna"], info["inv_na"]),
                ("May-Oct", info["pmo"], info["inv_mo"]),
            ]:
                row_z, row_t, row_h = [], [], []
                for d in range(10):
                    p = prob_src.get(d, 0)
                    row_z.append(p)
                    row_t.append(f"{p*100:.0f}%")
                    row_h.append(
                        f"끝자리 {d}<br>{sk}<br>수익확률: {p*100:.0f}%<br>"
                        f"{'✅ 투자' if d in inv_set else '💤 현금'}"
                    )
                z.append(row_z); texts.append(row_t); hovers.append(row_h)

            fig = go.Figure(go.Heatmap(
                z=z, x=[str(d) for d in range(10)], y=["Nov-Apr", "May-Oct"],
                text=texts, hovertext=hovers, texttemplate="%{text}",
                colorscale=[
                    [0.0, "#4b0000"], [0.35, "#991b1b"],
                    [0.60, "#064e3b"], [1.0,  "#022c22"],
                ],
                zmin=0, zmax=1, showscale=False,
            ))
            fig.add_shape(
                type="rect",
                x0=digit - 0.5, x1=digit + 0.5, y0=-0.5, y1=1.5,
                line=dict(color="#fbbf24", width=3), fillcolor="rgba(0,0,0,0)",
            )
            fig.add_annotation(
                x=digit, y=1.6, text=f"현재({digit})",
                font=dict(color="#fbbf24", size=11), showarrow=False,
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                height=200, margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(title="연도 끝자리", tickfont=dict(size=13, color="#f1f5f9")),
                yaxis=dict(tickfont=dict(size=12, color="#f1f5f9")),
                font=dict(color="#f1f5f9", size=13),
            )
            return fig

        hm_s, hm_n, hm_k, hm_d, hm_kq, hm_cn = st.tabs(["S&P500", "NASDAQ", "KOSPI", "DOW", "KOSDAQ", "CSI300"])
        for tab, key in [(hm_s,"sp500"), (hm_n,"nasdaq"), (hm_k,"kospi"), (hm_d,"dow"), (hm_kq,"kosdaq"), (hm_cn,"csi300")]:
            with tab:
                st.plotly_chart(make_heatmap(key), use_container_width=True)

        # ── 푸터 ──
        st.markdown("---")
        st.markdown(
            '<div class="footer-txt">본 대시보드는 교육·연구 목적이며 투자 권유가 아닙니다. '
            '과거 데이터 기반 백테스트로 미래 수익률을 보장하지 않습니다.</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════
    # TAB 2: 일일 최대낙폭
    # ════════════════════════════════════════
    with main_tab2:
        st.markdown('<div class="section-title">📉 역대 폭락일 TOP 30</div>', unsafe_allow_html=True)
        st.caption("전일 종가 대비 당일 종가 기준 하락률 상위 30일")

        drop_s, drop_n, drop_k, drop_d, drop_kq, drop_cn = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"])

        CATEGORIES = [
            "⚔️ 지정학/전쟁",
            "🏦 금융위기",
            "🦠 팬데믹",
            "📉 버블붕괴",
            "💱 외환위기",
            "📊 정책/긴축",
            "⚡ 기술충격",
            "🌏 글로벌경기",
        ]

        def get_category(reason):
            if " | " in reason:
                return reason.split(" | ")[0].strip()
            return "🌏 글로벌경기"

        def render_drop_tab(key):
            df = ohlc[key].copy()
            df_drop = df[df["daily_ret"] < 0].nsmallest(30, "daily_ret").reset_index()

            # Date 컬럼명 통일
            date_col = df_drop.columns[0]

            # 이후 수익률 계산용 전체 df (정수 인덱스)
            df_full = ohlc[key].copy().reset_index()
            df_full.columns = ["Date"] + list(df_full.columns[1:])

            def fwd_ret(idx, days):
                if idx + days < len(df_full):
                    c0 = df_full.loc[idx, "Close"]
                    cn = df_full.loc[idx + days, "Close"]
                    return (cn - c0) / c0
                return None

            def fmt_fwd(v):
                if v is None:
                    return "N/A"
                return f"{v*100:+.2f}%"

            rows = []
            for rank, row in enumerate(df_drop.itertuples(), start=1):
                dt = getattr(row, date_col)
                dt_ts  = pd.Timestamp(dt)
                dt_str = dt_ts.strftime("%Y-%m-%d")
                reason = get_crash_reason(key, dt_ts)

                pos = df_full[df_full["Date"] == dt_ts].index
                idx = pos[0] if len(pos) > 0 else None

                rows.append({
                    "순위":     rank,
                    "날짜":     dt_str,
                    "전일종가": f"{row.prev_close:,.2f}",
                    "종가":     f"{row.Close:,.2f}",
                    "하락률":   f"{row.daily_ret*100:.2f}%",
                    "+1일":     fmt_fwd(fwd_ret(idx, 1))  if idx is not None else "N/A",
                    "+5일":     fmt_fwd(fwd_ret(idx, 5))  if idx is not None else "N/A",
                    "+10일":    fmt_fwd(fwd_ret(idx, 10)) if idx is not None else "N/A",
                    "+20일":    fmt_fwd(fwd_ret(idx, 20)) if idx is not None else "N/A",
                    "사유":     reason,
                    "_category": get_category(reason),
                })

            result_df = pd.DataFrame(rows)

            # ── 카테고리 체크박스 필터 ──
            st.markdown(
                """
                <div style="
                    background:#1a2235;
                    border:1px solid #3b82f6;
                    border-radius:10px;
                    padding:14px 18px 6px 18px;
                    margin-bottom:14px;
                ">
                <div style="color:#60a5fa;font-size:13px;font-weight:700;margin-bottom:10px;">
                    🔽 카테고리 필터 (체크 해제 시 해당 항목 숨김)
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container():
                cb_cols = st.columns(4)
                selected_cats = []
                for i, cat in enumerate(CATEGORIES):
                    with cb_cols[i % 4]:
                        if st.checkbox(cat, value=True, key=f"cat_{key}_{i}"):
                            selected_cats.append(cat)
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

            if not selected_cats:
                st.warning("카테고리를 하나 이상 선택해주세요.")
                return

            filtered_df = result_df[result_df["_category"].isin(selected_cats)].drop(
                columns=["_category"]
            ).reset_index(drop=True)
            filtered_df["순위"] = range(1, len(filtered_df) + 1)

            st.caption(f"총 {len(filtered_df)}건 표시 중")

            def style_drop(df):
                def _c(val):
                    s = str(val)
                    if s.endswith("%"):
                        try:
                            v = float(s.replace("%", "").replace("+", ""))
                            if v < 0:
                                return "color:#f87171;font-weight:700"
                            elif v > 0:
                                return "color:#34d399;font-weight:600"
                        except:
                            pass
                    if s.isdigit():
                        return "color:#6b7280"
                    return "color:#d1d5db"
                return df.style.map(_c)

            st.dataframe(
                style_drop(filtered_df),
                use_container_width=True,
                height=min(100 + len(filtered_df) * 36, 980),
                hide_index=True,
            )

        for tab, key in [(drop_s, "sp500"), (drop_n, "nasdaq"), (drop_k, "kospi"), (drop_d, "dow"), (drop_kq, "kosdaq"), (drop_cn, "csi300")]:
            with tab:
                render_drop_tab(key)

    # ════════════════════════════════════════
    # TAB 3: 낙폭 후 투자 시뮬레이터
    # ════════════════════════════════════════
    with main_tab3:
        st.markdown('<div class="section-title">🔍 폭락 후 전략</div>', unsafe_allow_html=True)
        st.caption("폭락일 이후 N일 뒤 진입 시 수익률을 분석합니다. 모든 날짜는 거래일(영업일) 기준입니다.")

        sim_s, sim_n, sim_k, sim_d, sim_kq, sim_cn = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"])

        def render_sim_tab(key):
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                top_n = st.slider("분석할 낙폭 상위 N개", 5, 30, 10, key=f"topn_{key}")
            with c2:
                entry_input = st.text_input(
                    "진입 시점 (낙폭 후 N일 뒤 종가 기준, 여러 개 입력 시 TOP5에 모두 적용)",
                    value="1",
                    key=f"entry_{key}",
                )
                st.caption("0 = 낙폭 당일 종가  |  1 = 다음날 종가 (현실적)  |  여러 개: 예) 0, 1, 2, 3")
            with c3:
                days_input = st.text_input(
                    "수익률 확인 시점 — 쉼표로 구분 (예: 5, 10, 20, 40, 60)",
                    value="5, 10, 20, 40, 60",
                    key=f"days_{key}",
                )

            # ── 입력값 파싱 ──
            def parse_ints_pos(s):
                try:
                    return sorted(set(
                        int(x.strip()) for x in s.split(",")
                        if x.strip().lstrip("-").isdigit() and int(x.strip()) > 0
                    ))
                except Exception:
                    return []

            def parse_nonneg_ints(s):
                try:
                    return sorted(set(
                        int(x.strip()) for x in s.split(",")
                        if x.strip().lstrip("-").isdigit() and int(x.strip()) >= 0
                    ))
                except Exception:
                    return []

            entry_list = parse_nonneg_ints(entry_input) or [1]
            entry_days = entry_list[0]   # 메인 테이블·엑셀 기준

            fwd_days = parse_ints_pos(days_input)
            if not fwd_days:
                st.warning("수익률 확인 시점을 올바르게 입력해주세요. 예) 5, 10, 20, 40, 60")
                return

            df = ohlc[key].copy()
            df_drop = df[df["daily_ret"] < 0].nsmallest(top_n, "daily_ret").reset_index()
            date_col = df_drop.columns[0]

            df_full = ohlc[key].copy().reset_index()
            df_full.columns = ["Date"] + list(df_full.columns[1:])

            rows = []
            for rank, row in enumerate(df_drop.itertuples(), start=1):
                dt        = getattr(row, date_col)
                dt_ts     = pd.Timestamp(dt)
                dt_str    = dt_ts.strftime("%Y-%m-%d")
                reason    = get_crash_reason(key, dt_ts)

                pos = df_full[df_full["Date"] == dt_ts].index
                if len(pos) == 0:
                    continue
                crash_idx = pos[0]
                entry_idx = crash_idx + entry_days
                if entry_idx >= len(df_full):
                    continue

                entry_price = df_full.loc[entry_idx, "Close"]
                entry_date  = df_full.loc[entry_idx, "Date"].strftime("%Y-%m-%d")

                row_data = {
                    "순위":   rank,
                    "낙폭일": dt_str,
                    "하락률": f"{row.daily_ret*100:.2f}%",
                    "진입일": entry_date,
                    "진입가": f"{entry_price:,.2f}",
                }

                for d in fwd_days:
                    target_idx = entry_idx + d
                    if target_idx < len(df_full):
                        tp  = df_full.loc[target_idx, "Close"]
                        ret = (tp - entry_price) / entry_price
                        row_data[f"+{d}일"] = f"{ret*100:+.2f}%"
                    else:
                        row_data[f"+{d}일"] = "N/A"

                row_data["사유"] = reason
                rows.append(row_data)

            if not rows:
                st.warning("데이터가 부족합니다.")
                return

            result_df = pd.DataFrame(rows)

            def style_sim_tab(df):
                def _c(val):
                    s = str(val)
                    if s.endswith("%"):
                        try:
                            v = float(s.replace("%", "").replace("+", ""))
                            if v < 0: return "color:#f87171;font-weight:700"
                            if v > 0: return "color:#34d399;font-weight:600"
                        except: pass
                    return "color:#d1d5db"
                return df.style.map(_c)

            st.dataframe(style_sim_tab(result_df), use_container_width=True, height=420, hide_index=True)

            # ── 요약 통계 ──
            st.markdown(
                f'<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                f'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:20px 0 12px;">📊 요약 통계 '
                f'(낙폭 상위 {top_n}개 · 진입 {entry_days}일 뒤 종가 기준)</div>',
                unsafe_allow_html=True,
            )

            summary_rows = []
            for d in fwd_days:
                col_name = f"+{d}일"
                if col_name not in result_df.columns:
                    continue
                vals = []
                for v in result_df[col_name]:
                    if v != "N/A":
                        try:
                            vals.append(float(v.replace("%", "").replace("+", "")))
                        except Exception:
                            pass
                if vals:
                    pos_cnt = sum(1 for v in vals if v > 0)
                    summary_rows.append({
                        "기간":       col_name,
                        "평균 수익률": f"{np.mean(vals):+.2f}%",
                        "최대":       f"{max(vals):+.2f}%",
                        "최소":       f"{min(vals):+.2f}%",
                        "상승 횟수":  f"{pos_cnt}/{len(vals)}",
                        "상승 확률":  f"{pos_cnt/len(vals)*100:.0f}%",
                    })

            if summary_rows:
                st.dataframe(
                    style_sim_tab(pd.DataFrame(summary_rows)),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── 낙폭 후 200일 경로 차트 ──
            st.markdown(
                f'<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                f'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:24px 0 12px;">'
                f'📈 낙폭 후 200일 가격 경로 (낙폭일 = 0%)</div>',
                unsafe_allow_html=True,
            )
            st.caption("각 낙폭일 종가를 0% 기준으로 정규화 · 굵은 노란선 = 전체 평균")

            # 색상 팔레트 (이벤트별)
            palette = [
                "#60a5fa","#34d399","#f87171","#a78bfa","#fbbf24",
                "#38bdf8","#4ade80","#fb923c","#e879f9","#94a3b8",
                "#f472b6","#2dd4bf","#facc15","#818cf8","#86efac",
                "#7dd3fc","#fca5a5","#c4b5fd","#6ee7b7","#fed7aa",
            ]

            paths      = []   # (x_list, y_list, label)
            path_matrix = []  # for average

            for rank, cr in enumerate(df_drop.itertuples(), start=1):
                dt_cr  = getattr(cr, date_col)
                dt_ts  = pd.Timestamp(dt_cr)
                pos_cr = df_full[df_full["Date"] == dt_ts].index
                if len(pos_cr) == 0:
                    continue
                ci       = pos_cr[0]
                end_i    = min(ci + 201, len(df_full))
                segment  = df_full.loc[ci:end_i - 1, "Close"].values
                if len(segment) < 2:
                    continue
                base   = segment[0]
                y_vals = [(p - base) / base * 100 for p in segment]
                x_vals = list(range(len(y_vals)))
                reason = get_crash_reason(key, dt_ts)
                label  = f"#{rank} {dt_ts.strftime('%Y-%m-%d')}  {reason}"
                paths.append((x_vals, y_vals, label, rank))
                path_matrix.append(y_vals)

            # 200일 시점 최종 수익률 기준으로 내림차순 정렬
            paths.sort(key=lambda t: t[1][-1], reverse=True)

            fig2 = go.Figure()

            # 개별 이벤트 선
            for x_vals, y_vals, label, rank in paths:
                clr_line = palette[(rank - 1) % len(palette)]
                fig2.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    name=label,
                    mode="lines",
                    line=dict(color=clr_line, width=1.2),
                    opacity=0.55,
                    hovertemplate=f"<b>{label}</b><br>경과일: %{{x}}<br>수익률: %{{y:.2f}}%<extra></extra>",
                ))

            # 평균선
            if path_matrix:
                max_len = max(len(v) for v in path_matrix)
                avg_y   = [
                    np.mean([v[i] for v in path_matrix if i < len(v)])
                    for i in range(max_len)
                ]
                fig2.add_trace(go.Scatter(
                    x=list(range(len(avg_y))), y=avg_y,
                    name="━ 평균",
                    mode="lines",
                    line=dict(color="#fbbf24", width=3),
                    hovertemplate="<b>평균</b><br>경과일: %{x}<br>수익률: %{y:.2f}%<extra></extra>",
                ))

            # 0% 기준선
            fig2.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)

            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=520,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(
                    title="낙폭 후 경과일 (거래일 기준)",
                    showgrid=True, gridcolor="#1e2a3a",
                    tickfont=dict(size=11, color="#9ca3af"),
                ),
                yaxis=dict(
                    title="수익률 (%)",
                    showgrid=True, gridcolor="#1e2a3a",
                    tickfont=dict(size=11, color="#9ca3af"),
                    ticksuffix="%",
                ),
                showlegend=False,
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── 엑셀 다운로드 ──
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            fname_map = {"sp500": "SP500", "nasdaq": "NASDAQ", "kospi": "KOSPI", "dow": "DOW", "kosdaq": "KOSDAQ", "csi300": "CSI300"}
            fname = fname_map[key]

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                result_df.to_excel(writer, sheet_name="시뮬레이션 결과", index=False)
                if summary_rows:
                    pd.DataFrame(summary_rows).to_excel(
                        writer, sheet_name="요약 통계", index=False
                    )
                # 200일 경로 데이터
                if paths:
                    path_rows = []
                    for x_vals, y_vals, label, rank in paths:
                        for xi, yi in zip(x_vals, y_vals):
                            path_rows.append({"이벤트": label, "경과일": xi, "수익률(%)": round(yi, 4)})
                    pd.DataFrame(path_rows).to_excel(writer, sheet_name="200일 경로", index=False)

            st.download_button(
                label=f"📥 {fname} 엑셀 다운로드",
                data=buf.getvalue(),
                file_name=f"{fname}_낙폭시뮬레이션_상위{top_n}개_진입{entry_days}일.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{key}",
            )

        for tab, key in [(sim_s, "sp500"), (sim_n, "nasdaq"), (sim_k, "kospi"), (sim_d, "dow"), (sim_kq, "kosdaq"), (sim_cn, "csi300")]:
            with tab:
                render_sim_tab(key)

    # ════════════════════════════════════════
    # TAB 4: 시장 사이클
    # ════════════════════════════════════════
    with main_tab4:
        st.markdown('<div class="section-title">📈 시장 사이클 분석</div>', unsafe_allow_html=True)
        st.caption("직전 고점 대비 -20% 이하 → 하락장 시작 / 직전 저점 대비 +20% 이상 → 상승장 시작 (월가 Bull/Bear Market 표준 정의)")

        cyc_s, cyc_n, cyc_k, cyc_d, cyc_kq, cyc_cn = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"])

        def detect_cycles(prices: pd.Series):
            if len(prices) < 10:
                return pd.DataFrame()

            cycles = []
            state = "bull"
            cycle_start_date  = prices.index[0]
            cycle_start_price = prices.iloc[0]
            extreme_date  = prices.index[0]
            extreme_price = prices.iloc[0]

            for date, price in prices.items():
                if state == "bull":
                    if price >= extreme_price:
                        extreme_date  = date
                        extreme_price = price
                    elif price <= extreme_price * 0.80:
                        dur = (extreme_date - cycle_start_date).days
                        chg = (extreme_price - cycle_start_price) / cycle_start_price
                        cycles.append({
                            "구분": "🟢 상승장",
                            "시작일": cycle_start_date.strftime("%Y-%m-%d"),
                            "시작가": round(cycle_start_price, 2),
                            "종료일": extreme_date.strftime("%Y-%m-%d"),
                            "종료가": round(extreme_price, 2),
                            "기간(일)": dur,
                            "기간(월)": round(dur / 30.4, 1),
                            "변동률": f"+{chg*100:.1f}%",
                            "_chg": chg, "_type": "bull",
                            "_start": cycle_start_date, "_end": extreme_date,
                        })
                        state = "bear"
                        cycle_start_date  = extreme_date
                        cycle_start_price = extreme_price
                        extreme_date  = date
                        extreme_price = price
                else:
                    if price <= extreme_price:
                        extreme_date  = date
                        extreme_price = price
                    elif price >= extreme_price * 1.20:
                        dur = (extreme_date - cycle_start_date).days
                        chg = (extreme_price - cycle_start_price) / cycle_start_price
                        cycles.append({
                            "구분": "🔴 하락장",
                            "시작일": cycle_start_date.strftime("%Y-%m-%d"),
                            "시작가": round(cycle_start_price, 2),
                            "종료일": extreme_date.strftime("%Y-%m-%d"),
                            "종료가": round(extreme_price, 2),
                            "기간(일)": dur,
                            "기간(월)": round(dur / 30.4, 1),
                            "변동률": f"{chg*100:.1f}%",
                            "_chg": chg, "_type": "bear",
                            "_start": cycle_start_date, "_end": extreme_date,
                        })
                        state = "bull"
                        cycle_start_date  = extreme_date
                        cycle_start_price = extreme_price
                        extreme_date  = date
                        extreme_price = price

            # 현재 진행 중인 사이클
            last_date = prices.index[-1]
            dur = (extreme_date - cycle_start_date).days
            chg = (extreme_price - cycle_start_price) / cycle_start_price
            label = "🟢 상승장 (진행중)" if state == "bull" else "🔴 하락장 (진행중)"
            sign  = "+" if chg >= 0 else ""
            cycles.append({
                "구분": label,
                "시작일": cycle_start_date.strftime("%Y-%m-%d"),
                "시작가": round(cycle_start_price, 2),
                "종료일": extreme_date.strftime("%Y-%m-%d") + " ▶",
                "종료가": round(extreme_price, 2),
                "기간(일)": dur,
                "기간(월)": round(dur / 30.4, 1),
                "변동률": f"{sign}{chg*100:.1f}%",
                "_chg": chg, "_type": state,
                "_start": cycle_start_date, "_end": last_date,
            })
            return pd.DataFrame(cycles)

        def render_cycle_tab(key):
            prices = raw[key]
            if prices.empty:
                st.warning("데이터를 불러올 수 없습니다.")
                return

            cycles_df = detect_cycles(prices)
            if cycles_df.empty:
                st.warning("사이클 데이터를 계산할 수 없습니다.")
                return

            bull_df = cycles_df[cycles_df["_type"] == "bull"]
            bear_df = cycles_df[cycles_df["_type"] == "bear"]

            avg_bull_days = bull_df["기간(일)"].mean() if len(bull_df) else 0
            avg_bull_ret  = bull_df["_chg"].mean() * 100 if len(bull_df) else 0
            avg_bear_days = bear_df["기간(일)"].mean() if len(bear_df) else 0
            avg_bear_ret  = bear_df["_chg"].mean() * 100 if len(bear_df) else 0
            cur = cycles_df.iloc[-1]
            cur_label = "🟢 상승장" if cur["_type"] == "bull" else "🔴 하락장"
            cur_chg   = f"{'+' if cur['_chg']>=0 else ''}{cur['_chg']*100:.1f}%"
            cur_days  = cur["기간(일)"]

            # ── 사이클 요약 카드 ──
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("현재 상태", cur_label,
                          delta=f"{cur_chg}  ({cur_days}일 경과)")
            with m2:
                st.metric("🟢 상승장 횟수", f"{len(bull_df)}회")
            with m3:
                st.metric("🟢 평균 기간 / 상승률",
                          f"{avg_bull_days:.0f}일",
                          delta=f"+{avg_bull_ret:.1f}%")
            with m4:
                st.metric("🔴 하락장 횟수", f"{len(bear_df)}회")
            with m5:
                st.metric("🔴 평균 기간 / 하락률",
                          f"{avg_bear_days:.0f}일",
                          delta=f"{avg_bear_ret:.1f}%")

            # ── 신호 강도 계산 ──
            px2 = prices.copy()
            px2.index = pd.to_datetime(px2.index).tz_localize(None)

            drawdown  = (px2 - px2.cummax()) / px2.cummax() * 100
            delta     = px2.diff()
            gain      = delta.clip(lower=0).rolling(14).mean()
            loss      = (-delta.clip(upper=0)).rolling(14).mean()
            rsi_s     = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            ma200     = px2.rolling(200).mean()
            ma_gap    = (px2 - ma200) / ma200 * 100
            ma20      = px2.rolling(20).mean()
            std20     = px2.rolling(20).std()
            bb_up     = ma20 + 2 * std20
            bb_dn     = ma20 - 2 * std20
            bb_pct    = (px2 - bb_dn) / (bb_up - bb_dn) * 100

            if key in ("kospi", "kosdaq", "csi300"):
                vol   = px2.pct_change().rolling(20).std() * (252**0.5) * 100
                s_vol = (vol >= 35).astype(int)
            else:
                vix_h = load_vix_history()
                vol   = vix_h.reindex(px2.index, method="ffill") if not vix_h.empty else pd.Series(0, index=px2.index)
                s_vol = (vol >= 40).astype(int)

            sig_count = (
                (drawdown <= -30).astype(int) +
                (rsi_s     <= 30).astype(int) +
                (ma_gap    <= -15).astype(int) +
                (bb_pct    <= 0).astype(int) +
                s_vol
            ).fillna(0)

            # 하락장 구간 마스크
            in_bear = pd.Series(False, index=px2.index)
            for _, row in cycles_df.iterrows():
                if row["_type"] == "bear":
                    m = (px2.index >= row["_start"]) & (px2.index <= row["_end"])
                    in_bear[m] = True

            # DCA 구간: 하락장 + 신호 2개 이상
            dca_zone = in_bear & (sig_count >= 2)
            dca_periods = []
            in_p, st_d = False, None
            for dt, val in dca_zone.items():
                if val and not in_p:
                    st_d, in_p = dt, True
                elif not val and in_p:
                    dca_periods.append((st_d, dt))
                    in_p = False
            if in_p:
                dca_periods.append((st_d, px2.index[-1]))

            # ── 가격 + 음영 차트 ──
            # 매도 신호: 상승장에서 신호 ≥2  /  매수 신호: 하락장에서 신호 ≥4
            vol_label = "실현변동성≥35%" if key in ("kospi", "kosdaq", "csi300") else "VIX≥40"

            def _make_markers(cycle_type, thr):
                result = []
                for _, brow in cycles_df[cycles_df["_type"] == cycle_type].iterrows():
                    m = (sig_count.index >= brow["_start"]) & (sig_count.index <= brow["_end"])
                    p_sig = sig_count[m]
                    crossed = p_sig[p_sig >= thr]
                    if len(crossed) > 0:
                        dt = crossed.index[0]
                        sig_details = [
                            ("낙폭 ≤ -30%",  bool(drawdown.get(dt, 0)  <= -30)),
                            ("RSI ≤ 30",     bool(rsi_s.get(dt, 100)   <= 30)),
                            ("MA200 ≤ -15%", bool(ma_gap.get(dt, 0)    <= -15)),
                            ("BB %B ≤ 0",    bool(bb_pct.get(dt, 100)  <= 0)),
                            (vol_label,      bool(s_vol.get(dt, 0)     >= 1)),
                        ]
                        n_on = sum(v for _, v in sig_details)
                        hover = (
                            f"신호 {n_on}/5개 점등<br>"
                            + "<br>".join(f"{'✅' if v else '❌'} {name}" for name, v in sig_details)
                        )
                        result.append((dt, float(px2[dt]), hover))
                return result

            sell_markers = _make_markers("bear", 2)   # 하락장 초입(고점 근처) + 신호≥2 → 매도
            buy_markers  = _make_markers("bear", 4)   # 하락장 깊은 곳(저점 근처) + 신호≥4 → 매수

            st.markdown(
                '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:24px 0 12px;">'
                '📈 시장 사이클 + 매매 신호'
                '  <span style="color:#34d399;font-size:11px;">■ 상승장</span>'
                '  <span style="color:#f87171;font-size:11px;">■ 하락장</span>'
                '  <span style="color:#f87171;font-size:11px;">▼ 매도(신호≥2)</span>'
                '  <span style="color:#34d399;font-size:11px;">▲ 매수(신호≥4)</span></div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            # 상승/하락 음영
            for _, row in cycles_df.iterrows():
                fc = "rgba(52,211,153,0.13)" if row["_type"] == "bull" else "rgba(248,113,113,0.13)"
                fig.add_vrect(x0=row["_start"], x1=row["_end"],
                              fillcolor=fc, layer="below", line_width=0)

            # 가격선
            fig.add_trace(go.Scatter(
                x=px2.index, y=px2.values,
                mode="lines",
                line=dict(color="#60a5fa", width=1.5),
                name="종가",
                hovertemplate="%{x|%Y-%m-%d}  %{y:,.2f}<extra></extra>",
                showlegend=False,
            ))

            # 매도 마커 (▼ 빨간, 상승장 + 신호≥2)
            if sell_markers:
                sell_label = "▼ 매도 신호 (신호≥2)"
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in sell_markers],
                    y=[p[1] for p in sell_markers],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="#f87171",
                                size=14, line=dict(color="#0a0e1a", width=1)),
                    name=sell_label,
                    customdata=[p[2] for p in sell_markers],
                    hovertemplate=(
                        f"<b>{sell_label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "가격: %{y:,.2f}<br>"
                        "─────────────<br>"
                        "%{customdata}"
                        "<extra></extra>"
                    ),
                ))

            # 매수 마커 (▲ 초록, 하락장 + 신호≥4)
            if buy_markers:
                buy_label = "▲ 매수 신호 (신호≥4)"
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in buy_markers],
                    y=[p[1] for p in buy_markers],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#34d399",
                                size=14, line=dict(color="#0a0e1a", width=1)),
                    name=buy_label,
                    customdata=[p[2] for p in buy_markers],
                    hovertemplate=(
                        f"<b>{buy_label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "가격: %{y:,.2f}<br>"
                        "─────────────<br>"
                        "%{customdata}"
                        "<extra></extra>"
                    ),
                ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=500,
                margin=dict(l=0, r=0, t=60, b=0),
                xaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                           tickfont=dict(size=11, color="#9ca3af"),
                           rangeslider=dict(visible=False),
                           rangeselector=dict(
                               buttons=[
                                   dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                                   dict(count=10, label="10Y", step="year", stepmode="backward"),
                                   dict(count=20, label="20Y", step="year", stepmode="backward"),
                                   dict(step="all", label="전체"),
                               ],
                               bgcolor="#1e2a3a", activecolor="#3b82f6",
                               font=dict(color="#9ca3af", size=11),
                               y=1.12,
                           )),
                yaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                           tickfont=dict(size=11, color="#9ca3af")),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=11, color="#d1d5db"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── 저점 신호 강도 차트 ──
            st.markdown(
                '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:12px 0 4px;">'
                '📊 저점 신호 동시 점등 개수 (0~5개)</div>',
                unsafe_allow_html=True,
            )
            st.caption("낙폭·RSI·MA200·볼린저밴드·변동성 5개 지표 중 동시에 저점 기준을 충족하는 개수 | 🟡 노란선(2개) 이상 + 하락장 = 분할매수 구간")

            fig2 = go.Figure()

            # 색상 구간 배경
            fig2.add_hrect(y0=0, y1=2, fillcolor="rgba(52,211,153,0.06)", line_width=0)
            fig2.add_hrect(y0=2, y1=4, fillcolor="rgba(251,191,36,0.06)", line_width=0)
            fig2.add_hrect(y0=4, y1=5.5, fillcolor="rgba(248,113,113,0.06)", line_width=0)

            # 면적 차트
            fig2.add_trace(go.Scatter(
                x=sig_count.index, y=sig_count.values,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#60a5fa", width=1),
                fillcolor="rgba(96,165,250,0.15)",
                hovertemplate="%{x|%Y-%m-%d}  신호 %{y}개/5개<extra></extra>",
            ))

            # 기준선
            fig2.add_hline(y=2, line_dash="dot", line_color="#f87171", line_width=1.5,
                           annotation_text="매도 신호 기준", annotation_position="right",
                           annotation_font=dict(color="#f87171", size=10))
            fig2.add_hline(y=4, line_dash="dot", line_color="#34d399", line_width=1.5,
                           annotation_text="매수 신호 기준", annotation_position="right",
                           annotation_font=dict(color="#34d399", size=10))

            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=220,
                margin=dict(l=0, r=80, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                           tickfont=dict(size=10, color="#9ca3af"),
                           rangeselector=dict(
                               buttons=[
                                   dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                                   dict(count=10, label="10Y", step="year", stepmode="backward"),
                                   dict(step="all", label="전체"),
                               ],
                               bgcolor="#1e2a3a", activecolor="#3b82f6",
                               font=dict(color="#9ca3af", size=10),
                           )),
                yaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                           tickfont=dict(size=10, color="#9ca3af"),
                           range=[0, 5.5], dtick=1),
                showlegend=False,
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── 전략6 월별 포트폴리오 가치 계산 (사이클 기간 매핑용) ──
            bond_key  = "krbond" if key in ("kospi", "kosdaq") else "us30y"
            # csi300: 중국 채권 데이터 없으므로 미국채10년 대용 (bond_key = "us30y" 유지)
            stk_m     = monthly[key]
            gold_m    = monthly["gold"]
            if bond_key == "us30y":
                bond_m = (-DURATION_US * raw["us30y"].resample("ME").last().diff() / 100).dropna()
            else:
                bond_m = monthly["krbond"]

            inv_na_s = strategies[key]["inv_na"]
            inv_mo_s = strategies[key]["inv_mo"]

            _v   = 100.0
            _s6v = {}
            for _d in stk_m.index:
                _m = _d.month
                _sy = _d.year - 1 if _m in [1, 2, 3, 4] else _d.year
                _season  = "Nov-Apr" if _m in [11, 12, 1, 2, 3, 4] else "May-Oct"
                _inv_set = inv_na_s if _season == "Nov-Apr" else inv_mo_s
                _invested = (_sy % 10) in _inv_set
                _rs = float(stk_m[_d]) if _d in stk_m.index else 0.0
                _rg = float(gold_m[_d]) if _d in gold_m.index else 0.0
                _rb = float(bond_m[_d]) if _d in bond_m.index else 0.0
                _r  = (0.50*_rs + 0.25*_rg + 0.25*_rb) if _invested \
                      else (0.25*_rs + 0.25*_rg + 0.25*_rb)
                _v *= (1 + _r)
                _s6v[_d] = _v
            s6_monthly = pd.Series(_s6v)

            def s6_period_ret(start, end):
                """사이클 기간의 전략6 수익률(%) 반환"""
                try:
                    sv = s6_monthly.asof(pd.Timestamp(start))
                    ev = s6_monthly.asof(pd.Timestamp(end))
                    if pd.isna(sv) or pd.isna(ev) or sv == 0:
                        return None
                    return (ev / sv - 1) * 100
                except Exception:
                    return None

            # ── 사이클 테이블 (최신순) ──
            st.markdown(
                '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:24px 0 12px;">'
                '📋 사이클 목록 (최신순)</div>',
                unsafe_allow_html=True,
            )

            # 전략6 성과 컬럼 추가
            cycles_df["전략6(%)"] = cycles_df.apply(
                lambda r: s6_period_ret(r["_start"], r["_end"]), axis=1
            )
            cycles_df["전략6(%)"] = cycles_df["전략6(%)"].apply(
                lambda v: f"{v:+.1f}%" if v is not None and not pd.isna(v) else "—"
            )

            disp = cycles_df[["구분","시작일","시작가","종료일","종료가","기간(일)","기간(월)","변동률","전략6(%)"]].copy()
            disp = disp.iloc[::-1].reset_index(drop=True)
            disp.insert(0, "번호", range(1, len(disp) + 1))

            def style_cycle(df):
                def _c(val):
                    s = str(val)
                    if "상승" in s:  return "color:#34d399;font-weight:700"
                    if "하락" in s:  return "color:#f87171;font-weight:700"
                    if s.startswith("+"): return "color:#34d399;font-weight:600"
                    if s.startswith("-"): return "color:#f87171;font-weight:600"
                    return "color:#d1d5db"
                return df.style.map(_c)

            st.dataframe(
                style_cycle(disp),
                use_container_width=True,
                height=min(100 + len(disp) * 36, 820),
                hide_index=True,
            )

        for tab, key in [(cyc_s, "sp500"), (cyc_n, "nasdaq"), (cyc_k, "kospi"), (cyc_d, "dow"), (cyc_kq, "kosdaq"), (cyc_cn, "csi300")]:
            with tab:
                render_cycle_tab(key)


    # ════════════════════════════════════════
    # TAB 5: 저점 레이더
    # ════════════════════════════════════════
    with main_tab5:
        st.markdown('<div class="section-title">📡 저점 레이더 — 역사적 저점 신호 모니터링</div>', unsafe_allow_html=True)
        st.caption("각 지표가 역사적 저점 구간 기준을 충족하는지 모니터링합니다. ✏️ 표시 항목은 직접 입력하세요.")

        # ── 기술적 지표 계산 ──
        def calc_tech(key):
            prices = raw[key]
            if prices.empty or len(prices) < 200:
                return {}
            cur = float(prices.iloc[-1])

            ath = float(prices.max())
            drawdown = round((cur - ath) / ath * 100, 1)

            delta = prices.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            rsi   = round(100 - (100 / (1 + rs)), 1)

            ma200  = float(prices.rolling(200).mean().iloc[-1])
            ma_gap = round((cur - ma200) / ma200 * 100, 1)

            ma20   = float(prices.rolling(20).mean().iloc[-1])
            std20  = float(prices.rolling(20).std().iloc[-1])
            bb_up  = ma20 + 2 * std20
            bb_dn  = ma20 - 2 * std20
            bb_pct = round((cur - bb_dn) / (bb_up - bb_dn) * 100, 1) if (bb_up - bb_dn) != 0 else 50.0

            # 실현변동성: 20일 일별 수익률 표준편차 연율화 (VKOSPI 대용)
            rvol = round(float(prices.pct_change().rolling(20).std().iloc[-1]) * (252 ** 0.5) * 100, 1)

            return {"drawdown": drawdown, "rsi": rsi, "ma_gap": ma_gap, "bb_pct": bb_pct, "rvol": rvol}

        # ── 신호 판정 ──
        def get_sig(value, threshold, direction="below"):
            """(아이콘, 색상, 신호여부)"""
            if value is None:
                return "⚪", "#4b5563", False
            margin = abs(threshold) * 0.2
            if direction == "below":
                is_sig  = value <= threshold
                is_warn = (not is_sig) and value <= threshold + margin
            else:
                is_sig  = value >= threshold
                is_warn = (not is_sig) and value >= threshold - margin
            if is_sig:  return "🔴", "#f87171", True
            if is_warn: return "🟡", "#fbbf24", False
            return "🟢", "#34d399", False

        # ── 카드 HTML ──
        def ind_card(title, value, fmt, threshold, thr_label, direction, desc, is_manual=False):
            icon, color, _ = get_sig(value, threshold, direction)
            if value is None:
                val_str = "—"
            elif fmt == "pct":
                val_str = f"{value:+.1f}%"
            else:
                val_str = f"{value:.1f}"
            manual_tag = ' <span style="color:#6b7280;font-size:10px;">✏️</span>' if is_manual else ""
            return f"""<div style="background:#111827;border:1.5px solid {color}55;
                border-radius:12px;padding:16px 12px;text-align:center;min-height:140px;">
  <div style="font-size:20px;margin-bottom:4px;">{icon}</div>
  <div style="color:#9ca3af;font-size:11px;margin-bottom:4px;">{title}{manual_tag}</div>
  <div style="color:{color};font-size:22px;font-weight:700;line-height:1.1;">{val_str}</div>
  <div style="color:#4b5563;font-size:10px;margin-top:6px;">기준: {thr_label}</div>
  <div style="color:#374151;font-size:10px;margin-top:3px;">{desc}</div>
</div>"""

        # ── 외부 데이터 로드 ──
        with st.spinner("외부 지표 수집 중..."):
            vix_val   = load_vix_now()
            fg_val    = load_fear_greed()
            cape_val  = load_multpl("https://www.multpl.com/shiller-pe")
            sp_per    = load_multpl("https://www.multpl.com/s-p-500-pe-ratio")

        # ── 섹션1: 지수별 요약 (클릭 선택) ──
        tech_keys = ["sp500", "nasdaq", "kospi", "dow", "kosdaq", "csi300"]
        tech_all  = {k: calc_tech(k) for k in tech_keys}

        def index_signals(key, extra_indicators):
            t = tech_all[key]
            sigs = []
            for v, thr, d in [
                (t.get("drawdown"), -30, "below"),
                (t.get("rsi"),      30,  "below"),
                (t.get("ma_gap"),  -15,  "below"),
                (t.get("bb_pct"),    0,  "below"),
            ]:
                _, _, s = get_sig(v, thr, d)
                sigs.append(s)
            for v, thr, d in extra_indicators:
                _, _, s = get_sig(v, thr, d)
                sigs.append(s)
            return sigs

        sp500_sigs  = index_signals("sp500",  [(vix_val, 40, "above"), (fg_val, 25, "below"),
                                                (cape_val, 15, "below"), (sp_per, 15, "below")])
        nasdaq_sigs = index_signals("nasdaq", [(vix_val, 40, "above"), (fg_val, 25, "below")])
        dow_sigs    = index_signals("dow",    [(vix_val, 40, "above"), (fg_val, 25, "below"),
                                                (cape_val, 15, "below"), (sp_per, 15, "below")])
        kospi_rvol  = tech_all["kospi"].get("rvol")
        kospi_sigs  = index_signals("kospi",  [(kospi_rvol, 35, "above"), (fg_val, 25, "below")])
        kosdaq_rvol = tech_all["kosdaq"].get("rvol")
        kosdaq_sigs = index_signals("kosdaq", [(kosdaq_rvol, 35, "above"), (fg_val, 25, "below")])
        csi300_rvol = tech_all["csi300"].get("rvol")
        csi300_sigs = index_signals("csi300", [(csi300_rvol, 35, "above"), (fg_val, 25, "below")])
        sigs_map = {"sp500": sp500_sigs, "nasdaq": nasdaq_sigs, "kospi": kospi_sigs, "dow": dow_sigs, "kosdaq": kosdaq_sigs, "csi300": csi300_sigs}

        def summary_card_html(label, sigs, selected=False):
            n = sum(sigs)
            total = len(sigs)
            color  = "#f87171" if n >= 4 else "#fbbf24" if n >= 2 else "#34d399"
            icon   = "🔴" if n >= 4 else "🟡" if n >= 2 else "🟢"
            msg    = "저점 신호 다수" if n >= 4 else "주의 구간" if n >= 2 else "정상 구간"
            border = f"2.5px solid {color}" if selected else f"1.5px solid {color}44"
            bg     = "#1a2235" if selected else "#111827"
            return f"""<div style="background:{bg};border:{border};
                border-radius:12px;padding:18px 16px;text-align:center;">
  <div style="color:#9ca3af;font-size:12px;margin-bottom:6px;">{label}</div>
  <div style="font-size:32px;line-height:1;">{icon}</div>
  <div style="color:{color};font-size:18px;font-weight:700;margin-top:6px;">{n} / {total}개 점등</div>
  <div style="color:#6b7280;font-size:11px;margin-top:4px;">{msg}</div>
</div>"""

        # 선택 상태 (session state)
        if "radar_sel" not in st.session_state:
            st.session_state["radar_sel"] = "sp500"

        idx_list = [
            ("sp500",  "🇺🇸 S&P500"),
            ("nasdaq", "💻 NASDAQ"),
            ("kospi",  "🇰🇷 KOSPI"),
            ("dow",    "🏛️ DOW"),
            ("kosdaq", "📱 KOSDAQ"),
            ("csi300", "🇨🇳 CSI300"),
        ]

        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        for col, (key, label) in zip([sc1, sc2, sc3, sc4, sc5, sc6], idx_list):
            with col:
                selected = st.session_state["radar_sel"] == key
                st.markdown(summary_card_html(label, sigs_map[key], selected), unsafe_allow_html=True)
                if st.button("▼ 지표 보기" if not selected else "✅ 선택됨",
                             key=f"sel_{key}", use_container_width=True):
                    st.session_state["radar_sel"] = key
                    st.rerun()

        # ── 섹션2: 선택된 지수의 기술적 지표 ──
        sel_key   = st.session_state["radar_sel"]
        sel_label = dict(idx_list)[sel_key]
        t = tech_all[sel_key]

        st.markdown(
            f'<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
            f'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:20px 0 14px;">'
            f'📊 기술적 지표 — {sel_label}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(ind_card(
                "고점 대비 낙폭", t.get("drawdown"), "pct",
                -30, "-30% 이하", "below", "역사적 ATH 기준 현재 낙폭"
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(ind_card(
                "RSI (14일)", t.get("rsi"), "num",
                30, "30 이하", "below", "과매도 구간 진입 여부"
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(ind_card(
                "200일 MA 괴리율", t.get("ma_gap"), "pct",
                -15, "-15% 이하", "below", "장기 추세 대비 이탈 수준"
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(ind_card(
                "볼린저밴드 %B", t.get("bb_pct"), "pct",
                0, "0% 이하 (하단 이탈)", "below", "밴드 하단 이탈 시 과매도"
            ), unsafe_allow_html=True)

        # ── 섹션3: 시장 전반 지표 (지수별 구성 다름) ──
        st.markdown(
            '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
            'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:24px 0 14px;">'
            '🌐 시장 전반 지표</div>',
            unsafe_allow_html=True,
        )

        if sel_key in ("sp500", "nasdaq", "dow"):
            # S&P500 / NASDAQ / DOW / CSI300: VIX, Fear&Greed, CAPE, S&P500 PER
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.markdown(ind_card(
                    "VIX 공포지수", vix_val, "num",
                    40, "40 이상", "above", "시장 변동성·공포 지수"
                ), unsafe_allow_html=True)
            with g2:
                st.markdown(ind_card(
                    "Fear & Greed", fg_val, "num",
                    25, "25 이하 (극단공포)", "below", "CNN 공포탐욕지수 (0~100)"
                ), unsafe_allow_html=True)
            with g3:
                st.markdown(ind_card(
                    "CAPE (Shiller P/E)", cape_val, "num",
                    15, "15 이하", "below", "S&P500 물가조정 장기 PER"
                ), unsafe_allow_html=True)
            with g4:
                st.markdown(ind_card(
                    "S&P500 PER", sp_per, "num",
                    15, "15 이하", "below", "S&P500 현재 주가수익비율"
                ), unsafe_allow_html=True)

        else:  # kospi / kosdaq / csi300
            # KOSPI / KOSDAQ / CSI300: 실현변동성, Fear&Greed, PBR
            rvol_val = tech_all[sel_key].get("rvol")
            g1, g2, g3 = st.columns(3)
            _kr_name = "KOSDAQ" if sel_key == "kosdaq" else ("CSI300" if sel_key == "csi300" else "KOSPI")
            with g1:
                st.markdown(ind_card(
                    f"실현변동성 (V{_kr_name} 대용)", rvol_val, "pct",
                    35, "35% 이상", "above", f"{_kr_name} 20일 수익률 변동성 연율화"
                ), unsafe_allow_html=True)
            with g2:
                st.markdown(ind_card(
                    "Fear & Greed", fg_val, "num",
                    25, "25 이하 (극단공포)", "below", "CNN 공포탐욕지수 (참고용)"
                ), unsafe_allow_html=True)
            with g3:
                ks_pbr = st.number_input(
                    f"{_kr_name} PBR 직접 입력",
                    min_value=0.0, max_value=10.0, value=0.0,
                    step=0.01, key=f"ks_pbr_{sel_key}",
                )
                ks_pbr_val = ks_pbr if ks_pbr > 0 else None
                st.markdown(ind_card(
                    f"{_kr_name} PBR", ks_pbr_val, "num",
                    1.0, "1.0 이하", "below", f"{_kr_name} 주가순자산비율",
                    is_manual=True
                ), unsafe_allow_html=True)

        # ── 섹션4: 일별 지표 데이터 테이블 ──
        st.markdown(
            f'<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
            f'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:24px 0 14px;">'
            f'📅 일별 지표 데이터 — {sel_label}</div>',
            unsafe_allow_html=True,
        )

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            period = st.select_slider(
                "표시 기간",
                options=[30, 60, 90, 120],
                value=60,
                key="daily_period",
            )
        with pc2:
            date_col1, date_col2 = st.columns(2)
            _daily_min = date(1997, 1, 1) if sel_key == "kosdaq" else (date(2005, 1, 4) if sel_key == "csi300" else date(1985, 1, 1))
            with date_col1:
                custom_start = st.date_input(
                    "시작일 (직접 설정)",
                    value=None,
                    min_value=_daily_min,
                    max_value=date.today(),
                    key="daily_start",
                )
            with date_col2:
                custom_end = st.date_input(
                    "종료일",
                    value=date.today(),
                    min_value=_daily_min,
                    max_value=date.today(),
                    key="daily_end",
                )

        @st.cache_data(ttl=3600, show_spinner=False)
        def calc_daily_df(key):
            prices = raw[key]
            if prices.empty or len(prices) < 200:
                return pd.DataFrame()

            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            df = prices.to_frame(name="_close")

            # 고점 대비 낙폭
            df["고점대비낙폭(%)"] = (prices - prices.cummax()) / prices.cummax() * 100

            # RSI 14
            delta = prices.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            df["RSI"] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).round(1)

            # 200일 MA 괴리율
            ma200 = prices.rolling(200).mean()
            df["MA200 괴리율(%)"] = ((prices - ma200) / ma200 * 100).round(1)

            # 볼린저밴드 %B
            ma20  = prices.rolling(20).mean()
            std20 = prices.rolling(20).std()
            bb_up = ma20 + 2 * std20
            bb_dn = ma20 - 2 * std20
            df["볼린저밴드%B"] = ((prices - bb_dn) / (bb_up - bb_dn) * 100).round(1)

            # VIX or 실현변동성
            if key in ("kospi", "kosdaq", "csi300"):
                df["실현변동성(%)"] = (prices.pct_change().rolling(20).std() * (252**0.5) * 100).round(1)
            else:
                vix_h = load_vix_history()
                if not vix_h.empty:
                    df["VIX"] = vix_h.reindex(df.index, method="ffill").round(1)
                else:
                    df["VIX"] = np.nan

            return df.dropna(subset=["RSI", "MA200 괴리율(%)", "볼린저밴드%B"])

        daily_df = calc_daily_df(sel_key)

        if not daily_df.empty:
            vol_col = "실현변동성(%)" if sel_key in ("kospi", "kosdaq", "csi300") else "VIX"
            vol_thr = 35 if sel_key in ("kospi", "kosdaq", "csi300") else 40
            n_total_sigs = 5  # 낙폭, RSI, MA200, 볼린저, VIX/실현변동성

            # 날짜 범위 필터
            if custom_start:
                start_ts = pd.Timestamp(custom_start)
                end_ts   = pd.Timestamp(custom_end)
                disp = daily_df.loc[
                    (daily_df.index >= start_ts) & (daily_df.index <= end_ts)
                ].iloc[::-1].copy()
            else:
                disp = daily_df.tail(period).iloc[::-1].copy()

            # 신호 카운트 컬럼 (숫자 포함)
            def count_sigs(row):
                sigs = [
                    bool(row["고점대비낙폭(%)"] <= -30),
                    bool(row["RSI"] <= 30),
                    bool(row["MA200 괴리율(%)"] <= -15),
                    bool(row["볼린저밴드%B"] <= 0),
                    bool(row.get(vol_col, 0) >= vol_thr),
                ]
                n = sum(sigs)
                icon = "🔴" if n >= 4 else "🟡" if n >= 2 else "🟢"
                return f"{icon} {n}/{n_total_sigs}"

            disp["신호"] = disp.apply(count_sigs, axis=1)

            # 표시용 컬럼 정리
            disp.index = disp.index.strftime("%Y-%m-%d")
            disp.index.name = "날짜"
            disp["종가"] = disp["_close"].apply(lambda x: f"{x:,.2f}")
            disp = disp[["신호", "종가", "고점대비낙폭(%)", "RSI",
                          "MA200 괴리율(%)", "볼린저밴드%B", vol_col]]
            disp["고점대비낙폭(%)"] = disp["고점대비낙폭(%)"].round(1)

            def style_daily(df):
                def _c(val, col):
                    if not isinstance(val, (int, float)) or pd.isna(val):
                        return "color:#6b7280"
                    if col == "고점대비낙폭(%)":
                        if val <= -30: return "color:#f87171;font-weight:700"
                        if val <= -24: return "color:#fbbf24"
                        return "color:#34d399"
                    if col == "RSI":
                        if val <= 30: return "color:#f87171;font-weight:700"
                        if val <= 36: return "color:#fbbf24"
                        return "color:#d1d5db"
                    if col == "MA200 괴리율(%)":
                        if val <= -15: return "color:#f87171;font-weight:700"
                        if val <= -12: return "color:#fbbf24"
                        return "color:#34d399"
                    if col == "볼린저밴드%B":
                        if val <= 0:  return "color:#f87171;font-weight:700"
                        if val <= 20: return "color:#fbbf24"
                        return "color:#d1d5db"
                    if col in ("VIX", "실현변동성(%)"):
                        if val >= vol_thr:         return "color:#f87171;font-weight:700"
                        if val >= vol_thr * 0.8:   return "color:#fbbf24"
                        return "color:#d1d5db"
                    return "color:#d1d5db"

                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for col in df.columns:
                    styles[col] = [_c(v, col) for v in df[col]]
                return styles

            st.dataframe(
                disp.style.apply(style_daily, axis=None),
                use_container_width=True,
                height=min(100 + len(disp) * 36, 900),
            )
        else:
            st.warning("데이터를 불러올 수 없습니다.")

        # ── 섹션5: 지표 설명 ──
        with st.expander("📖 지표 설명 및 저점 기준 근거"):
            st.markdown("""
| 지표 | 저점 기준 | 근거 |
|------|-----------|------|
| **고점 대비 낙폭** | -30% 이하 | 역사적 Bear Market 진입 기준 (월가 표준) |
| **RSI (14일)** | 30 이하 | 과매도 구간, 기술적 반등 가능성 |
| **200일 MA 괴리율** | -15% 이하 | 장기 추세 대비 극단적 이탈 구간 |
| **볼린저밴드 %B** | 0% 이하 | 통계적 하단 이탈 (2σ 밖) |
| **VIX** | 40 이상 | 시장 극단적 공포 (2008년 80, 코로나 85) |
| **Fear & Greed** | 25 이하 | CNN 극단적 공포 구간 |
| **CAPE** | 15 이하 | 역사적 저평가 구간 (장기 평균 17) |
| **S&P500 PER** | 15 이하 | 역사적 저평가 구간 |
| **KOSPI PBR** | 1.0 이하 | 순자산 이하 거래 = 극단적 저평가 |
""")


    # ════════════════════════════════════════
    # TAB 6: 끝자리 사이클 누적 수익률
    # ════════════════════════════════════════
    with main_tab6:
        st.markdown('<div class="section-title">📅 끝자리 사이클 누적 수익률 분석</div>', unsafe_allow_html=True)
        st.caption("연도 끝자리(0~9)별 평균 수익률을 복리 누적하여 10년 사이클 패턴을 확인합니다.")
        st.caption("예) 끝자리 0 = 1990·2000·2010·2020년 평균 / 끝자리 1 = 1991·2001·2011·2021년 평균 ...")

        @st.cache_data(ttl=3600, show_spinner=False)
        def calc_digit_cycle(key):
            prices = raw[key]
            if prices.empty:
                return pd.DataFrame(), pd.DataFrame()
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            # 연말 종가 → 연간 수익률
            annual = prices.resample("YE").last()
            ann_ret = annual.pct_change().dropna()
            years   = ann_ret.index.year

            # 끝자리별 해당 연도 목록 및 평균 수익률
            rows = []
            for digit in range(10):
                yr_list  = [y for y in years if y % 10 == digit]
                yr_rets  = [(y, ann_ret[ann_ret.index.year == y].values[0] * 100)
                            for y in yr_list
                            if len(ann_ret[ann_ret.index.year == y]) > 0]
                rets     = [r for _, r in yr_rets]
                avg      = round(np.mean(rets), 2) if rets else np.nan
                up_cnt   = sum(1 for r in rets if r > 0)
                rows.append({
                    "끝자리":    digit,
                    "해당연도":  "·".join(str(y) for y in yr_list),
                    "연도수익률": yr_rets,   # [(year, ret%), ...]
                    "평균수익률": avg,
                    "데이터수":  len(rets),
                    "상승수":    up_cnt,
                })
            digit_df = pd.DataFrame(rows)

            # 복리 누적 수익률 (끝자리 0부터 9까지 순서대로)
            cum = 100.0
            cum_rows = []
            for _, r in digit_df.iterrows():
                if pd.notna(r["평균수익률"]):
                    cum *= (1 + r["평균수익률"] / 100)
                cum_rows.append({
                    "끝자리":    int(r["끝자리"]),
                    "평균수익률": r["평균수익률"],
                    "누적수익률": round(cum - 100, 2),
                    "해당연도":  r["해당연도"],
                    "연도수익률": r["연도수익률"],
                    "데이터수":  int(r["데이터수"]),
                    "상승수":    int(r["상승수"]),
                })
            cum_df = pd.DataFrame(cum_rows)
            return digit_df, cum_df

        def render_digit_tab(key):
            _, cum_df = calc_digit_cycle(key)
            if cum_df.empty:
                st.warning("데이터를 불러올 수 없습니다.")
                return

            # ── 요약 카드 ──
            best_row  = cum_df.loc[cum_df["평균수익률"].idxmax()]
            worst_row = cum_df.loc[cum_df["평균수익률"].idxmin()]
            total_cum = cum_df["누적수익률"].iloc[-1]
            avg_ann   = cum_df["평균수익률"].mean()

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("10년 사이클 누적 수익률", f"{total_cum:+.1f}%")
            with m2: st.metric("끝자리별 평균 연수익률", f"{avg_ann:+.2f}%")
            with m3: st.metric("최고 끝자리", f"끝자리 {int(best_row['끝자리'])}년  {best_row['평균수익률']:+.1f}%")
            with m4: st.metric("최저 끝자리", f"끝자리 {int(worst_row['끝자리'])}년  {worst_row['평균수익률']:+.1f}%")

            # ── 차트 ──
            st.markdown(
                '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:20px 0 12px;">'
                '📈 끝자리 사이클 누적 수익률 (복리)  '
                '<span style="color:#a78bfa;font-size:11px;">━ 누적수익률</span>'
                '  <span style="color:#60a5fa;font-size:11px;">━ 끝자리별 평균수익률</span></div>',
                unsafe_allow_html=True,
            )

            x_labels = [f"{d}년\n({str(yrs)[:4]}...)" for d, yrs in zip(cum_df["끝자리"], cum_df["해당연도"])]
            x_ticks  = [f"끝자리 {d}" for d in cum_df["끝자리"]]

            fig = go.Figure()

            # 누적수익률 꺽은선 (주축)
            fig.add_trace(go.Scatter(
                x=cum_df["끝자리"], y=cum_df["누적수익률"],
                mode="lines+markers",
                line=dict(color="#a78bfa", width=3),
                marker=dict(
                    size=10,
                    color=["#34d399" if v >= 0 else "#f87171" for v in cum_df["누적수익률"]],
                    line=dict(color="#0a0e1a", width=1.5),
                ),
                name="누적수익률",
                yaxis="y1",
                hovertemplate="끝자리 %{x}년<br>누적수익률: %{y:+.1f}%<extra></extra>",
            ))

            # 끝자리별 평균수익률 꺽은선 (보조축)
            fig.add_trace(go.Scatter(
                x=cum_df["끝자리"], y=cum_df["평균수익률"],
                mode="lines+markers",
                line=dict(color="#60a5fa", width=2, dash="dot"),
                marker=dict(
                    size=8,
                    color=["#34d399" if v >= 0 else "#f87171" for v in cum_df["평균수익률"]],
                    line=dict(color="#0a0e1a", width=1),
                ),
                name="끝자리별 평균수익률",
                yaxis="y2",
                hovertemplate="끝자리 %{x}년<br>평균수익률: %{y:+.1f}%<extra></extra>",
            ))

            fig.add_hline(y=0, line_color="#374151", line_width=1, yref="y2")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=460,
                margin=dict(l=0, r=60, t=20, b=0),
                xaxis=dict(
                    showgrid=True, gridcolor="#1e2a3a",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=x_ticks,
                    tickfont=dict(size=11, color="#9ca3af"),
                ),
                yaxis=dict(
                    title="누적수익률 (%)",
                    showgrid=True, gridcolor="#1e2a3a",
                    tickfont=dict(size=11, color="#9ca3af"),
                    ticksuffix="%",
                    title_font=dict(color="#a78bfa"),
                ),
                yaxis2=dict(
                    title="평균수익률 (%)",
                    overlaying="y", side="right",
                    showgrid=False,
                    tickfont=dict(size=11, color="#60a5fa"),
                    ticksuffix="%",
                    title_font=dict(color="#60a5fa"),
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color="#d1d5db"), bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── 테이블 ──
            st.markdown(
                '<div style="color:#5b9bd5;font-size:13px;font-weight:700;'
                'border-bottom:1px solid #1e2a3a;padding-bottom:6px;margin:20px 0 12px;">'
                '📋 끝자리별 상세 데이터</div>',
                unsafe_allow_html=True,
            )
            def _val_color(v):
                if pd.isna(v):   return "#d1d5db"
                return "#34d399" if v >= 0 else "#f87171"

            def _yr_cell(yr_rets):
                parts = []
                for y, r in yr_rets:
                    c   = "#34d399" if r >= 0 else "#f87171"
                    sgn = "+" if r >= 0 else ""
                    parts.append(
                        f'<span style="color:{c};white-space:nowrap">'
                        f'{y}({sgn}{r:.0f}%)</span>'
                    )
                return " &nbsp;·&nbsp; ".join(parts)

            html_rows = []
            for _, r in cum_df.iterrows():
                avg_c  = _val_color(r["평균수익률"])
                cum_c  = _val_color(r["누적수익률"])
                avg_s  = f'{r["평균수익률"]:+.2f}%' if pd.notna(r["평균수익률"]) else "—"
                cum_s  = f'{r["누적수익률"]:+.2f}%'
                sample = f'{int(r["상승수"])}/{int(r["데이터수"])}'
                yr_html = _yr_cell(r["연도수익률"])
                html_rows.append(
                    f'<tr>'
                    f'<td style="padding:7px 12px;color:#d1d5db;white-space:nowrap">끝자리 {int(r["끝자리"])}년</td>'
                    f'<td style="padding:7px 12px;color:{avg_c};font-weight:600;text-align:right">{avg_s}</td>'
                    f'<td style="padding:7px 12px;color:{cum_c};font-weight:600;text-align:right">{cum_s}</td>'
                    f'<td style="padding:7px 16px;font-size:12px;line-height:1.8">{yr_html}</td>'
                    f'<td style="padding:7px 12px;color:#9ca3af;text-align:center">{sample}</td>'
                    f'</tr>'
                )

            th = ("끝자리", "평균수익률", "누적수익률", "해당연도 (수익률)", "상승/표본수")
            thead = "".join(
                f'<th style="padding:8px 12px;color:#5b9bd5;font-weight:700;'
                f'border-bottom:1px solid #1e2a3a;text-align:{"right" if i in (1,2) else "center" if i==4 else "left"}">'
                f'{h}</th>'
                for i, h in enumerate(th)
            )
            html_tbl = (
                '<div style="overflow-x:auto">'
                '<table style="width:100%;border-collapse:collapse;font-size:13px">'
                f'<thead><tr>{thead}</tr></thead>'
                f'<tbody>{"".join(html_rows)}</tbody>'
                '</table></div>'
            )
            st.markdown(html_tbl, unsafe_allow_html=True)

        ann_s, ann_n, ann_k, ann_d, ann_kq, ann_cn = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"])
        for tab, key in [(ann_s, "sp500"), (ann_n, "nasdaq"), (ann_k, "kospi"), (ann_d, "dow"), (ann_kq, "kosdaq"), (ann_cn, "csi300")]:
            with tab:
                render_digit_tab(key)

    # ════════════════════════════════════════
    # TAB 7: 급락 패턴 분석
    # ════════════════════════════════════════
    with main_tab7:
        st.markdown('<div class="section-title">⚡ 급락 패턴 분석 — 낙폭 속도와 이후 수익률</div>', unsafe_allow_html=True)
        st.caption("역사적 하락 사이클을 '낙폭 속도' 기준으로 분류하고 저점 이후 수익률 패턴을 분석합니다 · S&P500 기준 (1970~현재)")

        # ── 사이클 탐지 ──
        @st.cache_data(ttl=3600)
        def _detect_crash_cycles(prices_tuple):
            px = pd.Series(dict(prices_tuple)).sort_index()
            px.index = pd.to_datetime(px.index).tz_localize(None)
            cycles = []
            peak_date = px.index[0]; peak_val = float(px.iloc[0])
            trough_date = px.index[0]; trough_val = float(px.iloc[0])
            in_dd = False
            for dt in px.index[1:]:
                val = float(px[dt])
                if not in_dd:
                    if val > peak_val:
                        peak_val = val; peak_date = dt
                    elif (val - peak_val) / peak_val * 100 <= -10:
                        in_dd = True; trough_date = dt; trough_val = val
                else:
                    if val < trough_val:
                        trough_date = dt; trough_val = val
                    if ((val-trough_val)/trough_val*100 >= 15 or
                            (val-peak_val)/peak_val*100 >= -5):
                        depth = (trough_val - peak_val) / peak_val * 100
                        dur   = (trough_date - peak_date).days
                        if depth <= -12:
                            fwd = {}
                            for m, d2 in [(1,21),(3,63),(6,126),(12,252)]:
                                fi = px.index[px.index > trough_date]
                                fwd[m] = round((float(px[fi[d2-1]])/trough_val-1)*100,1) if len(fi)>=d2 else None
                            cycles.append({
                                "고점일": peak_date, "저점일": trough_date,
                                "기간(일)": dur, "최대낙폭(%)": round(depth,1),
                                "낙폭속도(%/일)": round(abs(depth)/max(dur,1),3),
                                "1M(%)": fwd[1], "3M(%)": fwd[3],
                                "6M(%)": fwd[6], "12M(%)": fwd[12],
                                "고점가": round(peak_val,1), "저점가": round(trough_val,1),
                            })
                        in_dd = False
                        peak_val = val; peak_date = dt; trough_date = dt; trough_val = val
            return pd.DataFrame(cycles)

        _sp_raw = raw["sp500"].copy()
        _sp_raw.index = pd.to_datetime(_sp_raw.index).tz_localize(None)
        _sp_raw = _sp_raw.sort_index()
        cyc = _detect_crash_cycles(tuple(_sp_raw.items()))

        # ── 현재 S&P500 상태 계산 ──
        _cur_val   = float(_sp_raw.iloc[-1])
        _cur_ath   = float(_sp_raw.cummax().iloc[-1])
        _cur_dd    = (_cur_val - _cur_ath) / _cur_ath * 100
        _recent20  = _sp_raw.iloc[-21:]
        _peak20    = float(_recent20.iloc[0])
        _dd20      = min((_cur_val - _peak20) / _peak20 * 100, 0.0)
        _speed20   = round(abs(_dd20) / 20, 3)

        # 현재 하락 지속 기간
        _ath_date = _sp_raw[_sp_raw == _cur_ath].index[-1]
        _cur_dur  = (_sp_raw.index[-1] - _ath_date).days
        _cur_speed_full = round(abs(_cur_dd) / max(_cur_dur, 1), 3) if _cur_dd <= -5 else 0.0

        # ── 섹션 1: 현재 시장 상태 ──
        _is_dd = _cur_dd <= -5
        _dd_color  = "#f87171" if _cur_dd <= -20 else "#fbbf24" if _cur_dd <= -10 else "#34d399"
        _spd_color = "#f87171" if _speed20 > 0.30 else "#fbbf24" if _speed20 > 0.15 else "#6b7280"

        if _is_dd and _speed20 > 0.20:
            _pattern = "빠른 급락 — V자 반등 패턴"
            _pattern_color = "#f87171"
            _exp_12m = "+30% 내외 (역사적 평균)"
        elif _is_dd and _speed20 <= 0.20:
            _pattern = "느린 하락 — 횡보·침체 주의"
            _pattern_color = "#fbbf24"
            _exp_12m = "+27% 내외 (역사적 평균)"
        else:
            _pattern = "정상 범위 (ATH -5% 이내)"
            _pattern_color = "#34d399"
            _exp_12m = "—"

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d1117,#1a1f2e);'
            f'border:1px solid #334155;border-radius:14px;padding:20px 28px;margin-bottom:20px;">'
            f'<div style="color:#5b9bd5;font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:14px;">현재 S&P500 상태</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">'
            f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">ATH 대비 낙폭</div>'
            f'<div style="color:{_dd_color};font-size:28px;font-weight:900;">{_cur_dd:+.1f}%</div></div>'
            f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">낙폭속도 (20일)</div>'
            f'<div style="color:{_spd_color};font-size:28px;font-weight:900;">{_speed20:.3f}<span style="font-size:13px;">%/일</span></div></div>'
            f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">하락 지속일 (고점比)</div>'
            f'<div style="color:#e2e8f0;font-size:28px;font-weight:900;">{_cur_dur}<span style="font-size:13px;">일</span></div></div>'
            f'<div><div style="color:#4b5563;font-size:10px;margin-bottom:4px;">패턴 분류</div>'
            f'<div style="color:{_pattern_color};font-size:15px;font-weight:700;margin-top:6px;">{_pattern}</div>'
            f'<div style="color:#6b7280;font-size:11px;margin-top:4px;">역사적 12M 기대: {_exp_12m}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── 섹션 2: 요약 카드 (4개) ──
        def _grp_stats(df_g):
            r12 = [x for x in df_g["12M(%)"] if x is not None]
            r6  = [x for x in df_g["6M(%)"]  if x is not None]
            r1  = [x for x in df_g["1M(%)"]  if x is not None]
            return {
                "n": len(df_g),
                "avg_depth": df_g["최대낙폭(%)"].mean(),
                "avg_dur": df_g["기간(일)"].mean(),
                "r1_avg": np.mean(r1) if r1 else 0,
                "r6_avg": np.mean(r6) if r6 else 0,
                "r12_avg": np.mean(r12) if r12 else 0,
                "r12_pos": sum(x>0 for x in r12)/len(r12)*100 if r12 else 0,
            }

        fast = cyc[cyc["낙폭속도(%/일)"] > 0.20]
        slow = cyc[cyc["낙폭속도(%/일)"] <= 0.20]
        ultra= cyc[cyc["기간(일)"] <= 45]
        gs   = _grp_stats(fast); gs2 = _grp_stats(slow); gs3 = _grp_stats(ultra)

        st.markdown('<div class="section-title">패턴별 성과 요약</div>', unsafe_allow_html=True)
        _sc1, _sc2, _sc3 = st.columns(3)
        for _col, _title, _border, _gs, _desc in [
            (_sc1, "초단기 급락 (0~45일)", "#f87171", gs3,  f"{gs3['n']}건 · 평균속도 {cyc[cyc['기간(일)']<=45]['낙폭속도(%/일)'].mean():.3f}%/일"),
            (_sc2, "빠른 하락 (속도>0.20%/일)", "#fbbf24", gs,   f"{gs['n']}건 · 평균기간 {gs['avg_dur']:.0f}일"),
            (_sc3, "느린 하락 (속도≤0.20%/일)", "#6366f1", gs2,  f"{gs2['n']}건 · 평균기간 {gs2['avg_dur']:.0f}일"),
        ]:
            with _col:
                st.markdown(
                    f'<div style="background:#111827;border:1.5px solid {_border};border-radius:12px;padding:18px;">'
                    f'<div style="color:{_border};font-size:11px;font-weight:700;margin-bottom:4px;">{_title}</div>'
                    f'<div style="color:#6b7280;font-size:10px;margin-bottom:14px;">{_desc}</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center;">'
                    f'<div><div style="color:#4b5563;font-size:9px;">1개월</div>'
                    f'<div style="color:#34d399;font-size:18px;font-weight:700;">{_gs["r1_avg"]:+.1f}%</div></div>'
                    f'<div><div style="color:#4b5563;font-size:9px;">6개월</div>'
                    f'<div style="color:#34d399;font-size:18px;font-weight:700;">{_gs["r6_avg"]:+.1f}%</div></div>'
                    f'<div><div style="color:#4b5563;font-size:9px;">12개월</div>'
                    f'<div style="color:#34d399;font-size:22px;font-weight:900;">{_gs["r12_avg"]:+.1f}%</div></div>'
                    f'</div>'
                    f'<div style="margin-top:10px;border-top:1px solid #1e2a3a;padding-top:8px;'
                    f'color:#6b7280;font-size:10px;">12M 양수확률 '
                    f'<span style="color:#34d399;font-weight:700;">{_gs["r12_pos"]:.0f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 섹션 3: 버블 차트 ──
        st.markdown('<div class="section-title">하락 사이클 맵 — 기간 × 낙폭 × 속도 × 이후수익</div>', unsafe_allow_html=True)
        st.caption("버블 크기 = 낙폭 속도(%/일)  ·  버블 색상 = 12개월 이후 수익률 (빨강→초록)  ·  각 버블 = 하나의 하락 사이클")

        _valid = cyc.dropna(subset=["12M(%)"])
        _r12_vals = _valid["12M(%)"].tolist()
        _sizes    = [max(_valid["낙폭속도(%/일)"].iloc[i] * 280, 14) for i in range(len(_valid))]
        _labels   = [str(_valid["고점일"].iloc[i].year) for i in range(len(_valid))]
        _hover    = [
            f"<b>{str(_valid['고점일'].iloc[i])[:10]} → {str(_valid['저점일'].iloc[i])[:10]}</b><br>"
            f"기간: {_valid['기간(일)'].iloc[i]}일  |  낙폭: {_valid['최대낙폭(%)'].iloc[i]:.1f}%<br>"
            f"속도: {_valid['낙폭속도(%/일)'].iloc[i]:.3f}%/일<br>"
            f"─────────────<br>"
            f"1M: {_valid['1M(%)'].iloc[i]:+.1f}%  3M: {_valid['3M(%)'].iloc[i]:+.1f}%<br>"
            f"6M: {_valid['6M(%)'].iloc[i]:+.1f}%  12M: {_valid['12M(%)'].iloc[i]:+.1f}%"
            for i in range(len(_valid))
        ]

        _fig_bubble = go.Figure()
        _fig_bubble.add_trace(go.Scatter(
            x=_valid["기간(일)"], y=_valid["최대낙폭(%)"],
            mode="markers+text",
            marker=dict(
                size=_sizes, sizemode="area",
                color=_r12_vals,
                colorscale=[[0,"#991b1b"],[0.3,"#f87171"],[0.5,"#fbbf24"],[0.7,"#4ade80"],[1,"#16a34a"]],
                colorbar=dict(
                    title=dict(text="12M 수익%", font=dict(color="#9ca3af",size=10)),
                    tickfont=dict(color="#9ca3af",size=9),
                    thickness=12, len=0.7,
                ),
                cmin=-20, cmax=80,
                line=dict(color="#1e2a3a", width=1),
                opacity=0.85,
            ),
            text=_labels,
            textposition="top center",
            textfont=dict(color="#94a3b8", size=9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=_hover,
            name="하락 사이클",
        ))

        # 현재 시장 위치 표시 (하락 중일 때만)
        if _is_dd and _cur_dd <= -10:
            _fig_bubble.add_trace(go.Scatter(
                x=[_cur_dur], y=[_cur_dd],
                mode="markers+text",
                marker=dict(size=20, color="#ffffff", symbol="star",
                            line=dict(color="#f87171", width=2)),
                text=["현재"], textposition="top center",
                textfont=dict(color="#ffffff", size=10, family="Arial Black"),
                hovertemplate=f"<b>현재 시장</b><br>기간: {_cur_dur}일<br>낙폭: {_cur_dd:.1f}%<extra></extra>",
                name="현재 위치",
            ))

        # 속도 임계선 (0.20%/일)
        _x_max = max(_valid["기간(일)"].max() * 1.1, 700)
        _thr_x = [i for i in range(0, int(_x_max), 10)]
        _thr_y = [-0.20 * x for x in _thr_x]
        _fig_bubble.add_trace(go.Scatter(
            x=_thr_x, y=_thr_y, mode="lines",
            line=dict(color="#fbbf24", width=1, dash="dot"),
            hoverinfo="skip", showlegend=True, name="속도 0.20%/일 기준선",
        ))
        _fig_bubble.add_annotation(
            x=180, y=-0.20*180, text="← 빠름 | 느림 →",
            font=dict(color="#fbbf24", size=10), showarrow=False,
            bgcolor="#111827", borderpad=3,
        )

        _fig_bubble.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            height=480, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title=dict(text="하락 지속 기간 (일)", font=dict(color="#6b7280",size=11)),
                       showgrid=True, gridcolor="#1e2a3a", range=[0, _x_max]),
            yaxis=dict(title=dict(text="최대 낙폭 (%)", font=dict(color="#6b7280",size=11)),
                       showgrid=True, gridcolor="#1e2a3a"),
            legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
            hovermode="closest",
        )
        st.plotly_chart(_fig_bubble, use_container_width=True)

        # ── 섹션 4: 기간별 수익률 바 차트 ──
        st.markdown('<div class="section-title">기간별 그룹 수익률 비교</div>', unsafe_allow_html=True)
        _bins = [
            ("초단기\n(0~45일)",   cyc[cyc["기간(일)"] <= 45]),
            ("단기\n(45~120일)",   cyc[(cyc["기간(일)"]>45)&(cyc["기간(일)"]<=120)]),
            ("중기\n(120~300일)",  cyc[(cyc["기간(일)"]>120)&(cyc["기간(일)"]<=300)]),
            ("장기\n(300일+)",     cyc[cyc["기간(일)"] > 300]),
        ]
        _bar_labels = [b[0] for b in _bins]
        _bar_1m  = [np.mean([x for x in b[1]["1M(%)"]  if x is not None]) if len(b[1])>0 else 0 for b in _bins]
        _bar_3m  = [np.mean([x for x in b[1]["3M(%)"]  if x is not None]) if len(b[1])>0 else 0 for b in _bins]
        _bar_6m  = [np.mean([x for x in b[1]["6M(%)"]  if x is not None]) if len(b[1])>0 else 0 for b in _bins]
        _bar_12m = [np.mean([x for x in b[1]["12M(%)"] if x is not None]) if len(b[1])>0 else 0 for b in _bins]
        _bar_n   = [len(b[1]) for b in _bins]

        _fig_bar = go.Figure()
        for _vals, _name, _color in [
            (_bar_1m,  "1개월",  "#6366f1"),
            (_bar_3m,  "3개월",  "#5b9bd5"),
            (_bar_6m,  "6개월",  "#22c55e"),
            (_bar_12m, "12개월", "#f59e0b"),
        ]:
            _fig_bar.add_trace(go.Bar(
                x=[f"{_bar_labels[i]}\n({_bar_n[i]}건)" for i in range(len(_bar_labels))],
                y=_vals, name=_name,
                marker_color=_color,
                text=[f"{v:+.1f}%" for v in _vals],
                textposition="outside",
                textfont=dict(size=10, color="#d1d5db"),
            ))

        _fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            height=340, margin=dict(l=0,r=0,t=20,b=0),
            barmode="group", bargap=0.25, bargroupgap=0.08,
            legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#9ca3af")),
            yaxis=dict(showgrid=True, gridcolor="#1e2a3a", ticksuffix="%",
                       tickfont=dict(size=10)),
        )
        st.plotly_chart(_fig_bar, use_container_width=True)

        # ── 섹션 5: 상세 테이블 ──
        st.markdown('<div class="section-title">역대 하락 사이클 상세</div>', unsafe_allow_html=True)
        _disp = cyc.copy()
        _disp["고점일"] = _disp["고점일"].dt.strftime("%Y-%m-%d")
        _disp["저점일"] = _disp["저점일"].dt.strftime("%Y-%m-%d")
        _disp["유형"] = _disp["낙폭속도(%/일)"].apply(
            lambda s: "초단기급락" if s > 0.40 else ("빠른하락" if s > 0.20 else "느린하락")
        )
        _disp = _disp[["고점일","저점일","기간(일)","최대낙폭(%)","낙폭속도(%/일)","유형","1M(%)","3M(%)","6M(%)","12M(%)"]].sort_values("고점일", ascending=False).reset_index(drop=True)

        def _style_crash_table(df):
            def _c(val):
                if isinstance(val, float):
                    if val >= 30:  return "color:#16a34a;font-weight:700"
                    if val >= 10:  return "color:#34d399"
                    if val >= 0:   return "color:#86efac"
                    if val >= -20: return "color:#fbbf24"
                    return "color:#f87171;font-weight:700"
                if isinstance(val, str):
                    if "초단기" in str(val): return "color:#f87171;font-weight:700"
                    if "빠른"   in str(val): return "color:#fbbf24;font-weight:600"
                    if "느린"   in str(val): return "color:#6b7280"
                return "color:#d1d5db"
            return df.style.map(_c)

        def _tbl_summary(df_g):
            r12 = df_g["12M(%)"].dropna()
            r6  = df_g["6M(%)"].dropna()
            if len(r12) == 0: return ""
            return (
                f'<div style="display:flex;gap:24px;margin-bottom:10px;'
                f'background:#111827;border-radius:8px;padding:10px 16px;">'
                f'<span style="color:#4b5563;font-size:11px;">{len(df_g)}건</span>'
                f'<span style="color:#4b5563;font-size:11px;">|</span>'
                f'<span style="color:#9ca3af;font-size:11px;">평균낙폭 '
                f'<b style="color:#f87171;">{df_g["최대낙폭(%)"].mean():.1f}%</b></span>'
                f'<span style="color:#4b5563;font-size:11px;">|</span>'
                f'<span style="color:#9ca3af;font-size:11px;">평균기간 '
                f'<b style="color:#e2e8f0;">{df_g["기간(일)"].mean():.0f}일</b></span>'
                f'<span style="color:#4b5563;font-size:11px;">|</span>'
                f'<span style="color:#9ca3af;font-size:11px;">12M 평균 '
                f'<b style="color:#34d399;">{r12.mean():+.1f}%</b></span>'
                f'<span style="color:#4b5563;font-size:11px;">|</span>'
                f'<span style="color:#9ca3af;font-size:11px;">12M 양수확률 '
                f'<b style="color:#34d399;">{(r12>0).mean()*100:.0f}%</b></span>'
                f'</div>'
            )

        # ── N일차 평균 낙폭속도 꺾은선 차트 ──
        def _avg_speed_chart(df_group, group_color):
            """
            각 사이클의 N일차 일별 수익률을 수집 → N일차 평균 계산 → 꺾은선
            개별 사이클(얇은선) + 평균(굵은선) 겹쳐 표시
            """
            # ── 사이클별 거래일 기준 일별 수익률 수집 ──
            daily_by_day = {}   # {day_num: [ret_cycle1, ret_cycle2, ...]}
            cycle_data   = {}   # {label: {day_num: ret}}

            for _, row in df_group.iterrows():
                peak_dt   = pd.Timestamp(row["고점일"])
                trough_dt = pd.Timestamp(row["저점일"])
                _seg = _sp_raw[(_sp_raw.index >= peak_dt) & (_sp_raw.index <= trough_dt)]
                if len(_seg) < 2:
                    continue
                label = row["고점일"][:7]
                cycle_data[label] = {}
                for j in range(1, len(_seg)):
                    ret = (float(_seg.iloc[j]) / float(_seg.iloc[j-1]) - 1) * 100
                    cycle_data[label][j] = ret
                    daily_by_day.setdefault(j, []).append(ret)

            if not daily_by_day:
                return go.Figure()

            avg_days = sorted(daily_by_day.keys())
            avg_vals = [np.mean(daily_by_day[d]) for d in avg_days]
            # 평균 기준 최대 하락일
            peak_avg_day = avg_days[int(np.argmin(avg_vals))]

            fig = go.Figure()

            # ── 개별 사이클 (얇고 반투명) ──
            colors_ind = ["#f87171","#fb923c","#fbbf24","#a78bfa","#60a5fa",
                          "#34d399","#f472b6","#5b9bd5","#22c55e","#e879f9"]
            for ci, (label, ddata) in enumerate(cycle_data.items()):
                xs = sorted(ddata.keys())
                ys = [ddata[x] for x in xs]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=label,
                    line=dict(color=colors_ind[ci % len(colors_ind)], width=1.2),
                    opacity=0.45,
                    hovertemplate=f"<b>{label}</b><br>%{{x}}일차: %{{y:.2f}}%<extra></extra>",
                ))

            # ── 평균선 (굵고 선명) ──
            fig.add_trace(go.Scatter(
                x=avg_days, y=avg_vals,
                mode="lines+markers",
                name="■ 평균",
                line=dict(color=group_color, width=3),
                marker=dict(size=5, color=group_color),
                hovertemplate="<b>평균</b><br>%{x}일차: %{y:.2f}%<extra></extra>",
            ))

            # 0% 기준선
            fig.add_hline(y=0, line=dict(color="#374151", width=1, dash="dot"))

            # 평균 최대낙폭 날짜 수직선
            fig.add_vline(
                x=peak_avg_day,
                line=dict(color=group_color, width=1.5, dash="dash"),
                annotation_text=f"평균 최대낙폭일 ({peak_avg_day}일차)",
                annotation_font=dict(color=group_color, size=10),
                annotation_position="top right",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                title=dict(
                    text="N일차 평균 낙폭속도 — 얇은선: 개별 사이클 / 굵은선: 전체 평균",
                    font=dict(size=12, color="#f1f5f9"),
                ),
                legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
                hovermode="x unified",
                xaxis=dict(
                    title=dict(text="고점 이후 거래일 (N일차)", font=dict(color="#6b7280", size=10)),
                    showgrid=True, gridcolor="#1e2a3a",
                    tickfont=dict(size=10, color="#9ca3af"),
                    dtick=2,
                ),
                yaxis=dict(
                    title=dict(text="일별 낙폭률 (%)", font=dict(color="#6b7280", size=10)),
                    showgrid=True, gridcolor="#1e2a3a",
                    ticksuffix="%", tickfont=dict(size=10),
                    zeroline=True, zerolinecolor="#374151",
                ),
            )
            return fig

        # ── 일별 낙폭 속도 + 누적 이중 패널 차트 ──
        def _path_chart(df_group, title, line_color_list, show_days=120):
            """
            위: 일별 낙폭 속도 (그날 하루 얼마나 빠졌나, 막대)
            아래: 누적 낙폭 (합산이 얼마나 쌓였나, 선)
            — 고점 이후 경과일 기준, 각 사이클 겹쳐서 표시
            """
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.45, 0.55],
                vertical_spacing=0.06,
                subplot_titles=["일별 낙폭 속도 (%/일, 하락일만)", "누적 낙폭 (%)"],
            )

            for i, (_, row) in enumerate(df_group.iterrows()):
                peak_str   = row["고점일"]
                trough_str = row["저점일"]
                peak_dt    = pd.Timestamp(peak_str)
                trough_dt  = pd.Timestamp(trough_str)

                end_dt = trough_dt + pd.Timedelta(days=show_days)
                _seg = _sp_raw[(_sp_raw.index >= peak_dt) & (_sp_raw.index <= end_dt)]
                if len(_seg) < 2:
                    continue

                peak_val   = float(_seg.iloc[0])
                color      = line_color_list[i % len(line_color_list)]
                label      = peak_str[:7]   # YYYY-MM
                trough_day = (trough_dt - peak_dt).days

                # 경과일 / 일별수익률 / 누적낙폭
                days    = [(d - peak_dt).days for d in _seg.index]
                daily_r = [0.0] + [
                    (float(_seg.iloc[j]) / float(_seg.iloc[j-1]) - 1) * 100
                    for j in range(1, len(_seg))
                ]
                cum_dd  = [(float(v) / peak_val - 1) * 100 for v in _seg]

                # 하락일만 속도 표시 (양수 = 그날 하락폭)
                speed_y = [abs(r) if r < 0 else 0 for r in daily_r]

                # ── 위 패널: 일별 속도 막대 ──
                fig.add_trace(go.Bar(
                    x=days, y=speed_y,
                    name=label,
                    marker_color=color,
                    opacity=0.75,
                    legendgroup=label,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "경과: %{x}일<br>"
                        "당일낙폭: -%{y:.2f}%<extra></extra>"
                    ),
                ), row=1, col=1)

                # ── 아래 패널: 누적 낙폭 선 ──
                fig.add_trace(go.Scatter(
                    x=days, y=cum_dd,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "경과: %{x}일<br>"
                        "누적낙폭: %{y:.1f}%<extra></extra>"
                    ),
                ), row=2, col=1)

                # 저점 마커 (아래 패널)
                trough_idx = min(range(len(days)), key=lambda k: cum_dd[k])
                fig.add_trace(go.Scatter(
                    x=[days[trough_idx]], y=[cum_dd[trough_idx]],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="circle",
                                line=dict(color="#ffffff", width=1.5)),
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label} 저점</b><br>"
                        f"경과: {days[trough_idx]}일<br>"
                        f"최대낙폭: {cum_dd[trough_idx]:.1f}%<extra></extra>"
                    ),
                ), row=2, col=1)

            # 기준선
            fig.add_hline(y=0, line=dict(color="#374151", width=1, dash="dot"), row=2, col=1)

            # 저점 구분 수직선 (저점 이후 = 반등 구간)
            fig.add_vline(x=0, line=dict(color="#4b5563", width=1, dash="dot"))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                height=560,
                margin=dict(l=0, r=0, t=44, b=0),
                title=dict(text=title, font=dict(size=13, color="#f1f5f9")),
                legend=dict(orientation="h", y=1.04, x=0,
                            font=dict(size=11), traceorder="normal"),
                barmode="overlay",
                hovermode="x unified",
            )
            # 축 스타일
            axis_style = dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10, color="#9ca3af"))
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
            fig.update_yaxes(ticksuffix="%")
            fig.update_xaxes(title_text="고점 이후 경과일", title_font=dict(color="#6b7280", size=10), row=2, col=1)
            fig.update_yaxes(title_text="%/일", title_font=dict(color="#6b7280", size=10), row=1, col=1)
            fig.update_yaxes(title_text="누적낙폭 %", title_font=dict(color="#6b7280", size=10), row=2, col=1)

            # 서브플롯 제목 색상
            for ann in fig.layout.annotations:
                ann.font.color = "#6b7280"
                ann.font.size  = 10

            return fig

        # 유형별 색상 팔레트
        _colors_ultra = ["#f87171","#fb923c","#fbbf24","#a78bfa","#60a5fa","#34d399","#f472b6"]
        _colors_fast  = ["#fbbf24","#f59e0b","#d97706","#fb923c","#f87171",
                         "#a78bfa","#60a5fa","#34d399","#6ee7b7","#c4b5fd"]
        _colors_slow  = ["#6366f1","#5b9bd5","#06b6d4","#22c55e","#84cc16",
                         "#a78bfa","#e879f9","#94a3b8","#475569","#64748b",
                         "#0ea5e9","#f472b6","#10b981"]

        _ct1, _ct2, _ct3 = st.tabs([
            "🔴 초단기급락 (속도>0.40%/일)",
            "🟡 빠른하락 (0.20~0.40%/일)",
            "🔵 느린하락 (속도≤0.20%/일)",
        ])
        for _ctab, _label, _mask, _colors, _show, _avg_color in [
            (_ct1, "초단기급락", _disp["유형"] == "초단기급락", _colors_ultra, 90,  "#f87171"),
            (_ct2, "빠른하락",   _disp["유형"] == "빠른하락",   _colors_fast,  120, "#fbbf24"),
            (_ct3, "느린하락",   _disp["유형"] == "느린하락",   _colors_slow,  180, "#6366f1"),
        ]:
            with _ctab:
                _sub = _disp[_mask].reset_index(drop=True)
                st.markdown(_tbl_summary(_sub), unsafe_allow_html=True)

                # ① N일차 평균 낙폭속도 꺾은선
                st.plotly_chart(
                    _avg_speed_chart(_sub, _avg_color),
                    use_container_width=True,
                )

                # ② 누적 낙폭 경로 (기존)
                st.plotly_chart(
                    _path_chart(
                        _sub,
                        f"{_label} — 고점 기준 누적 낙폭 경로 (● = 저점)",
                        _colors,
                        show_days=_show,
                    ),
                    use_container_width=True,
                )

                st.dataframe(_style_crash_table(_sub), use_container_width=True,
                             height=min(80 + len(_sub)*36, 520), hide_index=True)

        st.markdown(
            '<div class="footer-txt">낙폭속도 = 최대낙폭(%) ÷ 하락기간(일) · '
            '기준선 0.20%/일 초과 = 빠른 하락 · 저점 판단은 사후적 기준 · 과거 성과가 미래를 보장하지 않습니다</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════
    # TAB 8: 메모장
    # ════════════════════════════════════════
    with main_tab8:
        MEMO_FILE = pathlib.Path(__file__).parent / "memo.txt"

        _DEFAULT_MEMO = """\
## 📌 현재 전략 요약
- 끝자리 전략: 수익 확률 60% 이상인 끝자리 해만 투자
- 투자 비중(투자시즌): 주식 50% / 금 25% / 채권 25%
- 투자 비중(현금보유): 주식 25% / 현금 25% / 금 25% / 채권 25%

## 📋 저점 레이더 기준값
| 지표 | 저점 기준 |
|------|----------|
| VIX | 40 이상 |
| Fear & Greed | 25 이하 |
| CAPE | 15 이하 |
| S&P500 PER | 15 이하 |
| ATH 대비 낙폭 | -30% 이하 |
| RSI(14) | 30 이하 |
| 200일 MA 괴리율 | -20% 이하 |
| 볼린저밴드 %B | 0 이하 |
| 실현변동성 | 35% 이상 |
| KOSPI PBR | 1.0 이하 |

## 🗒️ 나의 투자 메모
(여기에 자유롭게 작성하세요)

## 📅 매매 히스토리
| 날짜 | 지수 | 매수/매도 | 금액 | 비고 |
|------|------|---------|------|------|
|      |      |         |      |      |

## 💡 관찰 중 / 대기 중


## 🔖 참고 링크
- CNN Fear & Greed: https://money.cnn.com/data/fear-and-greed/
- multpl CAPE: https://www.multpl.com/shiller-pe
- multpl PER: https://www.multpl.com/s-p-500-pe-ratio
"""

        def _load_memo():
            try:
                if MEMO_FILE.exists():
                    return MEMO_FILE.read_text(encoding="utf-8")
            except Exception:
                pass
            return _DEFAULT_MEMO

        def _save_memo(text):
            try:
                MEMO_FILE.write_text(text, encoding="utf-8")
                return True
            except Exception:
                return False

        # 세션 내 최초 1회만 파일에서 로드
        if "memo_init" not in st.session_state:
            st.session_state["memo_init"] = _load_memo()

        st.markdown('<div class="section-title">📝 메모장</div>', unsafe_allow_html=True)
        st.caption("저장 버튼을 누르면 로컬 파일(memo.txt)에 저장됩니다. 새로고침 후에도 유지됩니다.")

        memo_text = st.text_area(
            label="memo",
            value=st.session_state["memo_init"],
            height=600,
            label_visibility="collapsed",
            key="memo_textarea",
        )

        c1, c2, c3 = st.columns([1, 1, 8])
        with c1:
            if st.button("💾 저장", use_container_width=True):
                if _save_memo(memo_text):
                    st.session_state["memo_init"] = memo_text
                    st.success("✅ 저장됐습니다!")
                else:
                    st.warning("⚠️ 저장 실패 (파일 쓰기 권한 확인)")
        with c2:
            if st.button("↺ 초기화", use_container_width=True):
                st.session_state["memo_init"] = _DEFAULT_MEMO
                if "memo_textarea" in st.session_state:
                    del st.session_state["memo_textarea"]
                st.rerun()


    # ════════════════════════════════════════
    # TAB 0: 실사용 전략 (전략8 가이드)
    # — main_tab1 내부 함수 재사용을 위해 맨 뒤에 배치
    # ════════════════════════════════════════
    with main_tab0:

        # ── 헬퍼: 월별 수익률 히트맵 HTML ──
        def _monthly_heatmap_html(series):
            """일별 가격 시리즈 → 월별 수익률 컬러 테이블 HTML"""
            if series is None or len(series) < 5:
                return "<p style='color:#9ca3af'>데이터 부족</p>"
            mo_r = series.resample("ME").last().pct_change().dropna() * 100
            df_m = mo_r.to_frame("v")
            df_m["y"] = df_m.index.year
            df_m["m"] = df_m.index.month
            pivot = df_m.pivot(index="y", columns="m", values="v")
            ann = series.resample("YE").last().pct_change().dropna() * 100
            ann.index = ann.index.year
            pivot["A"] = ann
            pivot = pivot.sort_index(ascending=False)

            MN = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월","연간"]
            th = "background:#1e2a3a;padding:6px 6px;text-align:center;color:#9ca3af;font-size:13px;font-weight:700"
            td_b = "padding:7px 5px;text-align:center;font-size:14px"

            def _sty(v, scale=10):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return f"{td_b};background:#0d111c;color:#374151", "─"
                t = min(abs(v) / scale, 1.0)
                if v > 0:
                    bg = f"rgb({int(6+22*t)},{int(78+100*t)},{int(57+3*t)})"
                else:
                    bg = f"rgb({int(127+100*t)},29,29)"
                return f"{td_b};background:{bg};color:#f1f5f9;font-weight:600", f"{v:+.1f}%"

            html = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;">'
            html += f'<tr><th style="{th};text-align:left">연도</th>'
            for nm in MN:
                html += f'<th style="{th}">{nm}</th>'
            html += '</tr>'
            for y in pivot.index:
                html += (f'<tr><td style="background:#111827;padding:7px 6px;'
                         f'color:#f1f5f9;font-weight:700;font-size:14px">{y}</td>')
                for m in range(1, 13):
                    val = None
                    if m in pivot.columns:
                        raw_v = pivot.loc[y, m]
                        val = None if pd.isna(raw_v) else float(raw_v)
                    sty, txt = _sty(val, scale=10)
                    html += f'<td style="{sty}">{txt}</td>'
                ann_v = None
                if "A" in pivot.columns:
                    av = pivot.loc[y, "A"]
                    ann_v = None if pd.isna(av) else float(av)
                sty, txt = _sty(ann_v, scale=20)
                html += f'<td style="{sty};font-weight:700">{txt}</td></tr>'
            html += '</table></div>'
            return html

        # ── 헬퍼: KODEX ETF 기반 전략8 일별 포트폴리오 ──
        def _daily_pf_series_kodex():
            """KODEX ETF 가격 기반 전략8 일별 시리즈 (2021~)"""
            kodex = load_kodex_etfs()

            def _pc(key):
                s = kodex.get(key, pd.Series(dtype=float))
                if s.empty:
                    return pd.Series(dtype=float)
                s2 = s.copy()
                s2.index = pd.to_datetime(s2.index).tz_localize(None)
                return s2.pct_change().dropna()

            d_sp = _pc("sp500")
            if d_sp.empty:
                return {}
            d_nq = _pc("nasdaq"); d_ko = _pc("kospi"); d_kq = _pc("kosdaq")
            d_cs = _pc("csi300"); d_g  = _pc("gold"); d_kr = _pc("krbond"); d_us = _pc("us10y")

            nq_idx = set(d_nq.index); ko_idx = set(d_ko.index); kq_idx = set(d_kq.index)
            cs_idx = set(d_cs.index); g_idx  = set(d_g.index)
            kr_idx = set(d_kr.index); us_idx = set(d_us.index)

            def _inv_map(key):
                inv_na = strategies[key]["inv_na"]
                inv_mo = strategies[key]["inv_mo"]
                return {
                    (yr, s): (yr % 10) in (inv_na if s == "Nov-Apr" else inv_mo)
                    for yr in range(2019, 2032)
                    for s in ["Nov-Apr", "May-Oct"]
                }

            sp_imap = _inv_map("sp500")
            ko_imap = _inv_map("kospi")
            cs_imap = _inv_map("csi300")

            cols = ["전략8_KODEX", "SP500", "KOSPI"]
            v = {c: 100.0 for c in cols}
            vals = {c: [] for c in cols}
            dates = []
            prev_sk = None

            for dt in d_sp.index:
                mo_dt = dt.month
                if mo_dt >= 11:   s_, sy = "Nov-Apr", dt.year
                elif mo_dt <= 4:  s_, sy = "Nov-Apr", dt.year - 1
                else:             s_, sy = "May-Oct", dt.year
                sk = (sy, s_)

                r_sp = float(d_sp[dt])
                r_nq = float(d_nq[dt]) if dt in nq_idx else r_sp
                r_ko = float(d_ko[dt]) if dt in ko_idx else r_sp
                r_kq = float(d_kq[dt]) if dt in kq_idx else r_ko
                r_cs = float(d_cs[dt]) if dt in cs_idx else r_sp
                r_g  = float(d_g[dt])  if dt in g_idx  else 0.0
                r_kr = float(d_kr[dt]) if dt in kr_idx else 0.0
                r_us = float(d_us[dt]) if dt in us_idx else 0.0

                sp_inv = sp_imap.get(sk, False)
                ko_inv = ko_imap.get(sk, sp_inv)
                cs_inv = cs_imap.get(sk, sp_inv)

                w_sp = 0.10; w_nq = 0.10 if sp_inv else 0.0
                w_ko = 0.10; w_kq = 0.10 if ko_inv else 0.0
                w_cs = 0.20 if cs_inv else 0.10
                prev_sk = sk

                w_cash = round(1.0 - w_sp - w_nq - w_ko - w_kq - w_cs - 0.20 - 0.10 - 0.10, 10)
                r_pf = (w_sp*r_sp + w_nq*r_nq + w_ko*r_ko + w_kq*r_kq
                        + w_cs*r_cs + 0.20*r_g + 0.10*r_kr + 0.10*r_us + w_cash*r_kr)
                v["전략8_KODEX"] *= (1 + r_pf)
                v["SP500"] *= (1 + r_sp)
                v["KOSPI"] *= (1 + r_ko)

                for c in cols:
                    vals[c].append(v[c])
                dates.append(dt)

            return {c: pd.Series(vals[c], index=dates) for c in cols}

        # ════════════════════
        # 현황판
        # ════════════════════
        st.markdown('<div class="section-title">💡 전략8 실사용 가이드</div>', unsafe_allow_html=True)

        sp_inv8 = digit in (strategies["sp500"]["inv_na"] if season == "Nov-Apr"
                            else strategies["sp500"]["inv_mo"])
        ko_inv8 = digit in (strategies["kospi"]["inv_na"] if season == "Nov-Apr"
                            else strategies["kospi"]["inv_mo"])
        cs_inv8 = digit in (strategies["csi300"]["inv_na"] if season == "Nov-Apr"
                            else strategies["csi300"]["inv_mo"])

        w_sp8 = 0.10; w_nq8 = 0.10 if sp_inv8 else 0.0
        w_ko8 = 0.10; w_kq8 = 0.10 if ko_inv8 else 0.0
        w_cs8 = 0.20 if cs_inv8 else 0.10
        w_cash8 = round(1.0 - w_sp8 - w_nq8 - w_ko8 - w_kq8 - w_cs8 - 0.20 - 0.10 - 0.10, 10)
        total_stk8 = int((w_sp8 + w_nq8 + w_ko8 + w_kq8 + w_cs8) * 100)

        # 다음 리밸런싱
        if season == "Nov-Apr":
            nrb_year = today.year + 1 if today.month <= 4 else today.year
            next_reb8 = date(nrb_year, 5, 1)
        else:
            next_reb8 = date(today.year, 11, 1)
        days_reb8 = (next_reb8 - today).days

        season_color8 = "#38bdf8" if season == "Nov-Apr" else "#fb923c"
        season_label8 = "🌨️ 투자시즌 (Nov~Apr)" if season == "Nov-Apr" else "☀️ 조정시즌 (May~Oct)"

        st.markdown(f"""
        <div style="background:#0d1117;border:1.5px solid {season_color8};border-radius:12px;
                    padding:16px 20px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
            <div>
              <div style="color:{season_color8};font-size:11px;font-weight:800;
                          letter-spacing:1px;margin-bottom:4px;">현재 시즌</div>
              <div style="color:#f1f5f9;font-size:18px;font-weight:900;">{season_label8}</div>
            </div>
            <div>
              <div style="color:#9ca3af;font-size:11px;margin-bottom:4px;">기준연도 끝자리</div>
              <div style="color:#fbbf24;font-size:18px;font-weight:900;">{sig_year}년 (끝자리 {digit})</div>
            </div>
            <div>
              <div style="color:#9ca3af;font-size:11px;margin-bottom:4px;">다음 리밸런싱</div>
              <div style="color:#f1f5f9;font-size:16px;font-weight:700;">
                {next_reb8.strftime("%Y-%m-%d")}
                <span style="color:#9ca3af;font-size:12px;margin-left:6px;">D-{days_reb8}일</span>
              </div>
            </div>
            <div>
              <div style="color:#9ca3af;font-size:11px;margin-bottom:4px;">현재 주식 비중</div>
              <div style="color:#34d399;font-size:20px;font-weight:900;">{total_stk8}%</div>
            </div>
            <div>
              <div style="color:#9ca3af;font-size:11px;margin-bottom:4px;">국고채3년 비중 (채권+현금)</div>
              <div style="color:#fb923c;font-size:20px;font-weight:900;">{int((0.10+w_cash8)*100)}%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 3대 지수 투자 신호 카드
        inv_card_html = ""
        for idx_label, inv, emoji, sub in [
            ("S&P500 / 나스닥", sp_inv8, "🇺🇸", "KODEX 미국S&P500 + 나스닥100"),
            ("KOSPI / 코스닥",  ko_inv8, "🇰🇷", "KODEX 200 + 코스닥150"),
            ("CSI300",          cs_inv8, "🇨🇳", "KODEX 차이나CSI300"),
        ]:
            clr_ = "#34d399" if inv else "#f87171"
            bg_  = "#052e16" if inv else "#2d0000"
            bdr_ = "#16a34a" if inv else "#991b1b"
            lbl_ = "✅ 투자시즌" if inv else "💤 비투자시즌"
            inv_card_html += (
                f'<div style="background:{bg_};border:1.5px solid {bdr_};border-radius:10px;'
                f'padding:14px;text-align:center;">'
                f'<div style="font-size:24px;margin-bottom:6px;">{emoji}</div>'
                f'<div style="color:#9ca3af;font-size:10px;font-weight:700;margin-bottom:4px;">{idx_label}</div>'
                f'<div style="color:{clr_};font-size:14px;font-weight:800;margin-bottom:4px;">{lbl_}</div>'
                f'<div style="color:#6b7280;font-size:10px;">{sub}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px;">'
            f'{inv_card_html}</div>',
            unsafe_allow_html=True,
        )

        # ETF 배분 테이블
        alloc_data = [
            ("🇺🇸 KODEX 미국S&P500",          "379800", w_sp8,   "SP500 투자시즌" if sp_inv8 else "─",               True),
            ("💻 KODEX 미국나스닥100",           "379810", w_nq8,   "SP500 투자시즌 추가" if sp_inv8 else "비투자 (0%)", sp_inv8),
            ("🇰🇷 KODEX 200",                   "069500", w_ko8,   "KOSPI 투자시즌" if ko_inv8 else "─",               True),
            ("📱 KODEX 코스닥150",               "229200", w_kq8,   "KOSPI 투자시즌 추가" if ko_inv8 else "비투자 (0%)", ko_inv8),
            ("🇨🇳 KODEX 차이나CSI300",          "168580", w_cs8,   "CSI300 투자시즌 20%" if cs_inv8 else "비투자 10%",  True),
            ("🥇 ACE KRX금현물",             "411060", 0.20,
             "고정 20%", True),
            ("🏦 KODEX 국고채3년",           "114820", round(0.10 + w_cash8, 4),
             f"채권 10% + 현금대체 {int(w_cash8*100)}%", True),
            ("🌐 KODEX 미국10년국채선물",     "304660", 0.10,
             "고정 10%", True),
        ]
        th_s = ("background:#1e2a3a;padding:6px 10px;color:#9ca3af;"
                "font-size:11px;font-weight:700")
        t_html = ('<table style="width:100%;border-collapse:collapse;margin-bottom:12px;font-size:12px;">'
                  f'<tr>'
                  f'<th style="{th_s};text-align:left">ETF</th>'
                  f'<th style="{th_s};text-align:center">티커</th>'
                  f'<th style="{th_s};text-align:center">비중</th>'
                  f'<th style="{th_s}">비고</th></tr>')
        for etf_nm, tkr, wt, note, active in alloc_data:
            row_bg = "#0d1117" if active else "#080c10"
            wt_clr = "#34d399" if wt > 0.05 else ("#fb923c" if wt > 0.001 else "#374151")
            nm_clr = "#f1f5f9" if active else "#4b5563"
            wt_txt = f"{wt*100:.0f}%" if wt > 0.001 else "0%"
            t_html += (
                f'<tr style="background:{row_bg};border-bottom:1px solid #1e2a3a;">'
                f'<td style="padding:6px 10px;color:{nm_clr}">{etf_nm}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:#94a3b8;'
                f'font-family:monospace">{tkr}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:{wt_clr};'
                f'font-weight:800;font-size:14px">{wt_txt}</td>'
                f'<td style="padding:6px 10px;color:#9ca3af">{note}</td>'
                f'</tr>'
            )
        t_html += '</table>'
        st.markdown(t_html, unsafe_allow_html=True)

        st.markdown("---")

        # ════════════════════
        # 서브탭: 과거 백테스트 | KODEX 실거래
        # ════════════════════
        stab_kodex, stab_hist = st.tabs(["🏦 KODEX 실거래 (2021~)", "📈 과거 백테스트 (장기 지수)"])

        with stab_hist:
            fee8 = st.session_state.get("applied_fee", 0.0)
            pf_h = _daily_pf_series_s8(fee_pct=fee8)
            if not pf_h:
                st.warning("데이터 없음")
            else:
                s8_h = pf_h["전략8"]
                y0h = s8_h.index[0].year; y1h = s8_h.index[-1].year

                # SP500 / KOSPI 벤치마크 (시작=100 리베이스)
                def _rebase(key, start_dt):
                    s = raw[key].copy()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s = s[s.index >= start_dt].dropna()
                    return s / s.iloc[0] * 100 if len(s) > 0 else pd.Series(dtype=float)

                sp_bh = _rebase("sp500", s8_h.index[0])
                ko_bh = _rebase("kospi", s8_h.index[0])

                c8, g8, m8 = _stats(s8_h)
                cs_, gs_, ms_ = _stats(sp_bh)
                ck_, gk_, mk_ = _stats(ko_bh)

                c1h, c2h, c3h = st.columns(3)
                with c1h:
                    st.markdown(
                        _card_html(f"🌏 전략8 백테스트 ({y0h}~{y1h})",
                                   "#0a1020", "#38bdf8", c8, g8, m8, large=True),
                        unsafe_allow_html=True)
                with c2h:
                    st.markdown(
                        _card_html("🇺🇸 S&P500 BH", "#0d0d20", "#6366f1", cs_, gs_, ms_),
                        unsafe_allow_html=True)
                with c3h:
                    st.markdown(
                        _card_html("🇰🇷 KOSPI BH", "#111827", "#22c55e", ck_, gk_, mk_),
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                fig_h = go.Figure()
                for s_plot, nm_h, clr_h, w_h in [
                    (s8_h,  "🌏 전략8",    "#38bdf8", 2.5),
                    (sp_bh, "🇺🇸 S&P500", "#6366f1", 1.0),
                    (ko_bh, "🇰🇷 KOSPI",  "#22c55e", 1.0),
                ]:
                    sh = s_plot.resample("W").last()
                    fig_h.add_trace(go.Scatter(
                        x=sh.index, y=sh.values, name=nm_h, mode="lines",
                        line=dict(color=clr_h, width=w_h),
                        hovertemplate=(f"<b>{nm_h}</b><br>%{{x|%Y-%m-%d}}"
                                       f"<br>%{{y:.1f}}<extra></extra>"),
                    ))
                fig_h.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    height=380, margin=dict(l=0, r=0, t=44, b=0),
                    title=dict(text=f"누적 성과 비교 (시작=100, {y0h}~{y1h})",
                               font=dict(size=13, color="#f1f5f9")),
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                    xaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                               tickfont=dict(size=10, color="#9ca3af")),
                    yaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                               tickfont=dict(size=10)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_h, use_container_width=True)

                st.markdown("#### 📅 월별 수익률 히트맵 (전략8)")
                st.markdown(_monthly_heatmap_html(s8_h), unsafe_allow_html=True)

        with stab_kodex:
            with st.spinner("KODEX ETF 데이터 로딩 중..."):
                pf_k = _daily_pf_series_kodex()

            if not pf_k:
                st.warning("KODEX ETF 데이터를 불러올 수 없습니다. 네트워크를 확인해 주세요.")
            else:
                s8_k = pf_k["전략8_KODEX"]
                sp_k = pf_k["SP500"]
                ko_k = pf_k["KOSPI"]
                ck, gk, mk   = _stats(s8_k)
                csp, gsp, msp = _stats(sp_k)
                cko, gko, mko = _stats(ko_k)
                y0k = s8_k.index[0].strftime("%Y-%m")
                y1k = s8_k.index[-1].strftime("%Y-%m")

                c1k, c2k, c3k = st.columns(3)
                with c1k:
                    st.markdown(
                        _card_html(f"🏦 전략8 KODEX ({y0k}~)",
                                   "#0a1020", "#38bdf8", ck, gk, mk, large=True),
                        unsafe_allow_html=True)
                with c2k:
                    st.markdown(
                        _card_html("🇺🇸 S&P500 BH (KODEX)", "#0d0d20", "#6366f1",
                                   csp, gsp, msp),
                        unsafe_allow_html=True)
                with c3k:
                    st.markdown(
                        _card_html("🇰🇷 KOSPI BH (KODEX)", "#111827", "#22c55e",
                                   cko, gko, mko),
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(
                    f"📌 KODEX/ACE ETF 실제 가격 기반 · {y0k} ~ {y1k} · 전략8 동일 로직 적용 "
                    f"· 벤치마크: KODEX 미국S&P500(379800) / KODEX 200(069500)")

                fig_k = go.Figure()
                for s_k, nm_k, clr_k, w_k in [
                    (s8_k, "🏦 전략8 KODEX", "#38bdf8", 2.5),
                    (sp_k, "🇺🇸 S&P500",    "#6366f1", 1.0),
                    (ko_k, "🇰🇷 KOSPI",     "#22c55e", 1.0),
                ]:
                    sk_s = s_k
                    fig_k.add_trace(go.Scatter(
                        x=sk_s.index, y=sk_s.values, name=nm_k, mode="lines",
                        line=dict(color=clr_k, width=w_k),
                        hovertemplate=(f"<b>{nm_k}</b><br>%{{x|%Y-%m-%d}}"
                                       f"<br>%{{y:.1f}}<extra></extra>"),
                    ))
                fig_k.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    height=380, margin=dict(l=0, r=0, t=44, b=0),
                    title=dict(text=f"KODEX ETF 기반 누적 성과 ({y0k}~{y1k})",
                               font=dict(size=13, color="#f1f5f9")),
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                    xaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                               tickfont=dict(size=10, color="#9ca3af")),
                    yaxis=dict(showgrid=True, gridcolor="#1e2a3a",
                               tickfont=dict(size=10)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_k, use_container_width=True)

                st.markdown("#### 📅 월별 수익률 히트맵 (전략8 KODEX)")
                st.markdown(_monthly_heatmap_html(s8_k), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
