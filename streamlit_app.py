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
                r_stk   = rs if inv else 0.0

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
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7 = st.tabs(["📊 자산배분", "📉 역대 폭락일", "🔍 폭락 후 전략", "📈 시장 사이클", "📡 저점 레이더", "📅 연간 수익률", "📝 메모장"])

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

        # ── 4. 누적 성과 차트 ──
        st.markdown('<div class="section-title">📊 누적 성과 차트</div>', unsafe_allow_html=True)

        def _chart_series(key):
            """전략7 / 비교군 전체를 일별 계산 → 주별 리샘플해서 반환"""
            sim = sims.get(key, pd.DataFrame())
            if sim.empty: return {}
            bond_key = "krbond" if key in ("kospi","kosdaq") else "us30y"
            d_stk = raw[key].pct_change().dropna()
            d_gld = raw["gold"].pct_change().dropna()
            d_bnd = (-DURATION_US*raw["us30y"].diff()/100).dropna() if bond_key=="us30y" \
                    else raw["krbond"].pct_change().dropna()
            for s in [d_stk, d_gld, d_bnd]:
                s.index = pd.to_datetime(s.index).tz_localize(None)
            px = raw[key].copy(); px.index = pd.to_datetime(px.index).tz_localize(None)
            dl = px.diff()
            rsi = 100 - 100/(1+dl.clip(lower=0).rolling(14).mean() /
                              (-dl.clip(upper=0)).rolling(14).mean().replace(0,np.nan))
            ma200=px.rolling(200).mean(); mag=(px-ma200)/ma200*100
            ma20=px.rolling(20).mean(); s20=px.rolling(20).std()
            bbr=(ma20+2*s20)-(ma20-2*s20)
            bbp=(px-(ma20-2*s20))/bbr.replace(0,np.nan)*100
            dd=(px-px.cummax())/px.cummax()*100
            if key in ("kospi","kosdaq","csi300"):
                sv=(px.pct_change().rolling(20).std()*(252**0.5)*100>=35).astype(int)
            else:
                vh=load_vix_history(); vh.index=pd.to_datetime(vh.index).tz_localize(None)
                sv=(vh.reindex(px.index,method="ffill").fillna(20)>=40).astype(int)
            sig=((dd<=-30).astype(int)+(rsi<=30).astype(int)+
                 (mag<=-15).astype(int)+(bbp<=0).astype(int)+sv).fillna(0).to_dict()
            im={(int(r["연도"]),r["시즌"]):bool(r["투자"]) for _,r in sim.iterrows()}
            cols=["전략7","BH_max","BH_min","주식단독"]
            v={c:100.0 for c in cols}; vals={c:[] for c in cols}; dates=[]; sell=False
            gi=set(d_gld.index); bi=set(d_bnd.index)
            for dt in d_stk.index:
                mo=dt.month
                if mo>=11: sea,sy="Nov-Apr",dt.year
                elif mo<=4: sea,sy="Nov-Apr",dt.year-1
                else: sea,sy="May-Oct",dt.year
                inv=im.get((sy,sea),False)
                rs=float(d_stk[dt]); rg=float(d_gld[dt]) if dt in gi else 0; rb=float(d_bnd[dt]) if dt in bi else 0
                s=float(sig.get(dt,0))
                if s>=2: sell=True
                elif s==0: sell=False
                w7=0 if sell else (0.5 if inv else 0.25)
                v["전략7"]   *= (1+w7*rs+0.25*rg+0.25*rb)
                v["BH_max"]  *= (1+0.50*rs+0.25*rg+0.25*rb)
                v["BH_min"]  *= (1+0.25*rs+0.25*rg+0.25*rb)
                v["주식단독"] *= (1+rs)   # 순수 주식 BH (시즌 무관)
                for c in cols: vals[c].append(v[c])
                dates.append(dt)
            return {c: pd.Series(vals[c], index=dates).resample("W").last() for c in cols}

        def make_perf_chart(chart_data, title, accent):
            if not chart_data:
                fig = go.Figure()
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    height=380, margin=dict(l=0, r=0, t=60, b=0),
                    annotations=[dict(text="데이터를 불러오는 중입니다. 잠시 후 새로고침 해주세요.",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(color="#9ca3af", size=14))],
                )
                return fig
            fig = go.Figure()
            cfg = [
                ("전략7",   "전략7 포트폴리오", accent,    3.0),
                ("BH_max",  "주식50% BH",      "#6b7280", 1.5),
                ("BH_min",  "주식25% BH",      "#4b5563", 1.5),
                ("주식단독", "주식단독",         "#fbbf24", 1.5),
            ]
            for col, name, c, w in cfg:
                s = chart_data.get(col)
                if s is None or s.empty: continue
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=name, mode="lines",
                    line=dict(color=c, width=w),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
                ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
                legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=9)),
                yaxis=dict(showgrid=True, gridcolor="#1e2a3a"),
                hovermode="x unified", height=380,
                margin=dict(l=0, r=0, t=60, b=0),
            )
            return fig

        tab_s, tab_n, tab_k, tab_d, tab_kq, tab_cn = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"])
        for tab, key in [(tab_s,"sp500"), (tab_n,"nasdaq"), (tab_k,"kospi"), (tab_d,"dow"), (tab_kq,"kosdaq"), (tab_cn,"csi300")]:
            with tab:
                m = meta[key]
                st.plotly_chart(
                    make_perf_chart(_chart_series(key),
                                    f"{m['name']} 자산배분 포트폴리오 누적성과 (시작=100)",
                                    m["color"]),
                    use_container_width=True,
                )

        # ── 5. 장기 백테스트 성과 카드 ──
        st.markdown('<div class="section-title">📊 장기 백테스트 성과</div>', unsafe_allow_html=True)

        def _daily_pf_series(key):
            """일별 포트폴리오 가치 시리즈 반환 — 전략6/전략7/BH/주식단독"""
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

            for dt in d_stk.index:
                mo = dt.month
                if mo >= 11:   season, sy = "Nov-Apr", dt.year
                elif mo <= 4:  season, sy = "Nov-Apr", dt.year - 1
                else:          season, sy = "May-Oct", dt.year

                invest = invest_map.get((sy, season), False)
                rs = float(d_stk[dt])
                rg = float(d_gold[dt]) if dt in gold_idx else 0.0
                rb = float(d_bond[dt]) if dt in bond_idx else 0.0

                # 전략7 매도 상태 갱신
                s = float(sig_dict.get(dt, 0))
                if   s >= 2: s7_sell = True
                elif s == 0: s7_sell = False
                # s == 1 → 상태 유지

                w6 = 0.50 if invest else 0.25
                w7 = 0.0  if s7_sell else w6

                v["전략6"]   *= (1 + w6*rs  + 0.25*rg + 0.25*rb)
                v["전략7"]   *= (1 + w7*rs  + 0.25*rg + 0.25*rb)
                v["BH_max"]  *= (1 + 0.50*rs + 0.25*rg + 0.25*rb)
                v["BH_min"]  *= (1 + 0.25*rs + 0.25*rg + 0.25*rb)
                v["주식단독"] *= (1 + rs)   # 순수 주식 BH (시즌 무관)
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

        def _render_perf_tab(key):
            pf = _daily_pf_series(key)
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

            # ── 전략7 히어로 카드 ──
            hero = _card_html("⚡ 전략7  (전략6 + 매도신호≥2 시 주식0%)",
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

            # ── 전략7 vs 전략6 비교 차트 ──
            fig_c = go.Figure()
            chart_cfg = [
                ("전략7", "⚡ 전략7", "#a78bfa", 2.5),
                ("전략6", "⚙️ 전략6", "#34d399", 1.5),
                ("BH_max","📈 주식50%BH", "#6366f1", 1.0),
                ("BH_min","📊 주식25%BH", "#6b7280", 1.0),
            ]
            for col, name, color, width in chart_cfg:
                s = pf[col].resample("W").last()  # 주별 샘플링
                fig_c.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=name, mode="lines",
                    line=dict(color=color, width=width),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
                ))
            fig_c.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                height=380, margin=dict(l=0, r=0, t=40, b=0),
                title=dict(text="전략7 vs 전략6 누적 성과 비교 (시작=100)", font=dict(size=13, color="#f1f5f9")),
                legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10, color="#9ca3af")),
                yaxis=dict(showgrid=True, gridcolor="#1e2a3a", tickfont=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_c, use_container_width=True)

        ptab_s, ptab_n, ptab_k, ptab_d, ptab_kq, ptab_cn = st.tabs(
            ["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI", "🏛️ DOW", "📱 KOSDAQ", "🇨🇳 CSI300"]
        )
        for ptab, pkey in [
            (ptab_s,"sp500"),(ptab_n,"nasdaq"),(ptab_k,"kospi"),
            (ptab_d,"dow"),(ptab_kq,"kosdaq"),(ptab_cn,"csi300")
        ]:
            with ptab:
                _render_perf_tab(pkey)

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
    # TAB 7: 메모장
    # ════════════════════════════════════════
    with main_tab7:
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


if __name__ == "__main__":
    main()
