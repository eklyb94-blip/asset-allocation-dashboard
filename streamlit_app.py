"""
자산배분 포트폴리오 대시보드
- 현금보유 시: 주식25% / 현금25% / 금25% / 국채30년25%
- 주식투자 시: 주식50% / 현금0%  / 금25% / 국채30년25%
- 신호: 끝자리 전략6 (수익확률 >= 60% 끝자리만 투자)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import io
import warnings
warnings.filterwarnings("ignore")

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
DURATION_US = 18.0   # 미국 30년채 Modified Duration 근사

# ═══════════════════════════════════════════
# 데이터 로딩 (1시간 캐시)
# ═══════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_raw():
    def dl(ticker, start):
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                         multi_level_index=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        return df["Close"].dropna()

    return {
        "sp500":  dl("^GSPC",     "1985-01-01"),
        "nasdaq": dl("^IXIC",     "1985-01-01"),
        "kospi":  dl("^KS11",     "1985-01-01"),
        "gold":   dl("GC=F",      "2000-01-01"),
        "us30y":  dl("^TYX",      "1985-01-01"),
        "krbond": dl("114820.KS", "2009-01-01"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_ohlc():
    tickers = {
        "sp500":  ("^GSPC", "1985-01-01"),
        "nasdaq": ("^IXIC", "1985-01-01"),
        "kospi":  ("^KS11", "1990-01-01"),
    }
    result = {}
    for key, (ticker, start) in tickers.items():
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                         multi_level_index=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            result[key] = pd.DataFrame(columns=["Close", "prev_close", "daily_ret"])
            continue
        df = df[["Close"]].dropna()
        df["prev_close"] = df["Close"].shift(1)
        df["daily_ret"]  = df["Close"].pct_change()
        df = df.dropna()
        result[key] = df
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
    for key in ["sp500", "nasdaq", "kospi"]:
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
        bond_key = "krbond" if stock_key == "kospi" else "us30y"
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

    sims = {k: simulate(k) for k in ["sp500", "nasdaq", "kospi"]}
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
        ("sp500",  "1987-10-19"): "블랙 먼데이 (프로그램 매매 폭주)",
        ("nasdaq", "1987-10-19"): "블랙 먼데이 (프로그램 매매 폭주)",
        ("kospi",  "1987-10-20"): "블랙 먼데이 글로벌 여파",
        ("sp500",  "2001-09-17"): "9/11 테러 후 시장 재개",
        ("nasdaq", "2001-09-17"): "9/11 테러 후 시장 재개",
        ("kospi",  "2001-09-12"): "9/11 테러 글로벌 여파",
        ("sp500",  "2008-09-29"): "금융위기 - 하원 TARP 법안 부결",
        ("sp500",  "2008-10-15"): "글로벌 금융위기 (리먼 파산 여파)",
        ("nasdaq", "2008-10-15"): "글로벌 금융위기 (리먼 파산 여파)",
        ("sp500",  "2020-03-16"): "COVID-19 - 미국 전국 비상사태 선포",
        ("nasdaq", "2020-03-16"): "COVID-19 - 미국 전국 비상사태 선포",
        ("sp500",  "2020-03-12"): "COVID-19 - 유럽 여행 금지 발표",
        ("nasdaq", "2020-03-12"): "COVID-19 - 유럽 여행 금지 발표",
        ("kospi",  "2020-03-19"): "COVID-19 - 서킷브레이커 발동",
        ("kospi",  "1997-11-24"): "IMF 외환위기 - 구제금융 신청",
        ("kospi",  "1997-12-12"): "IMF 외환위기 - 원화 급락",
        ("kospi",  "2008-10-24"): "글로벌 금융위기 - 원/달러 1,500원 돌파",
        ("nasdaq", "2000-04-14"): "닷컴 버블 붕괴 - 나스닥 역대 최대 주간 낙폭",
        ("nasdaq", "2000-04-03"): "닷컴 버블 붕괴 - 마이크로소프트 반독점 판결",
        ("sp500",  "2022-09-13"): "인플레이션 - 예상 상회한 CPI 충격",
        ("nasdaq", "2022-09-13"): "인플레이션 - 예상 상회한 CPI 충격",
    }

    k = (key, ds)
    if k in specific:
        return specific[k]

    # 기간별 분류
    if y == 1987 and m in [10, 11]:
        return "블랙 먼데이 여파"
    if y == 1989 and m == 10:
        return "미니 크래시 (블랙 프라이데이)"
    if key == "kospi" and y == 1997 and m >= 7:
        return "IMF 외환위기 (아시아 금융위기)"
    if key == "kospi" and y == 1998 and m <= 8:
        return "IMF 외환위기 구조조정"
    if y == 1997 and m in [10, 11] and key != "kospi":
        return "아시아 금융위기 글로벌 여파"
    if y == 1998 and m in [8, 9]:
        return "러시아 금융위기 / LTCM 헤지펀드 붕괴"
    if y == 2000 and key in ["sp500", "nasdaq"]:
        return "닷컴 버블 붕괴"
    if y == 2001 and m == 9:
        return "9/11 테러 여파"
    if y == 2001 and key in ["sp500", "nasdaq"]:
        return "닷컴 버블 붕괴 / 경기침체"
    if y == 2002 and key in ["sp500", "nasdaq"]:
        return "닷컴 버블 / 회계 스캔들 (엔론·월드컴)"
    if y == 2007:
        return "서브프라임 모기지 위기 시작"
    if y == 2008:
        return "글로벌 금융위기 (리먼 브라더스 파산)"
    if y == 2009 and m <= 3:
        return "글로벌 금융위기 여파"
    if y == 2010 and m == 5:
        return "플래시 크래시 (알고리즘 매매 오작동)"
    if y == 2011 and m == 8:
        return "미국 신용등급 강등 (S&P) / 유럽 재정위기"
    if y == 2015 and m == 8:
        return "중국 위안화 평가절하 / 경기 둔화 우려"
    if y == 2018 and m == 2:
        return "VIX 급등 / 인플레이션 우려 (볼마게돈)"
    if y == 2018 and m in [10, 11, 12]:
        return "미중 무역전쟁 / 연준 금리 인상"
    if y == 2020 and m in [2, 3]:
        return "COVID-19 팬데믹"
    if y == 2022:
        return "인플레이션 / 연준 공격적 금리 인상"
    if y == 2023 and m == 3:
        return "미국 지역은행 위기 (SVB 파산)"
    if y == 2024 and m == 8:
        return "엔 캐리 트레이드 청산 / 경기침체 우려"
    if y == 2025:
        return "미중 관세 전쟁 / 글로벌 경기 불확실성"

    return "글로벌 경기 불안 / 시장 조정"


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

    # ── 현재 시즌 수익률 ──
    cur = {}
    for key in ["sp500", "nasdaq", "kospi"]:
        rs = season_return(raw[key],   season_start)
        rg = season_return(raw["gold"], season_start)
        rb = season_return(raw["krbond"], season_start) if key == "kospi" \
             else season_return_bond(raw["us30y"], season_start)
        inv = digit in (strategies[key]["inv_na"] if season == "Nov-Apr"
                        else strategies[key]["inv_mo"])
        rp = (0.50*rs + 0.25*rg + 0.25*rb) if (inv and all(v is not None for v in [rs,rg,rb])) \
             else (0.25*rs + 0.25*rg + 0.25*rb) if all(v is not None for v in [rs,rg,rb]) \
             else None
        cur[key] = {"stock": rs, "gold": rg, "bond": rb, "port": rp, "invest": inv}

    meta = {
        "sp500":  {"name": "S&P500", "emoji": "🇺🇸", "bond": "미국채30년", "color": "#5b9bd5"},
        "nasdaq": {"name": "NASDAQ", "emoji": "💻",  "bond": "미국채30년", "color": "#ef4444"},
        "kospi":  {"name": "KOSPI",  "emoji": "🇰🇷", "bond": "한국채30년", "color": "#22c55e"},
    }

    # ════════════════════════════════════════
    # 최상위 탭
    # ════════════════════════════════════════
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 자산배분", "📉 역대 폭락일", "🔍 폭락 후 전략"])

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

        # ── 2. 현재 권장 자산배분 카드 ──
        st.markdown('<div class="section-title">🎯 현재 권장 자산배분</div>', unsafe_allow_html=True)

        cols3 = st.columns(3)
        for key, col in zip(["sp500", "nasdaq", "kospi"], cols3):
            with col:
                m       = meta[key]
                inv     = cur[key]["invest"]
                rp      = cur[key]["port"]
                stk_pct = 50 if inv else 25
                card    = "invest-card" if inv else "cash-card"
                icon    = "✅" if inv else "💤"
                status  = "투자 시즌" if inv else "현금 보유"
                sc      = "#34d399" if inv else "#fb923c"

                stk_bg  = "#065f46" if inv else "#374151"
                stk_clr = "#34d399" if inv else "#9ca3af"
                badge_stk  = f'<span style="background:{stk_bg};color:{stk_clr};border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">주식&nbsp;{stk_pct}%</span>'
                badge_cash = '' if inv else '<span style="background:#374151;color:#9ca3af;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">현금&nbsp;25%</span>'
                badge_gold = f'<span style="background:#78350f;color:#fbbf24;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">금&nbsp;25%</span>'
                badge_bond = f'<span style="background:#1e3a5f;color:#93c5fd;border-radius:6px;padding:3px 9px;font-size:12px;font-weight:700;">{m["bond"]}&nbsp;25%</span>'
                badges = f'{badge_stk} {badge_cash} {badge_gold} {badge_bond}'

                port_color = clr(rp)
                port_val   = fp(rp)
                st.markdown(
                    f'<div class="{card}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
                    f'<div><div style="color:#cbd5e1;font-size:16px;font-weight:700;letter-spacing:1px;">{m["emoji"]} {m["name"]}</div>'
                    f'<div style="color:#f1f5f9;font-size:24px;font-weight:800;margin-top:2px;">주식 {stk_pct}%</div></div>'
                    f'<div style="text-align:right;"><div style="color:{sc};font-size:13px;font-weight:700;">{icon} {status}</div>'
                    f'<div style="color:#6b7280;font-size:11px;margin-top:2px;">끝자리 {digit}</div></div></div>'
                    f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:14px;">{badges}</div>'
                    f'<div style="border-top:1px solid #1e2a3a;padding-top:12px;">'
                    f'<div style="color:#6b7280;font-size:11px;margin-bottom:4px;">이번 시즌 포트폴리오 수익률</div>'
                    f'<div style="color:{port_color};font-size:26px;font-weight:800;">{port_val}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # ── 3. 이번 시즌 자산별 수익률 ──
        st.markdown('<div class="section-title">📈 이번 시즌 자산별 수익률</div>', unsafe_allow_html=True)

        for key in ["sp500", "nasdaq", "kospi"]:
            m = meta[key]
            c1, c2, c3, c4 = st.columns(4)
            items = [
                ("주식",      cur[key]["stock"]),
                ("금",        cur[key]["gold"]),
                (m["bond"],   cur[key]["bond"]),
                ("포트폴리오", cur[key]["port"]),
            ]
            for col, (label, val) in zip([c1, c2, c3, c4], items):
                is_port = label == "포트폴리오"
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

        def make_perf_chart(sim_df, title, accent):
            if sim_df.empty or "전략6" not in sim_df.columns:
                fig = go.Figure()
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    height=380, margin=dict(l=0, r=0, t=60, b=0),
                    annotations=[dict(text="데이터를 불러오는 중입니다. 잠시 후 새로고침 해주세요.",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(color="#9ca3af", size=14))],
                )
                return fig
            x = [f"{r['연도']}-{r['시즌'][:3]}" for _, r in sim_df.iterrows()]
            cfg = [
                ("전략6",   "전략6 포트폴리오",    accent, 3),
                ("BH_min",  "BH_min (주식25%고정)", "#6b7280", 1.5),
                ("BH_max",  "BH_max (주식50%고정)", "#a3a3a3", 1.5),
                ("주식단독", "주식단독 (끝자리전략)", "#fbbf24", 1.5),
            ]
            fig = go.Figure()
            for col, name, c, w in cfg:
                fig.add_trace(go.Scatter(
                    x=x, y=sim_df[col], name=name, mode="lines",
                    line=dict(color=c, width=w),
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
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

        tab_s, tab_n, tab_k = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI"])
        for tab, key in [(tab_s,"sp500"), (tab_n,"nasdaq"), (tab_k,"kospi")]:
            with tab:
                m = meta[key]
                st.plotly_chart(
                    make_perf_chart(sims[key],
                                    f"{m['name']} 자산배분 포트폴리오 누적성과 (시작=100)",
                                    m["color"]),
                    use_container_width=True,
                )

        # ── 5. 과거 데이터 테이블 ──
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

            s_tab, n_tab, k_tab = st.tabs(["S&P500", "NASDAQ", "KOSPI"])
            for tab, key in [(s_tab,"sp500"), (n_tab,"nasdaq"), (k_tab,"kospi")]:
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

            ds_tab, dn_tab, dk_tab = st.tabs(["S&P500", "NASDAQ", "KOSPI"])
            for tab, key in [(ds_tab,"sp500"), (dn_tab,"nasdaq"), (dk_tab,"kospi")]:
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

        hm_s, hm_n, hm_k = st.tabs(["S&P500", "NASDAQ", "KOSPI"])
        for tab, key in [(hm_s,"sp500"), (hm_n,"nasdaq"), (hm_k,"kospi")]:
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

        drop_s, drop_n, drop_k = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI"])

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
                    "순위":   rank,
                    "날짜":   dt_str,
                    "전일종가": f"{row.prev_close:,.2f}",
                    "종가":    f"{row.Close:,.2f}",
                    "하락률": f"{row.daily_ret*100:.2f}%",
                    "+1일":   fmt_fwd(fwd_ret(idx, 1))  if idx is not None else "N/A",
                    "+5일":   fmt_fwd(fwd_ret(idx, 5))  if idx is not None else "N/A",
                    "+10일":  fmt_fwd(fwd_ret(idx, 10)) if idx is not None else "N/A",
                    "+20일":  fmt_fwd(fwd_ret(idx, 20)) if idx is not None else "N/A",
                    "사유":   reason,
                })

            result_df = pd.DataFrame(rows)

            def style_drop(df):
                def _c(val):
                    s = str(val)
                    if s.endswith("%"):
                        try:
                            v = float(s.replace("%", "").replace("+", ""))
                            col_name = ""
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
                style_drop(result_df),
                use_container_width=True,
                height=980,
                hide_index=True,
            )

        for tab, key in [(drop_s, "sp500"), (drop_n, "nasdaq"), (drop_k, "kospi")]:
            with tab:
                render_drop_tab(key)

    # ════════════════════════════════════════
    # TAB 3: 낙폭 후 투자 시뮬레이터
    # ════════════════════════════════════════
    with main_tab3:
        st.markdown('<div class="section-title">🔍 폭락 후 전략</div>', unsafe_allow_html=True)
        st.caption("폭락일 이후 N일 뒤 진입 시 수익률을 분석합니다. 모든 날짜는 거래일(영업일) 기준입니다.")

        sim_s, sim_n, sim_k = st.tabs(["🇺🇸 S&P500", "💻 NASDAQ", "🇰🇷 KOSPI"])

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
                legend=dict(
                    font=dict(size=10, color="#9ca3af"),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="v",
                    x=1.01, y=1,
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── 엑셀 다운로드 ──
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            fname_map = {"sp500": "SP500", "nasdaq": "NASDAQ", "kospi": "KOSPI"}
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

        for tab, key in [(sim_s, "sp500"), (sim_n, "nasdaq"), (sim_k, "kospi")]:
            with tab:
                render_sim_tab(key)


if __name__ == "__main__":
    main()
