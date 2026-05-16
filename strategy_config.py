"""
전략8 설정 파일 — 키움 자동매매 프로그램에서 import하여 사용
=============================================================
▶ 사용법:
    from strategy_config import get_signal

    signal = get_signal()
    print(signal)
    # {
    #   'season': 'May-Oct',
    #   'sig_year': 2026,
    #   'digit': 6,
    #   'sp_inv': True,
    #   'ko_inv': True,
    #   'cs_inv': True,
    #   'weights': { 'sp500':0.10, 'nasdaq':0.10, 'kospi':0.10, ... }
    # }

▶ 포트폴리오 구성 (전략8)
    - 주식 (변동): SP500 10% + 나스닥 10%(투자시즌만) + KOSPI 10% + 코스닥 10%(투자시즌만) + CSI300 10~20%
    - 금 (고정):   20%
    - 한국채10년 (고정): 10%  → KODEX 국채선물10년 (152380)
    - 미국채10년 (고정): 10%  → KODEX 미국10년국채선물 (308620)
    - 현금대체 (잔여): 나머지 → KODEX 국고채3년 (114260)

▶ 리밸런싱 시점
    - Nov-Apr 시즌 시작: 매년 11월 1일 (또는 첫 거래일)
    - May-Oct 시즌 시작: 매년 5월 1일 (또는 첫 거래일)

▶ 시즌 판단 기준
    - 기준연도(sig_year): Nov~Dec → 당해연도, Jan~Apr → 전년도, May~Oct → 당해연도
    - 끝자리(digit) = sig_year % 10
    - 해당 끝자리가 inv_na(Nov-Apr) 또는 inv_mo(May-Oct) 집합에 포함되면 투자시즌

▶ 끝자리별 투자시즌 집합 (과거 1970~2025, 양수확률 60% 이상 기준)
"""

import json
import pathlib
from datetime import date, timedelta

# ─────────────────────────────────────────────
# 투자시즌 끝자리 집합 (백테스트 기반 고정값)
# ─────────────────────────────────────────────
STRATEGY = {
    "sp500": {
        "inv_na": {0, 1, 2, 4, 5, 6, 7, 8, 9},   # Nov-Apr 투자 끝자리
        "inv_mo": {2, 3, 4, 5, 6, 7, 9},           # May-Oct 투자 끝자리
        # 끝자리별 양수확률 (참고용)
        "prob_na": {0:0.83, 1:0.67, 2:0.83, 3:0.50, 4:0.83,
                    5:0.83, 6:0.80, 7:0.80, 8:0.80, 9:0.60},
        "prob_mo": {0:0.50, 1:0.33, 2:0.67, 3:0.83, 4:0.83,
                    5:0.83, 6:1.00, 7:0.60, 8:0.40, 9:1.00},
    },
    "kospi": {
        "inv_na": {0, 2, 3, 4, 5, 6, 8},
        "inv_mo": {0, 3, 5, 7},
        "prob_na": {0:0.80, 1:0.40, 2:0.80, 3:0.80, 4:0.80,
                    5:0.60, 6:0.75, 7:0.25, 8:1.00, 9:0.25},
        "prob_mo": {0:0.60, 1:0.40, 2:0.20, 3:0.60, 4:0.40,
                    5:0.80, 6:0.50, 7:0.75, 8:0.25, 9:0.50},
    },
    "csi300": {
        "inv_na": {2, 5, 6, 8},
        "inv_mo": {0, 4, 6, 7},
        "prob_na": {0:0.50, 1:0.00, 2:1.00, 3:0.50, 4:0.50,
                    5:0.67, 6:1.00, 7:0.00, 8:1.00, 9:0.50},
        "prob_mo": {0:1.00, 1:0.00, 2:0.00, 3:0.00, 4:1.00,
                    5:0.33, 6:1.00, 7:1.00, 8:0.00, 9:0.50},
    },
}

# ─────────────────────────────────────────────
# ETF 티커 (키움 6자리)
# ─────────────────────────────────────────────
ETF_TICKERS = {
    "sp500":  "379800",   # KODEX 미국S&P500
    "nasdaq": "379810",   # KODEX 미국나스닥100
    "kospi":  "069500",   # KODEX 200
    "kosdaq": "229200",   # KODEX 코스닥150
    "csi300": "283580",   # KODEX 차이나CSI300
    "gold":   "411060",   # ACE KRX금현물
    "krbond": "114260",   # KODEX 국고채3년 (현금대체)
    "kr10y":  "152380",   # KODEX 국채선물10년 (한국, 고정 10%)
    "us10y":  "308620",   # KODEX 미국10년국채선물 (고정 10%)
}

# ─────────────────────────────────────────────
# 현재 시즌 판단
# ─────────────────────────────────────────────
def current_season(today=None):
    """현재 시즌 정보 반환"""
    if today is None:
        today = date.today()
    m, y = today.month, today.year
    if m in [11, 12]:
        return "Nov-Apr", y
    elif m in [1, 2, 3, 4]:
        return "Nov-Apr", y - 1
    else:
        return "May-Oct", y


# ─────────────────────────────────────────────
# 현재 투자 신호 + 목표 비중 계산
# ─────────────────────────────────────────────
def get_signal(today=None):
    """
    전략8 현재 투자 신호 및 목표 비중 반환

    Returns:
        dict: {
            'season'   : 'Nov-Apr' or 'May-Oct',
            'sig_year' : int,
            'digit'    : int (0~9),
            'sp_inv'   : bool,
            'ko_inv'   : bool,
            'cs_inv'   : bool,
            'weights'  : {ticker: float, ...}  ← 합계 = 1.0
        }
    """
    season, sig_year = current_season(today)
    digit = sig_year % 10

    inv_key = "inv_na" if season == "Nov-Apr" else "inv_mo"
    sp_inv  = digit in STRATEGY["sp500"][inv_key]
    ko_inv  = digit in STRATEGY["kospi"][inv_key]
    cs_inv  = digit in STRATEGY["csi300"][inv_key]

    # 비중 계산
    w_sp  = 0.10
    w_nq  = 0.10 if sp_inv else 0.0
    w_ko  = 0.10
    w_kq  = 0.10 if ko_inv else 0.0
    w_cs  = 0.20 if cs_inv else 0.10
    w_kr10 = 0.10   # 한국채10년 고정
    w_us10 = 0.10   # 미국채10년 고정
    w_gold = 0.20   # 금 고정
    w_cash = round(1.0 - w_sp - w_nq - w_ko - w_kq - w_cs - w_gold - w_kr10 - w_us10, 6)

    weights = {
        ETF_TICKERS["sp500"]:  w_sp,
        ETF_TICKERS["nasdaq"]: w_nq,
        ETF_TICKERS["kospi"]:  w_ko,
        ETF_TICKERS["kosdaq"]: w_kq,
        ETF_TICKERS["csi300"]: w_cs,
        ETF_TICKERS["gold"]:   w_gold,
        ETF_TICKERS["kr10y"]:  w_kr10,
        ETF_TICKERS["us10y"]:  w_us10,
        ETF_TICKERS["krbond"]: w_cash,   # 현금대체 (잔여 전액)
    }

    return {
        "season":   season,
        "sig_year": sig_year,
        "digit":    digit,
        "sp_inv":   sp_inv,
        "ko_inv":   ko_inv,
        "cs_inv":   cs_inv,
        "weights":  weights,
    }


def next_rebalance_date(today=None):
    """다음 리밸런싱 날짜 반환"""
    if today is None:
        today = date.today()
    season, _ = current_season(today)
    if season == "Nov-Apr":
        y = today.year + 1 if today.month <= 4 else today.year
        return date(y, 5, 1)
    else:
        return date(today.year, 11, 1)


# ─────────────────────────────────────────────
# 단독 실행 시 현재 신호 출력
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 전략8+ 트레일링 스탑
# ─────────────────────────────────────────────
TRAIL_STATE_FILE = pathlib.Path(__file__).parent / "trail_state.json"

TRAIL_TICKERS = {
    "sp500":  "^GSPC",
    "nasdaq": "^IXIC",
    "kospi":  "^KS11",
    "kosdaq": "^KQ11",
    "csi300": "000300.SS",
}

_TRAIL_ASSETS  = ["sp500", "nasdaq", "kospi", "kosdaq", "csi300"]
_STOP_PCT      = 0.15   # rolling 고점 대비 -15% → 손절
_RESUME_PCT    = 0.05   # rolling 저점 대비 +5%  → 복귀
_SAFE_ASSETS   = ["gold", "kr10y", "us10y"]   # 손절 비중 1/3씩 배분


def load_trail_state() -> dict:
    if TRAIL_STATE_FILE.exists():
        try:
            return json.loads(TRAIL_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {k: {"state": "normal", "peak": 0.0, "trough": 0.0}
            for k in _TRAIL_ASSETS}


def save_trail_state(state: dict):
    out = {k: v for k, v in state.items()}
    out["last_updated"] = date.today().isoformat()
    TRAIL_STATE_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_prices_yf() -> dict:
    try:
        import yfinance as yf
    except ImportError:
        return {}
    prices = {}
    for asset, ticker in TRAIL_TICKERS.items():
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True,
                             progress=False, multi_level_index=False)
            if not df.empty:
                prices[asset] = float(df["Close"].dropna().iloc[-1])
        except Exception:
            pass
    return prices


def _apply_one_price(s: dict, price: float) -> dict:
    """상태 dict에 가격 하나를 적용해 peak/trough/state 갱신 후 반환."""
    if s["state"] == "normal":
        if s["peak"] <= 0:
            s["peak"] = price
        elif price > s["peak"]:
            s["peak"] = price
        if s["peak"] > 0 and price < s["peak"] * (1 - _STOP_PCT):
            s["state"]  = "bear"
            s["trough"] = price
    else:  # bear
        if s["trough"] <= 0 or price < s["trough"]:
            s["trough"] = price
        if s["trough"] > 0 and price > s["trough"] * (1 + _RESUME_PCT):
            s["state"]  = "normal"
            s["peak"]   = price
            s["trough"] = price
    return s


def init_trail_state(start_date: str = "2020-01-01") -> dict:
    """최초 1회 — start_date부터 오늘까지 전체 히스토리로 rolling 상태 계산 후 저장."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    state = {k: {"state": "normal", "peak": 0.0, "trough": 0.0}
             for k in _TRAIL_ASSETS}

    for asset, ticker in TRAIL_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if df.empty:
                continue
            s = {"state": "normal", "peak": 0.0, "trough": 0.0}
            for price in df["Close"].dropna():
                s = _apply_one_price(s, float(price))
            state[asset] = s
        except Exception:
            pass

    save_trail_state(state)
    return state


def catchup_trail_state() -> dict:
    """프로그램 시작 시 — last_updated 이후 누락된 거래일 데이터를 일괄 반영."""
    state = load_trail_state()
    last_str = state.get("last_updated", "")

    if not last_str:
        return init_trail_state()

    try:
        last_date = date.fromisoformat(last_str)
    except ValueError:
        return init_trail_state()

    if (date.today() - last_date).days <= 1:
        return state  # 최신 상태 — 갱신 불필요

    try:
        import yfinance as yf
    except ImportError:
        return state

    start = (last_date + timedelta(days=1)).isoformat()
    for asset, ticker in TRAIL_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if df.empty:
                continue
            s = state.get(asset, {"state": "normal", "peak": 0.0, "trough": 0.0})
            for price in df["Close"].dropna():
                s = _apply_one_price(s, float(price))
            state[asset] = s
        except Exception:
            pass

    save_trail_state(state)
    return state


def update_trail_state(prices: dict = None) -> dict:
    """매일 장 마감 후 — 오늘 종가 1개로 상태 갱신 후 저장.
    prices가 None이면 yfinance 자동 조회."""
    if prices is None:
        prices = _fetch_prices_yf()

    state = load_trail_state()

    for asset in _TRAIL_ASSETS:
        price = prices.get(asset)
        if not price or price <= 0:
            continue
        s = state.get(asset, {"state": "normal", "peak": 0.0, "trough": 0.0})
        state[asset] = _apply_one_price(s, price)

    save_trail_state(state)
    return state


def get_signal_plus(today=None) -> dict:
    """전략8 + trailing stop 반영 비중 반환."""
    sig     = get_signal(today)
    weights = dict(sig["weights"])
    state   = load_trail_state()

    for asset in _TRAIL_ASSETS:
        if state.get(asset, {}).get("state") != "bear":
            continue
        code = ETF_TICKERS.get(asset)
        if not code or weights.get(code, 0.0) == 0.0:
            continue
        freed = weights[code]
        weights[code] = 0.0
        per_safe = freed / len(_SAFE_ASSETS)
        for safe in _SAFE_ASSETS:
            safe_code = ETF_TICKERS.get(safe)
            if safe_code:
                weights[safe_code] = round(weights.get(safe_code, 0.0) + per_safe, 6)

    result = dict(sig)
    result["weights"]     = weights
    result["trail_state"] = state
    return result


if __name__ == "__main__":
    sig = get_signal()
    nxt = next_rebalance_date()

    print("=" * 55)
    print("  전략8 현재 투자 신호")
    print("=" * 55)
    print(f"  시즌       : {sig['season']}")
    print(f"  기준연도   : {sig['sig_year']}년 (끝자리 {sig['digit']})")
    print(f"  SP500/나스닥: {'✅ 투자시즌' if sig['sp_inv'] else '💤 비투자시즌'}")
    print(f"  KOSPI/코스닥: {'✅ 투자시즌' if sig['ko_inv'] else '💤 비투자시즌'}")
    print(f"  CSI300     : {'✅ 투자시즌' if sig['cs_inv'] else '💤 비투자시즌'}")
    print(f"  다음 리밸런싱: {nxt.strftime('%Y-%m-%d')}")
    print()
    print("  목표 비중:")
    for ticker, w in sig['weights'].items():
        name = {v: k for k, v in ETF_TICKERS.items()}[ticker]
        bar = "█" * int(w * 100 // 2)
        print(f"    {ticker} ({name:7s}): {w*100:5.1f}%  {bar}")
    print(f"  {'합계':>25}: {sum(sig['weights'].values())*100:.1f}%")
    print("=" * 55)
