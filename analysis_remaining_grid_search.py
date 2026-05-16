"""
NASDAQ / KOSDAQ / CSI300 트레일링 스탑 그리드서치
- NASDAQ  : 미국채 수익률(^TNX), Duration 8.5yr  — SP500과 동일 방식
- KOSDAQ  : 한국채 수익률(FRED), Duration 2.7yr  — KOSPI와 동일 방식
- CSI300  : 미국채 수익률(^TNX), Duration 8.5yr  — 전략8 실운용 기준
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pathlib, os
import yfinance as yf
import pandas_datareader as pdr

BASE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD    = 0.60
FEE_ONE_WAY  = 0.0035
DURATION_US  = 8.5   # 미국 10년채
DURATION_KR  = 2.7   # 한국 국고채3년

STOP_LEVELS    = [-10, -12, -15, -17, -20, -22, -25, -28, -30, -35]
RECOVER_LEVELS = [+5, +8, +10, +12, +15, +17, +20, +22, +25, +28, +30]

# ══════════════════════════════════════════════════════════
# 공통 데이터 로드
# ══════════════════════════════════════════════════════════
def load_csv(fname):
    df = pd.read_csv(BASE / fname, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df["Close"].dropna().sort_index()

print("공통 데이터 로딩 중...")
gold_d = load_csv("gold_history.csv")

# 미국 10년 국채수익률 (^TNX)
tnx = yf.download("^TNX", start="1970-01-01", auto_adjust=True,
                  progress=False, multi_level_index=False)
if isinstance(tnx.columns, pd.MultiIndex):
    tnx.columns = tnx.columns.get_level_values(0)
us10y_d = tnx["Close"].dropna()
us10y_d.index = pd.to_datetime(us10y_d.index).tz_localize(None)

# 한국 국채수익률 (FRED: INTGSBKRM193N)
print("  한국 국채수익률 (FRED) 다운로드 중...")
kr_yield_raw = pdr.get_data_fred("INTGSBKRM193N", start="1985-01-01")
kr_yield_raw.columns = ["yield"]
kr_yield_d = kr_yield_raw["yield"].dropna()
kr_yield_d.index = pd.to_datetime(kr_yield_d.index).tz_localize(None)
print("  완료\n")

# ══════════════════════════════════════════════════════════
# 공통 함수
# ══════════════════════════════════════════════════════════
def make_bond_us(start):
    """미국채 월별 수익률 (Duration 변환)"""
    m = us10y_d.resample("ME").last()
    rb = (-DURATION_US * m.diff() / 100).dropna()
    return rb[rb.index >= start]

def make_bond_kr(start):
    """한국채 월별 수익률 (쿠폰+Duration 변환)"""
    m = kr_yield_d.resample("ME").last()
    rb = (m / 12 / 100) - (DURATION_KR * m.diff() / 100)
    return rb.dropna()[rb.dropna().index >= start]

def make_gold(start):
    m = gold_d.resample("ME").last()
    rg = m.pct_change().dropna()
    return rg[rg.index >= start]

def digit_invest_sets(stk_daily):
    """끝자리별 양수확률 ≥ THRESHOLD 인 집합 반환"""
    m   = stk_daily.resample("ME").last()
    ret = m.pct_change().dropna()
    df  = pd.DataFrame({"ret": ret})
    df["month"]    = df.index.month
    df["season"]   = df["month"].apply(lambda m: "Nov-Apr" if m in [11,12,1,2,3,4] else "May-Oct")
    df["sig_year"] = df.apply(lambda r: r.name.year-1 if r["month"] in [1,2,3,4] else r.name.year, axis=1)
    na_rows, mo_rows = [], []
    for (sn, sy), g in df.groupby(["season","sig_year"]):
        ret_s = (1 + g["ret"]).prod() - 1
        row   = {"year": sy, "ret": ret_s, "digit": sy % 10}
        (na_rows if sn == "Nov-Apr" else mo_rows).append(row)
    def inv_set(rows):
        if not rows: return set()
        d = pd.DataFrame(rows)
        pmap = {dig: float((d[d["digit"]==dig]["ret"] > 0).mean())
                for dig in range(10) if len(d[d["digit"]==dig]) > 0}
        return set(dig for dig, p in pmap.items() if p >= THRESHOLD)
    return inv_set(na_rows), inv_set(mo_rows)

def is_invest_fn(inv_na, inv_mo):
    def _fn(dt):
        m, y = dt.month, dt.year
        if m in [11,12]:     season, sy = "Nov-Apr", y
        elif m in [1,2,3,4]: season, sy = "Nov-Apr", y-1
        else:                season, sy = "May-Oct",  y
        return (sy % 10) in (inv_na if season == "Nov-Apr" else inv_mo)
    return _fn

def run_grid(stk_m, rs_m, rg_m, rb_m, is_invest):
    """전략6 기준선 계산"""
    common = rs_m.index.intersection(rg_m.index).intersection(rb_m.index)
    rs_m = rs_m.reindex(common); rg_m = rg_m.reindex(common)
    rb_m = rb_m.reindex(common); stk_m = stk_m.reindex(common)

    # 전략6 기준선
    v, vals = 100.0, []
    for dt in common:
        w = 0.50 if is_invest(dt) else 0.25
        v *= (1 + w*float(rs_m[dt]) + 0.25*float(rg_m[dt]) + 0.25*float(rb_m[dt]))
        vals.append(v)
    s6 = pd.Series(vals, index=common)
    n_y = (common[-1]-common[0]).days/365.25
    s6_cagr = ((s6.iloc[-1]/s6.iloc[0])**(1/n_y)-1)*100
    s6_mdd  = ((s6-s6.cummax())/s6.cummax()).min()*100

    def simulate(sp, rp):
        v_n = 100.0
        state = "normal"
        peak = trough = float(stk_m.iloc[0])
        prev = "normal"
        vals_n = []
        for dt in common:
            px = float(stk_m[dt])
            rs = float(rs_m[dt]); rg = float(rg_m[dt]); rb = float(rb_m[dt])
            if state == "normal":
                if px >= peak: peak = px
                if px <= peak * (1 + sp/100): state="bear"; trough=px
            else:
                if px <= trough: trough = px
                if px >= trough * (1 + rp/100): state="normal"; peak=trough=px
            bw = 0.50 if is_invest(dt) else 0.25
            if state == "bear":
                s_ = bw/3
                r_n = (0.25+s_)*rg + (0.25+s_)*rb - (bw*FEE_ONE_WAY if state!=prev else 0)
            else:
                r_n = bw*rs + 0.25*rg + 0.25*rb - (bw*FEE_ONE_WAY if state!=prev else 0)
            prev = state
            v_n *= (1+r_n); vals_n.append(v_n)
        s = pd.Series(vals_n, index=common)
        cagr = (s.iloc[-1]/s.iloc[0])**(1/n_y)-1
        mdd  = ((s-s.cummax())/s.cummax()).min()
        return cagr*100, mdd*100

    results = []
    for sp in STOP_LEVELS:
        for rp in RECOVER_LEVELS:
            cagr, mdd = simulate(sp, rp)
            results.append({"손절":sp,"복귀":rp,"CAGR":cagr,"MDD":mdd,
                            "CAGR차이":cagr-s6_cagr,"MDD차이":mdd-s6_mdd})
    results.sort(key=lambda x: -x["CAGR"])
    return common, s6_cagr, s6_mdd, results

def detail_events(stk_m, rs_m, rg_m, rb_m, is_invest, sp, rp):
    common = rs_m.index.intersection(rg_m.index).intersection(rb_m.index)
    state = "normal"; peak = trough = float(stk_m.reindex(common).iloc[0])
    events = []
    for dt in common:
        px = float(stk_m[dt]) if dt in stk_m.index else np.nan
        if np.isnan(px): continue
        if state == "normal":
            if px >= peak: peak = px
            if px <= peak*(1+sp/100):
                state="bear"; trough=px
                events.append((dt,"매도",px,px/peak-1,peak))
        else:
            if px <= trough: trough=px
            if px >= trough*(1+rp/100):
                events.append((dt,"복귀",px,px/trough-1,trough))
                state="normal"; peak=trough=px
    return events

def count_events(stk_m, rs_m, rg_m, rb_m, is_invest):
    common = rs_m.index.intersection(rg_m.index).intersection(rb_m.index)
    stk_m = stk_m.reindex(common)
    rows = []
    for sp in STOP_LEVELS:
        state="normal"; peak=trough=float(stk_m.iloc[0])
        ns=nb=0
        for dt in common:
            px=float(stk_m[dt])
            if state=="normal":
                if px>=peak: peak=px
                if px<=peak*(1+sp/100): state="bear"; trough=px; ns+=1
            else:
                if px<=trough: trough=px
                if px>=trough*(1+20/100): state="normal"; peak=trough=px; nb+=1
        yr=len(common)/12
        rows.append({"손절":sp,"매도":ns,"복귀":nb,"미복귀":ns-nb,"avg":yr/ns if ns else 0})
    return rows

def print_results(name, common, s6_cagr, s6_mdd, results):
    print()
    print("=" * 84)
    print(f"  {name}  트레일링 스탑 그리드서치  [수수료 편도 0.35%]")
    print(f"  기간: {common[0].date()} ~ {common[-1].date()} ({len(common)}개월, 약{len(common)//12}년)")
    print(f"  전략6 기준: CAGR {s6_cagr:.2f}%  MDD {s6_mdd:.2f}%")
    print("=" * 84)
    print(f"\n  {'손절':>6s} {'복귀':>6s}  {'CAGR':>7s} {'CAGR↑':>7s}  {'MDD':>8s} {'MDD↑':>7s}")
    print(f"  {'-'*55}")
    for r in results[:15]:
        cf = "✅" if r["CAGR차이"]>0 else "❌"
        mf = "✅" if r["MDD차이"]>0 else "❌"
        print(f"  {r['손절']:>5.0f}% {r['복귀']:>5.0f}%  "
              f"{r['CAGR']:>+6.2f}%  {r['CAGR차이']:>+6.2f}%{cf}  "
              f"{r['MDD']:>+7.2f}%  {r['MDD차이']:>+6.2f}%{mf}")
    # -15%/+5% 하이라이트
    r15 = next((r for r in results if r["손절"]==-15 and r["복귀"]==5), None)
    if r15:
        print(f"\n  ★ -15%/+5% 기준:  CAGR {r15['CAGR']:+.2f}% ({r15['CAGR차이']:+.2f}%)  "
              f"MDD {r15['MDD']:+.2f}% ({r15['MDD차이']:+.2f}%)")

def print_heatmap(results):
    df_g = pd.DataFrame(results)
    pivot = df_g.pivot(index="손절", columns="복귀", values="CAGR차이")
    print(f"\n  {'손절↓복귀→':>10s}", end="")
    for rp in RECOVER_LEVELS: print(f"  {rp:>+4.0f}%", end="")
    print()
    print(f"  {'-'*78}")
    for sp in STOP_LEVELS:
        print(f"  {sp:>+9.0f}%", end="")
        for rp in RECOVER_LEVELS:
            val = pivot.loc[sp, rp]
            mk = "★" if val==pivot.values.max() else " "
            print(f"  {val:>+4.2f}{mk}", end="")
        print()

HIST_EVENTS_US = {
    1987:"블랙먼데이 📉", 1990:"걸프전 침체", 1998:"LTCM/러시아",
    2000:"닷컴버블 💥", 2001:"9.11 테러", 2002:"닷컴버블",
    2008:"글로벌금융위기 🔥", 2009:"글로벌금융위기 🔥",
    2011:"유럽재정위기", 2020:"코로나 🦠", 2022:"금리인상쇼크",
    2025:"트럼프관세 📉",
}
HIST_EVENTS_KR = {
    2000:"닷컴버블", 2001:"9.11 테러", 2008:"글로벌금융위기 🔥",
    2011:"유럽재정위기", 2015:"중국경기둔화", 2020:"코로나 🦠",
    2022:"금리인상쇼크", 2024:"계엄령 🇰🇷", 2025:"트럼프관세 📉",
}
HIST_EVENTS_CN = {
    2008:"글로벌금융위기 🔥", 2015:"중국증시급락 📉",
    2018:"미중무역전쟁", 2020:"코로나 🦠",
    2022:"봉쇄/금리충격", 2025:"트럼프관세 📉",
}

def print_events(events, hist):
    n_sell = sum(1 for e in events if e[1]=="매도")
    n_buy  = sum(1 for e in events if e[1]=="복귀")
    print(f"\n  이벤트 로그 (매도 {n_sell}회 / 복귀 {n_buy}회):")
    print(f"  {'날짜':12s} {'액션':6s} {'가격':>10s} {'변동률':>8s} {'기준가':>10s}  배경")
    print(f"  {'-'*68}")
    for dt, action, px, chg, ref in events:
        lbl = "고점" if action=="매도" else "저점"
        h = hist.get(dt.year, "")
        print(f"  {str(dt.date()):12s} {action:6s} {px:>10,.1f} {chg:>+7.1%}  ({lbl} {ref:>9,.1f})  {h}")

def print_counts(counts):
    print(f"\n  {'손절기준':>8s}  {'매도':>4s}  {'복귀':>4s}  {'미복귀':>5s}  평균간격")
    print(f"  {'-'*42}")
    for r in counts:
        os_ = f"({r['미복귀']})" if r['미복귀']>0 else ""
        print(f"  {r['손절']:>+7.0f}%  {r['매도']:>3d}회  {r['복귀']:>3d}회  {os_:>5s}  약{r['avg']:.1f}년마다")


# ══════════════════════════════════════════════════════════
# 1. NASDAQ
# ══════════════════════════════════════════════════════════
print("=" * 84)
print("  [1/3] NASDAQ 분석 중...")
print("=" * 84)

nq_d = load_csv("nasdaq_history.csv")
nq_m = nq_d.resample("ME").last()
rs_nq = nq_m.pct_change().dropna()
rg_nq = make_gold("1970-01-01")
rb_nq = make_bond_us("1970-01-01")
inv_na_nq, inv_mo_nq = digit_invest_sets(nq_d)
is_inv_nq = is_invest_fn(inv_na_nq, inv_mo_nq)
print(f"  NASDAQ 투자시즌 Nov-Apr: {sorted(inv_na_nq)}")
print(f"  NASDAQ 투자시즌 May-Oct: {sorted(inv_mo_nq)}")
print(f"  그리드서치 {len(STOP_LEVELS)*len(RECOVER_LEVELS)}개 조합 계산 중...")

common_nq, s6c_nq, s6m_nq, res_nq = run_grid(nq_m, rs_nq, rg_nq, rb_nq, is_inv_nq)
print_results("NASDAQ", common_nq, s6c_nq, s6m_nq, res_nq)
print_heatmap(res_nq)
r15_nq = next(r for r in res_nq if r["손절"]==-15 and r["복귀"]==5)
evs_nq = detail_events(nq_m, rs_nq, rg_nq, rb_nq, is_inv_nq, -15, 5)
print_events(evs_nq, HIST_EVENTS_US)
cnt_nq = count_events(nq_m, rs_nq, rg_nq, rb_nq, is_inv_nq)
print_counts(cnt_nq)


# ══════════════════════════════════════════════════════════
# 2. KOSDAQ
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 84)
print("  [2/3] KOSDAQ 분석 중...")
print("=" * 84)

kq_raw = yf.download("^KQ11", start="2000-01-01", auto_adjust=True,
                     progress=False, multi_level_index=False)
if isinstance(kq_raw.columns, pd.MultiIndex): kq_raw.columns = kq_raw.columns.get_level_values(0)
kq_d = kq_raw["Close"].dropna()
kq_d.index = pd.to_datetime(kq_d.index).tz_localize(None)

kq_m = kq_d.resample("ME").last()
rs_kq = kq_m.pct_change().dropna()
rg_kq = make_gold("2000-01-01")
rb_kq = make_bond_kr("2000-01-01")
inv_na_kq, inv_mo_kq = digit_invest_sets(kq_d)
is_inv_kq = is_invest_fn(inv_na_kq, inv_mo_kq)
print(f"  KOSDAQ 투자시즌 Nov-Apr: {sorted(inv_na_kq)}")
print(f"  KOSDAQ 투자시즌 May-Oct: {sorted(inv_mo_kq)}")
print(f"  그리드서치 {len(STOP_LEVELS)*len(RECOVER_LEVELS)}개 조합 계산 중...")

common_kq, s6c_kq, s6m_kq, res_kq = run_grid(kq_m, rs_kq, rg_kq, rb_kq, is_inv_kq)
print_results("KOSDAQ", common_kq, s6c_kq, s6m_kq, res_kq)
print_heatmap(res_kq)
evs_kq = detail_events(kq_m, rs_kq, rg_kq, rb_kq, is_inv_kq, -15, 5)
print_events(evs_kq, HIST_EVENTS_KR)
cnt_kq = count_events(kq_m, rs_kq, rg_kq, rb_kq, is_inv_kq)
print_counts(cnt_kq)


# ══════════════════════════════════════════════════════════
# 3. CSI300
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 84)
print("  [3/3] CSI300 분석 중...")
print("=" * 84)

cs_d = load_csv("csi300_history.csv")
cs_m = cs_d.resample("ME").last()
rs_cs = cs_m.pct_change().dropna()
rg_cs = make_gold("2005-01-01")
rb_cs = make_bond_us("2005-01-01")
inv_na_cs, inv_mo_cs = digit_invest_sets(cs_d)
is_inv_cs = is_invest_fn(inv_na_cs, inv_mo_cs)
print(f"  CSI300 투자시즌 Nov-Apr: {sorted(inv_na_cs)}")
print(f"  CSI300 투자시즌 May-Oct: {sorted(inv_mo_cs)}")
print(f"  그리드서치 {len(STOP_LEVELS)*len(RECOVER_LEVELS)}개 조합 계산 중...")

common_cs, s6c_cs, s6m_cs, res_cs = run_grid(cs_m, rs_cs, rg_cs, rb_cs, is_inv_cs)
print_results("CSI300", common_cs, s6c_cs, s6m_cs, res_cs)
print_heatmap(res_cs)
evs_cs = detail_events(cs_m, rs_cs, rg_cs, rb_cs, is_inv_cs, -15, 5)
print_events(evs_cs, HIST_EVENTS_CN)
cnt_cs = count_events(cs_m, rs_cs, rg_cs, rb_cs, is_inv_cs)
print_counts(cnt_cs)


# ══════════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════════
print()
print("=" * 84)
print("  ▶▶▶ 전체 요약 (-15%/+5% 기준, 전략6 대비)")
print("=" * 84)
print(f"  {'지수':10s}  {'기간':>6s}  {'전략6 CAGR':>10s}  {'전략6+ CAGR':>11s}  {'CAGR↑':>7s}  "
      f"{'전략6 MDD':>10s}  {'전략6+ MDD':>10s}  {'MDD↑':>7s}")
print(f"  {'-'*90}")
summaries = [
    ("NASDAQ",  common_nq, s6c_nq, s6m_nq, res_nq),
    ("KOSDAQ",  common_kq, s6c_kq, s6m_kq, res_kq),
    ("CSI300",  common_cs, s6c_cs, s6m_cs, res_cs),
]
# SP500, KOSPI 참고값 추가 (이전 분석 결과)
ref = [
    ("SP500(참고)",  "56년", 8.32, -29.87, 9.18, -25.96),
    ("KOSPI(참고)", "41년", 10.51, -37.18, 14.36, -14.37),
]
for name, cm, s6c, s6m, res in summaries:
    r15 = next((r for r in res if r["손절"]==-15 and r["복귀"]==5), None)
    yrs = f"{len(cm)//12}년"
    if r15:
        print(f"  {name:10s}  {yrs:>6s}  {s6c:>+9.2f}%  {r15['CAGR']:>+10.2f}%  "
              f"{r15['CAGR차이']:>+6.2f}%  {s6m:>+9.2f}%  {r15['MDD']:>+9.2f}%  "
              f"{r15['MDD차이']:>+6.2f}%")
print(f"  {'-'*90}")
for name, yrs, s6c, s6m, tp_c, tp_m in ref:
    print(f"  {name:10s}  {yrs:>6s}  {s6c:>+9.2f}%  {tp_c:>+10.2f}%  "
          f"{tp_c-s6c:>+6.2f}%  {s6m:>+9.2f}%  {tp_m:>+9.2f}%  "
          f"{tp_m-s6m:>+6.2f}%")
print()
print("=" * 84)
print("분석 완료")
print("=" * 84)
