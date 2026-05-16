import sys, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pathlib
import os
import pandas_datareader as pdr

# BASE: 스크립트 위치 기준으로 동적 설정 (경로 인코딩 문제 방지)
BASE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD    = 0.60
FEE_ONE_WAY  = 0.0035   # 편도 수수료 0.35% (왕복 0.70%)
DURATION_KR  = 2.7      # 한국 국고채3년 Modified Duration 근사

# ── 데이터 로드 ──
def load_csv(fname):
    df = pd.read_csv(BASE / fname, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df["Close"].dropna().sort_index()

print("데이터 로딩 중...")
kospi_d = load_csv("kospi_history.csv")
gold_d  = load_csv("gold_history.csv")

# 한국 국채수익률 (IMF/FRED: INTGSBKRM193N, 1988~)
# SP500 분석의 ^TNX와 동일한 방식으로 수익률→채권가격 변환
print("  한국 국채수익률 (FRED: INTGSBKRM193N) 다운로드 중...")
kr_yield_raw = pdr.get_data_fred("INTGSBKRM193N", start="1985-01-01")
kr_yield_raw.columns = ["yield"]
kr_yield_d = kr_yield_raw["yield"].dropna()
kr_yield_d.index = pd.to_datetime(kr_yield_d.index).tz_localize(None)

# ── 월말 리샘플 ──
ko_m     = kospi_d.resample("ME").last()
gold_m   = gold_d.resample("ME").last()
kr_yld_m = kr_yield_d.resample("ME").last()

rs_m = ko_m.pct_change().dropna()
rg_m = gold_m.pct_change().dropna()
# 채권 수익률 = 쿠폰수익 + 가격변화
# 월쿠폰 = 연수익률/12/100, 가격변화 = -Duration × Δ수익률/100
rb_m = (kr_yld_m / 12 / 100) - (DURATION_KR * kr_yld_m.diff() / 100)
rb_m = rb_m.dropna()

common = rs_m.index.intersection(rg_m.index).intersection(rb_m.index)
rs_m = rs_m.reindex(common)
rg_m = rg_m.reindex(common)
rb_m = rb_m.reindex(common)
ko_m = ko_m.reindex(common)

print(f"  ✓ 시뮬레이션 기간: {common[0].date()} ~ {common[-1].date()} ({len(common)}개월, 약{len(common)//12}년)")
print(f"  ✓ 1997 IMF 외환위기 포함, 2008 글로벌 금융위기 포함\n")

# ── KOSPI 전략6 투자시즌 (전체 KOSPI 역사 기반) ──
def digit_invest_sets():
    df_full = pd.DataFrame({"ret": kospi_d.resample("ME").last().pct_change().dropna()})
    df_full["month"]    = df_full.index.month
    df_full["season"]   = df_full["month"].apply(
        lambda m: "Nov-Apr" if m in [11,12,1,2,3,4] else "May-Oct")
    df_full["sig_year"] = df_full.apply(
        lambda r: r.name.year-1 if r["month"] in [1,2,3,4] else r.name.year, axis=1)
    na_rows, mo_rows = [], []
    for (sn, sy), g in df_full.groupby(["season","sig_year"]):
        ret = (1 + g["ret"]).prod() - 1
        row = {"year": sy, "ret": ret, "digit": sy % 10}
        (na_rows if sn == "Nov-Apr" else mo_rows).append(row)
    def inv_set(rows):
        if not rows: return set()
        d = pd.DataFrame(rows)
        pmap = {dig: float((d[d["digit"]==dig]["ret"] > 0).mean())
                for dig in range(10) if len(d[d["digit"]==dig]) > 0}
        return set(dig for dig, p in pmap.items() if p >= THRESHOLD)
    return inv_set(na_rows), inv_set(mo_rows)

inv_na, inv_mo = digit_invest_sets()
print(f"  KOSPI 투자시즌 끝자리 Nov-Apr: {sorted(inv_na)}")
print(f"  KOSPI 투자시즌 끝자리 May-Oct: {sorted(inv_mo)}\n")

def is_invest(dt):
    m, y = dt.month, dt.year
    if m in [11,12]:     season, sy = "Nov-Apr", y
    elif m in [1,2,3,4]: season, sy = "Nov-Apr", y-1
    else:                season, sy = "May-Oct",  y
    return (sy % 10) in (inv_na if season == "Nov-Apr" else inv_mo)

def base_weights(dt):
    if is_invest(dt):
        return {"stock": 0.50, "gold": 0.25, "bond": 0.25, "cash": 0.00}
    else:
        return {"stock": 0.25, "gold": 0.25, "bond": 0.25, "cash": 0.25}

# ── 핵심 시뮬레이션 ──
def simulate(stop_pct, recover_pct, fee=FEE_ONE_WAY):
    v_new = 100.0
    state = "normal"
    peak_p = trough_p = float(ko_m.iloc[0])
    vals_new = []
    prev_state = "normal"

    for dt in common:
        ko_px = float(ko_m[dt])
        rs = float(rs_m[dt])
        rg = float(rg_m[dt])
        rb = float(rb_m[dt])

        if state == "normal":
            if ko_px >= peak_p: peak_p = ko_px
            if ko_px <= peak_p * (1 + stop_pct/100):
                state = "bear"; trough_p = ko_px
        else:
            if ko_px <= trough_p: trough_p = ko_px
            if ko_px >= trough_p * (1 + recover_pct/100):
                state = "normal"; peak_p = ko_px; trough_p = ko_px

        bw = base_weights(dt)
        if state == "bear":
            s  = bw["stock"] / 3.0
            wN = {"stock": 0, "gold": bw["gold"]+s, "bond": bw["bond"]+s, "cash": bw["cash"]+s}
        else:
            wN = bw

        cost = bw["stock"] * fee if state != prev_state else 0.0
        prev_state = state

        r_new = wN["stock"]*rs + wN["gold"]*rg + wN["bond"]*rb - cost
        v_new *= (1 + r_new)
        vals_new.append(v_new)

    s = pd.Series(vals_new, index=common)
    n_years = (common[-1] - common[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1/n_years) - 1
    cum  = s.iloc[-1] / s.iloc[0] - 1
    mdd  = ((s - s.cummax()) / s.cummax()).min()
    return cum*100, cagr*100, mdd*100

# ── 전략6 기준선 ──
v = 100.0
vals = []
for dt in common:
    bw = base_weights(dt)
    r  = bw["stock"]*float(rs_m[dt]) + bw["gold"]*float(rg_m[dt]) + bw["bond"]*float(rb_m[dt])
    v *= (1 + r)
    vals.append(v)
s6 = pd.Series(vals, index=common)
n_y = (common[-1]-common[0]).days/365.25
s6_cagr = ((s6.iloc[-1]/s6.iloc[0])**(1/n_y)-1)*100
s6_cum  = (s6.iloc[-1]/s6.iloc[0]-1)*100
s6_mdd  = ((s6-s6.cummax())/s6.cummax()).min()*100

# ── 그리드 서치 ──
stop_levels    = [-10, -12, -15, -17, -20, -22, -25, -28, -30, -35]
recover_levels = [+5, +8, +10, +12, +15, +17, +20, +22, +25, +28, +30]

results = []
print(f"그리드 서치: {len(stop_levels)*len(recover_levels)}개 조합 계산 중...")

for sp in stop_levels:
    for rp in recover_levels:
        cum, cagr, mdd = simulate(sp, rp)
        results.append({
            "손절": sp, "복귀": rp,
            "CAGR": cagr, "누적": cum, "MDD": mdd,
            "CAGR차이": cagr - s6_cagr,
            "MDD차이":  mdd  - s6_mdd,
        })

results.sort(key=lambda x: -x["CAGR"])

# ══════════════════════════════════════════════════════════
print()
print("=" * 82)
print("  KOSPI 트레일링 스탑 + 복귀 기준 그리드 서치  [수수료 편도 0.35% / 왕복 0.70%]")
print(f"  채권: 한국국채수익률(FRED:INTGSBKRM193N) × Duration {DURATION_KR}yr  (쿠폰+가격변화)")
print(f"  기간: {common[0].date()} ~ {common[-1].date()}")
print(f"  KOSPI 전략6 기준: CAGR {s6_cagr:.2f}%  MDD {s6_mdd:.2f}%")
print("=" * 82)

print(f"\n  {'손절':>6s} {'복귀':>6s}  {'CAGR':>7s} {'CAGR↑':>7s}  {'누적':>9s}  {'MDD':>8s} {'MDD↑':>7s}")
print(f"  {'-'*65}")
for r in results[:20]:
    cagr_flag = "✅" if r["CAGR차이"] > 0 else "❌"
    mdd_flag  = "✅" if r["MDD차이"]  > 0 else "❌"
    print(f"  {r['손절']:>5.0f}% {r['복귀']:>5.0f}%  "
          f"{r['CAGR']:>+6.2f}%  {r['CAGR차이']:>+6.2f}%{cagr_flag}  "
          f"{r['누적']:>+8.1f}%  "
          f"{r['MDD']:>+7.2f}%  {r['MDD차이']:>+6.2f}%{mdd_flag}")

# ── CAGR 히트맵 ──
print()
print("=" * 82)
print("  CAGR 히트맵 (행=손절, 열=복귀)  [전략6 대비 차이]")
print("=" * 82)

df_grid = pd.DataFrame(results)
pivot = df_grid.pivot(index="손절", columns="복귀", values="CAGR차이")

print(f"\n  {'손절↓복귀→':>10s}", end="")
for rp in recover_levels:
    print(f"  {rp:>+5.0f}%", end="")
print()
print(f"  {'-'*82}")
for sp in stop_levels:
    print(f"  {sp:>+9.0f}%", end="")
    for rp in recover_levels:
        val = pivot.loc[sp, rp]
        marker = "★" if val == pivot.values.max() else " "
        print(f"  {val:>+4.2f}{marker}", end="")
    print()

# ── 최적 조합 ──
best = results[0]
print()
print("=" * 82)
print(f"  ▶ 최적 조합: 손절 {best['손절']}% / 복귀 +{best['복귀']}%")
print(f"     CAGR {best['CAGR']:+.2f}%  (전략6 대비 {best['CAGR차이']:+.2f}%)")
print(f"     누적 {best['누적']:+.1f}%  MDD {best['MDD']:+.2f}%")
print("=" * 82)

# ── 이벤트 상세 ──
def simulate_detail(stop_pct, recover_pct):
    state = "normal"
    peak_p = trough_p = float(ko_m.iloc[0])
    events = []
    for dt in common:
        ko_px = float(ko_m[dt])
        if state == "normal":
            if ko_px >= peak_p: peak_p = ko_px
            if ko_px <= peak_p * (1 + stop_pct/100):
                state = "bear"; trough_p = ko_px
                events.append((dt, "매도", ko_px, ko_px/peak_p-1, peak_p))
        else:
            if ko_px <= trough_p: trough_p = ko_px
            if ko_px >= trough_p * (1 + recover_pct/100):
                events.append((dt, "복귀", ko_px, ko_px/trough_p-1, trough_p))
                state = "normal"; peak_p = ko_px; trough_p = ko_px
    return events

HIST_EVENTS = {
    1990: "걸프전 침체",
    1992: "신용위기",
    1994: "금리인상 쇼크",
    1997: "IMF 외환위기 💥",
    1998: "IMF 외환위기 💥",
    2000: "닷컴버블/IT 거품",
    2001: "9.11 테러",
    2002: "닷컴버블",
    2008: "글로벌 금융위기 🔥",
    2009: "글로벌 금융위기 🔥",
    2011: "유럽 재정위기",
    2015: "중국 경기둔화",
    2018: "미연준 금리인상",
    2020: "코로나 팬데믹 🦠",
    2022: "금리인상 쇼크",
    2024: "계엄령 사태 🇰🇷",
    2025: "트럼프 관세전쟁 📉",
}

events = simulate_detail(best["손절"], best["복귀"])
n_sell = sum(1 for e in events if e[1] == "매도")
n_buy  = sum(1 for e in events if e[1] == "복귀")
print(f"\n  최적 조합 이벤트 로그 (매도 {n_sell}회 / 복귀 {n_buy}회):")
print(f"  {'날짜':12s} {'액션':6s} {'KOSPI':>8s} {'변동률':>8s} {'기준가':>9s}  배경")
print(f"  {'-'*68}")
for dt, action, px, chg, ref in events:
    ref_label = "고점" if action == "매도" else "저점"
    hist = HIST_EVENTS.get(dt.year, "")
    print(f"  {str(dt.date()):12s} {action:6s} {px:>8,.1f} {chg:>+7.1%}  ({ref_label} {ref:,.1f})  {hist}")

# ── 손절 횟수 분포 ──
print()
print("=" * 82)
print("  손절 기준별 매도/복귀 횟수")
print("=" * 82)
print(f"  {'손절기준':>8s}  {'매도':>5s}  {'복귀':>5s}  {'미복귀':>6s}  평균간격")
print(f"  {'-'*48}")
total_yrs = len(common) / 12
for sp in stop_levels:
    evs = simulate_detail(sp, 20)
    n_sell = sum(1 for e in evs if e[1] == "매도")
    n_buy  = sum(1 for e in evs if e[1] == "복귀")
    n_open = n_sell - n_buy
    avg_gap = total_yrs / n_sell if n_sell > 0 else 0
    open_str = f"({n_open}건)" if n_open > 0 else ""
    print(f"  {sp:>+7.0f}%  {n_sell:>4d}회  {n_buy:>4d}회  {open_str:>6s}  약 {avg_gap:.1f}년마다")

print()
print("=" * 82)
print("분석 완료")
print("=" * 82)
