"""
자산배분 대시보드 일일 백업 스크립트
- 실행 시 백업 폴더에 날짜별로 저장:
    1. streamlit_app.py / requirements.txt / memo.txt  (소스코드)
    2. *_history.csv  (SP500·NASDAQ·DOW·금 장기 히스토리 CSV)
    3. 지수/자산 가격 데이터 CSV (KOSPI, KOSDAQ, 한국채, VIX - yfinance)
    4. 미국채10년 (FRED DGS10)
"""

import shutil
import pathlib
import sys
from datetime import datetime, date

# Windows 콘솔 UTF-8 출력 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 경로 설정 ───────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
BACKUP_DIR = BASE_DIR / "backup"
TODAY      = datetime.now().strftime("%Y%m%d")
DEST       = BACKUP_DIR / TODAY
DEST.mkdir(parents=True, exist_ok=True)

LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)


# ════════════════════════════════════════════════════════
# 1. 소스 파일 백업
# ════════════════════════════════════════════════════════
log("=" * 50)
log(f"[{TODAY}] 백업 시작")
log("=" * 50)

files_to_copy = [
    BASE_DIR / "streamlit_app.py",
    BASE_DIR / "requirements.txt",
    BASE_DIR / "memo.txt",
]
for f in files_to_copy:
    if f.exists():
        shutil.copy2(f, DEST / f.name)
        log(f"  ✅ 파일 복사: {f.name}")
    else:
        log(f"  ⚠️  파일 없음 (건너뜀): {f.name}")


# ════════════════════════════════════════════════════════
# 2. 장기 히스토리 CSV 백업 (STOOQ 기반)
# ════════════════════════════════════════════════════════
history_csvs = [
    "sp500_history.csv",
    "nasdaq_history.csv",
    "dow_history.csv",
    "gold_history.csv",
    "kospi_history.csv",
]
log("\n📂 히스토리 CSV 복사")
for fname in history_csvs:
    src = BASE_DIR / fname
    if src.exists():
        shutil.copy2(src, DEST / fname)
        size_kb = src.stat().st_size / 1024
        log(f"  ✅ {fname}  ({size_kb:.0f} KB)")
    else:
        log(f"  ⚠️  없음 (건너뜀): {fname}")


# ════════════════════════════════════════════════════════
# 3. 데이터 다운로드 → CSV 저장
# ════════════════════════════════════════════════════════
try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    log("\n❌ yfinance / pandas 미설치")
    yf = None

DATA_DIR = DEST / "data"
DATA_DIR.mkdir(exist_ok=True)

if yf:
    # yfinance 티커 (KOSPI·KOSDAQ·한국채·VIX)
    yf_tickers = {
        "KOSPI":   ("^KS11",      "1990-01-01"),
        "KOSDAQ":  ("^KQ11",      "1997-01-01"),
        "KRBOND":  ("114820.KS",  "2009-01-01"),
        "VIX":     ("^VIX",       "1990-01-01"),
    }
    log(f"\n📥 yfinance 다운로드 → {DATA_DIR}")
    for name, (ticker, start) in yf_tickers.items():
        try:
            df = yf.download(ticker, start=start, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                log(f"  ⚠️  {name} ({ticker}): 데이터 없음")
                continue
            out = DATA_DIR / f"{name}.csv"
            df[["Close"]].dropna().to_csv(out, encoding="utf-8-sig")
            log(f"  ✅ {name:8s} ({ticker:12s}): {len(df):,}행  →  {out.name}")
        except Exception as e:
            log(f"  ❌ {name} ({ticker}): 오류 — {e}")

# FRED DGS10 (미국채 10년물)
try:
    import pandas_datareader.data as pdr
    log(f"\n📥 FRED 다운로드")
    s = pdr.DataReader("DGS10", "fred", start=date(1962, 1, 1))["DGS10"].dropna()
    out = DATA_DIR / "US10Y.csv"
    s.to_frame("Close").to_csv(out, encoding="utf-8-sig")
    log(f"  ✅ US10Y    (FRED DGS10  ): {len(s):,}행  →  {out.name}")
except Exception as e:
    log(f"  ❌ US10Y (FRED DGS10): 오류 — {e}")


# ════════════════════════════════════════════════════════
# 4. 오래된 백업 정리 (30일 초과 폴더 삭제)
# ════════════════════════════════════════════════════════
KEEP_DAYS = 30
deleted = []
for d in BACKUP_DIR.iterdir():
    if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
        try:
            folder_date = datetime.strptime(d.name, "%Y%m%d")
            age = (datetime.now() - folder_date).days
            if age > KEEP_DAYS:
                shutil.rmtree(d)
                deleted.append(d.name)
        except ValueError:
            pass

if deleted:
    log(f"\n🗑️  오래된 백업 삭제 ({KEEP_DAYS}일 초과): {', '.join(deleted)}")
else:
    log(f"\n🗑️  삭제할 오래된 백업 없음 (보관 기준: {KEEP_DAYS}일)")


# ════════════════════════════════════════════════════════
# 5. 로그 저장
# ════════════════════════════════════════════════════════
log(f"\n✅ 백업 완료 → {DEST}")
log("=" * 50)

log_file = DEST / "backup_log.txt"
log_file.write_text("\n".join(LOG), encoding="utf-8")
print(f"\n로그 저장: {log_file}")
