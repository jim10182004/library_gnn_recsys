"""
資料前處理：把 Excel 轉成 Parquet，並建立統一的 book_id / user_id。

執行：
    # 1) 設環境變數指到原始 Excel 資料夾
    PowerShell:  $env:LIBRARY_RAW_DIR = "C:\\path\\to\\raw\\excels"
    Bash:        export LIBRARY_RAW_DIR=/path/to/raw/excels

    # 2) 執行
    python src/preprocess.py

輸出：data/processed/borrows.parquet, reservations.parquet, books.parquet, users.parquet
"""
from __future__ import annotations
import os
import re
import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# 原始 Excel 資料夾（含 借閱*.xlsx, 預約2025原檔.xlsx, 讀者id對照表.xlsx）
# 優先用環境變數 LIBRARY_RAW_DIR；找不到時 fallback 到 .env 檔；都沒有則報錯
def _resolve_raw_dir() -> Path:
    env_path = os.environ.get("LIBRARY_RAW_DIR", "").strip()
    if env_path:
        return Path(env_path)
    # 嘗試從專案根目錄的 .env 檔讀取
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LIBRARY_RAW_DIR="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return Path(val)
    print(
        "[ERROR] 找不到原始資料路徑。請設定環境變數 LIBRARY_RAW_DIR，例如：\n"
        '  PowerShell: $env:LIBRARY_RAW_DIR = "C:\\path\\to\\raw\\excels"\n'
        "  Bash:       export LIBRARY_RAW_DIR=/path/to/raw/excels\n"
        "或在專案根目錄建立 .env 檔（參考 .env.example）",
        file=sys.stderr,
    )
    sys.exit(1)


RAW_DIR = _resolve_raw_dir()
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists():
    print(f"[ERROR] LIBRARY_RAW_DIR 路徑不存在：{RAW_DIR}", file=sys.stderr)
    sys.exit(1)

ISBN_RE = re.compile(r"(\d{13}|\d{10})")


def clean_isbn(raw) -> str | None:
    """從 ISBN 欄位抽出第一個 10 或 13 位數字串，當作標準 ISBN。"""
    if pd.isna(raw):
        return None
    s = str(raw)
    m = ISBN_RE.search(s.replace("-", ""))
    return m.group(1) if m else None


def book_key(row) -> str:
    """每本書的唯一 key：優先 ISBN，否則用 hash(題名+作者)。"""
    isbn = row["isbn_clean"]
    if isbn:
        return f"ISBN:{isbn}"
    title = "" if pd.isna(row["題名"]) else str(row["題名"]).strip()
    author = "" if pd.isna(row["作者"]) else str(row["作者"]).strip()
    h = hashlib.md5(f"{title}|{author}".encode("utf-8")).hexdigest()[:16]
    return f"H:{h}"


def load_borrows() -> pd.DataFrame:
    """合併兩個半年度的借閱資料。"""
    print("[Borrows] 讀取 借閱202501_07 ...")
    h1 = pd.read_excel(RAW_DIR / "借閱202501_07.xlsx", sheet_name="借閱202501_07")
    print(f"   -> {len(h1):,} rows")
    print("[Borrows] 讀取 借閱202508_12 ...")
    h2 = pd.read_excel(RAW_DIR / "借閱202508_12.xlsx", sheet_name="借閱202508_12")
    print(f"   -> {len(h2):,} rows")
    df = pd.concat([h1, h2], ignore_index=True)
    print(f"[Borrows] 合計 {len(df):,} rows")
    return df


def load_reservations() -> pd.DataFrame:
    print("[Reservations] 讀取 預約2025 ...")
    df = pd.read_excel(RAW_DIR / "預約2025原檔.xlsx", sheet_name="預約2025")
    print(f"   -> {len(df):,} rows")
    return df


def _to_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _to_int_or_none(x):
    """嘗試轉成整數，失敗回傳 None（適合年份這種有 '20uu' 雜訊的欄位）。"""
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


def normalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """共用的清洗：清 ISBN、建 book_key、保留必要欄位。"""
    df = df.copy()
    # 統一字串型欄位的 dtype（混型會讓 parquet 報錯）
    for c in ("題名", "作者", "ISBN", "分類號", "讀者性別"):
        if c in df.columns:
            df[c] = df[c].apply(_to_str)
    # 出版年有 '20uu' 等不明值，轉成 nullable Int
    if "出版年" in df.columns:
        df["出版年"] = df["出版年"].apply(_to_int_or_none).astype("Int64")
    df["isbn_clean"] = df["ISBN"].apply(clean_isbn)
    df["book_key"] = df.apply(book_key, axis=1)
    # 重命名欄位（英文方便後續處理）
    rename = {
        "識別碼": "record_id",
        "讀者ID": "user_orig",
        "讀者性別": "gender",
        "年齡": "age",
        "題名": "title",
        "作者": "author",
        "出版年": "pub_year",
        "ISBN": "isbn_raw",
        "分類號": "category",
    }
    if kind == "borrow":
        rename["借閱日期"] = "ts"
        rename["還書日期"] = "return_ts"
    else:
        rename["預約日期"] = "ts"
    df = df.rename(columns=rename)
    return df


def build_books(borrows: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    """從借閱+預約資料中萃取每本書的代表資訊。"""
    print("[Books] 萃取書籍資訊 ...")
    cols = ["book_key", "title", "author", "pub_year", "isbn_clean", "category"]
    all_b = pd.concat([borrows[cols], reservations[cols]], ignore_index=True)
    # 同一 book_key 取第一筆作代表
    books = all_b.drop_duplicates(subset=["book_key"], keep="first").reset_index(drop=True)
    books["book_id"] = np.arange(len(books))
    print(f"[Books] 共 {len(books):,} 本不同的書")
    return books


def build_users(borrows: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    """從借閱+預約資料中萃取每位讀者的代表資訊。"""
    print("[Users] 萃取讀者資訊 ...")
    cols = ["user_orig", "gender", "age"]
    all_u = pd.concat([borrows[cols], reservations[cols]], ignore_index=True)
    users = all_u.drop_duplicates(subset=["user_orig"], keep="first").reset_index(drop=True)
    users["user_id"] = np.arange(len(users))
    print(f"[Users] 共 {len(users):,} 位不同讀者")
    return users


def main():
    borrows = normalize(load_borrows(), "borrow")
    reservations = normalize(load_reservations(), "reservation")

    books = build_books(borrows, reservations)
    users = build_users(borrows, reservations)

    book_map = dict(zip(books["book_key"], books["book_id"]))
    user_map = dict(zip(users["user_orig"], users["user_id"]))

    print("[Map] 套用 user_id / book_id ...")
    for df in (borrows, reservations):
        df["user_id"] = df["user_orig"].map(user_map).astype("int32")
        df["book_id"] = df["book_key"].map(book_map).astype("int32")

    # 只保留必要欄位 + 互動資料
    borrow_cols = ["user_id", "book_id", "ts", "return_ts", "gender", "age", "category"]
    reserv_cols = ["user_id", "book_id", "ts", "gender", "age", "category"]
    borrows_out = borrows[borrow_cols].copy()
    reserv_out = reservations[reserv_cols].copy()

    # 保留借閱天數（之後可做 churn / 還書天數預測）
    borrows_out["borrow_days"] = (
        borrows_out["return_ts"] - borrows_out["ts"]
    ).dt.total_seconds() / 86400.0

    # 寫出 parquet
    print("[Write] 寫出 Parquet ...")
    borrows_out.to_parquet(OUT_DIR / "borrows.parquet", index=False)
    reserv_out.to_parquet(OUT_DIR / "reservations.parquet", index=False)
    books.to_parquet(OUT_DIR / "books.parquet", index=False)
    users.to_parquet(OUT_DIR / "users.parquet", index=False)

    print("\n=== 完成 ===")
    print(f"borrows.parquet      {len(borrows_out):>10,} rows")
    print(f"reservations.parquet {len(reserv_out):>10,} rows")
    print(f"books.parquet        {len(books):>10,} books")
    print(f"users.parquet        {len(users):>10,} users")


if __name__ == "__main__":
    main()
