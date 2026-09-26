import sqlite3
import time
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA temp_store=MEMORY")

print("Creating FTS5 trigram index...")

conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS name_fts
    USING fts5(
        name_core,
        name_translit,
        country UNINDEXED,
        content='records',
        content_rowid='rowid',
        tokenize='trigram',
        detail='none',
        columnsize=0
    )
""")

conn.commit()

# Check whether the index is already populated.
count = conn.execute(
    "SELECT COUNT(*) FROM name_fts"
).fetchone()[0]

if count == 0:

    print("Building index from 10.32M records...")
    print("This is a ONE-TIME operation.")

    start = time.time()

    conn.execute(
        "INSERT INTO name_fts(name_fts) VALUES ('rebuild')"
    )

    conn.commit()

    elapsed = time.time() - start

    count = conn.execute(
        "SELECT COUNT(*) FROM name_fts"
    ).fetchone()[0]

    print(
        f"Build complete: {count:,} rows"
    )

    print(
        f"Time: {elapsed / 60:.1f} minutes"
    )

else:

    print(
        f"FTS index already populated: {count:,} rows"
    )

conn.close()

print("Done.")