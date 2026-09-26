import sqlite3
import time
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA temp_store=MEMORY")

print("=" * 90)
print("BUILDING FULL MATERIALIZED NAME FTS5 INDEX")
print("=" * 90)

# Remove only our full index if it already exists.
conn.execute("DROP TABLE IF EXISTS name_fts_full")
conn.commit()

print("\nCreating materialized trigram index...")

conn.execute("""
    CREATE VIRTUAL TABLE name_fts_full
    USING fts5(
        search_text,
        tokenize='trigram',
        detail='none',
        columnsize=0
    )
""")

conn.commit()

print("Table created.")

print("\nPopulating from 10,320,219 records...")

start = time.time()

conn.execute("""
    INSERT INTO name_fts_full(rowid, search_text)
    SELECT
        rowid,
        CASE
            WHEN name_translit != ''
                THEN name_core || ' ' || name_translit
            ELSE name_core
        END
    FROM records
    WHERE name_core != ''
""")

conn.commit()

elapsed = time.time() - start

count = conn.execute(
    "SELECT COUNT(*) FROM name_fts_full"
).fetchone()[0]

print(f"\nIndexed rows: {count:,}")
print(f"Build time:   {elapsed / 60:.2f} minutes")

print("\nChecking vocabulary...")

conn.execute("""
    DROP TABLE IF EXISTS name_fts_full_vocab
""")

conn.execute("""
    CREATE VIRTUAL TABLE name_fts_full_vocab
    USING fts5vocab(name_fts_full, row)
""")

vocab_count = conn.execute(
    "SELECT COUNT(*) FROM name_fts_full_vocab"
).fetchone()[0]

print(
    f"Vocabulary terms: {vocab_count:,}"
)

conn.close()

print("\nFULL FTS INDEX READY.")