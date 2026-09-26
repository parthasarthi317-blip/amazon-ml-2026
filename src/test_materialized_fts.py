import sqlite3
import time
from pathlib import Path


DB_PATH = Path("models/blocking_index.db")

TEST_ROWS = 200_000


conn = sqlite3.connect(DB_PATH)

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA temp_store=MEMORY")

print("=" * 90)
print("MATERIALIZED FTS5 TEST")
print("=" * 90)

# Remove previous test table if it exists.
conn.execute("DROP TABLE IF EXISTS name_fts_test")
conn.commit()

print("\nCreating test FTS5 table...")

conn.execute("""
    CREATE VIRTUAL TABLE name_fts_test
    USING fts5(
        name_text,
        tokenize='trigram'
    )
""")

conn.commit()

print("Table created.")


# ---------------------------------------------------------
# Populate with first 200k records
# ---------------------------------------------------------

print(f"\nLoading {TEST_ROWS:,} records...")

start = time.time()

conn.execute(
    """
    INSERT INTO name_fts_test(rowid, name_text)
    SELECT
        rowid,
        CASE
            WHEN name_translit != ''
                THEN name_core || ' ' || name_translit
            ELSE name_core
        END
    FROM records
    WHERE rowid <= ?
      AND name_core != ''
    """,
    (TEST_ROWS,)
)

conn.commit()

elapsed = time.time() - start

count = conn.execute(
    "SELECT COUNT(*) FROM name_fts_test"
).fetchone()[0]

print(
    f"Indexed rows: {count:,}"
)

print(
    f"Build time: {elapsed:.2f} sec"
)


# ---------------------------------------------------------
# Check vocabulary
# ---------------------------------------------------------

print("\nChecking vocabulary...")

conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS name_fts_test_vocab
    USING fts5vocab(name_fts_test, row)
""")

vocab_count = conn.execute(
    "SELECT COUNT(*) FROM name_fts_test_vocab"
).fetchone()[0]

print(
    f"Vocabulary terms: {vocab_count:,}"
)


# ---------------------------------------------------------
# Get a known record
# ---------------------------------------------------------

sample = conn.execute("""
    SELECT rowid, name_core, name_translit
    FROM records
    WHERE rowid <= ?
      AND name_core != ''
    LIMIT 1
""", (TEST_ROWS,)).fetchone()

rowid, name_core, name_translit = sample

print("\nKnown record:")
print("Row ID       :", rowid)
print("Name core    :", name_core)
print("Transliterated:", name_translit)


# ---------------------------------------------------------
# Test trigram from known name
# ---------------------------------------------------------

if len(name_core) >= 3:

    trigram = name_core[:3]

    print(
        "\nSearching trigram:",
        repr(trigram)
    )

    results = conn.execute(
        """
        SELECT rowid
        FROM name_fts_test
        WHERE name_fts_test MATCH ?
        LIMIT 10
        """,
        (trigram,)
    ).fetchall()

    print(
        "Search results:",
        results
    )

else:

    print(
        "\nKnown name is shorter than 3 characters."
    )


# ---------------------------------------------------------
# Test English examples
# ---------------------------------------------------------

print("\nEnglish trigram tests:")

for trigram in ["ore", "bar", "sum"]:

    results = conn.execute(
        """
        SELECT rowid
        FROM name_fts_test
        WHERE name_fts_test MATCH ?
        LIMIT 5
        """,
        (trigram,)
    ).fetchall()

    print(
        f"{trigram!r}: {results}"
    )


# ---------------------------------------------------------
# Test actual record lookup
# ---------------------------------------------------------

print("\nExample matching records:")

results = conn.execute("""
    SELECT
        f.rowid,
        r.entity_id,
        r.country,
        r.name_core
    FROM name_fts_test AS f
    JOIN records AS r
        ON r.rowid = f.rowid
    WHERE name_fts_test MATCH ?
    LIMIT 10
""", ("ore",)).fetchall()

for result in results:
    print(result)


print("\n" + "=" * 90)
print("TEST COMPLETE")
print("=" * 90)

conn.close()