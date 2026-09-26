import sqlite3
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)

print("=" * 90)
print("FTS5 INDEX DIAGNOSTIC")
print("=" * 90)

# ---------------------------------------------------------
# 1. FTS external row count
# ---------------------------------------------------------

count = conn.execute(
    "SELECT COUNT(*) FROM name_fts"
).fetchone()[0]

print("\nFTS external row count:", count)


# ---------------------------------------------------------
# 2. Check actual records
# ---------------------------------------------------------

samples = conn.execute("""
    SELECT rowid, entity_id, country, name_core, name_translit
    FROM records
    WHERE name_core != ''
    LIMIT 5
""").fetchall()

print("\nSample records:")

for row in samples:
    print(row)


# ---------------------------------------------------------
# 3. Test simple English trigram
# ---------------------------------------------------------

print("\n--- RAW FTS TESTS ---")

for term in ["ore", "bar", "ram"]:

    result = conn.execute(
        """
        SELECT rowid
        FROM name_fts
        WHERE name_fts MATCH ?
        LIMIT 10
        """,
        (term,)
    ).fetchall()

    print(
        f"Search {term!r:10}: {result}"
    )


# ---------------------------------------------------------
# 4. Test a known exact English name if present
# ---------------------------------------------------------

english = conn.execute("""
    SELECT rowid, entity_id, name_core
    FROM records
    WHERE name_core LIKE '%ore%'
    LIMIT 5
""").fetchall()

print("\nRecords containing 'ore':")

for row in english:
    print(row)


# ---------------------------------------------------------
# 5. FTS vocabulary
# ---------------------------------------------------------

print("\n--- FTS VOCABULARY ---")

try:

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS name_vocab
        USING fts5vocab(name_fts, row)
    """)

    vocab_count = conn.execute(
        "SELECT COUNT(*) FROM name_vocab"
    ).fetchone()[0]

    print("Vocabulary terms:", vocab_count)

    print("\nFirst 20 terms:")

    terms = conn.execute("""
        SELECT term, doc
        FROM name_vocab
        LIMIT 20
    """).fetchall()

    for term, doc in terms:
        print(
            repr(term),
            "docs=",
            doc
        )

    print("\nTerms containing 'ore':")

    terms = conn.execute("""
        SELECT term, doc
        FROM name_vocab
        WHERE term LIKE '%ore%'
        LIMIT 20
    """).fetchall()

    for term, doc in terms:
        print(
            repr(term),
            "docs=",
            doc
        )

except sqlite3.Error as e:

    print("Vocabulary error:", e)


conn.close()

print("\nDiagnostic complete.")