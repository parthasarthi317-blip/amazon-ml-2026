import sqlite3
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)


def make_query(text):
    text = text.strip()

    grams = sorted({
        text[i:i + 3]
        for i in range(len(text) - 2)
    })

    return " OR ".join(
        '"' + g.replace('"', '""') + '"'
        for g in grams
    )


# ---------------------------------------------------------
# Pick a known record
# ---------------------------------------------------------

target = conn.execute("""
    SELECT rowid, entity_id, country, name_core
    FROM records
    WHERE rowid <= 200000
      AND name_core = 'core welfare society'
    LIMIT 1
""").fetchone()

if target is None:
    print("Target record not found.")
    conn.close()
    raise SystemExit

target_rowid, target_id, target_country, target_name = target

print("=" * 90)
print("FULL TRIGRAM QUERY TEST")
print("=" * 90)

print("\nTarget:")
print("rowid   :", target_rowid)
print("ID      :", target_id)
print("country :", target_country)
print("name    :", target_name)


# ---------------------------------------------------------
# Build query from the WHOLE name
# ---------------------------------------------------------

query = make_query(target_name)

print("\nGenerated query:")
print(query[:500] + ("..." if len(query) > 500 else ""))


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

results = conn.execute("""
    SELECT
        f.rowid,
        r.entity_id,
        r.country,
        r.name_core,
        bm25(name_fts_test) AS score
    FROM name_fts_test AS f
    JOIN records AS r
        ON r.rowid = f.rowid
    WHERE name_fts_test MATCH ?
      AND r.country = ?
    ORDER BY bm25(name_fts_test)
    LIMIT 100
""", (
    query,
    target_country
)).fetchall()


# ---------------------------------------------------------
# Locate target
# ---------------------------------------------------------

target_rank = None

for rank, result in enumerate(results, 1):

    if result[0] == target_rowid:
        target_rank = rank

    print(
        f"{rank:3d}. "
        f"{result[1]} | "
        f"{result[3]} | "
        f"score={result[4]:.4f}"
    )


print("\n" + "=" * 90)

if target_rank is None:
    print("TARGET WAS NOT IN TOP 100")
else:
    print(
        f"TARGET RANK: {target_rank}"
    )

print("=" * 90)

conn.close()