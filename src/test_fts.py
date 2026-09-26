import sqlite3

conn = sqlite3.connect(":memory:")

conn.execute("""
    CREATE VIRTUAL TABLE t
    USING fts5(x, tokenize='trigram')
""")

conn.execute(
    "INSERT INTO t(x) VALUES (?)",
    ("orelee barbershop",)
)

result = conn.execute(
    "SELECT rowid FROM t WHERE t MATCH ?",
    ("bar",)
).fetchall()

print("FTS5 trigram result:", result)

conn.close()