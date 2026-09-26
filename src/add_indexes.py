import sqlite3
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)

indexes = [
    (
        "idx_name_translit",
        "country, name_translit"
    ),
    (
        "idx_address_translit",
        "country, address_translit"
    ),
]

for index_name, columns in indexes:
    print(f"Creating {index_name}...")

    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON records ({columns})"
    )

    conn.commit()

conn.close()

print("Additional indexes complete.")