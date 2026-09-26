import sqlite3
from pathlib import Path

DB_PATH = Path("models/blocking_index.db")

conn = sqlite3.connect(DB_PATH)

indexes = [
    (
        "idx_core_prefix_len",
        """
        CREATE INDEX IF NOT EXISTS idx_core_prefix_len
        ON records(
            country,
            substr(name_core, 1, 3),
            length(name_core) / 5
        )
        """
    ),
    (
        "idx_translit_prefix_len",
        """
        CREATE INDEX IF NOT EXISTS idx_translit_prefix_len
        ON records(
            country,
            substr(name_translit, 1, 3),
            length(name_translit) / 5
        )
        """
    ),
    (
        "idx_core_suffix_len",
        """
        CREATE INDEX IF NOT EXISTS idx_core_suffix_len
        ON records(
            country,
            substr(name_core, -3),
            length(name_core) / 5
        )
        """
    ),
    (
        "idx_translit_suffix_len",
        """
        CREATE INDEX IF NOT EXISTS idx_translit_suffix_len
        ON records(
            country,
            substr(name_translit, -3),
            length(name_translit) / 5
        )
        """
    ),
    (
        "idx_token_prefix_len",
        """
        CREATE INDEX IF NOT EXISTS idx_token_prefix_len
        ON records(
            country,
            substr(name_token_sorted, 1, 3),
            length(name_token_sorted) / 5
        )
        """
    ),
]

for name, sql in indexes:
    print(f"Creating {name}...")
    conn.execute(sql)
    conn.commit()

conn.close()

print("Approximate blocking indexes ready.")