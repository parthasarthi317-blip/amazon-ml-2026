import csv
import sqlite3
import time
from pathlib import Path

from normalize import normalize_name, normalize_address


ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource"
)

DATASET = ROOT / "dataset"
DB_DIR = Path("models")
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "blocking_index.db"

SOURCES = [
    ("S2", DATASET / "train/train_source2.tsv"),
    ("S3", DATASET / "train/train_source3.tsv"),
]

BATCH_SIZE = 10_000


def prepare_database():
    if DB_PATH.exists():
        print(f"Index already exists: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT,
            business_name TEXT,
            business_address TEXT,
            name_basic TEXT,
            name_core TEXT,
            name_token_sorted TEXT,
            name_translit TEXT,
            address_canonical TEXT,
            address_token_sorted TEXT,
            address_translit TEXT,
            numbers TEXT
        )
    """)

    conn.commit()

    return conn


def insert_batch(conn, batch):

    conn.executemany(
        """
        INSERT INTO records (
            entity_id,
            source,
            country,
            business_name,
            business_address,
            name_basic,
            name_core,
            name_token_sorted,
            name_translit,
            address_canonical,
            address_token_sorted,
            address_translit,
            numbers
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def process_source(conn, source_name, path):

    print(f"\nProcessing {source_name}: {path.name}")

    start = time.time()
    rows = 0
    batch = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            name = normalize_name(
                row["business_name"]
            )

            address = normalize_address(
                row["business_address"]
            )

            numbers = "|".join(
                sorted(set(address["numbers"]))
            )

            batch.append(
                (
                    row["entity_id"],
                    source_name,
                    row["country"].casefold().strip(),
                    row["business_name"],
                    row["business_address"],
                    name["basic"],
                    name["core"],
                    name["token_sorted"],
                    name["translit"],
                    address["canonical"],
                    address["token_sorted"],
                    address["translit"],
                    numbers,
                )
            )

            rows += 1

            if len(batch) >= BATCH_SIZE:

                insert_batch(conn, batch)
                conn.commit()
                batch.clear()

                if rows % 100_000 == 0:

                    elapsed = time.time() - start

                    print(
                        f"{source_name}: "
                        f"{rows:,} rows | "
                        f"{elapsed / 60:.1f} min"
                    )

    if batch:
        insert_batch(conn, batch)
        conn.commit()

    elapsed = time.time() - start

    print(
        f"{source_name} complete: "
        f"{rows:,} rows | "
        f"{elapsed / 60:.1f} min"
    )


def create_indexes(conn):

    print("\nCreating SQLite indexes...")

    indexes = [
        (
            "idx_basic_name",
            "country, name_basic"
        ),
        (
            "idx_core_name",
            "country, name_core"
        ),
        (
            "idx_token_name",
            "country, name_token_sorted"
        ),
        (
            "idx_address_canonical",
            "country, address_canonical"
        ),
        (
            "idx_address_token",
            "country, address_token_sorted"
        ),
    ]

    for index_name, columns in indexes:

        print(f"Creating {index_name}...")

        conn.execute(
            f"""
            CREATE INDEX {index_name}
            ON records ({columns})
            """
        )

        conn.commit()

    print("Indexes complete.")


def main():

    print("=" * 80)
    print("BUILDING REUSABLE BLOCKING INDEX")
    print("=" * 80)

    conn = prepare_database()

    if not conn:
        conn = sqlite3.connect(DB_PATH)

    try:

        # Only populate if the table is empty.
        count = conn.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]

        if count == 0:

            for source_name, path in SOURCES:
                process_source(
                    conn,
                    source_name,
                    path
                )

            create_indexes(conn)

        else:

            print(
                f"Existing records: {count:,}"
            )

        final_count = conn.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]

        print(
            f"\nTotal indexed records: "
            f"{final_count:,}"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()