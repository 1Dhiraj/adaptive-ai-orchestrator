"""Look inside the orchestrator's SQLite state without a GUI.

The database is binary, so opening it in an editor shows nothing useful, and
its interesting columns are JSON blobs that a generic table viewer renders as
one long unreadable line. This prints the three things anyone actually wants:
which runs exist, what happened inside one, and what a raw table holds.

    python scripts/db.py runs                 # recent runs
    python scripts/db.py runs --limit 40
    python scripts/db.py run 394e1c4f9091     # one run's steps and output
    python scripts/db.py tables               # table names and row counts
    python scripts/db.py table events --limit 20
    python scripts/db.py sql "select ..."     # anything else
"""

from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "orchestrator_state.db"


def connect(path: str) -> sqlite3.Connection:
    if not Path(path).exists():
        raise SystemExit(f"no database at {path}")
    # Read-only, so inspecting a database never disturbs a run using it.
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def when(value) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(value)).strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return "-"


def cmd_runs(connection, args) -> None:
    rows = connection.execute(
        "select * from runs order by created_at desc limit ?", (args.limit,)
    ).fetchall()
    print(f"{'run_id':14} {'when':13} {'status':9} description")
    print("-" * 96)
    for row in rows:
        print(f"{row['run_id']:14} {when(row['created_at']):13} "
              f"{str(row['status']):9} {str(row['description'])[:56]}")
    print(f"\n{len(rows)} run(s). Inspect one with: python scripts/db.py run <run_id>")


def cmd_run(connection, args) -> None:
    run = connection.execute(
        "select * from runs where run_id like ?", (args.run_id + "%",)).fetchone()
    if run is None:
        raise SystemExit(f"no run starting with '{args.run_id}'")

    print(f"run     : {run['run_id']}")
    print(f"started : {when(run['created_at'])}")
    print(f"status  : {run['status']}")
    print(f"task    : {run['description']}")
    print()

    results = connection.execute(
        "select * from step_results where run_id = ?", (run["run_id"],)).fetchall()
    if not results:
        print("(no step results stored yet)")
        return

    for row in results:
        try:
            result = json.loads(row["result_json"])
        except (TypeError, ValueError):
            print(f"  {row['step_id']}: (unreadable result)")
            continue
        status = result.get("status", "?")
        print(f"  [{status}] {row['step_id']}")
        if result.get("error"):
            print(f"        error : {str(result['error'])[:150]}")
        output = (result.get("output") or "").strip()
        if output:
            head = output if args.full else output[:400]
            for line in head.splitlines():
                print(f"        {line}")
            if not args.full and len(output) > 400:
                print(f"        ... ({len(output)} chars; --full to see it all)")
        print()


def cmd_tables(connection, args) -> None:
    names = [r[0] for r in connection.execute(
        "select name from sqlite_master where type='table' order by name")]
    for name in names:
        count = connection.execute(f"select count(*) from {name}").fetchone()[0]
        print(f"  {name:22} {count:>7} rows")


def cmd_table(connection, args) -> None:
    rows = connection.execute(
        f"select * from {args.name} order by rowid desc limit ?", (args.limit,)
    ).fetchall()
    if not rows:
        print(f"{args.name} is empty")
        return
    for row in rows:
        print("-" * 78)
        for key in row.keys():
            value = str(row[key])
            if len(value) > 160:
                value = value[:160] + f"... ({len(str(row[key]))} chars)"
            print(f"  {key:16} {value}")


def cmd_sql(connection, args) -> None:
    rows = connection.execute(args.query).fetchall()
    if not rows:
        print("(no rows)")
        return
    print(" | ".join(rows[0].keys()))
    for row in rows:
        print(" | ".join(str(row[key])[:40] for key in row.keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("runs", help="recent runs")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    one = sub.add_parser("run", help="one run's steps and output")
    one.add_argument("run_id")
    one.add_argument("--full", action="store_true", help="do not truncate output")
    one.set_defaults(func=cmd_run)

    tables = sub.add_parser("tables", help="table names and row counts")
    tables.set_defaults(func=cmd_tables)

    table = sub.add_parser("table", help="rows of one table")
    table.add_argument("name")
    table.add_argument("--limit", type=int, default=10)
    table.set_defaults(func=cmd_table)

    query = sub.add_parser("sql", help="run a read-only query")
    query.add_argument("query")
    query.set_defaults(func=cmd_sql)

    args = parser.parse_args()
    with connect(args.db) as connection:
        args.func(connection, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
