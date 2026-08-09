"""Small SQLite persistence layer for the paper-trading application state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class StateStore:
    """Persist portfolio, risk and audit state without changing domain models."""

    def __init__(self, path: str = "data/omnimarket.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def get(self, key: str) -> Any | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":"))
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO app_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )
            conn.commit()

    def clear(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM app_state")
            conn.commit()
