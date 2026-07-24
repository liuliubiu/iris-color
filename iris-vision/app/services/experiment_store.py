"""实验记录存储层（支持 MySQL / SQLite，共享校验与命名规范）。"""

from __future__ import annotations

import re
import sqlite3
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # pragma: no cover
    pymysql = None  # type: ignore
    DictCursor = None  # type: ignore

GRADE_VALUES = ("Grade1", "Grade2", "Grade3", "Grade4", "Grade5")

COLOR_VALUES = (
    "浅蓝",
    "蓝",
    "深蓝",
    "浅绿",
    "绿",
    "深绿",
    "浅棕",
    "棕",
    "深棕",
)

# 实验大组：第一大组、第二大组…（兼容旧版 G20260722-001）
GROUP_CN_RE = re.compile(r"^第(.+?)大组$")
GROUP_LEGACY_RE = re.compile(r"^G\d{8}-\d{3}$")
# 实验小组：第一小组、第二小组…（兼容旧版 S01；留空表示不分组）
SUBGROUP_CN_RE = re.compile(r"^第(.+?)小组$")
SUBGROUP_LEGACY_RE = re.compile(r"^S\d{2}$")

GROUP_FORMAT_HINT = "第一大组、第二大组…"
SUBGROUP_FORMAT_HINT = "第一小组、第二小组…（可留空）"

_CN_DIGITS = "零一二三四五六七八九"


def _int_to_chinese(n: int) -> str:
    if n <= 0:
        return str(n)
    if n < 10:
        return _CN_DIGITS[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    if n < 100:
        tens, ones = divmod(n, 10)
        s = _CN_DIGITS[tens] + "十"
        if ones:
            s += _CN_DIGITS[ones]
        return s
    return str(n)


def _chinese_to_int(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if text.startswith("十"):
        rest = text[1:]
        return 10 + (_CN_DIGITS.index(rest) if rest in _CN_DIGITS else 0)
    if "十" in text:
        parts = text.split("十", 1)
        tens = _CN_DIGITS.index(parts[0]) if parts[0] in _CN_DIGITS else 0
        ones = _CN_DIGITS.index(parts[1]) if len(parts) > 1 and parts[1] in _CN_DIGITS else 0
        return tens * 10 + ones
    if text in _CN_DIGITS:
        return _CN_DIGITS.index(text)
    return 0

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name TEXT NOT NULL,
    subgroup_name TEXT,
    experiment_date TEXT NOT NULL,
    operator TEXT NOT NULL,
    camera_device TEXT,
    light_device TEXT,
    illuminance INTEGER,
    color TEXT,
    grade_before TEXT,
    lstar_before REAL,
    grade_after TEXT,
    lstar_after REAL,
    notes TEXT,
    image_rel TEXT,
    debug_run_id TEXT,
    skip_quality INTEGER NOT NULL DEFAULT 0,
    manual_adjusted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_group ON experiment_records(group_name);
CREATE INDEX IF NOT EXISTS idx_exp_date ON experiment_records(experiment_date);
CREATE INDEX IF NOT EXISTS idx_exp_operator ON experiment_records(operator);
"""

_MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    group_name VARCHAR(32) NOT NULL,
    subgroup_name VARCHAR(16) NULL,
    experiment_date DATE NOT NULL,
    operator VARCHAR(64) NOT NULL,
    camera_device VARCHAR(128) NULL,
    light_device VARCHAR(128) NULL,
    illuminance INT NULL,
    color VARCHAR(16) NULL,
    grade_before VARCHAR(16) NULL,
    lstar_before DOUBLE NULL,
    grade_after VARCHAR(16) NULL,
    lstar_after DOUBLE NULL,
    notes TEXT NULL,
    image_rel VARCHAR(512) NULL,
    debug_run_id VARCHAR(32) NULL,
    skip_quality TINYINT(1) NOT NULL DEFAULT 0,
    manual_adjusted TINYINT(1) NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_exp_group (group_name),
    INDEX idx_exp_date (experiment_date),
    INDEX idx_exp_operator (operator)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _parse_group_seq(group_name: str) -> int:
    m = GROUP_CN_RE.match(group_name)
    if m:
        return _chinese_to_int(m.group(1))
    if GROUP_LEGACY_RE.match(group_name):
        return int(group_name.split("-")[-1])
    return 0


def _parse_subgroup_seq(subgroup_name: str) -> int:
    if not subgroup_name:
        return 0
    m = SUBGROUP_CN_RE.match(subgroup_name)
    if m:
        return _chinese_to_int(m.group(1))
    if SUBGROUP_LEGACY_RE.match(subgroup_name):
        return int(subgroup_name[1:])
    return 0


def _sqlite_migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(experiment_records)")}
    if "image_rel" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN image_rel TEXT")
    if "debug_run_id" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN debug_run_id TEXT")
    if "image_before_rel" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN image_before_rel TEXT")
    if "image_after_rel" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN image_after_rel TEXT")
    if "skip_quality" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN skip_quality INTEGER NOT NULL DEFAULT 0")
    if "manual_adjusted" not in cols:
        conn.execute("ALTER TABLE experiment_records ADD COLUMN manual_adjusted INTEGER NOT NULL DEFAULT 0")


def _mysql_migrate(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'image_rel'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN image_rel VARCHAR(512) NULL")
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'debug_run_id'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN debug_run_id VARCHAR(32) NULL")
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'image_before_rel'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN image_before_rel VARCHAR(512) NULL")
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'image_after_rel'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN image_after_rel VARCHAR(512) NULL")
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'skip_quality'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN skip_quality TINYINT(1) NOT NULL DEFAULT 0")
        cur.execute("SHOW COLUMNS FROM experiment_records LIKE 'manual_adjusted'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE experiment_records ADD COLUMN manual_adjusted TINYINT(1) NOT NULL DEFAULT 0")
    conn.commit()


class ExperimentStore(ABC):
    @abstractmethod
    def list_records(self, **filters: Any) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_by_id(self, record_id: int) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    def create_record(self, data: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def update_record(self, record_id: int, data: dict[str, Any]) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    def update_record_images(
        self,
        record_id: int,
        *,
        image_before_rel: Optional[str],
        image_after_rel: Optional[str],
    ) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    def delete_record(self, record_id: int) -> bool: ...

    @abstractmethod
    def bulk_delete(self, ids: list[int]) -> int: ...

    @abstractmethod
    def get_distinct_options(self) -> dict[str, list[str]]: ...

    @abstractmethod
    def suggest_group_name(self, date_yyyymmdd: Optional[str] = None) -> str: ...

    @abstractmethod
    def suggest_subgroup_name(self, group_name: str) -> str: ...

    def get_subgroup_defaults(self, group_name: str, subgroup_name: str) -> dict[str, Any]:
        """同一大组统一日期；同一大组+小组统一实验人/设备/照度（取各自最近一条）。"""
        if not group_name:
            return {}
        out: dict[str, Any] = {}

        group_records = self.list_records(group_name=group_name)
        if group_records:
            latest_group = max(group_records, key=lambda r: int(r.get("id") or 0))
            out["experiment_date"] = latest_group.get("experiment_date") or ""

        if subgroup_name:
            sub_records = self.list_records(group_name=group_name, subgroup_name=subgroup_name)
            if sub_records:
                latest_sub = max(sub_records, key=lambda r: int(r.get("id") or 0))
                out.update(
                    {
                        "operator": latest_sub.get("operator") or "",
                        "camera_device": latest_sub.get("camera_device") or "",
                        "light_device": latest_sub.get("light_device") or "",
                        "illuminance": latest_sub.get("illuminance"),
                    }
                )
        return out

    @staticmethod
    def _validate_grade(value: Optional[str], field: str) -> None:
        if value is not None and value not in GRADE_VALUES:
            raise ValueError(f"invalid_{field}")

    @staticmethod
    def _validate_color(value: Optional[str]) -> None:
        if value is not None and value not in COLOR_VALUES:
            raise ValueError("invalid_color")

    @staticmethod
    def _validate_group_name(value: str) -> None:
        if not GROUP_CN_RE.match(value) and not GROUP_LEGACY_RE.match(value):
            raise ValueError("invalid_group_name_format")

    @staticmethod
    def _validate_subgroup_name(value: Optional[str]) -> None:
        if value is not None and not SUBGROUP_CN_RE.match(value) and not SUBGROUP_LEGACY_RE.match(value):
            raise ValueError("invalid_subgroup_name_format")

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(value)

    def _normalize_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["group_name"] = (data.get("group_name") or "").strip()
        if not out["group_name"]:
            raise ValueError("missing_group_name")
        self._validate_group_name(out["group_name"])

        subgroup = data.get("subgroup_name")
        out["subgroup_name"] = (
            subgroup.strip() if isinstance(subgroup, str) and subgroup.strip() else None
        )
        self._validate_subgroup_name(out["subgroup_name"])

        out["experiment_date"] = (data.get("experiment_date") or "").strip()
        if not out["experiment_date"]:
            raise ValueError("missing_experiment_date")

        out["operator"] = (data.get("operator") or "").strip()
        if not out["operator"]:
            raise ValueError("missing_operator")

        for field in ("camera_device", "light_device", "notes", "image_rel", "debug_run_id"):
            val = data.get(field)
            out[field] = val.strip() if isinstance(val, str) and val.strip() else None

        color = data.get("color")
        out["color"] = color.strip() if isinstance(color, str) and color.strip() else None
        self._validate_color(out["color"])

        illuminance = data.get("illuminance")
        if illuminance is None or illuminance == "":
            out["illuminance"] = None
        else:
            try:
                out["illuminance"] = int(illuminance)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_illuminance") from exc

        for field in ("grade_before", "grade_after"):
            val = data.get(field)
            out[field] = val.strip() if isinstance(val, str) and val.strip() else None
            self._validate_grade(out[field], field)

        for field in ("lstar_before", "lstar_after"):
            val = data.get(field)
            if val is None or val == "":
                out[field] = None
            else:
                try:
                    out[field] = round(float(val), 2)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid_{field}") from exc

        out["skip_quality"] = self._coerce_bool(data.get("skip_quality"), False)
        out["manual_adjusted"] = self._coerce_bool(data.get("manual_adjusted"), False)

        return out


class SqliteExperimentStore(ExperimentStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SQLITE_SCHEMA)
            _sqlite_migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = {k: row[k] for k in row.keys()}
        if d.get("experiment_date"):
            d["experiment_date"] = str(d["experiment_date"])[:10]
        for field in ("skip_quality", "manual_adjusted"):
            if field in d and d[field] is not None:
                d[field] = bool(d[field])
        return d

    def list_records(
        self,
        *,
        group_name: Optional[str] = None,
        subgroup_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        operator: Optional[str] = None,
        camera_device: Optional[str] = None,
        light_device: Optional[str] = None,
        color: Optional[str] = None,
        grade_before: Optional[str] = None,
        grade_after: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def _add(field: str, value: Optional[str]) -> None:
            if value:
                clauses.append(f"{field} = ?")
                params.append(value)

        _add("group_name", group_name)
        _add("subgroup_name", subgroup_name)
        _add("operator", operator)
        _add("camera_device", camera_device)
        _add("light_device", light_device)
        _add("color", color)
        _add("grade_before", grade_before)
        _add("grade_after", grade_after)
        if date_from:
            clauses.append("experiment_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("experiment_date <= ?")
            params.append(date_to)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experiment_records {where} ORDER BY experiment_date DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_id(self, record_id: int) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiment_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_payload(data)
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiment_records (
                    group_name, subgroup_name, experiment_date, operator,
                    camera_device, light_device, illuminance, color,
                    grade_before, lstar_before, grade_after, lstar_after,
                    notes, image_rel, debug_run_id, skip_quality, manual_adjusted,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["group_name"],
                    payload["subgroup_name"],
                    payload["experiment_date"],
                    payload["operator"],
                    payload["camera_device"],
                    payload["light_device"],
                    payload["illuminance"],
                    payload["color"],
                    payload["grade_before"],
                    payload["lstar_before"],
                    payload["grade_after"],
                    payload["lstar_after"],
                    payload["notes"],
                    payload["image_rel"],
                    payload["debug_run_id"],
                    int(payload["skip_quality"]),
                    int(payload["manual_adjusted"]),
                    now,
                    now,
                ),
            )
            record_id = cur.lastrowid
        result = self.get_by_id(record_id)
        assert result is not None
        return result

    def update_record_images(
        self,
        record_id: int,
        *,
        image_before_rel: Optional[str],
        image_after_rel: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not self.get_by_id(record_id):
            return None
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE experiment_records SET
                    image_before_rel = ?, image_after_rel = ?, updated_at = ?
                WHERE id = ?
                """,
                (image_before_rel, image_after_rel, now, record_id),
            )
        return self.get_by_id(record_id)

    def update_record(self, record_id: int, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.get_by_id(record_id):
            return None
        payload = self._normalize_payload(data)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE experiment_records SET
                    group_name = ?, subgroup_name = ?, experiment_date = ?, operator = ?,
                    camera_device = ?, light_device = ?, illuminance = ?, color = ?,
                    grade_before = ?, lstar_before = ?, grade_after = ?, lstar_after = ?,
                    notes = ?, image_rel = ?, debug_run_id = ?,
                    skip_quality = ?, manual_adjusted = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    payload["group_name"],
                    payload["subgroup_name"],
                    payload["experiment_date"],
                    payload["operator"],
                    payload["camera_device"],
                    payload["light_device"],
                    payload["illuminance"],
                    payload["color"],
                    payload["grade_before"],
                    payload["lstar_before"],
                    payload["grade_after"],
                    payload["lstar_after"],
                    payload["notes"],
                    payload["image_rel"],
                    payload["debug_run_id"],
                    int(payload["skip_quality"]),
                    int(payload["manual_adjusted"]),
                    now,
                    record_id,
                ),
            )
        return self.get_by_id(record_id)

    def delete_record(self, record_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM experiment_records WHERE id = ?", (record_id,))
        return cur.rowcount > 0

    def bulk_delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM experiment_records WHERE id IN ({placeholders})", ids
            )
        return cur.rowcount

    def get_distinct_options(self) -> dict[str, list[str]]:
        fields = [
            "group_name",
            "subgroup_name",
            "operator",
            "camera_device",
            "light_device",
        ]
        out: dict[str, list[str]] = {
            "grades": list(GRADE_VALUES),
            "colors": list(COLOR_VALUES),
            "group_format": GROUP_FORMAT_HINT,
            "subgroup_format": SUBGROUP_FORMAT_HINT,
        }
        with self._connect() as conn:
            for field in fields:
                rows = conn.execute(
                    f"SELECT DISTINCT {field} FROM experiment_records "
                    f"WHERE {field} IS NOT NULL AND {field} != '' "
                    f"ORDER BY {field}"
                ).fetchall()
                out[field] = [r[0] for r in rows]
        return out

    def suggest_group_name(self, date_yyyymmdd: Optional[str] = None) -> str:
        del date_yyyymmdd  # 中文序号命名不再绑定日期
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT group_name FROM experiment_records").fetchall()
        max_seq = max((_parse_group_seq(r[0]) for r in rows), default=0)
        return f"第{_int_to_chinese(max_seq + 1)}大组"

    def suggest_subgroup_name(self, group_name: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT subgroup_name FROM experiment_records "
                "WHERE group_name = ? AND subgroup_name IS NOT NULL",
                (group_name,),
            ).fetchall()
        max_seq = max((_parse_subgroup_seq(r[0]) for r in rows), default=0)
        return f"第{_int_to_chinese(max_seq + 1)}小组"


class MysqlExperimentStore(ExperimentStore):
    def __init__(self, mysql_cfg: dict[str, Any]) -> None:
        if pymysql is None:
            raise RuntimeError("pymysql_not_installed")
        self._cfg = mysql_cfg
        self._ensure_database()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_MYSQL_SCHEMA)
            _mysql_migrate(conn)
            conn.commit()
        finally:
            conn.close()

    def _mysql_password(self) -> str:
        pwd = self._cfg.get("password")
        if pwd is not None and pwd != "":
            return str(pwd)
        return os.environ.get("EXPERIMENT_MYSQL_PASSWORD", "")

    def _ensure_database(self) -> None:
        db_name = self._cfg.get("database", "iris_experiment")
        conn = pymysql.connect(
            host=self._cfg.get("host", "127.0.0.1"),
            port=int(self._cfg.get("port", 3306)),
            user=self._cfg.get("user", "root"),
            password=self._mysql_password(),
            charset=self._cfg.get("charset", "utf8mb4"),
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            conn.close()

    def _connect(self):
        return pymysql.connect(
            host=self._cfg.get("host", "127.0.0.1"),
            port=int(self._cfg.get("port", 3306)),
            user=self._cfg.get("user", "root"),
            password=self._mysql_password(),
            database=self._cfg.get("database", "iris_experiment"),
            charset=self._cfg.get("charset", "utf8mb4"),
            cursorclass=DictCursor,
            autocommit=False,
        )

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if out.get("experiment_date") is not None:
            out["experiment_date"] = str(out["experiment_date"])[:10]
        for k in ("created_at", "updated_at"):
            if out.get(k) is not None:
                out[k] = str(out[k])
        for k in ("lstar_before", "lstar_after"):
            if out.get(k) is not None:
                out[k] = round(float(out[k]), 2)
        for field in ("skip_quality", "manual_adjusted"):
            if field in out and out[field] is not None:
                out[field] = bool(out[field])
        return out

    def list_records(
        self,
        *,
        group_name: Optional[str] = None,
        subgroup_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        operator: Optional[str] = None,
        camera_device: Optional[str] = None,
        light_device: Optional[str] = None,
        color: Optional[str] = None,
        grade_before: Optional[str] = None,
        grade_after: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def _add(field: str, value: Optional[str]) -> None:
            if value:
                clauses.append(f"{field} = %s")
                params.append(value)

        _add("group_name", group_name)
        _add("subgroup_name", subgroup_name)
        _add("operator", operator)
        _add("camera_device", camera_device)
        _add("light_device", light_device)
        _add("color", color)
        _add("grade_before", grade_before)
        _add("grade_after", grade_after)
        if date_from:
            clauses.append("experiment_date >= %s")
            params.append(date_from)
        if date_to:
            clauses.append("experiment_date <= %s")
            params.append(date_to)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experiment_records {where} ORDER BY experiment_date DESC, id DESC"
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        return [self._normalize_row(r) for r in rows]

    def get_by_id(self, record_id: int) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM experiment_records WHERE id = %s", (record_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        return self._normalize_row(row) if row else None

    def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_payload(data)
        now = _utc_now()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO experiment_records (
                        group_name, subgroup_name, experiment_date, operator,
                        camera_device, light_device, illuminance, color,
                        grade_before, lstar_before, grade_after, lstar_after,
                        notes, image_rel, debug_run_id, skip_quality, manual_adjusted,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        payload["group_name"],
                        payload["subgroup_name"],
                        payload["experiment_date"],
                        payload["operator"],
                        payload["camera_device"],
                        payload["light_device"],
                        payload["illuminance"],
                        payload["color"],
                        payload["grade_before"],
                        payload["lstar_before"],
                        payload["grade_after"],
                        payload["lstar_after"],
                        payload["notes"],
                        payload["image_rel"],
                        payload["debug_run_id"],
                        int(payload["skip_quality"]),
                        int(payload["manual_adjusted"]),
                        now,
                        now,
                    ),
                )
                record_id = cur.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        result = self.get_by_id(record_id)
        assert result is not None
        return result

    def update_record_images(
        self,
        record_id: int,
        *,
        image_before_rel: Optional[str],
        image_after_rel: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not self.get_by_id(record_id):
            return None
        now = _utc_now()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE experiment_records SET
                        image_before_rel = %s, image_after_rel = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (image_before_rel, image_after_rel, now, record_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.get_by_id(record_id)

    def update_record(self, record_id: int, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.get_by_id(record_id):
            return None
        payload = self._normalize_payload(data)
        now = _utc_now()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE experiment_records SET
                        group_name = %s, subgroup_name = %s, experiment_date = %s, operator = %s,
                        camera_device = %s, light_device = %s, illuminance = %s, color = %s,
                        grade_before = %s, lstar_before = %s, grade_after = %s, lstar_after = %s,
                        notes = %s, image_rel = %s, debug_run_id = %s,
                        skip_quality = %s, manual_adjusted = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (
                        payload["group_name"],
                        payload["subgroup_name"],
                        payload["experiment_date"],
                        payload["operator"],
                        payload["camera_device"],
                        payload["light_device"],
                        payload["illuminance"],
                        payload["color"],
                        payload["grade_before"],
                        payload["lstar_before"],
                        payload["grade_after"],
                        payload["lstar_after"],
                        payload["notes"],
                        payload["image_rel"],
                        payload["debug_run_id"],
                        int(payload["skip_quality"]),
                        int(payload["manual_adjusted"]),
                        now,
                        record_id,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.get_by_id(record_id)

    def delete_record(self, record_id: int) -> bool:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM experiment_records WHERE id = %s", (record_id,))
                deleted = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return deleted > 0

    def bulk_delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join(["%s"] * len(ids))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM experiment_records WHERE id IN ({placeholders})", ids
                )
                deleted = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return deleted

    def get_distinct_options(self) -> dict[str, list[str]]:
        fields = [
            "group_name",
            "subgroup_name",
            "operator",
            "camera_device",
            "light_device",
        ]
        out: dict[str, list[str]] = {
            "grades": list(GRADE_VALUES),
            "colors": list(COLOR_VALUES),
            "group_format": GROUP_FORMAT_HINT,
            "subgroup_format": SUBGROUP_FORMAT_HINT,
        }
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                for field in fields:
                    cur.execute(
                        f"SELECT DISTINCT {field} FROM experiment_records "
                        f"WHERE {field} IS NOT NULL AND {field} != '' "
                        f"ORDER BY {field}"
                    )
                    out[field] = [r[field] for r in cur.fetchall()]
        finally:
            conn.close()
        return out

    def suggest_group_name(self, date_yyyymmdd: Optional[str] = None) -> str:
        del date_yyyymmdd
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT group_name FROM experiment_records")
                rows = cur.fetchall()
        finally:
            conn.close()
        max_seq = max((_parse_group_seq(r["group_name"]) for r in rows), default=0)
        return f"第{_int_to_chinese(max_seq + 1)}大组"

    def suggest_subgroup_name(self, group_name: str) -> str:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT subgroup_name FROM experiment_records "
                    "WHERE group_name = %s AND subgroup_name IS NOT NULL",
                    (group_name,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        max_seq = max((_parse_subgroup_seq(r["subgroup_name"]) for r in rows), default=0)
        return f"第{_int_to_chinese(max_seq + 1)}小组"


def create_experiment_store(exp_cfg: dict[str, Any], root: Path) -> ExperimentStore:
    backend = (exp_cfg.get("backend") or "sqlite").lower()
    if backend == "mysql":
        return MysqlExperimentStore(exp_cfg.get("mysql") or {})
    db_rel = exp_cfg.get("db_path", "data/experiment_records.db")
    return SqliteExperimentStore(root / db_rel)
