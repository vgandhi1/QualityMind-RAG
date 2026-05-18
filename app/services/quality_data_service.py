"""
Parameterized read-only SQL for manufacturing quality endpoints (plan.md).
Uses a thread-safe connection pool to avoid creating a new TCP connection per query.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from app.config import settings

logger = logging.getLogger("rag_app.quality_data")

_PART_NUMBER_RE = re.compile(r"^[a-zA-Z0-9._\-]{1,64}$")

_quality_data_singleton: Optional["QualityDataService"] = None


def init_quality_data_service() -> None:
    global _quality_data_singleton
    if not settings.DATABASE_URL:
        _quality_data_singleton = None
        return
    try:
        _quality_data_singleton = QualityDataService()
        logger.info("Quality data service initialized")
    except Exception as e:
        logger.warning("Quality data service not initialized: %s", type(e).__name__)
        _quality_data_singleton = None


def get_quality_data_service() -> "QualityDataService":
    if _quality_data_singleton is None:
        raise RuntimeError("Quality data service unavailable")
    return _quality_data_singleton


def _validate_part_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if not _PART_NUMBER_RE.match(cleaned):
        raise ValueError("Invalid part_number")
    return cleaned


def _validate_supplier_id(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1 or value > 50_000_000:
        raise ValueError("Invalid supplier_id")
    return value


class QualityDataService:
    """Safe, parameterized queries against the quality schema with connection pooling."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or settings.DATABASE_URL
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for quality data endpoints")

        self._pool = ThreadedConnectionPool(
            settings.DB_POOL_MIN_CONNECTIONS,
            settings.DB_POOL_MAX_CONNECTIONS,
            self.database_url,
        )
        logger.info(
            "DB connection pool created (min=%d, max=%d)",
            settings.DB_POOL_MIN_CONNECTIONS,
            settings.DB_POOL_MAX_CONNECTIONS,
        )

    @contextmanager
    def _get_conn(self):
        """Yield a pooled connection; commit on success, rollback on error, return to pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Return all connections to the pool and close it (call on shutdown)."""
        self._pool.closeall()

    def capa_status(
        self,
        *,
        supplier_id: Optional[int] = None,
        overdue_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        supplier_id = _validate_supplier_id(supplier_id)
        limit = max(1, min(limit, 500))
        clauses = ["1=1"]
        params: list[Any] = []
        if supplier_id is not None:
            params.append(supplier_id)
            clauses.append("c.supplier_id = %s")
        if overdue_only:
            clauses.append("c.status = 'open' AND c.due_date < CURRENT_DATE")
        where_sql = " AND ".join(clauses)
        params.append(limit)
        sql = f"""
            SELECT c.capa_number, c.title, c.status, c.due_date, c.opened_date,
                   s.supplier_code, s.name AS supplier_name
            FROM capa_log c
            LEFT JOIN suppliers s ON s.id = c.supplier_id
            WHERE {where_sql}
            ORDER BY c.due_date NULLS LAST
            LIMIT %s
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, tuple(params))
                return [dict(row) for row in cur.fetchall()]

    def ncr_history(
        self,
        *,
        part_number: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        part_number = _validate_part_number(part_number)
        limit = max(1, min(limit, 500))
        if part_number:
            sql = """
                SELECT ncr_number, part_number, status, opened_date, closed_date, description
                FROM ncr
                WHERE part_number = %s
                ORDER BY opened_date DESC
                LIMIT %s
            """
            params = (part_number, limit)
        else:
            sql = """
                SELECT ncr_number, part_number, status, opened_date, closed_date, description
                FROM ncr
                ORDER BY opened_date DESC
                LIMIT %s
            """
            params = (limit,)
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def spc_aggregate(
        self,
        *,
        part_number: str,
        characteristic: Optional[str] = None,
        station: Optional[str] = None,
    ) -> dict[str, Any]:
        part_number = _validate_part_number(part_number)
        if not part_number:
            raise ValueError("part_number is required")
        if characteristic:
            characteristic = characteristic.strip()
            if len(characteristic) > 200 or not characteristic:
                raise ValueError("Invalid characteristic")
        if station:
            station = station.strip()
            if len(station) > 120 or not station:
                raise ValueError("Invalid station")

        clauses = ["part_number = %s"]
        params: list[Any] = [part_number]
        if characteristic:
            clauses.append("characteristic = %s")
            params.append(characteristic)
        if station:
            clauses.append("station = %s")
            params.append(station)
        where_sql = " AND ".join(clauses)

        sql = f"""
            SELECT
                COUNT(*) AS sample_count,
                AVG(cpk) AS avg_cpk,
                MIN(cpk) AS min_cpk,
                MAX(cpk) AS max_cpk,
                AVG(cp) AS avg_cp
            FROM inspection_results
            WHERE {where_sql}
        """
        by_char_sql = f"""
            SELECT characteristic,
                   COUNT(*) AS samples,
                   AVG(cpk) AS avg_cpk,
                   MIN(cpk) AS min_cpk
            FROM inspection_results
            WHERE {where_sql}
            GROUP BY characteristic
            ORDER BY min_cpk ASC NULLS LAST
            LIMIT 20
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, tuple(params))
                summary = dict(cur.fetchone() or {})
                cur.execute(by_char_sql, tuple(params))
                by_char = [dict(r) for r in cur.fetchall()]

        filters = {"part_number": part_number}
        if station:
            filters["station"] = station
        if characteristic:
            filters["characteristic"] = characteristic

        return {"summary": summary, "by_characteristic": by_char, "filters": filters}
