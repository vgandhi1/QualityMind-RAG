"""
Manufacturing quality synthetic data (plan.md Phase 1).

Aligned with upstream multidata-rag-project pattern:
  https://github.com/sourangshupal/multidata-rag-project
  (see data/generate_sample_data.py there for the original e-commerce script.)

Usage:
  export DATABASE_URL=postgresql://...
  python data/generate_sample_data.py

Or: hatch run generate-sample-data
"""

from __future__ import annotations

import os
import random
from datetime import date, datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set. Configure .env or the environment.")
        raise SystemExit(1)

    random.seed(42)
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        TRUNCATE corrective_actions, inspection_results, defects, capa_log, eight_d, ncr, suppliers
        RESTART IDENTITY CASCADE;
        """
    )

    suppliers = []
    for i in range(1, 26):
        code = f"S{i:04d}"
        suppliers.append(
            (
                code,
                f"Supplier {code} Components",
                random.choice([1, 2, 3]),
                random.choice(["fasteners", "seals", "electronics", "plastics"]),
                round(random.uniform(72, 99), 2),
                True,
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO suppliers (supplier_code, name, tier, commodity, quality_rating, active)
        VALUES %s
        """,
        suppliers,
    )

    ncrs = []
    for i in range(1, 81):
        opened = date.today() - timedelta(days=random.randint(1, 400))
        status = random.choice(["open", "closed", "on-hold", "open"])
        closed = opened + timedelta(days=random.randint(5, 120)) if status == "closed" else None
        ncrs.append(
            (
                f"NCR-{2024 + (i % 2)}-{i:04d}",
                random.choice(["EDV-FASCIA-01", "BRACKET-A12", "SEAL-DOOR", "PCB-CTRL", "FAST-M6"]),
                random.choice(
                    [
                        "Dimensional variation on critical feature",
                        "Surface defect cosmetic",
                        "Torque failure at assembly",
                        "Delamination observed",
                    ]
                ),
                random.randint(1, 500),
                random.choice(["rework", "scrap", "use-as-is"]),
                status,
                opened,
                closed,
                "Under investigation" if status == "open" else "Resolved per disposition",
                round(random.uniform(100, 50000), 2),
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO ncr (ncr_number, part_number, description, quantity, disposition, status,
            opened_date, closed_date, root_cause, cost_impact)
        VALUES %s
        """,
        ncrs,
    )

    cur.execute("SELECT id FROM ncr")
    ncr_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM suppliers")
    supplier_ids = [r[0] for r in cur.fetchall()]

    defects = []
    stations = [f"Station {s}" for s in range(1, 21)]
    modes = [
        "Torque failure",
        "Seal leakage",
        "Delamination",
        "Fastener strip",
        "Dimensional out of spec",
    ]
    for _ in range(400):
        defects.append(
            (
                random.choice(["EDV-FASCIA-01", "BRACKET-A12", "SEAL-DOOR", "PCB-CTRL", "FAST-M6"]),
                "Defect logged during production",
                random.choice(modes),
                random.choice(stations),
                random.choice(["rework", "scrap", "use-as-is"]),
                random.randint(1, 10),
                date.today() - timedelta(days=random.randint(0, 365)),
                random.choice(["A", "B", "C"]),
                f"OP-{random.randint(1000, 9999)}",
                random.choice(supplier_ids),
                random.choice(ncr_ids) if random.random() > 0.3 else None,
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO defects (part_number, description, failure_mode, detection_station, disposition,
            severity, date_found, shift, operator_id, supplier_id, ncr_id)
        VALUES %s
        """,
        defects,
    )

    capas = []
    for i in range(1, 121):
        opened = date.today() - timedelta(days=random.randint(1, 500))
        due = opened + timedelta(days=random.randint(30, 120))
        status = random.choice(["open", "closed", "verified", "open"])
        closed = due - timedelta(days=random.randint(1, 20)) if status != "open" else None
        capas.append(
            (
                f"CAPA-{2024 + (i % 2)}-{i:04d}",
                f"Corrective action for issue {i}",
                "Recurring failure mode observed in production",
                "Process variation suspected",
                "Update control plan and retrain operators",
                "Add SPC monitoring on critical parameter",
                f"owner{i % 20}@example.com",
                random.choice(supplier_ids),
                status,
                due,
                opened,
                closed,
                "QE Lead" if status != "open" else None,
                random.random() < 0.1,
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO capa_log (capa_number, title, problem_statement, root_cause, corrective_action,
            preventive_action, owner, supplier_id, status, due_date, opened_date, closed_date,
            verified_by, recurrence_flag)
        VALUES %s
        """,
        capas,
    )

    eight_ds = []
    for i in range(1, 41):
        opened = date.today() - timedelta(days=random.randint(30, 600))
        eight_ds.append(
            (
                f"8D-{2024}-{i:04d}",
                random.choice(["EDV-FASCIA-01", "BRACKET-A12", "FAST-M6"]),
                "Customer complaint on field performance",
                "Cross-functional team assigned",
                "Symptoms quantified from warranty data",
                "Interim sorting at warehouse",
                "Root cause linked to weld parameter drift",
                "Permanent corrective action implemented on line",
                "Verification runs completed",
                "PFMEA and control plan updated",
                "Team recognized; lessons learned captured",
                random.choice(["open", "closed", "open"]),
                opened,
                opened + timedelta(days=90) if random.random() > 0.4 else None,
                random.choice(supplier_ids),
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO eight_d (report_number, part_number, problem_statement, d1_team, d2_problem_desc,
            d3_containment, d4_root_cause, d5_perm_action, d6_implemented, d7_prevention, d8_closure,
            status, opened_date, closed_date, supplier_id)
        VALUES %s
        """,
        eight_ds,
    )

    cur.execute("SELECT id FROM capa_log")
    capa_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM eight_d")
    eight_ids = [r[0] for r in cur.fetchall()]

    inspections = []
    for _ in range(800):
        nominal = round(random.uniform(10.0, 50.0), 3)
        usl = nominal + 0.5
        lsl = nominal - 0.5
        measured = nominal + random.uniform(-0.4, 0.4)
        cpk = round(random.uniform(0.8, 1.8), 3)
        inspections.append(
            (
                random.choice(["EDV-FASCIA-01", "BRACKET-A12", "SEAL-DOOR"]),
                random.choice(["flange thickness", "torque residual", "seal compression"]),
                round(measured, 6),
                nominal,
                usl,
                lsl,
                round(random.uniform(1.0, 2.0), 3),
                cpk,
                datetime.utcnow() - timedelta(days=random.randint(0, 180)),
                random.choice(stations),
                f"G-{random.randint(10, 99)}",
                f"OP-{random.randint(1000, 9999)}",
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO inspection_results (part_number, characteristic, measured_value, nominal, usl, lsl,
            cp, cpk, measurement_date, station, gauge_id, operator_id)
        VALUES %s
        """,
        inspections,
    )

    actions = []
    for _ in range(150):
        use_capa = random.random() > 0.5
        actions.append(
            (
                "Implement containment and verification",
                f"owner{random.randint(0, 19)}@example.com",
                date.today() + timedelta(days=random.randint(1, 60)),
                date.today() - timedelta(days=random.randint(0, 30)) if random.random() > 0.4 else None,
                random.choice(["open", "closed", "verified"]),
                random.choice(capa_ids) if use_capa else None,
                random.choice(eight_ids) if not use_capa else None,
                random.random() > 0.3,
            )
        )
    execute_values(
        cur,
        """
        INSERT INTO corrective_actions (action_text, owner, due_date, completed_date, status,
            capa_id, eight_d_id, verified)
        VALUES %s
        """,
        actions,
    )

    cur.close()
    conn.close()
    print("Manufacturing quality seed completed successfully.")


if __name__ == "__main__":
    main()
