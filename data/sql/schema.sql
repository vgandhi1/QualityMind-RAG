-- Manufacturing Quality Engineering schema (plan.md §4.2)
-- Matches upstream multidata-rag-project layout: data/sql/schema.sql

DROP TABLE IF EXISTS corrective_actions, inspection_results, defects, capa_log, eight_d, ncr, suppliers CASCADE;

CREATE TABLE suppliers (
    id              SERIAL PRIMARY KEY,
    supplier_code   VARCHAR(20) UNIQUE NOT NULL,
    name            VARCHAR(200) NOT NULL,
    tier            INTEGER,
    commodity       VARCHAR(100),
    quality_rating  DECIMAL(4,2),
    active          BOOLEAN DEFAULT TRUE
);

CREATE TABLE ncr (
    id              SERIAL PRIMARY KEY,
    ncr_number      VARCHAR(30) UNIQUE NOT NULL,
    part_number     VARCHAR(50),
    description     TEXT,
    quantity        INTEGER,
    disposition     VARCHAR(50),
    status          VARCHAR(20),
    opened_date     DATE,
    closed_date     DATE,
    root_cause      TEXT,
    cost_impact     DECIMAL(10,2)
);

CREATE TABLE defects (
    id              SERIAL PRIMARY KEY,
    part_number     VARCHAR(50),
    description     TEXT,
    failure_mode    VARCHAR(200),
    detection_station VARCHAR(100),
    disposition     VARCHAR(50),
    severity        INTEGER,
    date_found      DATE,
    shift           VARCHAR(10),
    operator_id     VARCHAR(20),
    supplier_id     INTEGER REFERENCES suppliers(id),
    ncr_id          INTEGER REFERENCES ncr(id)
);

CREATE TABLE capa_log (
    id              SERIAL PRIMARY KEY,
    capa_number     VARCHAR(30) UNIQUE NOT NULL,
    title           TEXT,
    problem_statement TEXT,
    root_cause      TEXT,
    corrective_action TEXT,
    preventive_action TEXT,
    owner           VARCHAR(100),
    supplier_id     INTEGER REFERENCES suppliers(id),
    status          VARCHAR(20),
    due_date        DATE,
    opened_date     DATE,
    closed_date     DATE,
    verified_by     VARCHAR(100),
    recurrence_flag BOOLEAN DEFAULT FALSE
);

CREATE TABLE eight_d (
    id              SERIAL PRIMARY KEY,
    report_number   VARCHAR(30) UNIQUE NOT NULL,
    part_number     VARCHAR(50),
    problem_statement TEXT,
    d1_team         TEXT,
    d2_problem_desc TEXT,
    d3_containment  TEXT,
    d4_root_cause   TEXT,
    d5_perm_action  TEXT,
    d6_implemented  TEXT,
    d7_prevention   TEXT,
    d8_closure      TEXT,
    status          VARCHAR(20),
    opened_date     DATE,
    closed_date     DATE,
    supplier_id     INTEGER REFERENCES suppliers(id)
);

CREATE TABLE inspection_results (
    id              SERIAL PRIMARY KEY,
    part_number     VARCHAR(50),
    characteristic  VARCHAR(200),
    measured_value  DECIMAL(12,6),
    nominal         DECIMAL(12,6),
    usl             DECIMAL(12,6),
    lsl             DECIMAL(12,6),
    cp              DECIMAL(6,3),
    cpk             DECIMAL(6,3),
    measurement_date TIMESTAMP,
    station         VARCHAR(100),
    gauge_id        VARCHAR(50),
    operator_id     VARCHAR(20)
);

CREATE TABLE corrective_actions (
    id              SERIAL PRIMARY KEY,
    action_text     TEXT,
    owner           VARCHAR(100),
    due_date        DATE,
    completed_date  DATE,
    status          VARCHAR(20),
    capa_id         INTEGER REFERENCES capa_log(id),
    eight_d_id      INTEGER REFERENCES eight_d(id),
    verified        BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_defects_part ON defects(part_number);
CREATE INDEX idx_defects_station ON defects(detection_station);
CREATE INDEX idx_capa_status ON capa_log(status);
CREATE INDEX idx_capa_supplier ON capa_log(supplier_id);
CREATE INDEX idx_ncr_part ON ncr(part_number);
CREATE INDEX idx_insp_part_char ON inspection_results(part_number, characteristic);
