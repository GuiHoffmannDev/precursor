-- Precursor schema. Idempotent: safe to re-run.
--
-- As-of rule: every query that feeds detection, retrieval or evaluation must filter
-- on `available_at <= :as_of` (complaints) or `report_received_date <= :as_of`
-- (recalls). Otherwise backtests see the future.

create extension if not exists pg_search;
create extension if not exists vector;

-- One row per NHTSA complaint (ODINO). The flat file repeats a complaint once per
-- product/component; those rows live in complaint_products.
create table if not exists complaints (
    odino           text primary key,
    received_date   date not null,          -- LDATE: received by NHTSA
    available_at    date not null,          -- greatest(LDATE, DATEA): first day it was public
    incident_date   date,                   -- FAILDATE
    narrative       text not null,          -- CDESCR
    crash           boolean,
    fire            boolean,
    injured         integer,
    deaths          integer,
    medical_attn    boolean,
    towed           boolean,
    police_report   boolean,
    miles           integer,
    vehicle_speed   integer,
    state           text,
    -- CMPL_TYPE. 'RC'/'RP'/'DP' complaints are produced by recall or petition
    -- processes, so they leak labels and must be excluded from signal detection.
    source_type     text,
    loaded_at       timestamptz not null default now()
);

create table if not exists complaint_products (
    odino           text not null references complaints (odino) on delete cascade,
    cmplid          text not null,          -- NHTSA says this id can change between files
    mfr_name        text,
    make            text,
    model           text,
    model_year      integer,                -- null when NHTSA reports 9999 (unknown)
    component       text,
    product_type    text,                   -- V vehicle, T tires, E equipment, C child restraint
    primary key (odino, cmplid)
);

create index if not exists complaint_products_vehicle_idx
    on complaint_products (make, model, model_year);
create index if not exists complaint_products_component_idx
    on complaint_products (component);
create index if not exists complaints_available_at_idx
    on complaints (available_at);

-- BM25 over complaint narratives. available_at is indexed so the as-of filter is
-- pushed into the search instead of applied after top-k.
create index if not exists complaints_bm25_idx
    on complaints using bm25 (odino, narrative, available_at);

-- Narrative embeddings. text_hash detects narratives changed by a later NHTSA
-- release, so the embed job re-embeds them. Dimension is fixed by the model.
create table if not exists complaint_embeddings (
    odino       text primary key references complaints (odino) on delete cascade,
    model       text not null,
    text_hash   text not null,              -- md5(narrative) at embedding time
    embedding   halfvec(768) not null,
    created_at  timestamptz not null default now()
);

create index if not exists complaint_embeddings_hnsw_idx
    on complaint_embeddings using hnsw (embedding halfvec_cosine_ops);

-- LLM weak labels (hazard, severity, entity spans) used to train and evaluate the
-- small models. label is null and error set when the LLM never produced a valid answer.
create table if not exists llm_labels (
    odino           text not null references complaints (odino) on delete cascade,
    prompt_version  text not null,
    model           text not null,
    label           jsonb,
    error           text,
    attempts        integer not null,
    created_at      timestamptz not null default now(),
    primary key (odino, prompt_version)
);

-- LLM hazard labels for recall campaigns (same taxonomy as complaints), the backtest
-- labels. Built from defect/consequence text, which is public from the report date.
-- label is null and error set when the LLM never produced a valid answer.
create table if not exists recall_labels (
    campno          text not null,
    prompt_version  text not null,
    model           text not null,
    label           jsonb,
    error           text,
    attempts        integer not null,
    created_at      timestamptz not null default now(),
    primary key (campno, prompt_version)
);

-- Human-reviewed labels on held-out complaints: the ground truth for evaluation.
create table if not exists gold_labels (
    odino       text primary key references complaints (odino) on delete cascade,
    hazard      text not null,
    severity    text not null,
    notes       text,
    created_at  timestamptz not null default now()
);

-- Distilled classifier output for every embedded complaint.
create table if not exists complaint_predictions (
    odino           text primary key references complaints (odino) on delete cascade,
    model_version   text not null,          -- prompt version : model kinds : training date
    hazard          text not null,
    hazard_conf     real not null,
    severity        text not null,
    severity_conf   real not null,
    created_at      timestamptz not null default now()
);

-- One row per (campaign, make, model, year, component), as in FLAT_RCL.
create table if not exists recalls (
    campno                  text not null,      -- NHTSA recall number
    record_id               bigint not null,
    make                    text,
    model                   text,
    model_year              integer,
    component               text,
    mfr_name                text,
    report_received_date    date,               -- RCDATE: Part 573 report received; backtest label date
    owner_notification_date date,               -- ODATE
    manufacture_begin       date,
    manufacture_end         date,
    recall_type             text,               -- V, E, C, T
    influenced_by           text,               -- MFR / OVSC / ODI
    potentially_affected    integer,
    defect_summary          text,
    consequence_summary     text,
    corrective_action       text,
    notes                   text,
    loaded_at               timestamptz not null default now(),
    primary key (campno, record_id)
);

create index if not exists recalls_vehicle_idx on recalls (make, model, model_year);
create index if not exists recalls_received_idx on recalls (report_received_date);

-- Agent investigations of signals (Phase 4). One row per run as of a date. The report's
-- claims cite complaint ODINOs that were verified in code; the trace keeps every LLM and
-- tool call with tokens and latency, for the evaluation harness.
create table if not exists investigations (
    id                  bigserial primary key,
    as_of               date not null,
    make                text not null,
    model               text not null,
    model_year          integer not null,
    hazard              text not null,
    signal              jsonb not null,         -- detector window, complaint count, expected, score
    tag                 text not null,          -- 'adhoc', or an evaluation run such as 'triage-dev:agent'
    prompt_version      text not null,
    llm                 text not null,
    blind               boolean not null,       -- make and model masked in everything the LLM saw
    use_tools           boolean not null,
    status              text not null,          -- 'ok', or 'invalid' when no valid report was produced
    recommendation      text,                   -- escalate / monitor / dismiss
    confidence          text,
    report              jsonb,
    markdown            text,
    error               text,
    trace               jsonb not null,
    llm_calls           integer not null,
    tool_calls          integer not null,
    prompt_tokens       integer not null,
    completion_tokens   integer not null,
    latency_ms          integer not null,
    created_at          timestamptz not null default now()
);

create index if not exists investigations_lookup_idx
    on investigations (tag, make, model, model_year, hazard, as_of);
