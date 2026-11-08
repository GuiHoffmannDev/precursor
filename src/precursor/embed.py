"""Embed complaint narratives into pgvector. Resumable: only missing or changed rows are embedded."""

from collections.abc import Callable

import psycopg

MODEL = "Alibaba-NLP/gte-modernbert-base"  # 768 dims; must match halfvec(768) in schema.sql
MAX_TOKENS = 512  # p95 narrative is ~1.4k chars (~350 tokens)

PENDING_SQL = """
    select c.odino, c.narrative, md5(c.narrative)
      from complaints c
      left join complaint_embeddings e using (odino)
     where (e.odino is null or e.text_hash <> md5(c.narrative) or e.model <> %(model)s)
       and c.odino > %(after)s
       and (not %(labeled_only)s or exists (select 1 from llm_labels l where l.odino = c.odino))
     order by c.odino
     limit %(n)s
"""
UPSERT_SQL = """
    insert into complaint_embeddings (odino, model, text_hash, embedding)
    values (%s, %s, %s, %s::halfvec)
    on conflict (odino) do update
       set model = excluded.model, text_hash = excluded.text_hash,
           embedding = excluded.embedding, created_at = now()
"""


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    model.max_seq_length = MAX_TOKENS
    return model


def encode(model, texts: list[str]) -> list[str]:
    """Normalized embeddings as pgvector text literals."""
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
    return ["[" + ",".join(f"{x:.5f}" for x in v) + "]" for v in vectors]


def embed_pending(
    conn: psycopg.Connection,
    encode_fn: Callable[[list[str]], list[str]],
    *,
    batch: int = 1024,
    limit: int | None = None,
    labeled_only: bool = False,
    on_batch: Callable[[int], None] = lambda n: None,
) -> int:
    """Embed complaints that have no embedding, a stale one, or one from another model.

    labeled_only restricts the run to complaints with LLM labels, so training can start
    before the full corpus is embedded.
    """
    done, after = 0, ""
    while limit is None or done < limit:
        n = batch if limit is None else min(batch, limit - done)
        params = {"model": MODEL, "after": after, "n": n, "labeled_only": labeled_only}
        rows = conn.execute(PENDING_SQL, params).fetchall()
        if not rows:
            break
        vectors = encode_fn([r[1] for r in rows])
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, [(r[0], MODEL, r[2], v) for r, v in zip(rows, vectors, strict=True)])
        conn.commit()  # commit per batch so an interrupted run keeps its progress
        done += len(rows)
        after = rows[-1][0]
        on_batch(done)
    return done
