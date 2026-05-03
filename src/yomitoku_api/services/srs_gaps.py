"""SRS knowledge-gap persistence via Supabase (JSONB rows).

Expect a table::

    create table srs_knowledge_gaps (
      id text primary key,
      gap jsonb not null,
      practice_results jsonb not null default '[]'::jsonb
    );
"""

from __future__ import annotations

from typing import Any

from supabase import Client

from yomitoku_api.schemas import KnowledgeGap, KnowledgeGapPartial, PracticeResult

SRS_GAPS_TABLE = "srs_knowledge_gaps"


def _practice_list_from_cell(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


def upsert_gap(client: Client, gap: KnowledgeGap) -> None:
    res = (
        client.table(SRS_GAPS_TABLE)
        .select("practice_results")
        .eq("id", gap.id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    preserved: list[dict[str, Any]] = []
    if rows and isinstance(rows[0], dict):
        preserved = _practice_list_from_cell(rows[0].get("practice_results"))
    payload = {
        "id": gap.id,
        "gap": gap.model_dump(mode="json"),
        "practice_results": preserved,
    }
    client.table(SRS_GAPS_TABLE).upsert(payload, on_conflict="id").execute()


def merge_gap(existing: KnowledgeGap, patch: KnowledgeGapPartial) -> KnowledgeGap:
    base = existing.model_dump()
    for field, value in patch.model_dump(exclude_unset=True).items():
        if field in base:
            base[field] = value
    return KnowledgeGap.model_validate(base)


def update_gap_partial(
    client: Client,
    gap_id: str,
    patch: KnowledgeGapPartial,
) -> KnowledgeGap | None:
    res = (
        client.table(SRS_GAPS_TABLE)
        .select("gap,practice_results")
        .eq("id", gap_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    if not rows or not isinstance(rows[0], dict):
        return None
    cell = rows[0].get("gap")
    if not isinstance(cell, dict):
        return None
    merged = merge_gap(KnowledgeGap.model_validate(cell), patch)
    upsert_gap(client, merged)
    return merged


def list_gaps(client: Client) -> list[KnowledgeGap]:
    res = client.table(SRS_GAPS_TABLE).select("gap").execute()
    rows = getattr(res, "data", None) or []
    out: list[KnowledgeGap] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gap_cell = row.get("gap")
        if isinstance(gap_cell, dict):
            out.append(KnowledgeGap.model_validate(gap_cell))
    out.sort(key=lambda g: g.createdAtIso or "")
    return out


def delete_gap(client: Client, gap_id: str) -> None:
    client.table(SRS_GAPS_TABLE).delete().eq("id", gap_id).execute()


def append_practice_result(
    client: Client,
    gap_id: str,
    result: PracticeResult,
) -> KnowledgeGap | None:
    res = (
        client.table(SRS_GAPS_TABLE)
        .select("gap,practice_results")
        .eq("id", gap_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    if not rows or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    gap_cell = row.get("gap")
    if not isinstance(gap_cell, dict):
        return None
    gap = KnowledgeGap.model_validate(gap_cell)
    history = _practice_list_from_cell(row.get("practice_results"))
    history.append(result.model_dump(mode="json"))
    client.table(SRS_GAPS_TABLE).upsert(
        {"id": gap_id, "gap": gap.model_dump(mode="json"), "practice_results": history},
        on_conflict="id",
    ).execute()
    return gap
