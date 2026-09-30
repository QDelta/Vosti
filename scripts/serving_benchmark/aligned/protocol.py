"""Fail-closed identities and geometry for an aligned prefill cohort."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cohort:
    name: str
    cached: int
    query: int

    @property
    def outputs(self):
        return 128 if self.query == 1 else 1

    def ids(self):
        return [f'aligned_{self.name}_{self.cached}_{self.query}_{i}' for i in range(4)]


def identify(rid):
    if not rid.startswith('aligned_'):
        return None
    parts = rid.split('_')
    if len(parts) != 5:
        raise ValueError('malformed aligned request ID')
    _, name, cached, query, index = parts
    cached, query, index = int(cached), int(query), int(index)
    if not name.isascii() or not name.isalnum() or cached < 0 or query <= 0 or index not in range(4):
        raise ValueError('invalid aligned request geometry')
    return Cohort(name, cached, query)


def admission(ids):
    """None means no cohort, False means hold, Cohort means release all four."""
    cohorts = [identify(rid) for rid in ids]
    if not any(c is not None for c in cohorts):
        return None
    if any(c is None for c in cohorts) or len(set(cohorts)) != 1:
        raise ValueError('mixed aligned/unrelated requests or cohorts')
    cohort = cohorts[0]
    if len(ids) != len(set(ids)) or not set(ids) <= set(cohort.ids()):
        raise ValueError('duplicate or unexpected aligned request')
    return cohort if len(ids) == 4 else False


def check_batch(cohort, ids, prefix_lens, query_lens, *, is_extend):
    if len(ids) != 4 or set(ids) != set(cohort.ids()):
        raise ValueError('aligned cohort split or mixed in actual scheduled batch')
    if not is_extend or list(prefix_lens) != [cohort.cached]*4 or list(query_lens) != [cohort.query]*4:
        raise ValueError('actual scheduled cache/query geometry differs')
