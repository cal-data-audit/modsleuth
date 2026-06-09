"""Post-merge dedup pipeline.

Four stages run on the merged JSON graph (produced by ``modsleuth run merge``)
to remove residual duplicates and intermediate research checkpoints before
evaluation:

    heuristic   signature clustering + fuzzy surface-form merge (no LLM)
    hub-audit   per-hub LLM audit; drops dup / hallucinated / vacuous edges
    node-dedup  whole-graph LLM-verified node merges; conflict-guarded union
    release     LLM KEEP / DROP per node + transitive rewiring

Run as ``python -m modsleuth.dedup --source merge.json --dest graph.json``
or via ``modsleuth dedup --source merge.json --dest graph.json``.
"""

from .lib import (  # noqa: F401
    ConflictGuardedUnionFind,
    StageLogger,
    assert_invariants,
    load_graph,
    save_graph,
)
