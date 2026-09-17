"""Pure token-trie construction and scoring for parallel constrained decisions.

No MLX here: the trie and the scoring math are plain Python so they can be
unit-tested without a model (see tests/test_trie.py).

The probability model is the *constrained path probability*: the probability
that masked greedy-per-branch decoding yields a choice, i.e. the product over
the choice's branch points of the locally-masked next-token distribution. It
is a proper distribution over the schema's choices, but it is NOT a normalized
full-sequence likelihood — unary-token evidence and full-vocabulary
normalizers outside the allowed continuations are discarded.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def build_trie(remainders: list[list[int]]) -> list[dict]:
    """Build the branch-point list for one field's choice token remainders.

    Each remainder is the token continuation that distinguishes one choice
    after the field's shared token prefix. A node is a *branch point* when two
    or more choices diverge there (>= 2 distinct next tokens). The rows for a
    field are exactly its branch points: one row per node, holding
    ``shared_ids + path``.

    Returns a list of branch nodes in trie order, each::

        {"path": [token ids from the remainder start to this node, inclusive],
         "children": {token_id: [choice indices reaching that child]}}

    ``children`` is ordered by ascending token id so scoring is deterministic.
    Choices whose remainders never branch keep log-probability 0 (uniquely
    determined once their field's branch factors are applied). A strict-prefix
    remainder (one that is a prefix of another) can never be scored this way
    and is rejected at plan-compile time.
    """
    root: dict = {"children": {}, "choices": []}
    for choice_index, remainder in enumerate(remainders):
        root["choices"].append(choice_index)
        node = root
        for token in remainder:
            node = node["children"].setdefault(token, {"children": {}, "choices": []})
            node["choices"].append(choice_index)

    branch_nodes: list[dict] = []

    def walk(node: dict, path: list[int]) -> None:
        if len(node["children"]) >= 2:
            branch_nodes.append(
                {
                    "path": list(path),
                    "children": {
                        token: node["children"][token]["choices"]
                        for token in sorted(node["children"])
                    },
                }
            )
        for token in sorted(node["children"]):
            walk(node["children"][token], [*path, token])

    walk(root, [])
    return branch_nodes


def logsumexp(values: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    largest = max(values)
    return largest + math.log(sum(math.exp(v - largest) for v in values))


def log_softmax(values: list[float]) -> list[float]:
    """Numerically stable log-softmax (never takes log(0))."""
    lse = logsumexp(values)
    return [v - lse for v in values]


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax; temperature must be finite and > 0."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    scaled = [v / temperature for v in values]
    lse = logsumexp(scaled)
    return [math.exp(v - lse) for v in scaled]


def score_trie(
    branch_nodes: list[dict],
    n_choices: int,
    logits_at_node: Callable[[dict], list[float]],
) -> list[float]:
    """Natural-log constrained-path probability per choice, at temperature 1.

    ``logits_at_node(node)`` must return the child logits in the same order as
    ``list(node["children"])``. At each branch node the children's full-vocab
    logits are log-softmaxed over the allowed continuations and every choice
    under a child accumulates that child's log-probability. Temperature is NOT
    applied here: callers score at T=1 and apply their confidence temperature
    once to the final per-choice scores (see calibrate.py), so ranking is
    invariant to it.

    This is the constrained path probability — the probability that masked
    greedy-per-branch decoding yields the choice — not a normalized
    full-sequence likelihood. It is a proper distribution over the choices:
    the probabilities exp(scores) sum to 1.

    Choices with no branch nodes on their path score log P = 0.0: their value
    is fully determined by the field's other branch decisions. A field with a
    single choice therefore scores log P = 0 (probability 1.0) with no rows.
    """
    log_probs = [0.0] * n_choices
    for node in branch_nodes:
        child_logits = logits_at_node(node)
        if not all(math.isfinite(value) for value in child_logits):
            raise ValueError(f"non-finite logits at branch node {node['path']!r}: {child_logits!r}")
        child_log_probs = log_softmax(child_logits)
        for token, log_prob in zip(node["children"], child_log_probs, strict=True):
            for choice_index in node["children"][token]:
                log_probs[choice_index] += log_prob
    return log_probs
