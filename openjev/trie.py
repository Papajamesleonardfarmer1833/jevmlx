"""Pure token-trie construction and scoring for parallel constrained decisions.

No MLX here: the trie and the scoring math are plain Python so they can be
unit-tested without a model (see tests/test_trie.py).
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
    determined once their field's branch factors are applied).
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


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over a list of floats (temperature must be > 0)."""
    largest = max(values)
    exps = [math.exp((v - largest) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def score_trie(
    branch_nodes: list[dict],
    n_choices: int,
    logits_at_node: Callable[[dict], list[float]],
    temperature: float = 1.0,
) -> list[float]:
    """Natural-log probability per choice: the product of branch factors.

    ``logits_at_node(node)`` must return the child logits in the same order as
    ``list(node["children"])``. At each branch node the children's logits
    (divided by ``temperature``, which must be > 0) are softmaxed and every
    choice under a child accumulates ``log P(child)``. The result is a proper
    distribution over the choices: the probabilities sum to 1, so the winner's
    probability is the confidence and no extra softmax is applied.

    Choices with no branch nodes on their path score log P = 0.0: their value
    is fully determined by the field's other branch decisions.
    """
    log_probs = [0.0] * n_choices
    for node in branch_nodes:
        child_tokens = list(node["children"])
        child_logits = logits_at_node(node)
        probs = softmax(child_logits, temperature=temperature)
        for token, prob in zip(child_tokens, probs, strict=True):
            log_prob = math.log(prob)
            for choice_index in node["children"][token]:
                log_probs[choice_index] += log_prob
    return log_probs
