"""Tiny helpers for word-level tokenization and branch / prefix analysis."""


def tokenize(s: str) -> list[str]:
    return s.strip().lower().split()


def build_vocab(sentences: list[str]) -> tuple[dict[str, int], list[str]]:
    words: list[str] = []
    seen: set[str] = set()
    for sent in sentences:
        for w in tokenize(sent):
            if w not in seen:
                seen.add(w)
                words.append(w)
    stoi = {w: i for i, w in enumerate(words)}
    itos = words
    return stoi, itos


def encode(words: list[str], stoi: dict[str, int]) -> list[int]:
    return [stoi[w] for w in words]


def decode(ids: list[int], itos: list[str]) -> list[str]:
    return [itos[i] for i in ids]


def first_divergence_index(a: list[str], b: list[str]) -> int | None:
    """
    Index of the first token where two sequences disagree.
    If one is a strict prefix of the other, returns len(shorter).
    Identical sequences return None.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def shared_prefix_until_divergence(a: list[str], b: list[str]) -> list[str]:
    """Tokens common to both sequences up to (but not including) the first differing index."""
    idx = first_divergence_index(a, b)
    if idx is None:
        return list(a)
    return a[:idx]
