"""Prefix-cache models.

A prompt is a sequence of segments (system+tools+task, then per-step outputs/observations, or
synthetic segments such as a compaction summary). Each prompt carries a hash chain over its
segment ids, so "entry E is a prefix of prompt P" is an O(1) check:
``P.keys[E.n_segs - 1] == E.key``.

Cloud semantics (Anthropic and OpenAI GPT-5.6+, see configs/tiers.yaml):
  * every request places a breakpoint at the end of the prompt (automatic caching) and,
    optionally, an explicit breakpoint after the static prefix (system+tools+task);
  * lookup returns the longest live entry that is a prefix of the prompt;
  * an entry is live at time t iff t < expiry; a request starting at t refreshes the entries it
    reads (expiry = t + ttl) and writes the new full-prompt entry with expiry = t + ttl;
  * billing: read tokens = hit, write tokens = prompt - hit (only the delta past the highest hit);
  * prompts (or breakpoints) shorter than the minimum cacheable length are neither read nor
    written.

A parallel "ghost" cache with infinite TTL gives the counterfactual hit for the same request,
which defines cold delegations and the re-prefill excess (K1, K5, K6).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

SegId = Hashable


@dataclass(frozen=True, slots=True)
class Prompt:
    keys: tuple[int, ...]  # hash chain over segment ids
    cum_tokens: tuple[int, ...]
    cum_bytes: tuple[int, ...]
    static_segs: int  # number of leading segments that form the static prefix

    @property
    def n_segs(self) -> int:
        return len(self.keys)

    @property
    def tokens(self) -> int:
        return self.cum_tokens[-1] if self.cum_tokens else 0

    @property
    def nbytes(self) -> int:
        return self.cum_bytes[-1] if self.cum_bytes else 0

    def tokens_upto(self, n_segs: int) -> int:
        return self.cum_tokens[n_segs - 1] if n_segs > 0 else 0

    def bytes_upto(self, n_segs: int) -> int:
        return self.cum_bytes[n_segs - 1] if n_segs > 0 else 0

    def prefix(self, n_segs: int) -> Prompt:
        return Prompt(
            self.keys[:n_segs],
            self.cum_tokens[:n_segs],
            self.cum_bytes[:n_segs],
            min(self.static_segs, n_segs),
        )


class PromptBuilder:
    """Incrementally build prompts that share prefixes (O(1) per appended segment)."""

    __slots__ = ("_bytes", "_keys", "_tokens", "static_segs")

    def __init__(self) -> None:
        self._keys: list[int] = []
        self._tokens: list[int] = []
        self._bytes: list[int] = []
        self.static_segs = 0

    def append(self, seg_id: SegId, tokens: int, nbytes: int) -> PromptBuilder:
        prev = self._keys[-1] if self._keys else 0
        self._keys.append(hash((prev, seg_id)))
        self._tokens.append((self._tokens[-1] if self._tokens else 0) + tokens)
        self._bytes.append((self._bytes[-1] if self._bytes else 0) + nbytes)
        return self

    def mark_static(self) -> PromptBuilder:
        self.static_segs = len(self._keys)
        return self

    def build(self) -> Prompt:
        return Prompt(tuple(self._keys), tuple(self._tokens), tuple(self._bytes), self.static_segs)

    def copy(self) -> PromptBuilder:
        b = PromptBuilder()
        b._keys = list(self._keys)
        b._tokens = list(self._tokens)
        b._bytes = list(self._bytes)
        b.static_segs = self.static_segs
        return b

    def __len__(self) -> int:
        return len(self._keys)


def build_prompt(segments: Sequence[tuple[SegId, int, int]], static_segs: int = 0) -> Prompt:
    b = PromptBuilder()
    for i, (sid, tok, nb) in enumerate(segments):
        b.append(sid, tok, nb)
        if i + 1 == static_segs:
            b.mark_static()
    return b.build()


@dataclass(slots=True)
class _Entry:
    n_segs: int
    key: int
    tokens: int
    expiry: float  # inf for ghost / persistent entries
    last_used: float = 0.0


@dataclass(frozen=True, slots=True)
class CacheResult:
    prompt_tokens: int
    hit_tokens: int  # billed at read price
    write_tokens: int  # billed at write price
    uncached_tokens: int  # billed at base price (not cacheable)
    ghost_hit_tokens: int  # hit under infinite TTL (same request history)

    @property
    def expired_tokens(self) -> int:
        """Tokens that were cached earlier in the session but had expired."""
        return max(0, self.ghost_hit_tokens - self.hit_tokens)

    @property
    def cold(self) -> bool:
        """Cold delegation: a previously cached prefix of this prompt had expired."""
        return self.hit_tokens < self.ghost_hit_tokens

    @property
    def compulsory(self) -> bool:
        """Nothing of this prompt was ever cached before (first contact)."""
        return self.ghost_hit_tokens == 0


@dataclass
class PromptCache:
    """TTL prefix cache with refresh-on-hit (cloud) or persistent LRU cache (ttl_s=None, edge)."""

    ttl_s: float | None
    min_tokens: int = 0
    static_breakpoint: bool = True
    capacity_tokens: int | None = None
    _live: dict[tuple[int, int], _Entry] = field(default_factory=dict)
    _ghost: dict[tuple[int, int], _Entry] = field(default_factory=dict)

    def _expiry(self, t: float) -> float:
        return float("inf") if self.ttl_s is None else t + self.ttl_s

    @staticmethod
    def _longest(entries: dict[tuple[int, int], _Entry], p: Prompt, t: float) -> _Entry | None:
        best: _Entry | None = None
        for e in entries.values():
            if e.expiry <= t or e.n_segs > p.n_segs or p.keys[e.n_segs - 1] != e.key:
                continue
            if best is None or e.tokens > best.tokens:
                best = e
        return best

    def peek(self, p: Prompt, t: float) -> tuple[int, int]:
        """(live hit tokens, ghost hit tokens) for ``p`` at time ``t`` without mutating state."""
        if p.tokens < self.min_tokens:
            return 0, 0
        live = self._longest(self._live, p, t)
        ghost = self._longest(self._ghost, p, float("-inf"))
        return (live.tokens if live else 0), (ghost.tokens if ghost else 0)

    def live_expiry(self, p: Prompt, t: float) -> float:
        """Expiry time of the longest live prefix entry of ``p`` (or ``-inf``)."""
        e = self._longest(self._live, p, t)
        return e.expiry if e else float("-inf")

    def request(self, p: Prompt, t: float, write: bool = True) -> CacheResult:
        """Process a request starting at time ``t``; returns billing breakdown and mutates state."""
        total = p.tokens
        if total < self.min_tokens:
            return CacheResult(total, 0, 0, total, 0)
        self._prune(t)
        live = self._longest(self._live, p, t)
        ghost = self._longest(self._ghost, p, float("-inf"))
        hit = live.tokens if live else 0
        ghost_hit = ghost.tokens if ghost else 0
        exp = self._expiry(t)
        if live is not None:
            live.expiry = exp
            live.last_used = t
        breakpoints = [p.n_segs]
        if self.static_breakpoint and 0 < p.static_segs < p.n_segs:
            breakpoints.append(p.static_segs)
        for n in breakpoints:
            tok = p.tokens_upto(n)
            if tok < self.min_tokens:
                continue
            k = (n, p.keys[n - 1])
            e = self._live.get(k)
            if e is not None and e.expiry > t:
                e.expiry = exp
                e.last_used = t
            elif write:
                self._live[k] = _Entry(n, p.keys[n - 1], tok, exp, t)
            if write:
                self._ghost.setdefault(k, _Entry(n, p.keys[n - 1], tok, float("inf"), t))
        # The final breakpoint covers everything past the hit; writes bill only that delta.
        write_tokens = total - hit if write else 0
        uncached = total - hit - write_tokens
        if self.capacity_tokens is not None:
            self._evict_lru()
        return CacheResult(total, hit, write_tokens, uncached, ghost_hit)

    def _prune(self, t: float) -> None:
        if self.ttl_s is None:
            return
        dead = [k for k, e in self._live.items() if e.expiry <= t]
        for k in dead:
            del self._live[k]

    def _evict_lru(self) -> None:
        assert self.capacity_tokens is not None
        # Entries are nested prefixes stored in a radix tree in practice; approximate the resident
        # size by the largest entries and evict least-recently-used until under capacity.
        while self._live and sum(e.tokens for e in self._live.values()) > self.capacity_tokens:
            if len(self._live) == 1:
                break
            k = min(self._live, key=lambda kk: self._live[kk].last_used)
            del self._live[k]

    def clear(self) -> None:
        self._live.clear()
