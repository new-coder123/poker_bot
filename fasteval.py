"""
Fast hand evaluation for 5 to 7 cards, pure Python.

Cards are ints 0..51: rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3.
score(cards) returns an int; higher wins, equal splits. The category is
score >> 20 (0 high card .. 8 straight flush), as in card_utils.

Tables are built once when this module is imported (well under a second).
"""

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "CDHS"

# Per card: base-5 rank key and base-8 suit key.
KEY = [5 ** (c >> 2) for c in range(52)]
SKEY = [8 ** (c & 3) for c in range(52)]
BIT = [1 << (c >> 2) for c in range(52)]
SUIT = [c & 3 for c in range(52)]


def card_int(card) -> int:
    """A card_utils.Card to an int."""
    return RANK_CHARS.index(card.rank) * 4 + SUIT_CHARS.index(card.suit)


def card_str(c: int) -> str:
    return RANK_CHARS[c >> 2] + SUIT_CHARS[c & 3]


def _pack(cat, ranks):
    s = cat
    for i in range(5):
        s = (s << 4) | (ranks[i] if i < len(ranks) else 0)
    return s


def _straight_high(mask):
    for hi in range(12, 3, -1):
        m = 0b11111 << (hi - 4)
        if mask & m == m:
            return hi
    if mask & 0b1000000001111 == 0b1000000001111:  # wheel
        return 3
    return -1


STRAIGHT = [_straight_high(m) for m in range(8192)]


def _flush_score(mask):
    sh = STRAIGHT[mask]
    if sh >= 0:
        return _pack(8, [sh])
    ranks = [r for r in range(12, -1, -1) if mask >> r & 1][:5]
    return _pack(5, ranks)


FLUSH = [0] * 8192
for _m in range(8192):
    if bin(_m).count("1") >= 5:
        FLUSH[_m] = _flush_score(_m)

# Which suit (if any) has five or more cards, by summed suit key.
FLUSH_SUIT = [-1] * 4096
for _k in range(4096):
    for _s in range(4):
        if (_k >> (3 * _s)) & 7 >= 5:
            FLUSH_SUIT[_k] = _s


def _counts_score(counts):
    """Best non-flush 5-card score from rank counts (index 0..12)."""
    present = 0
    for r in range(13):
        if counts[r]:
            present |= 1 << r
    by_rank = [r for r in range(12, -1, -1) if counts[r]]
    quads = [r for r in by_rank if counts[r] == 4]
    trips = [r for r in by_rank if counts[r] == 3]
    pairs = [r for r in by_rank if counts[r] == 2]
    if quads:
        q = quads[0]
        kick = [r for r in by_rank if r != q][:1]
        return _pack(7, [q] + kick)
    if trips and (len(trips) >= 2 or pairs):
        t = trips[0]
        p = max([r for r in trips[1:]] + pairs)
        return _pack(6, [t, p])
    sh = STRAIGHT[present]
    if sh >= 0:
        return _pack(4, [sh])
    if trips:
        t = trips[0]
        kick = [r for r in by_rank if r != t][:2]
        return _pack(3, [t] + kick)
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kick = [r for r in by_rank if r != p1 and r != p2][:1]
        return _pack(2, [p1, p2] + kick)
    if pairs:
        p = pairs[0]
        kick = [r for r in by_rank if r != p][:3]
        return _pack(1, [p] + kick)
    return _pack(0, by_rank[:5])


NONFLUSH = {}


def _build_nonflush():
    counts = [0] * 13

    def rec(r, left, key):
        if r == 13:
            n = sum(counts)
            if n >= 5:
                NONFLUSH[key] = _counts_score(counts)
            return
        for c in range(min(4, left) + 1):
            counts[r] = c
            rec(r + 1, left - c, key + c * 5 ** r)
        counts[r] = 0

    rec(0, 7, 0)


_build_nonflush()


def score(cards) -> int:
    """Score of the best 5-card hand among 5 to 7 int cards."""
    k = 0
    sk = 0
    for c in cards:
        k += KEY[c]
        sk += SKEY[c]
    fs = FLUSH_SUIT[sk]
    if fs >= 0:
        m = 0
        for c in cards:
            if c & 3 == fs:
                m |= BIT[c]
        f = FLUSH[m]
        n = NONFLUSH[k]
        return f if f > n else n
    return NONFLUSH[k]


def category(s: int) -> int:
    return s >> 20
