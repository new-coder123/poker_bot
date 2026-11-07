"""
An adaptive heads-up no-limit hold'em bot.

The opponent model: in each kind of spot (first in preflop, facing my
open, facing my bet of some size, checked to, facing an all-in, ...) the
opponent is assumed to pick its move by thresholds on absolute hand
strength: the strongest hands take the most aggressive move, then the
next, and so on, plus a "leak" share of moves taken at random. The
thresholds and the leak are fitted per spot by maximum likelihood to
everything seen so far (the hand-strength distribution of their range
each time they acted, sharpened to their exact hand once it is shown),
starting from priors. So it learns within a few dozen hands who folds too
much, who never folds, and who bets at random.

The opponent's range is a weight for every hand they could hold; each of
their moves multiplies it by the chance of that move given each hand.

Decisions: for each candidate move (fold, check/call, a few bet sizes,
all-in) it estimates the expected chips, using how the current range
would answer and its equity against each slice, and plays the best one.

Preflop equities come from a precomputed 169 x 169 table (preflop_data);
postflop equities from Monte Carlo with a fast evaluator (fasteval).
"""

import math
import random

import fasteval as fe
from player_action import PlayerAction
from preflop_data import EQ

# -- static tables (built on import, not pickled) ----------------------------

ALL_COMBOS = []  # (c1, c2), c1 > c2
COMBO_CLASS = []
COMBO_INDEX = {}
for _c1 in range(52):
    for _c2 in range(_c1):
        COMBO_INDEX[(_c1, _c2)] = len(ALL_COMBOS)
        ALL_COMBOS.append((_c1, _c2))
        _r1, _r2 = _c1 >> 2, _c2 >> 2
        if _r1 < _r2:
            _r1, _r2 = _r2, _r1
        if _r1 == _r2:
            COMBO_CLASS.append(_r1 * 13 + _r1)
        elif (_c1 & 3) == (_c2 & 3):
            COMBO_CLASS.append(_r1 * 13 + _r2)
        else:
            COMBO_CLASS.append(_r2 * 13 + _r1)

CLASS_N = [0] * 169
for _cl in COMBO_CLASS:
    CLASS_N[_cl] += 1

# Strength of a class: equity against a random hand; ABS = its quantile
# among all 1326 hands (0 weakest .. 1 strongest).
CLASS_STRENGTH = [
    sum(EQ[i][j] * CLASS_N[j] for j in range(169)) / 1326.0
    for i in range(169)
]
CLASS_ORDER = sorted(range(169), key=lambda i: CLASS_STRENGTH[i])
ABS = [0.0] * 169
_acc = 0
for _i in CLASS_ORDER:
    ABS[_i] = (_acc + CLASS_N[_i] / 2) / 1326.0
    _acc += CLASS_N[_i]

BINS = 20


def hand_class(c1, c2):
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 * 13 + r1
    if (c1 & 3) == (c2 & 3):
        return r1 * 13 + r2
    return r2 * 13 + r1


# Spots, their moves strongest first, and prior move frequencies.
SPOTS = {
    "pre_open": (("shove", "raise", "limp", "fold"), (0.1, 0.4, 0.3, 0.2)),
    "pre_vs_open": (("raise", "call", "fold"), (0.2, 0.45, 0.35)),
    "pre_vs_limp": (("raise", "check"), (0.35, 0.65)),
    "pre_vs_raise": (("raise", "call", "fold"), (0.15, 0.4, 0.45)),
    "pre_vs_shove": (("call", "fold"), (0.4, 0.6)),
    "nobet": (("bet", "check"), (0.4, 0.6)),
    "vsbet_S": (("raise", "call", "fold"), (0.12, 0.5, 0.38)),
    "vsbet_M": (("raise", "call", "fold"), (0.1, 0.45, 0.45)),
    "vsbet_L": (("raise", "call", "fold"), (0.08, 0.4, 0.52)),
    "vsraise": (("raise", "call", "fold"), (0.1, 0.45, 0.45)),
    "post_vs_shove": (("call", "fold"), (0.4, 0.6)),
}
# How the equilibrium (what top heads-up players study and solvers play)
# acts at 25bb, from a CFR+ solve of this event's preflop: blended into
# the priors so an unknown opponent starts out assumed to play well.
EXPERT_PRIORS = {
    "pre_open": (0.003, 0.213, 0.784, 0.0),
    "pre_vs_open": (0.233, 0.431, 0.336),
    "pre_vs_limp": (0.347, 0.653),
    "pre_vs_raise": (0.149, 0.454, 0.397),
    "pre_vs_shove": (0.22, 0.78),
}
# How professional players act postflop in heads-up pots: 24,000+ human
# decisions from the 10,000-hand Pluribus vs pros match (Brown & Sandholm
# 2019; PHH dataset), tools/pro_analysis.py. Order as in SPOTS.
PRO_PRIORS = {
    "nobet": (0.29, 0.71),
    "vsbet_S": (0.102, 0.456, 0.442),
    "vsbet_M": (0.052, 0.408, 0.539),
    "vsbet_L": (0.028, 0.285, 0.688),
    "vsraise": (0.056, 0.446, 0.498),
}
EPS_GRID = (0.02, 0.06, 0.12, 0.25, 0.45, 0.7)
EPS_LOGPRIOR = (-1.2, -0.3, 0.0, -0.8, -1.8, -3.0)
PRIOR_OBS = 3.0
EDGE = 0.025


def _grid(k):
    """Threshold vectors (descending, in bins) for k moves."""
    if k == 2:
        return [(t,) for t in range(BINS + 1)]
    if k == 3:
        return [(a, b) for a in range(BINS + 1) for b in range(a + 1)]
    step = 2
    pts = list(range(0, BINS + 1, step))
    return [(a, b, c) for a in pts for b in pts if b <= a
            for c in pts if c <= b]


GRIDS = {k: _grid(k) for k in (2, 3, 4)}
# Bluff band (in bins): the weakest hands that also take the most
# aggressive move, as polarised players do. Not fitted for 4-move spots.
BLUFF_GRID = (0, 2, 4, 6)
BLUFF_LOGPRIOR = (0.0, -0.7, -1.0, -1.3)


def _ramp(v):
    return 0.0 if v <= 0 else (1.0 if v >= 1 else v)


def move_prob(s, idx, th, eps, k):
    """Chance of move idx (0 = strongest) for strength s. th holds the
    k - 1 descending thresholds as fractions, then optionally the bluff
    threshold: hands below it take move 0 too."""
    tb = th[k - 1] if len(th) >= k else 0.0
    if idx == 0:
        inside = 1.0 if k == 1 else _ramp((s - th[0]) / EDGE + 0.5)
        if tb > 0:
            inside += _ramp((tb - s) / EDGE + 0.5)
    else:
        upper = th[idx - 1]
        lower = th[idx] if idx < k - 1 else tb
        inside = _ramp((upper - s) / EDGE + 0.5)
        if inside > 0 and (idx < k - 1 or tb > 0):
            inside *= _ramp((s - lower) / EDGE + 0.5)
    return (1.0 - eps) * inside + eps / k


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = random.Random(12345)
        # team -> spot -> {(cum tuple, move index): count}
        self.obs = {}
        # team -> spot -> (n_seen_at_fit, thresholds, eps)
        self.fits = {}
        # Tunables.
        self.r_ip = 1.0  # share of equity realised in position
        self.r_oop = 0.95  # out of position
        self.samples = 1500
        self.prior_obs = PRIOR_OBS
        self.limp_penalty = 0.05
        self.bet_sizes = (0.33, 0.66, 1.0)
        self.open_sizes = (2.0, 2.5, 3.0)
        self.debug = False
        self.bluff_model = True
        self.strength_mode = "equity"
        self.strength_runs = 30
        self.overbet_cap = True
        self.table_model = True
        self.expert_priors = True
        self.pro_priors = False
        self.table_eps_cut = 0.0
        self.real_slope = 0.4
        self.hail_mary = 0  # off
        self.table_prior = 2.0
        self.em_iters = 15
        self.tables = {}
        self.h = None
        self.my = []

    # -- engine hooks ----------------------------------------------------------

    def seed(self, hand, player_id):
        self.hand = hand
        self.player_id = player_id
        self.h = None
        self.my = []

    def run(self, actions, game_state):
        try:
            self._observe(actions, game_state)
            return self._decide(game_state)
        except Exception:
            if self.debug:
                raise
            return self._safe(game_state)

    def end_game(self, actions, game_state):
        try:
            self._observe(actions, game_state)
            self._learn_showdown()
        except Exception:
            if self.debug:
                raise

    def _safe(self, gs):
        me = self.player_id
        if gs.call_amount == 0:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    # -- the opponent model ------------------------------------------------------

    def _spot_obs(self, team, spot):
        return self.obs.setdefault(team, {}).setdefault(spot, {})

    def table(self, spot, team=None):
        """P(move | strength bin) for spot: BINS rows of k probabilities.

        Starts from the threshold model (strongest hands take the most
        aggressive move) and is refitted by EM to the observations, so
        hands that break the pattern (strong limps, slow-plays, bluffs)
        are learned once showdowns reveal them."""
        team = self.h["team"] if team is None else team
        th, eps = self.policy(spot, team)
        k = len(SPOTS[spot][0])
        if not self.table_model or (
                self.table_eps_cut and eps >= self.table_eps_cut):
            # Off, or the opponent looks random here: nothing to learn
            # beyond the threshold model, so do not fit noise.
            return [[move_prob((b + 0.5) / BINS, a, th, eps, k)
                     for a in range(k)] for b in range(BINS)]
        data = self.obs.get(team, {}).get(spot, {})
        n = sum(data.values())
        cache = self.tables.setdefault(team, {}).get(spot)
        if cache is not None and (cache[0] == n or (
                n - cache[0] < max(1, cache[0] // 8))):
            return cache[1]
        prior = [[move_prob((b + 0.5) / BINS, a, th, eps, k)
                  for a in range(k)] for b in range(BINS)]
        lam = self.table_prior
        tab = [row[:] for row in prior]
        items = []
        for (cum, a), cnt in data.items():
            masses = [cum[b + 1] - cum[b] for b in range(BINS)]
            nz = [(b, m) for b, m in enumerate(masses) if m > 0]
            if nz:
                items.append((nz, a, cnt))
        for _ in range(self.em_iters if items else 0):
            counts = [[lam * prior[b][a] for a in range(k)]
                      for b in range(BINS)]
            for nz, a, cnt in items:
                tot = 0.0
                for b, m in nz:
                    tot += m * tab[b][a]
                if tot <= 0:
                    continue
                f = cnt / tot
                for b, m in nz:
                    counts[b][a] += f * m * tab[b][a]
            for b in range(BINS):
                s = sum(counts[b])
                tab[b] = [c / s for c in counts[b]]
        self.tables[team][spot] = (n, tab)
        return tab

    def policy(self, spot, team=None):
        """(thresholds as fractions, eps) for spot, fitted lazily."""
        team = self.h["team"] if team is None else team
        data = self.obs.get(team, {}).get(spot, {})
        n = sum(data.values())
        fit = self.fits.setdefault(team, {}).get(spot)
        if fit is not None and (fit[0] == n or (
                n - fit[0] < max(1, fit[0] // 8))):
            return fit[1], fit[2]
        th, eps = self._fit(spot, data)
        self.fits[team][spot] = (n, th, eps)
        return th, eps

    def _fit(self, spot, data):
        moves, prior = SPOTS[spot]
        if self.expert_priors and spot in EXPERT_PRIORS:
            prior = tuple((a + b) / 2 for a, b in
                          zip(prior, EXPERT_PRIORS[spot]))
        if self.pro_priors and spot in PRO_PRIORS:
            prior = tuple((a + b) / 2 for a, b in
                          zip(prior, PRO_PRIORS[spot]))
        k = len(moves)
        uni = tuple(j / BINS for j in range(BINS + 1))
        items = list(data.items())
        items += [((uni, a), self.prior_obs * prior[a]) for a in range(k)]
        best = None
        best_ll = -1e18
        log = math.log
        bluffs = BLUFF_GRID if (k <= 3 and self.bluff_model) else (0,)
        for th in GRIDS[k]:
            for bi, tb in enumerate(bluffs):
                if tb > th[-1]:
                    break
                # Band masses per observation do not depend on eps.
                masses = []
                for (cum, a), cnt in items:
                    if a == 0:
                        m = 1.0 - cum[th[0]] + cum[tb]
                    else:
                        lo = th[a] if a < k - 1 else tb
                        m = cum[th[a - 1]] - cum[lo]
                    masses.append((m, cnt))
                for ei, eps in enumerate(EPS_GRID):
                    base = eps / k
                    keep = 1.0 - eps
                    ll = EPS_LOGPRIOR[ei] + BLUFF_LOGPRIOR[bi]
                    for m, cnt in masses:
                        ll += cnt * log(keep * m + base)
                    if ll > best_ll:
                        best_ll = ll
                        best = (th + (tb,), eps)
        th, eps = best
        return tuple(t / BINS for t in th), eps

    def _record(self, spot, move_idx, cum, strength_ref):
        h = self.h
        data = self._spot_obs(h["team"], spot)
        key = (cum, move_idx)
        data[key] = data.get(key, 0) + 1
        h["obs"].append((spot, key, move_idx, strength_ref))
        # Keep the data bounded: drop the rarest when it grows large.
        if len(data) > 400:
            for kk in sorted(data, key=data.get)[:100]:
                del data[kk]

    # -- following the hand -------------------------------------------------------

    def _start(self, gs):
        me = self.player_id
        opp = 1 - me if gs.num_players == 2 else (0 if me != 0 else 1)
        self.my = [fe.card_int(c) for c in self.hand]
        start = [gs.stacks[p] + gs.bets[p] for p in range(gs.num_players)]
        sb_seat = gs.button
        bb_seat = (gs.button + 1) % gs.num_players
        bets = [0] * gs.num_players
        bets[sb_seat] = min(gs.small_blind, start[sb_seat])
        bets[bb_seat] = min(gs.big_blind, start[bb_seat])
        self.h = {
            "me": me,
            "opp": opp,
            "team": gs.teams[opp],
            "sb": sb_seat,
            "bb_amt": gs.big_blind,
            "start": start,
            "bets": bets,
            "committed": list(bets),
            "cur": max(bets),
            "board": [],
            "street": 0,
            "pf_actions": 0,
            "my_last": None,  # my last move this street: (tag, size/pot)
            "cw": [1.0] * 169,  # preflop range, per class
            "w": None,  # postflop range, combo index -> weight
            "strength": None,  # combo index -> strength on this board
            "obs": [],  # this hand's observations, for showdowns
            "opp_cards": [],
        }

    def _observe(self, actions, gs):
        if self.h is None:
            self._start(gs)
        h = self.h
        for a in actions:
            kind = a.kind
            if kind == "reveal_card":
                if a.player_id < 0:
                    self._board_card(fe.card_int(a.card))
                elif a.player_id == h["opp"]:
                    h["opp_cards"].append(fe.card_int(a.card))
                continue
            if kind == "win":
                continue
            if a.player_id == h["opp"]:
                self._opp_action(a)
            self._apply(a)

    def _board_card(self, c):
        h = self.h
        h["board"].append(c)
        n = len(h["board"])
        if n in (3, 4, 5):
            h["street"] = n - 2
            h["bets"] = [0] * len(h["bets"])
            h["cur"] = 0
            h["my_last"] = None
            if h["w"] is None:
                self._range_to_combos()
        h["strength"] = None

    def _apply(self, a):
        h = self.h
        p = a.player_id
        if a.kind == "call":
            room = h["start"][p] - h["committed"][p]
            add = min(h["cur"] - h["bets"][p], room)
            h["bets"][p] += add
            h["committed"][p] += add
        elif a.kind == "raise":
            h["committed"][p] += a.amount - h["bets"][p]
            h["bets"][p] = a.amount
            h["cur"] = a.amount
        if h["street"] == 0:
            h["pf_actions"] += 1
        h["n_actions"] = h.get("n_actions", 0) + 1

    def _pot(self):
        return sum(self.h["committed"])

    def _opp_action(self, a):
        """Classify the opponent's move, record it, narrow their range."""
        h = self.h
        opp = h["opp"]
        owed = h["cur"] - h["bets"][opp]
        kind = a.kind
        mine = h["my_last"]
        if h["street"] == 0:
            if h["pf_actions"] == 0:
                eff = min(h["start"])
                spot = "pre_open"
                if kind == "raise":
                    label = "shove" if a.amount >= 0.85 * eff else "raise"
                else:
                    label = {"fold": "fold", "call": "limp"}[kind]
            elif mine is not None and mine[0] == "shove":
                spot, label = "pre_vs_shove", (
                    "call" if kind == "call" else "fold")
            elif mine is not None and mine[0] == "open":
                spot, label = "pre_vs_open", kind
            elif mine is not None and mine[0] == "limp":
                spot, label = "pre_vs_limp", (
                    "raise" if kind == "raise" else "check")
            else:
                spot, label = "pre_vs_raise", kind
        else:
            if owed <= 0:
                spot, label = "nobet", ("bet" if kind == "raise" else "check")
            elif mine is not None and mine[0] == "shove":
                spot, label = "post_vs_shove", (
                    "call" if kind == "call" else "fold")
            elif mine is not None and mine[0] == "bet":
                spot, label = self._bet_spot(mine[1]), kind
            else:
                spot, label = "vsraise", kind
        moves = SPOTS[spot][0]
        if label not in moves:
            return
        idx = moves.index(label)
        cum, ref = self._range_cum()
        self._record(spot, idx, cum, ref)
        self._narrow(spot, idx)

    @staticmethod
    def _bet_spot(ratio):
        return ("vsbet_S" if ratio < 0.55 else
                "vsbet_M" if ratio < 1.05 else "vsbet_L")

    def _range_items(self):
        """[(strength, weight)] for the opponent's current range."""
        h = self.h
        if h["street"] == 0:
            cw = h["cw"]
            return [(ABS[i], cw[i] * CLASS_N[i]) for i in range(169)
                    if cw[i] > 0]
        w = self._live_weights()
        st = self._strengths()
        return [(st[i], v) for i, v in w.items()]

    def _range_cum(self):
        """Cumulative strength histogram of the range (BINS + 1 values),
        and what is needed to place their real hand later."""
        h = self.h
        hist = [0.0] * BINS
        total = 0.0
        for s, v in self._range_items():
            b = int(s * BINS)
            if b >= BINS:
                b = BINS - 1
            hist[b] += v
            total += v
        cum = [0.0]
        acc = 0.0
        for x in hist:
            acc += x
            cum.append(round(acc / total, 3) if total > 0 else 0.0)
        cum[-1] = 1.0
        ref = ("pre", None) if h["street"] == 0 else ("post", h["strength"])
        return tuple(cum), ref

    def _narrow(self, spot, idx):
        h = self.h
        col = [row[idx] for row in self.table(spot)]
        top = BINS - 1
        if h["street"] == 0:
            cw = h["cw"]
            for i in range(169):
                if cw[i] > 0:
                    b = int(ABS[i] * BINS)
                    cw[i] *= col[b if b < top else top]
        else:
            w = self._live_weights()
            st = self._strengths()
            for i in w:
                b = int(st[i] * BINS)
                w[i] *= col[b if b < top else top]

    def _range_to_combos(self):
        h = self.h
        dead = set(self.my) | set(h["board"])
        cw = h["cw"]
        w = {}
        for i, (c1, c2) in enumerate(ALL_COMBOS):
            if c1 in dead or c2 in dead:
                continue
            v = cw[COMBO_CLASS[i]]
            if v > 0:
                w[i] = v
        h["w"] = w

    def _live_weights(self):
        h = self.h
        if h["w"] is None:
            self._range_to_combos()
        w = h["w"]
        board = h["board"]
        if board:
            last = board[-1]
            for i in [i for i in w if last in ALL_COMBOS[i]]:
                del w[i]
        return w

    def _strengths(self):
        """Absolute strength of every live combo on this board (0..1): its
        equity against a random hand, the scale most bots decide by.
        Estimated by ranking every combo on the same sampled runouts."""
        h = self.h
        if h["strength"] is not None:
            return h["strength"]
        if self.strength_mode != "equity":
            return self._strengths_heur()
        board = h["board"]
        dead = set(self.my) | set(board)
        live = [i for i, (c1, c2) in enumerate(ALL_COMBOS)
                if c1 not in dead and c2 not in dead]
        need = 5 - len(board)
        runs = 1 if need == 0 else self.strength_runs
        deck = [c for c in range(52) if c not in set(board)]
        rng = self.rng
        score = fe.score
        tot = dict.fromkeys(live, 0.0)
        cnt = dict.fromkeys(live, 0)
        for _ in range(runs):
            extra = rng.sample(deck, need) if need else []
            full = board + extra
            ex = set(extra)
            sc = []
            for i in live:
                c1, c2 = ALL_COMBOS[i]
                if c1 in ex or c2 in ex:
                    continue
                sc.append((score([c1, c2] + full), i))
            sc.sort()
            n = len(sc)
            j = 0
            while j < n:
                k = j
                v0 = sc[j][0]
                while k + 1 < n and sc[k + 1][0] == v0:
                    k += 1
                pct = ((j + k) / 2 + 0.5) / n
                for t in range(j, k + 1):
                    i = sc[t][1]
                    tot[i] += pct
                    cnt[i] += 1
                j = k + 1
        st = {i: (tot[i] / cnt[i] if cnt[i] else 0.5) for i in live}
        h["strength"] = st
        return st

    def _strengths_heur(self):
        """Absolute strength of every live combo on this board (0..1):
        made-hand quantile among all live combos, lifted for draws."""
        h = self.h
        if h["strength"] is not None:
            return h["strength"]
        board = h["board"]
        dead = set(self.my) | set(board)
        score = fe.score
        raw = {}
        for i, (c1, c2) in enumerate(ALL_COMBOS):
            if c1 in dead or c2 in dead:
                continue
            raw[i] = score([c1, c2] + board)
        ordered = sorted(raw, key=raw.get)
        n = len(ordered)
        hs = {}
        j = 0
        while j < n:
            k = j
            v0 = raw[ordered[j]]
            while k + 1 < n and raw[ordered[k + 1]] == v0:
                k += 1
            pct = ((j + k) / 2 + 0.5) / n
            for t in range(j, k + 1):
                hs[ordered[t]] = pct
            j = k + 1
        if len(board) < 5:
            fd_b, oe_b, gs_b = ((0.30, 0.22, 0.10) if len(board) == 3
                                else (0.18, 0.13, 0.06))
            bsuits = [0, 0, 0, 0]
            bmask = 0
            for c in board:
                bsuits[c & 3] += 1
                bmask |= 1 << (c >> 2)
            straight = fe.STRAIGHT
            for i, v in hs.items():
                c1, c2 = ALL_COMBOS[i]
                cat = raw[i] >> 20
                bonus = 0.0
                if cat < 5:
                    s1, s2 = c1 & 3, c2 & 3
                    if bsuits[s1] + 1 + (s2 == s1) == 4 or (
                            s2 != s1 and bsuits[s2] + 1 == 4):
                        bonus = fd_b
                if cat < 4:
                    m = bmask | (1 << (c1 >> 2)) | (1 << (c2 >> 2))
                    if m != bmask:
                        outs = 0
                        for r in range(13):
                            if not m >> r & 1 and straight[m | (1 << r)] >= 0:
                                outs += 1
                        if outs >= 2:
                            bonus = max(bonus, oe_b) + (0.05 if bonus else 0)
                        elif outs == 1:
                            bonus = max(bonus, gs_b) + (0.03 if bonus else 0)
                if bonus:
                    hs[i] = v + bonus * (1.0 - v)
        h["strength"] = hs
        return hs

    # -- learning from showdowns -----------------------------------------------------

    def _learn_showdown(self):
        """Replace this hand's observations by exact ones once the
        opponent's cards are known."""
        h = self.h
        if h is None or len(h["opp_cards"]) != 2 or not h["obs"]:
            return
        c1, c2 = sorted(h["opp_cards"], reverse=True)
        cls = hand_class(c1, c2)
        idx = COMBO_INDEX[(c1, c2)]
        for spot, key, move_idx, ref in h["obs"]:
            if ref[0] == "pre":
                s = ABS[cls]
            else:
                if ref[1] is None or idx not in ref[1]:
                    continue
                s = ref[1][idx]
            b = min(BINS - 1, int(s * BINS))
            exact = tuple(0.0 if j <= b else 1.0 for j in range(BINS + 1))
            data = self._spot_obs(h["team"], spot)
            if data.get(key, 0) > 0:
                data[key] -= 1
                if data[key] <= 0:
                    del data[key]
            nk = (exact, move_idx)
            data[nk] = data.get(nk, 0) + 1
        h["obs"] = []

    # -- deciding ----------------------------------------------------------------------

    def _decide(self, gs):
        h = self.h
        me = self.player_id
        if gs.street == "preflop":
            act, total = self._decide_preflop(gs)
        else:
            act, total = self._decide_postflop(gs)
        if (self.hail_mary and h.get("n_actions", 0) >= self.hail_mary
                and gs.can_raise):
            # Optional: all-in once the hand has gone on this long.
            act, total = "raise", gs.max_raise_to
        pot_before = gs.pot
        if act == "raise" and gs.can_raise:
            lo, hi = gs.raise_range
            total = int(min(hi, max(lo, total)))
            if total >= 0.85 * hi:
                total = hi
            if total >= hi:
                tag = "shove"
            elif gs.street == "preflop":
                tag = "open" if h["pf_actions"] == 0 else "raise"
            elif gs.current_bet == 0:
                tag = "bet"
            else:
                tag = "raise"
            ratio = (total - gs.bets[me]) / max(1, pot_before)
            h["my_last"] = (tag, ratio)
            return PlayerAction.raise_to(me, total)
        if act == "fold" and gs.call_amount > 0:
            return PlayerAction.fold(me)
        if gs.street == "preflop" and h["pf_actions"] == 0:
            h["my_last"] = ("limp", 0)
        else:
            h["my_last"] = ("call", 0)
        return PlayerAction.call(me)

    @staticmethod
    def _shove_ok_pre(gs, S):
        """All-in preflop only when stacks are short (always, at this
        event's 25bb) or the betting is already big."""
        return S <= 40 * gs.big_blind or gs.current_bet >= 0.2 * S

    @staticmethod
    def _shove_ok_post(A, P, bb):
        """An all-in bet of A into P: fine when short (30bb or less) or no
        bigger than three pots."""
        return A <= 30 * bb or A <= 3 * P

    @staticmethod
    def _pick(cands):
        best = None
        best_ev = -1e18
        for name, (ev, total) in cands.items():
            if ev > best_ev + 1e-9:
                best_ev = ev
                best = (name, total)
        name, total = best
        if name == "fold":
            return "fold", 0
        if name in ("call", "check", "limp"):
            return "call", 0
        return "raise", total

    def _responses(self, items, spot):
        """For the range items [(strength, weight, outcome)], each move's
        (probability, equity against the hands that take it)."""
        tab = self.table(spot)
        moves = SPOTS[spot][0]
        k = len(moves)
        out = {}
        # Bucket the items by strength bin once.
        bw = [0.0] * BINS
        bo = [0.0] * BINS
        top = BINS - 1
        for s, wt, o in items:
            b = int(s * BINS)
            if b > top:
                b = top
            bw[b] += wt
            bo[b] += wt * o
        tot = sum(bw)
        for idx, name in enumerate(moves):
            m = 0.0
            e = 0.0
            for b in range(BINS):
                if bw[b]:
                    p = tab[b][idx]
                    m += p * bw[b]
                    e += p * bo[b]
            out[name] = (m / tot if tot > 0 else 1.0 / k,
                         e / m if m > 0 else 0.5)
        return out

    @staticmethod
    def _eq(items):
        m = 0.0
        e = 0.0
        for s, wt, o in items:
            m += wt
            e += wt * o
        return e / m if m > 0 else 0.5

    # Preflop: equity from the table against the class weights.

    def _decide_preflop(self, gs):
        h = self.h
        me = self.player_id
        row = EQ[hand_class(self.my[0], self.my[1])]
        cw = h["cw"]
        items = [(ABS[j], cw[j] * CLASS_N[j], row[j]) for j in range(169)
                 if cw[j] > 0]
        S = min(h["start"])  # everyone's all-in total
        my_bet = gs.bets[me]
        P = gs.pot
        owed = gs.call_amount
        bb = gs.big_blind
        ip = me == h["sb"]
        R = self.r_ip if ip else self.r_oop
        allin_put = S - my_bet  # chips I add to be all-in (effective)
        eq_all = self._eq(items)
        cands = {}
        if owed > 0:
            cands["fold"] = (0.0, 0)

        def shove_ev():
            r = self._responses(items, "pre_vs_shove")
            pf = r["fold"][0]
            pc, eqc = r["call"]
            return pf * P + pc * (eqc * 2 * S - allin_put)

        def raise_ev(T, spot):
            x = T - my_bet
            r = self._responses(items, spot)
            pf = r["fold"][0]
            pc, eqc = r["call"]
            pr, eqr = r["raise"]
            return (pf * P + pc * (eqc * R * 2 * T - x)
                    + pr * max(-x, eqr * 2 * S - allin_put))

        can_raise = gs.can_raise and gs.stacks[h["opp"]] > 0
        maxr = gs.max_raise_to if gs.can_raise else 0
        if h["pf_actions"] == 0:
            # Small blind, first in.
            if can_raise:
                if self._shove_ok_pre(gs, S):
                    cands["shove"] = (shove_ev(), maxr)
                for mult in self.open_sizes:
                    T = int(mult * bb)
                    if T < 0.5 * maxr:
                        cands["open%.1f" % mult] = (
                            raise_ev(T, "pre_vs_open"), T)
            r = self._responses(items, "pre_vs_limp")
            pr, eqr = r["raise"]
            pk, eqk = r["check"]
            ev_limp = (pk * (eqk * R * 2 * bb - owed)
                       + pr * max(-owed, eqr * 2 * S - allin_put))
            cands["limp"] = (ev_limp - self.limp_penalty * bb, 0)
        elif owed == 0:
            # Big blind after a limp.
            cands["check"] = (eq_all * R * P, 0)
            if can_raise:
                if self._shove_ok_pre(gs, S):
                    cands["shove"] = (shove_ev(), maxr)
                for mult in (3.0, 4.0):
                    T = int(mult * bb)
                    if T < 0.5 * maxr:
                        cands["iso%.1f" % mult] = (
                            raise_ev(T, "pre_vs_raise"), T)
        else:
            # Facing a raise or an all-in.
            T_opp = gs.current_bet
            my_room = my_bet + gs.stacks[me]
            allin = (gs.stacks[h["opp"]] == 0 or owed >= gs.stacks[me]
                     or T_opp >= 0.5 * S)
            Rc = 1.0 if allin else R
            final = 2 * min(T_opp, my_room)
            cands["call"] = (eq_all * Rc * final - min(owed, gs.stacks[me]),
                             0)
            if can_raise:
                if self._shove_ok_pre(gs, S):
                    cands["shove"] = (shove_ev(), maxr)
                # Deep stacks only: a normal re-raise instead of all-in.
                if S > 60 * bb:
                    T = int((3.0 if ip else 3.5) * T_opp)
                    if gs.min_raise_to <= T < 0.4 * maxr:
                        cands["reraise"] = (
                            raise_ev(T, "pre_vs_raise"), T)
        return self._pick(cands)

    # Postflop: Monte Carlo equity against weighted combo samples.

    def _mc(self, n):
        """[(strength, weight, outcome)] samples of the current range."""
        h = self.h
        w = self._live_weights()
        st = self._strengths()
        keys = [i for i in w if w[i] > 1e-12]
        if not keys:
            return []
        board = h["board"]
        need = 5 - len(board)
        dead = set(self.my) | set(board)
        deck = [c for c in range(52) if c not in dead]
        score = fe.score
        mine_base = self.my + board
        out = []
        if need == 0:
            ms = score(mine_base)
            for i in keys:
                c1, c2 = ALL_COMBOS[i]
                os_ = score([c1, c2] + board)
                o = 1.0 if ms > os_ else 0.5 if ms == os_ else 0.0
                out.append((st[i], w[i], o))
            return out
        rng = self.rng
        # Group samples per combo: weight 1 each.
        picks = rng.choices(keys, weights=[w[i] for i in keys], k=n)
        for i in picks:
            c1, c2 = ALL_COMBOS[i]
            extra = []
            for c in rng.sample(deck, need + 2):
                if c != c1 and c != c2:
                    extra.append(c)
                    if len(extra) == need:
                        break
            ms = score(mine_base + extra)
            os_ = score([c1, c2] + board + extra)
            o = 1.0 if ms > os_ else 0.5 if ms == os_ else 0.0
            out.append((st[i], 1.0, o))
        return out

    def _decide_postflop(self, gs):
        h = self.h
        me = self.player_id
        opp = h["opp"]
        items = self._mc(self.samples)
        P = gs.pot
        owed = gs.call_amount
        my_stack = gs.stacks[me]
        opp_stack = gs.stacks[opp]
        my_bet = gs.bets[me]
        opp_bet = gs.bets[opp]
        river = gs.street == "river"
        ip = me == h["sb"]
        R = 1.0 if river else (self.r_ip if ip else self.r_oop)
        eq_all = self._eq(items)
        if not river and self.real_slope:
            # Strong hands realise more of their equity, weak ones less.
            adj = 1.0 + self.real_slope * (eq_all - 0.5)
            R = min(R * adj, 1.0 / max(eq_all, 1e-6))
        # Chips I can still add this hand, capped by the opponent.
        A = min(my_stack + my_bet, opp_stack + opp_bet) - my_bet
        cands = {}
        if owed == 0:
            Rchk = 1.0 if (river and ip) else R
            cands["check"] = (eq_all * Rchk * P, 0)
            if gs.can_raise and A > 0 and opp_stack > 0:
                for frac in self.bet_sizes:
                    x = max(int(frac * P), gs.min_raise_to - my_bet)
                    if x >= 0.6 * A:
                        continue
                    cands["bet%.2f" % frac] = (
                        self._bet_ev(items, x, P, A, R), my_bet + x)
                if self._shove_ok_post(A, P, gs.big_blind):
                    cands["shove"] = (self._bet_ev(items, A, P, A, R),
                                      gs.max_raise_to)
        else:
            cands["fold"] = (0.0, 0)
            call_amt = min(owed, my_stack)
            allin = opp_stack == 0 or owed >= my_stack
            Rc = 1.0 if (allin or river) else R * 0.97
            cands["call"] = (eq_all * Rc * (P + call_amt) - call_amt, 0)
            if gs.can_raise and opp_stack > 0 and A > owed:
                r = self._responses(items, "post_vs_shove")
                x = A
                y = A - owed  # what they add to call
                pf = r["fold"][0]
                pc, eqc = r["call"]
                if self._shove_ok_post(A, P, gs.big_blind):
                    cands["shove"] = (
                        pf * P + pc * (eqc * (P + x + y) - x),
                        gs.max_raise_to)
                T = 3 * gs.current_bet
                if T - my_bet < 0.6 * A and T >= gs.min_raise_to:
                    x = T - my_bet
                    y = T - opp_bet
                    r = self._responses(items, "vsraise")
                    pf = r["fold"][0]
                    pc, eqc = r["call"]
                    pr, eqr = r["raise"]
                    ev = (pf * P + pc * (eqc * R * (P + x + y) - x)
                          + pr * max(-x, eqr * (P + x + y + 2 * (A - x))
                                     - A))
                    cands["raise3x"] = (ev, T)
        return self._pick(cands)

    def _bet_ev(self, items, x, P, A, R):
        if x >= A:
            r = self._responses(items, "post_vs_shove")
            pf = r["fold"][0]
            pc, eqc = r["call"]
            ev = pf * P + pc * (eqc * (P + 2 * x) - x)
            if self.overbet_cap and x > 2.5 * P:
                # Never count on an overbet shove folding more than a
                # big bet does: most bots call by equity or pot odds.
                rl = self._responses(items, "vsbet_L")
                pfl = rl["fold"][0]
                if pfl < pf:
                    (pcl, eql), (prl, eqr) = rl["call"], rl["raise"]
                    cont = pcl + prl
                    eqk = (pcl * eql + prl * eqr) / cont if cont > 0 else eqc
                    ev = min(ev, pfl * P + (1 - pfl) * (eqk * (P + 2 * x) - x))
            return ev
        r = self._responses(items, self._bet_spot(x / max(1, P)))
        pf = r["fold"][0]
        pc, eqc = r["call"]
        pr, eqr = r["raise"]
        ev_r = max(-x, eqr * (P + 2 * A) - A)
        return pf * P + pc * (eqc * R * (P + 2 * x) - x) + pr * ev_r
