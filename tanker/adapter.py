"""Tanker Run solver.

1. Clarke-Wright savings gives a good valid plan fast (submitted at once, so the
   5% checkpoint already has a strong cost).
2. SISR (string-removal ruin & greedy-blink recreate) with simulated annealing
   moves villages *between* routes, which is where the gains are.
3. Fleet limit is handled lexicographically: (routes over fleet, cost). Only
   plans with no excess routes are ever submitted/returned.
"""
import math
import random
import time
from math import isqrt

from adapter import Solver

try:
    from data import distance_matrix
except Exception:  # pragma: no cover
    distance_matrix = None

TIME_LIMIT = 5.55   # seconds of the 6 s budget we actually use
SUBMIT_GAP = 0.02   # throttle between improving submissions
SAFETY = 0.25       # seconds kept in hand before the evaluator's deadline


def _savings(n, D, dem, cap):
    routes = {i: [i] for i in range(1, n + 1)}
    rid = list(range(n + 1))
    load = {i: dem[i] for i in routes}
    sv = sorted(((D[0][i] + D[0][j] - D[i][j], i, j)
                 for i in range(1, n + 1) for j in range(i + 1, n + 1)),
                reverse=True)
    for _, i, j in sv:
        a, b = rid[i], rid[j]
        if a == b or load[a] + load[b] > cap:
            continue
        A, B = routes[a], routes[b]
        if A[-1] != i:
            if A[0] == i:
                A.reverse()
            else:
                continue
        if B[0] != j:
            if B[-1] == j:
                B.reverse()
            else:
                continue
        A.extend(B)
        load[a] += load[b]
        for v in B:
            rid[v] = a
        del routes[b]
        del load[b]
    return list(routes.values())


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        t0 = time.perf_counter()
        n = instance.size
        cap = instance.capacity
        fleet = instance.fleet
        pts = instance.coords
        dem = [0] + [instance.demand[v] for v in range(1, n + 1)]
        D = None
        if distance_matrix is not None:
            try:
                D = [list(row) for row in distance_matrix(instance)]
            except Exception:
                D = None
        if D is None:
            D = [[isqrt((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2)
                  for b in range(n + 1)] for a in range(n + 1)]
        if n == 0:
            return {"routes": []}

        def cost_of(routes):
            t = 0
            for r in routes:
                p = 0
                for v in r:
                    t += D[p][v]
                    p = v
                t += D[p][0]
            return t

        rnd = random.Random(12345)
        state = {"best": None, "bc": float("inf"), "last": 0.0,
                 "dl": t0 + TIME_LIMIT}

        def offer(routes, c, force=False):
            if c < state["bc"]:
                state["bc"] = c
                state["best"] = [r[:] for r in routes if r]
                now = time.perf_counter()
                if force or now - state["last"] >= SUBMIT_GAP:
                    state["last"] = now
                    rec = self._push(submit_candidate, state["best"])
                    rem = self._remaining(rec)
                    if rem is not None:   # trust the evaluator's own clock
                        state["dl"] = min(state["dl"],
                                          time.perf_counter() + rem - SAFETY)
            return

        # ---- initial solution
        cur = [r for r in _savings(n, D, dem, cap) if r]
        cur_c = cost_of(cur)
        cur_ex = max(0, len(cur) - fleet)
        if cur_ex == 0:
            offer(cur, cur_c, force=True)

        adj = [None] + [sorted(range(1, n + 1), key=lambda w, v=v: D[v][w])
                        for v in range(1, n + 1)]
        T0, Tf = 25.0, 0.4
        total = TIME_LIMIT
        cbar, Lmax, blink = 10.0, 10.0, 0.01
        far_key = lambda c: -D[0][c]

        it = 0
        while True:
            now = time.perf_counter()
            if now >= state["dl"]:
                break
            it += 1
            T = T0 * (Tf / T0) ** ((now - t0) / total)

            # ---- ruin
            new = [r[:] for r in cur]
            rof = {}
            for k, r in enumerate(new):
                for v in r:
                    rof[v] = k
            avg = n / max(1, len(new))
            Lsmax = min(Lmax, avg)
            ksmax = 4.0 * cbar / (1.0 + Lsmax) - 1.0
            ks = int(rnd.uniform(1.0, ksmax + 1.0))
            seed = rnd.randint(1, n)
            removed = []
            rem_set = set()
            ruined = set()
            for c in adj[seed]:
                if len(ruined) >= ks:
                    break
                if c in rem_set:
                    continue
                k = rof[c]
                if k in ruined:
                    continue
                r = new[k]
                lt = min(len(r), Lsmax)
                l = int(rnd.uniform(1.0, lt + 1.0))
                l = max(1, min(l, len(r)))
                pos = r.index(c)
                lo = max(0, pos - l + 1)
                hi = min(pos, len(r) - l)
                st = rnd.randint(lo, hi)
                seg = r[st:st + l]
                del r[st:st + l]
                removed.extend(seg)
                rem_set.update(seg)
                ruined.add(k)

            # ---- recreate
            u = rnd.random() * 8
            if u < 4:
                rnd.shuffle(removed)
            elif u < 8 and u >= 4 and rnd.random() < 0.5:
                removed.sort(key=lambda c: -dem[c])
            elif rnd.random() < 0.5:
                removed.sort(key=far_key)
            else:
                removed.sort(key=lambda c: D[0][c])
            loads = [sum(dem[v] for v in r) for r in new]
            for c in removed:
                dc = dem[c]
                bd, bk, bi = None, -1, 0
                Dc = D[c]
                for k, r in enumerate(new):
                    if loads[k] + dc > cap or not r:
                        continue
                    prev = 0
                    L = len(r)
                    for idx in range(L + 1):
                        nxt = r[idx] if idx < L else 0
                        if rnd.random() >= blink:
                            d = D[prev][c] + Dc[nxt] - D[prev][nxt]
                            if bd is None or d < bd:
                                bd, bk, bi = d, k, idx
                        prev = nxt
                if bd is None:
                    new.append([c])
                    loads.append(dc)
                else:
                    new[bk].insert(bi, c)
                    loads[bk] += dc
            new = [r for r in new if r]
            new_c = cost_of(new)
            new_ex = max(0, len(new) - fleet)

            # ---- accept (lexicographic: fleet excess first, then annealed cost)
            if new_ex < cur_ex or (
                    new_ex == cur_ex and
                    new_c < cur_c - T * math.log(1.0 - rnd.random())):
                cur, cur_c, cur_ex = new, new_c, new_ex
                if cur_ex == 0:
                    offer(cur, cur_c)

        best = state["best"]
        if best is None:
            best = cur
        else:
            self._push(submit_candidate, best)   # make sure the final best is in
        return {"routes": best}

    @staticmethod
    def _push(submit_candidate, routes):
        try:
            return submit_candidate({"routes": [r[:] for r in routes]})
        except Exception:
            return None

    @staticmethod
    def _remaining(rec):
        if rec is None:
            return None
        try:
            v = rec["remaining_s"] if isinstance(rec, dict) else rec.remaining_s
            return float(v)
        except Exception:
            return None
