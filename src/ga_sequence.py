# ga_sequence.py
# ============================================================
# Genetic Algorithm (GA) over sequences of local operations (flip/transpose)
# Initial population is loaded from sa_generation.py JSONL dataset.
#
# This version:
#   - NO MUTATION (mutation disabled)
#   - Uses crossover only
#   - Keeps MORE from the first population (strong elitism + parent carryover)
#   - Logs ONLY TRUE/ACCEPTED crossover-children into Dataset2/Children.jsonl
#   - Writes ALL best-tie individuals (min crossings) into Dataset2/Dataset.jsonl
#     while avoiding duplicates by exact sequence signature
#
# Supports zone_mode:
#   "left_right" | "islands" | "stripes" | "voronoi"
#
# Requires:
#   - operations.py  (HamiltonianSTL with get_subgrid, transpose_subgrid, flip_subgrid)
# If using stripes/voronoi:
#   - Zones.py with: zones_stripes, zones_voronoi
# ============================================================

import os
import json
import time
import copy
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt

from operations import HamiltonianSTL
from zones import zones_stripes, zones_voronoi

Point = Tuple[int, int]

# -----------------------
# GA Hyperparameters (defaults)
# -----------------------
GENOME_LEN = 100

DEBUG_SUMMARY_EVERY = 10

# Acceptance gate (child must not be much worse than best parent)
EPS_CROSSINGS = 2

# How many to keep from previous gen (strongly keep initial population quality)
KEEP_RATE = 0.60  # keep 60% of pop unchanged into next generation

# Strong elitism (subset of best parents always kept)
ELITE_K = 6

# Crossover
CX_RATE = 0.90
CX_RATIO = 0.60  # cut point = ratio*L (prefix from parent A)

# Minimum number of successfully-applied ops in evaluation to consider the individual non-degenerate
MIN_APPLIED_VALID = 1

# For filling population, limit number of crossover attempts per slot
MAX_TRIES_PER_SLOT = 80


# ============================================================
# Zones
# ============================================================
def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}


def zones_islands(
    W: int,
    H: int,
    *,
    num_islands: int,
    island_size: int,
    seed: int,
    allow_touch: bool,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    """
    Background zone=1, islands zone=2 (square islands). Deterministic by seed.
    """
    rng = random.Random(seed)
    zones = {(x, y): 1 for y in range(H) for x in range(W)}

    S = int(island_size)
    if S <= 0 or S > W or S > H:
        raise ValueError("island_size must be in [1, min(W,H)].")

    anchors: List[Tuple[int, int]] = []
    tries, max_tries = 0, 50_000

    def overlaps(ax, ay, bx, by) -> bool:
        pad = 0 if allow_touch else 1
        aL, aR = ax - pad, ax + S - 1 + pad
        aT, aB = ay - pad, ay + S - 1 + pad

        bL, bR = bx, bx + S - 1
        bT, bB = by, by + S - 1
        return not (aR < bL or bR < aL or aB < bT or bB < aT)

    max_x = W - S
    max_y = H - S

    while len(anchors) < num_islands and tries < max_tries:
        tries += 1
        ax = rng.randint(0, max_x)
        ay = rng.randint(0, max_y)
        if all(not overlaps(ax, ay, bx, by) for (bx, by) in anchors):
            anchors.append((ax, ay))

    if len(anchors) < num_islands:
        raise RuntimeError(
            f"Could not place {num_islands} non-overlapping {S}x{S} islands. "
            f"Try allow_touch=True or reduce num_islands/island_size."
        )

    for (ax, ay) in anchors:
        for yy in range(ay, ay + S):
            for xx in range(ax, ax + S):
                zones[(xx, yy)] = 2

    meta = {"mode": "islands", "anchors": anchors, "num_islands": num_islands, "island_size": S}
    return zones, meta


def build_zones(
    W: int,
    H: int,
    *,
    zone_mode: str,
    seed: int,
    # islands
    num_islands: int,
    island_size: int,
    allow_touch: bool,
    # stripes
    stripe_direction: str,
    stripe_k: int,
    # voronoi
    voronoi_k: int,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    m = str(zone_mode).lower()

    if m in ("left_right", "leftright", "lr"):
        return zones_left_right(W, H), {"mode": "left_right"}

    if m == "islands":
        z, meta = zones_islands(
            W,
            H,
            num_islands=num_islands,
            island_size=island_size,
            seed=seed,
            allow_touch=allow_touch,
        )
        return z, meta

    if m == "stripes":
        return zones_stripes(W, H, direction=stripe_direction, k=stripe_k), {
            "mode": "stripes",
            "stripe_direction": stripe_direction,
            "stripe_k": stripe_k,
        }

    if m == "voronoi":
        z, meta = zones_voronoi(W, H, k=voronoi_k)
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["mode"] = "voronoi"
        meta["voronoi_k"] = voronoi_k
        return z, meta

    raise ValueError(f"Unknown zone_mode='{zone_mode}'")


# ============================================================
# Crossings
# ============================================================
def compute_crossings(h: HamiltonianSTL, zones: Dict[Point, int]) -> int:
    W, H = h.width, h.height
    c = 0

    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x] and zones[(x, y)] != zones[(x + 1, y)]:
                c += 1

    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x] and zones[(x, y)] != zones[(x, y + 1)]:
                c += 1

    return c


# ============================================================
# Snapshot / restore + validity
# ============================================================
def snapshot_edges(h: HamiltonianSTL):
    return copy.deepcopy(h.H), copy.deepcopy(h.V)


def restore_edges(h: HamiltonianSTL, edges):
    h.H, h.V = copy.deepcopy(edges[0]), copy.deepcopy(edges[1])


def is_valid_cycle(h: HamiltonianSTL) -> bool:
    if hasattr(h, "validate_full_path_cycle"):
        return bool(h.validate_full_path_cycle())
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    return True


# ============================================================
# GA genome
# ============================================================
@dataclass
class Op:
    kind: str  # "T" transpose, "F" flip, "N" no-op
    x: int
    y: int
    variant: str  # transpose variant OR flip dir n/s/e/w OR "noop"


@dataclass
class Individual:
    ops: List[Op]
    # metrics
    fitness: Optional[float] = None  # fitness = -best_seen_crossings
    applied: int = 0
    best_seen: int = 10**9
    uid: str = field(default_factory=lambda: f"ind_{int(time.time()*1e6)}_{random.randint(0,999999)}")
    # bookkeeping
    is_crossover: bool = False


# ============================================================
# Operation application (preserve Hamiltonian)
# ============================================================
def _infer_success(ret, changed: bool) -> bool:
    """
    Robustly infer whether an op succeeded.
    """
    if ret is None:
        return True

    if isinstance(ret, bool):
        return ret

    if isinstance(ret, dict) and "ok" in ret:
        return bool(ret["ok"])

    if isinstance(ret, tuple) and len(ret) > 0:
        a = ret[0]
        if isinstance(a, bool):
            return bool(a)
        msg = " ".join(str(x) for x in ret if x is not None).lower()
        if any(k in msg for k in ("transposed", "flipped", "success", "succeeded", "ok", "done")):
            return True
        if any(k in msg for k in ("fail", "failed", "invalid", "error", "mismatch", "not flippable")):
            return False
        return changed

    if isinstance(ret, str):
        msg = ret.lower()
        if any(k in msg for k in ("transposed", "flipped", "success", "succeeded", "ok", "done")):
            return True
        if any(k in msg for k in ("fail", "failed", "invalid", "error", "mismatch", "not flippable")):
            return False
        return changed

    return changed


def apply_op(h: HamiltonianSTL, op: Op) -> bool:
    # IMPORTANT: NOP should NOT count as applied, so return False
    if op.kind == "N":
        return False

    H_before, V_before = snapshot_edges(h)

    try:
        if op.kind == "T":
            # transpose is 3x3
            if op.x < 0 or op.y < 0 or op.x + 2 >= h.width or op.y + 2 >= h.height:
                restore_edges(h, (H_before, V_before))
                return False
            sub = h.get_subgrid((op.x, op.y), (op.x + 2, op.y + 2))
            ret = h.transpose_subgrid(sub, op.variant)

        elif op.kind == "F":
            # match SA flip mapping:
            # w/e => 3x2 , n/s => 2x3
            if op.variant in ("w", "e"):
                w, hh = 3, 2
            elif op.variant in ("n", "s"):
                w, hh = 2, 3
            else:
                restore_edges(h, (H_before, V_before))
                return False

            if op.x < 0 or op.y < 0 or op.x + (w - 1) >= h.width or op.y + (hh - 1) >= h.height:
                restore_edges(h, (H_before, V_before))
                return False

            sub = h.get_subgrid((op.x, op.y), (op.x + w - 1, op.y + hh - 1))
            ret = h.flip_subgrid(sub, op.variant)

        else:
            restore_edges(h, (H_before, V_before))
            return False

        changed = (h.H != H_before) or (h.V != V_before)
        ok = _infer_success(ret, changed)

        if (not changed) or (not ok) or (not is_valid_cycle(h)):
            restore_edges(h, (H_before, V_before))
            return False

        return True

    except Exception:
        restore_edges(h, (H_before, V_before))
        return False


# ============================================================
# Variants + padding
# ============================================================
def _transpose_variants(W: int, H: int) -> List[str]:
    tmp = HamiltonianSTL(W, H)
    tp = getattr(tmp, "transpose_patterns", [])
    if hasattr(tp, "keys"):
        return list(tp.keys())
    try:
        return list(tp)
    except Exception:
        return ["a", "b", "c", "d", "e", "f", "g", "h"]


def force_length_noop_pad(ops: List[Op], L: int) -> List[Op]:
    ops2 = ops[:L]
    while len(ops2) < L:
        ops2.append(Op("N", -1, -1, "noop"))
    return ops2


# ============================================================
# Dataset (SA JSONL) -> population
# ============================================================
def load_dataset_jsonl(dataset_jsonl: str) -> List[Dict[str, Any]]:
    p = Path(dataset_jsonl)
    if not p.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl}")

    recs: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln}: {e}")
    return recs


def record_matches(rec: Dict[str, Any], W: int, H: int, zone_pattern: str) -> bool:
    try:
        return (
            int(rec.get("grid_W")) == int(W)
            and int(rec.get("grid_H")) == int(H)
            and str(rec.get("zone_pattern")).lower() == str(zone_pattern).lower()
        )
    except Exception:
        return False


def record_to_individual(rec: Dict[str, Any], genome_len: int) -> Individual:
    seq = rec.get("sequence_ops", [])
    ops: List[Op] = []
    for item in seq:
        kind = str(item.get("kind", "N"))
        x = int(item.get("x", -1))
        y = int(item.get("y", -1))
        variant = str(item.get("variant", "noop"))

        if kind not in ("T", "F"):
            kind, x, y, variant = "N", -1, -1, "noop"

        ops.append(Op(kind=kind, x=x, y=y, variant=variant))

    ops = force_length_noop_pad(ops, genome_len)
    ind = Individual(ops=ops)

    # make UID stable-ish for seed records
    if "seed" in rec:
        ind.uid = f"seed_{rec.get('seed')}_final_{rec.get('final_crossings')}_{random.randint(0,999999)}"
    return ind


def build_population_from_dataset(
    dataset_jsonl: str,
    *,
    W: int,
    H: int,
    zone_pattern: str,
    pop_size: int,
    genome_len: int,
    choose: str = "best",  # "best" or "random"
    sample_seed: int = 0,
) -> List[Tuple[Dict[str, Any], Individual]]:
    """
    Returns list of (record, individual) pairs (record kept for seed/meta).
    """
    all_recs = load_dataset_jsonl(dataset_jsonl)
    filtered = [r for r in all_recs if record_matches(r, W, H, zone_pattern)]

    if not filtered:
        raise RuntimeError(
            f"No dataset records match W={W}, H={H}, zone_pattern='{zone_pattern}'. "
            f"Dataset file: {dataset_jsonl}"
        )

    filtered.sort(key=lambda r: int(r.get("final_crossings", 10**9)))
    rng = random.Random(sample_seed)

    if choose == "random":
        if len(filtered) >= pop_size:
            chosen = rng.sample(filtered, pop_size)
        else:
            chosen = [rng.choice(filtered) for _ in range(pop_size)]
    else:
        chosen = filtered[:pop_size]
        if len(chosen) < pop_size:
            chosen += [filtered[i % len(filtered)] for i in range(pop_size - len(chosen))]

    pop_pairs = [(rec, record_to_individual(rec, genome_len)) for rec in chosen]

    print(
        f"[Dataset->Pop] loaded={len(all_recs)} matched={len(filtered)} picked={len(pop_pairs)} "
        f"best_final={filtered[0].get('final_crossings')} worst_final={filtered[-1].get('final_crossings')}"
    )
    return pop_pairs


# ============================================================
# Evaluation + GA operators
# ============================================================
def evaluate_individual(
    ind: Individual,
    base_edges,
    W: int,
    H: int,
    zones: Dict[Point, int],
    *,
    verbose: bool = False,
    tag: str = "",
    print_every: int = 50,
) -> float:
    h = HamiltonianSTL(W, H)
    restore_edges(h, base_edges)

    applied = 0
    best_seen = compute_crossings(h, zones)

    for i, op in enumerate(ind.ops, start=1):
        if apply_op(h, op):
            applied += 1

        c = compute_crossings(h, zones)
        if c < best_seen:
            best_seen = c

        # (kept as comment, as you had)
        # if verbose and (i % print_every == 0):
        #     ...

    ind.applied = applied
    ind.best_seen = best_seen
    ind.fitness = -float(best_seen)
    return ind.fitness


def tournament_select(pop: List[Individual], k: int = 3) -> Individual:
    cand = random.sample(pop, k)
    cand.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
    return cand[0]


def crossover_ratio(a: Individual, b: Individual, cx_rate: float, ratio: float) -> Individual:
    """
    One-point crossover at cut = ratio*L.
    """
    child = Individual(ops=a.ops[:])  # default clone
    child.is_crossover = False

    if random.random() > cx_rate:
        return child  # clone, not crossover

    L = len(a.ops)
    cut = max(1, min(L - 1, int(round(ratio * L))))
    child.ops = a.ops[:cut] + b.ops[cut:]
    child.is_crossover = True
    return child


def crossings(ind: Individual) -> int:
    return int(-ind.fitness) if ind.fitness is not None else 10**9


# ============================================================
# Child logger: Dataset2/Children.jsonl (ACCEPTED ONLY) - FAST (file handle)
# ============================================================
def write_accepted_child_jsonl(
    f,  # open file handle
    *,
    run_id: str,
    generation: int,
    child: Individual,
    parent1: Individual,
    parent2: Individual,
    W: int,
    H: int,
    zone_pattern: str,
    zone_params: Dict[str, Any],
):
    seq = [{"kind": op.kind, "x": int(op.x), "y": int(op.y), "variant": str(op.variant)} for op in child.ops]

    rec = {
        "run_id": str(run_id),
        "algorithm": "GA_CHILD",
        "generation": int(generation),
        "grid_W": int(W),
        "grid_H": int(H),
        "zone_pattern": str(zone_pattern),
        "zone_params": dict(zone_params),
        "parent1_uid": str(parent1.uid),
        "parent2_uid": str(parent2.uid),
        "child_uid": str(child.uid),
        "parent1_best_crossings": int(parent1.best_seen),
        "parent2_best_crossings": int(parent2.best_seen),
        "child_best_crossings": int(child.best_seen),
        "child_applied_count": int(child.applied),
        "sequence_len": int(len(seq)),
        "sequence_ops": seq,
        "timestamp": int(time.time()),
    }

    f.write(json.dumps(rec) + "\n")


# ============================================================
# NEW: Save ALL best-tie GA individuals into Dataset2/Dataset.jsonl (dedup)
# ============================================================
def _sequence_signature(ind: Individual) -> str:
    return "|".join(f"{op.kind}:{op.x}:{op.y}:{op.variant}" for op in ind.ops)


def save_ga_dataset_records_top_all(
    dataset_dir: str,
    *,
    run_id: str,
    seed: int,
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    zone_params: Dict[str, Any],
    initial_crossings: int,
    population: List[Individual],
    population_size: int,
    generations: int,
    genome_len: int,
    elite_k: int,
    cx_rate: float,
    tourn_k: int,
    ratio: float,
    top_k: Optional[int] = None,  # None = write all ties
):
    """
    Append ALL individuals that achieve the minimum crossings in 'population'
    into Dataset2/Dataset.jsonl. De-duplicates by exact op-sequence signature.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, "Dataset.jsonl")

    pop2 = [p for p in population if p.fitness is not None]
    if not pop2:
        print("[GA Dataset] No evaluated individuals to save.")
        return

    min_cross = min(int(p.best_seen) for p in pop2)
    top = [p for p in pop2 if int(p.best_seen) == min_cross]

    if top_k is not None:
        top = top[: int(top_k)]

    # de-duplicate against existing dataset file + within this write
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seq = r.get("sequence_ops", [])
                    sig = "|".join(
                        f"{o.get('kind')}:{o.get('x')}:{o.get('y')}:{o.get('variant')}" for o in seq
                    )
                    existing.add(sig)
                except Exception:
                    pass

    written = 0
    local_seen = set()

    with open(path, "a", encoding="utf-8") as f:
        for ind in top:
            sig = _sequence_signature(ind)
            if sig in existing or sig in local_seen:
                continue
            local_seen.add(sig)

            sequence_ops = [
                {"kind": op.kind, "x": int(op.x), "y": int(op.y), "variant": str(op.variant)} for op in ind.ops
            ]

            rec = {
                "run_id": str(run_id),
                "algorithm": "GA_TOP",
                "seed": int(seed),
                "grid_W": int(grid_W),
                "grid_H": int(grid_H),
                "zone_pattern": str(zone_pattern),
                "zone_params": dict(zone_params),
                "initial_crossings": int(initial_crossings),
                "best_crossings": int(ind.best_seen),
                "population_size": int(population_size),
                "generations": int(generations),
                "genome_len": int(genome_len),
                "elite_k": int(elite_k),
                "cx_rate": float(cx_rate),
                "tourn_k": int(tourn_k),
                "ratio": float(ratio),
                "sequence_len": int(len(sequence_ops)),
                "applied_count": int(ind.applied),
                "sequence_ops": sequence_ops,
                "uid": str(ind.uid),
                "timestamp": int(time.time()),
            }

            f.write(json.dumps(rec) + "\n")
            written += 1

    print(f"[GA Dataset] min_crossings={min_cross} | top_ties={len(top)} | appended={written} -> {path}")


# ============================================================
# Main GA runner (dataset init + zones + Dataset2 save)
# ============================================================
def run_ga_sequences_dataset_init(
    *,
    dataset_jsonl: str,  # SA dataset: Dataset/Dataset.jsonl
    W: int,
    H: int,
    zone_pattern: str,  # left_right / islands / stripes / voronoi
    pop_size: int = 30,
    generations: int = 30,
    tourn_k: int = 3,
    genome_len: int = GENOME_LEN,
    # zone params (MUST match sa_generation.py used when dataset created)
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    stripe_direction: str = "v",
    stripe_k: int = 3,
    voronoi_k: int = 3,
    # dataset selection
    dataset_choose: str = "best",  # "best" or "random"
    dataset_sample_seed: int = 0,
    # GA output dataset
    ga_out_dir: str = "Dataset2",
):
    random.seed(dataset_sample_seed)

    # Base cycle edges
    base = HamiltonianSTL(W, H)
    if not is_valid_cycle(base):
        raise RuntimeError("Initial Hamiltonian cycle invalid (base). Check HamiltonianSTL init.")
    base_edges = snapshot_edges(base)

    # Build initial population from SA dataset
    pop_pairs = build_population_from_dataset(
        dataset_jsonl,
        W=W,
        H=H,
        zone_pattern=zone_pattern,
        pop_size=pop_size,
        genome_len=genome_len,
        choose=dataset_choose,
        sample_seed=dataset_sample_seed,
    )

    # Build zones (use first record seed for islands)
    first_seed = int(pop_pairs[0][0].get("seed", 0))
    zones, zones_meta = build_zones(
        W,
        H,
        zone_mode=zone_pattern,
        seed=first_seed,
        num_islands=num_islands,
        island_size=island_size,
        allow_touch=allow_touch,
        stripe_direction=stripe_direction,
        stripe_k=stripe_k,
        voronoi_k=voronoi_k,
    )

    base_cross = compute_crossings(base, zones)
    print(f"\n[GA] grid={W}x{H} zone_pattern={zone_pattern} zones_meta={zones_meta}")
    print(f"[GA] base crossings (zigzag init) = {base_cross}")

    # Evaluate generation 0
    pop: List[Individual] = []
    print(f"[GA] evaluating generation 0 ... (pop_size={pop_size}, genome_len={genome_len})")
    t_gen0 = time.time()

    for _, (rec, ind) in enumerate(pop_pairs, start=1):
        evaluate_individual(ind, base_edges, W, H, zones, verbose=True, tag="", print_every=50)
        pop.append(ind)

    print(f"[GA] generation 0 done in {time.time() - t_gen0:.2f}s")

    # GA run id (shared for child logging)
    ga_seed = int(dataset_sample_seed)
    run_id = f"ga_{zone_pattern}_W{W}H{H}_seed{ga_seed}_{int(time.time())}"

    zone_params = {
        "num_islands": num_islands,
        "island_size": island_size,
        "allow_touch": allow_touch,
        "stripe_direction": stripe_direction,
        "stripe_k": stripe_k,
        "voronoi_k": voronoi_k,
        "zones_meta": zones_meta,
    }

    # GA loop (NO mutation)
    for gen in range(1, generations + 1):
        pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)

        keep_n = max(ELITE_K, int(round(KEEP_RATE * pop_size)))
        keep_n = min(keep_n, len(pop))

        # Preserve uid when cloning (so parent tracking is consistent)
        elites = [
            Individual(
                ops=pop[i].ops[:],
                fitness=pop[i].fitness,
                applied=pop[i].applied,
                best_seen=pop[i].best_seen,
                uid=pop[i].uid,
            )
            for i in range(min(ELITE_K, len(pop)))
        ]

        carry = [
            Individual(
                ops=pop[i].ops[:],
                fitness=pop[i].fitness,
                applied=pop[i].applied,
                best_seen=pop[i].best_seen,
                uid=pop[i].uid,
            )
            for i in range(keep_n)
        ]

        new_pop: List[Individual] = []
        new_pop.extend(elites)
        new_pop.extend(carry)

        if len(new_pop) > pop_size:
            new_pop = new_pop[:pop_size]

        # Open Children.jsonl ONCE per generation
        os.makedirs(ga_out_dir, exist_ok=True)
        children_path = os.path.join(ga_out_dir, "Children.jsonl")
        children_f = open(children_path, "a", encoding="utf-8")

        try:
            # Fill remaining slots with accepted crossover children only
            while len(new_pop) < pop_size:
                placed = False

                for _try in range(MAX_TRIES_PER_SLOT):
                    p1 = tournament_select(pop, k=tourn_k)
                    p2 = tournament_select(pop, k=tourn_k)

                    child = crossover_ratio(p1, p2, cx_rate=CX_RATE, ratio=CX_RATIO)
                    evaluate_individual(child, base_edges, W, H, zones, verbose=False)

                    parent_best = min(p1.best_seen, p2.best_seen)
                    child_best = child.best_seen

                    valid = (child.applied >= MIN_APPLIED_VALID)
                    not_too_worse = (child_best <= parent_best + EPS_CROSSINGS)

                    accepted = bool(valid and not_too_worse)
                    if accepted:
                        write_accepted_child_jsonl(
                            children_f,
                            run_id=run_id,
                            generation=gen,
                            child=child,
                            parent1=p1,
                            parent2=p2,
                            W=W,
                            H=H,
                            zone_pattern=zone_pattern,
                            zone_params=zone_params,
                        )
                        new_pop.append(child)
                        placed = True
                        break

                if not placed:
                    # If we fail to find an accepted child, we must still fill pop.
                    # (This does NOT get logged as a child; only accepted children get logged.)
                    p = tournament_select(pop, k=tourn_k)
                    clone = Individual(ops=p.ops[:], uid=p.uid)
                    evaluate_individual(clone, base_edges, W, H, zones, verbose=False)
                    new_pop.append(clone)

        finally:
            children_f.close()

        pop = new_pop

        if gen % DEBUG_SUMMARY_EVERY == 0 or gen == generations:
            pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
            best = pop[0]
            uniq = len(set(int(p.best_seen) for p in pop))
            print(f"[GEN {gen:03d}] best_cross={best.best_seen}  uniq_best_seen={uniq}")

    # ============================================================
    # FINAL: save ALL best-tie individuals (dedup) into Dataset2/Dataset.jsonl
    # ============================================================
    pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
    evaluated = [p for p in pop if p.fitness is not None]
    if not evaluated:
        raise RuntimeError("No evaluated individuals at the end (unexpected).")

    best_crossings = min(int(p.best_seen) for p in evaluated)
    best_ties = [p for p in evaluated if int(p.best_seen) == best_crossings]
    print(f"\nFINAL best crossings = {best_crossings} | ties={len(best_ties)}")

    save_ga_dataset_records_top_all(
        dataset_dir=ga_out_dir,
        run_id=run_id,
        seed=ga_seed,
        grid_W=W,
        grid_H=H,
        zone_pattern=zone_pattern,
        zone_params=zone_params,
        initial_crossings=base_cross,
        population=pop,
        population_size=pop_size,
        generations=generations,
        genome_len=genome_len,
        elite_k=ELITE_K,
        cx_rate=CX_RATE,
        tourn_k=tourn_k,
        ratio=CX_RATIO,
        top_k=None,  # write all ties
    )

    # return one representative best
    best_final = best_ties[0]
    plt.show()
    return best_final


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # SA dataset path produced by sa_generation.py:
    dataset_jsonl = os.path.join("Dataset", "Dataset.jsonl")

    run_ga_sequences_dataset_init(
        dataset_jsonl=dataset_jsonl,
        W=30,
        H=30,
        zone_pattern="voronoi",
        pop_size=50,
        generations=50,
        tourn_k=3,
        genome_len=400,  # must match your SA sequence length
        # Match SA params you used when building dataset:
        num_islands=3,
        island_size=8,
        allow_touch=False,
        stripe_direction="v",
        stripe_k=3,
        voronoi_k=3,
        dataset_choose="best",
        dataset_sample_seed=0,
        ga_out_dir="Dataset2",
    )
