# ga_sequence.py
# ============================================================
# Genetic Algorithm (GA) over sequences of local operations (flip/transpose)
# Initial population is loaded from SA_generation.py JSONL dataset.
# GA output is stored into Dataset2/Dataset.jsonl (SA-like JSONL record).
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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt

from operations import HamiltonianSTL
from Zones import zones_stripes, zones_voronoi

Point = Tuple[int, int]

# -----------------------
# GA Hyperparameters (defaults)
# -----------------------
GENOME_LEN = 100

SAVE_TOP10_PNGS = True
TOP10_DIR = "top10_plots"

DEBUG_SUMMARY_EVERY = 20

MIN_APPLIED_VALID = 1
MAX_TRIES_PER_SLOT = 60

EPS_CROSSINGS = 2

# Mutation behaviour
NOOP_MUT_PROB = 0.20
TRANSPOSE_MUT_PROB = 0.45
# remaining -> flip


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
            W, H,
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
    kind: str      # "T" transpose, "F" flip, "N" no-op
    x: int
    y: int
    variant: str   # transpose variant OR flip dir n/s/e/w OR "noop"


@dataclass
class Individual:
    ops: List[Op]
    fitness: Optional[float] = None  # fitness = -best_seen_crossings
    applied: int = 0
    best_seen: int = 10**9


# ============================================================
# Operation application (preserve Hamiltonian)
# ============================================================
def _msg_from_result(ret) -> str:
    if isinstance(ret, tuple) and len(ret) >= 2:
        return str(ret[1])
    return str(ret)


def apply_op(h: HamiltonianSTL, op: Op) -> bool:
    if op.kind == "N":
        return True

    before = snapshot_edges(h)

    try:
        if op.kind == "T":
            if op.x < 0 or op.y < 0 or op.x + 2 >= h.width or op.y + 2 >= h.height:
                restore_edges(h, before)
                return False
            sub = h.get_subgrid((op.x, op.y), (op.x + 2, op.y + 2))
            ret = h.transpose_subgrid(sub, op.variant)
            ok = _msg_from_result(ret).startswith("transposed")

        elif op.kind == "F":
            if op.variant in ("n", "s"):
                w, hh = 3, 2
            elif op.variant in ("e", "w"):
                w, hh = 2, 3
            else:
                restore_edges(h, before)
                return False

            if op.x < 0 or op.y < 0 or op.x + (w - 1) >= h.width or op.y + (hh - 1) >= h.height:
                restore_edges(h, before)
                return False

            sub = h.get_subgrid((op.x, op.y), (op.x + w - 1, op.y + hh - 1))
            ret = h.flip_subgrid(sub, op.variant)
            ok = _msg_from_result(ret).startswith("flipped")

        else:
            restore_edges(h, before)
            return False

        if not ok:
            restore_edges(h, before)
            return False

        if not is_valid_cycle(h):
            restore_edges(h, before)
            return False

        return True

    except Exception:
        restore_edges(h, before)
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
    return Individual(ops=ops)


def build_population_from_dataset(
    dataset_jsonl: str,
    *,
    W: int,
    H: int,
    zone_pattern: str,
    pop_size: int,
    genome_len: int,
    choose: str = "best",     # "best" or "random"
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
# Boundary anchors for mutation (works for any zone map)
# ============================================================
def boundary_anchors_from_zones(W: int, H: int, zones: Dict[Point, int]) -> List[Point]:
    anchors: List[Point] = []

    for y in range(H):
        for x in range(W - 1):
            if zones[(x, y)] != zones[(x + 1, y)]:
                anchors.append((x, y))
                anchors.append((x + 1, y))

    for y in range(H - 1):
        for x in range(W):
            if zones[(x, y)] != zones[(x, y + 1)]:
                anchors.append((x, y))
                anchors.append((x, y + 1))

    if not anchors:
        anchors = [(W // 2, H // 2)]
    return anchors


def mutate_boundary_biased(ind: Individual, mut_rate: float, W: int, H: int, anchors: List[Point]):
    tvars = _transpose_variants(W, H)
    fvars = ["n", "s", "e", "w"]

    for i in range(len(ind.ops)):
        if random.random() > mut_rate:
            continue

        r = random.random()
        if r < NOOP_MUT_PROB:
            ind.ops[i] = Op("N", -1, -1, "noop")
            continue

        ax, ay = random.choice(anchors)
        x = int(round(random.gauss(mu=ax, sigma=max(2.0, W / 12.0))))
        y = int(round(random.gauss(mu=ay, sigma=max(2.0, H / 12.0))))

        if r < NOOP_MUT_PROB + TRANSPOSE_MUT_PROB:
            x = max(0, min(W - 3, x))
            y = max(0, min(H - 3, y))
            ind.ops[i] = Op("T", x, y, random.choice(tvars))
        else:
            v = random.choice(fvars)
            if v in ("n", "s"):
                x = max(0, min(W - 3, x))
                y = max(0, min(H - 2, y))
            else:
                x = max(0, min(W - 2, x))
                y = max(0, min(H - 3, y))
            ind.ops[i] = Op("F", x, y, v)


# ============================================================
# Evaluation + GA operators
# ============================================================
def evaluate_individual(ind: Individual, base_edges, W: int, H: int, zones: Dict[Point, int]) -> float:
    h = HamiltonianSTL(W, H)
    restore_edges(h, base_edges)

    applied = 0
    best_seen = compute_crossings(h, zones)

    for op in ind.ops:
        if apply_op(h, op):
            applied += 1
        c = compute_crossings(h, zones)
        if c < best_seen:
            best_seen = c

    ind.applied = applied
    ind.best_seen = best_seen
    ind.fitness = -float(best_seen)
    return ind.fitness


def tournament_select(pop: List[Individual], k: int = 3) -> Individual:
    cand = random.sample(pop, k)
    cand.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
    return cand[0]


def crossover_ratio(a: Individual, b: Individual, cx_rate: float, ratio: float = 0.80) -> Individual:
    if random.random() > cx_rate:
        return Individual(ops=a.ops[:])

    L = len(a.ops)
    cut = max(1, min(L - 1, int(round(ratio * L))))
    return Individual(ops=a.ops[:cut] + b.ops[cut:])


def _cross(ind: Individual) -> int:
    return int(-ind.fitness) if ind.fitness is not None else 10**9


# ============================================================
# Plotting (BEST snapshot on trajectory)
# ============================================================
def plot_individual_best_snapshot(
    ind: Individual,
    base_edges,
    W: int,
    H: int,
    zones: Dict[Point, int],
    title: str,
    save_path: Optional[str] = None,
):
    h = HamiltonianSTL(W, H)
    restore_edges(h, base_edges)

    best_cross = compute_crossings(h, zones)
    best_edges = snapshot_edges(h)

    for op in ind.ops:
        apply_op(h, op)
        c = compute_crossings(h, zones)
        if c < best_cross:
            best_cross = c
            best_edges = snapshot_edges(h)

    restore_edges(h, best_edges)

    zone_vals = sorted(set(zones.values()))
    z_to_idx = {z: i for i, z in enumerate(zone_vals)}
    cmap = plt.get_cmap("tab20")

    plt.figure()
    for y in range(H):
        for x in range(W):
            z = zones[(x, y)]
            color = cmap(z_to_idx[z] % 20)
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.25))

    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                col = "red" if zones[(x, y)] != zones[(x + 1, y)] else "black"
                plt.plot([x, x + 1], [y, y], color=col, linewidth=2)

    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                col = "red" if zones[(x, y)] != zones[(x, y + 1)] else "black"
                plt.plot([x, x], [y, y + 1], color=col, linewidth=2)

    plt.title(f"{title} | best_crossings={best_cross} | applied={ind.applied}/{len(ind.ops)}")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


# ============================================================
# GA output dataset writer (Dataset2)
# ============================================================
def save_ga_dataset_record(
    dataset_dir: str,
    *,
    run_id: str,
    seed: int,
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    zone_params: Dict[str, Any],
    initial_crossings: int,
    best_crossings: int,
    best_individual: Individual,
    population_size: int,
    generations: int,
    genome_len: int,
    elite_k: int,
    cx_rate: float,
    mut_rate: float,
    tourn_k: int,
    ratio: float,
):
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, "Dataset.jsonl")

    sequence_ops = []
    for op in best_individual.ops:
        if op.kind in ("T", "F"):
            sequence_ops.append({
                "kind": op.kind,
                "x": int(op.x),
                "y": int(op.y),
                "variant": str(op.variant),
            })
        else:
            # keep fixed length (NOPs included) – consistent with genomes
            sequence_ops.append({"kind": "N", "x": -1, "y": -1, "variant": "noop"})

    rec = {
        "run_id": str(run_id),
        "algorithm": "GA",
        "seed": int(seed),
        "grid_W": int(grid_W),
        "grid_H": int(grid_H),
        "zone_pattern": str(zone_pattern),
        "zone_params": dict(zone_params),

        "initial_crossings": int(initial_crossings),
        "best_crossings": int(best_crossings),

        "population_size": int(population_size),
        "generations": int(generations),
        "genome_len": int(genome_len),
        "elite_k": int(elite_k),
        "cx_rate": float(cx_rate),
        "mut_rate": float(mut_rate),
        "tourn_k": int(tourn_k),
        "ratio": float(ratio),

        "sequence_len": int(len(sequence_ops)),
        "applied_count": int(best_individual.applied),
        "sequence_ops": sequence_ops,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"[GA Dataset] appended 1 record -> {path}")


# ============================================================
# Main GA runner (dataset init + zones + Dataset2 save)
# ============================================================
def run_ga_sequences_dataset_init(
    *,
    dataset_jsonl: str,           # SA dataset: Dataset/Dataset.jsonl
    W: int,
    H: int,
    zone_pattern: str,            # left_right / islands / stripes / voronoi
    pop_size: int = 30,
    generations: int = 200,
    elite_k: int = 2,
    cx_rate: float = 0.8,
    mut_rate: float = 0.20,
    tourn_k: int = 3,
    genome_len: int = GENOME_LEN,
    ratio: float = 0.40,
    # zone params (MUST match SA_generation.py used when dataset created)
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
        W=W, H=H,
        zone_pattern=zone_pattern,
        pop_size=pop_size,
        genome_len=genome_len,
        choose=dataset_choose,
        sample_seed=dataset_sample_seed,
    )

    # Build zones for this GA run.
    # Use first record seed (important for islands if your SA islands used seeded generation).
    first_seed = int(pop_pairs[0][0].get("seed", 0))
    zones, zones_meta = build_zones(
        W, H,
        zone_mode=zone_pattern,
        seed=first_seed,
        num_islands=num_islands,
        island_size=island_size,
        allow_touch=allow_touch,
        stripe_direction=stripe_direction,
        stripe_k=stripe_k,
        voronoi_k=voronoi_k,
    )
    anchors = boundary_anchors_from_zones(W, H, zones)

    base_cross = compute_crossings(base, zones)
    print(f"\n[GA] grid={W}x{H} zone_pattern={zone_pattern} zones_meta={zones_meta}")
    print(f"[GA] base crossings (zigzag init) = {base_cross}")
    print(f"[GA] boundary anchors = {len(anchors)}")

    # Evaluate generation 0
    pop: List[Individual] = []
    for rec, ind in pop_pairs:
        evaluate_individual(ind, base_edges, W, H, zones)
        pop.append(ind)

    # GA loop
    for gen in range(1, generations + 1):
        pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)

        elites = [
            Individual(ops=pop[i].ops[:], fitness=pop[i].fitness, applied=pop[i].applied, best_seen=pop[i].best_seen)
            for i in range(min(elite_k, len(pop)))
        ]
        new_pop: List[Individual] = elites[:]

        while len(new_pop) < pop_size:
            placed = False

            for _try in range(MAX_TRIES_PER_SLOT):
                p1 = tournament_select(pop, k=tourn_k)
                p2 = tournament_select(pop, k=tourn_k)

                child = crossover_ratio(p1, p2, cx_rate=cx_rate, ratio=ratio)
                mutate_boundary_biased(child, mut_rate, W, H, anchors)

                evaluate_individual(child, base_edges, W, H, zones)

                parent_best = min(_cross(p1), _cross(p2))
                child_best = _cross(child)

                valid = (child.applied >= MIN_APPLIED_VALID)
                not_too_worse = (child_best <= parent_best + EPS_CROSSINGS)

                if valid and not_too_worse:
                    new_pop.append(child)
                    placed = True
                    break

            if not placed:
                # fallback: clone best parent
                p = tournament_select(pop, k=tourn_k)
                new_pop.append(Individual(ops=p.ops[:]))

        pop = new_pop

        if gen % DEBUG_SUMMARY_EVERY == 0 or gen == generations:
            pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
            best = pop[0]
            uniq = len(set(int(-p.fitness) for p in pop if p.fitness is not None))
            print(f"[GEN {gen:03d}] best_cross={-best.fitness:.0f}  uniq_fitness={uniq}")

    # Final top 10
    pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
    top10 = pop[:10]

    print("\nTOP 10 best_crossings:")
    for i, ind in enumerate(top10, start=1):
        print(f"  #{i:02d}: crossings={-ind.fitness:.0f} applied={ind.applied}/{len(ind.ops)}")

    if SAVE_TOP10_PNGS:
        os.makedirs(TOP10_DIR, exist_ok=True)
        for rank, ind in enumerate(top10, start=1):
            save_path = os.path.join(TOP10_DIR, f"top{rank:02d}_cross{int(-ind.fitness)}.png")
            plot_individual_best_snapshot(
                ind, base_edges, W, H, zones,
                title=f"TOP {rank:02d}",
                save_path=save_path,
            )

    best_final = top10[0]
    best_crossings = int(-best_final.fitness)
    print(f"\nFINAL best crossings = {best_crossings} | applied={best_final.applied}/{len(best_final.ops)}")

    # ===============================
    # Save GA output to Dataset2
    # ===============================
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

    save_ga_dataset_record(
        dataset_dir=ga_out_dir,
        run_id=run_id,
        seed=ga_seed,
        grid_W=W,
        grid_H=H,
        zone_pattern=zone_pattern,
        zone_params=zone_params,
        initial_crossings=base_cross,
        best_crossings=best_crossings,
        best_individual=best_final,
        population_size=pop_size,
        generations=generations,
        genome_len=genome_len,
        elite_k=elite_k,
        cx_rate=cx_rate,
        mut_rate=mut_rate,
        tourn_k=tourn_k,
        ratio=ratio,
    )

    plt.show()
    return best_final


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # SA dataset path produced by SA_generation.py:
    dataset_jsonl = os.path.join("Dataset", "Dataset.jsonl")

    # Example: GA for 32x32 and left_right
    run_ga_sequences_dataset_init(
        dataset_jsonl=dataset_jsonl,
        W=32,
        H=32,
        zone_pattern="left_right",

        pop_size=30,
        generations=200,
        elite_k=2,
        cx_rate=0.8,
        mut_rate=0.20,
        tourn_k=3,
        genome_len=100,                 # Must change for each size
        ratio=0.40,

        # Match the SA_generation.py params you used when building the dataset:
        num_islands=3,
        island_size=8,
        allow_touch=False,
        stripe_direction="v",
        stripe_k=3,
        voronoi_k=3,

        dataset_choose="best",        # "best" or "random"
        dataset_sample_seed=0,        # GA randomness seed

        ga_out_dir="Dataset2",        # GA output dataset folder
    )
