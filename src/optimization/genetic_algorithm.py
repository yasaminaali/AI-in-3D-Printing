"""
Genetic Algorithm over sequences of local operations.

Population is built from SA runs. Uses one-point crossover on operation sequences.
Keeps a child ONLY IF:
    - valid (applied >= MIN_APPLIED_VALID), AND
    - does NOT increase crossings (child_best_seen <= min(parent_best_seen))
"""

import os
import copy
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt

from src.core.hamiltonian import HamiltonianSTL
from src.optimization import simulated_annealing as SA

GENOME_LEN = 100
SAVE_TOP10_PNGS = True
TOP10_DIR = "top10_plots"

DEBUG_SUMMARY_EVERY = 20

MIN_APPLIED_VALID = 1           
MAX_TRIES_PER_SLOT = 60  


PRINT_SELECTION = True 
PRINT_ELITES = True      
PRINT_TRIES = False  

DEBUG_SELECTION = True 

# Zones + Crossings (LEFT/RIGHT)
Point = Tuple[int, int]

def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}

def compute_crossings(h: HamiltonianSTL, zones: Dict[Point, int]) -> int:
    W, H = h.width, h.height
    c = 0

    # horizontal edges
    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                if zones[(x, y)] != zones[(x + 1, y)]:
                    c += 1

    # vertical edges
    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                if zones[(x, y)] != zones[(x, y + 1)]:
                    c += 1

    return c


# Edge snapshot/restore
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

# GA genome
@dataclass
class Op:
    kind: str      # "T" transpose, "F" flip, "N" no-op
    x: int
    y: int
    variant: str   # transpose OR flip directions n/s/e/w OR "noop"


@dataclass
class Individual:
    ops: List[Op]
    fitness: Optional[float] = None  # fitness = -best_crossings
    applied: int = 0
    best_seen: Optional[int] = None


# Apply operation
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
            else:
                w, hh = 2, 3

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

# SA -> GA 
def sa_run_best_ops(
    W: int,
    H: int,
    seed: int,
    iterations: int,
    Tmax: float,
    Tmin: float,
    dataset_dir: str = "Dataset1",
) -> Tuple[int, List[Dict[str, Any]]]:

    out = SA.run_sa(
        width=W,
        height=H,
        iterations=iterations,
        Tmax=Tmax,
        Tmin=Tmin,
        seed=seed,
        plot_live=False,
        show_every_accepted=0,
        pause_seconds=0.0,
        dataset_dir=dataset_dir,
    )
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], list)):
        raise RuntimeError()
    return int(out[0]), out[1]


def sa_ops_to_ga_ops(sa_ops: List[Dict[str, Any]]) -> List[Op]:
    ops: List[Op] = []
    for mv in sa_ops:
        if mv.get("op") == "transpose":
            ops.append(Op("T", int(mv["x"]), int(mv["y"]), str(mv["variant"])))
        elif mv.get("op") == "flip":
            ops.append(Op("F", int(mv["x"]), int(mv["y"]), str(mv["variant"])))
    return ops


def force_length_noop_pad(ops: List[Op], L: int) -> List[Op]:
    if len(ops) >= L:
        return ops[:L]
    padded = ops[:]
    while len(padded) < L:
        padded.append(Op("N", 0, 0, "noop"))
    return padded


def _transpose_variants(W: int, H: int) -> List[str]:
    tmp = HamiltonianSTL(W, H)
    tp = getattr(tmp, "transpose_patterns", [])
    if hasattr(tp, "keys"):
        return list(tp.keys())
    try:
        return list(tp)
    except Exception:
        return ["a", "b", "c", "d", "e", "f", "g", "h"]


def build_population_from_sa(
    W: int,
    H: int,
    pop_size: int,
    sa_iters: int,
    seed0: int,
    Tmax: float,
    Tmin: float,
    genome_len: int,
) -> Tuple[Tuple, List[Individual]]:
    base = HamiltonianSTL(W, H)
    base_edges = snapshot_edges(base)

    pop: List[Individual] = []
    for i in range(pop_size):
        seed = seed0 + i
        print(f"[SA->Pop] individual {i+1}/{pop_size} seed={seed} ...")
        best_cross, best_sa_ops = sa_run_best_ops(
            W=W, H=H, seed=seed,
            iterations=sa_iters,
            Tmax=Tmax, Tmin=Tmin,
            dataset_dir="Dataset1",
        )
        ga_ops = sa_ops_to_ga_ops(best_sa_ops)
        ga_ops = force_length_noop_pad(ga_ops, genome_len)
        pop.append(Individual(ops=ga_ops))
        print(f"    SA reported best crossings={best_cross} | genome_len={len(ga_ops)}")

    return base_edges, pop


# Evaluation
def evaluate_individual(ind: Individual, base_edges, W: int, H: int, zones) -> float:
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

# DEBUG HELPERS
def _cross(ind: Individual) -> int:
    return int(-ind.fitness) if ind.fitness is not None else 10**9


def _sid(ind: Individual) -> str:
    return f"{id(ind)%100000:05d}"


def tournament_select_debug(pop: List[Individual], k: int = 3, tag: str = "") -> Individual:  
    cand = random.sample(pop, k)  
    cand_sorted = sorted(
        cand, key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True
    )  
    winner = cand_sorted[0]  

    if DEBUG_SELECTION: 
        print(f"    [TOURN {tag}] candidates:", end=" ")  
        for c in cand:  
            print(f"{_sid(c)} cross={_cross(c)}", end=" | ") 
        print(f" -> WIN {_sid(winner)} cross={_cross(winner)}") 
    return winner  


def crossover_ratio_debug(
    a: Individual, b: Individual, cx_rate: float, ratio: float = 0.80, tag: str = ""
) -> Individual:  
    r = random.random()  
    L = len(a.ops) 
    cut = max(1, min(L - 1, int(round(ratio * L))))  

    if r > cx_rate: 
        if DEBUG_SELECTION: 
            print(
                f"    [XOVER {tag}] skipped (r={r:.3f} > cx_rate={cx_rate}) -> child=clone(A)"
            )  
        return Individual(ops=a.ops[:]) 

    if DEBUG_SELECTION: 
        print(
            f"    [XOVER {tag}] applied (r={r:.3f} <= cx_rate={cx_rate}) "
            f"cut={cut}/{L} (A:{cut} genes, B:{L-cut} genes)"
        )  
    return Individual(ops=a.ops[:cut] + b.ops[cut:])  

# GA operators
def tournament_select(pop: List[Individual], k: int = 3) -> Individual:
    cand = random.sample(pop, k)
    cand.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)
    return cand[0]

"""def one_point_crossover(a: Individual, b: Individual, cx_rate: float) -> Individual:
    if random.random() > cx_rate:
        return Individual(ops=a.ops[:])

    cut = random.randint(1, len(a.ops) - 1)
    return Individual(a.ops[:cut] + b.ops[cut:])"""

def crossover_ratio(a: Individual, b: Individual, cx_rate: float, ratio: float = 0.80) -> Individual:
    if random.random() > cx_rate:
        return Individual(ops=a.ops[:])

    L = len(a.ops)
    cut = max(1, min(L - 1, int(round(ratio * L))))  # keep cut in [1, L-1]
    return Individual(ops=a.ops[:cut] + b.ops[cut:])


def mutate_boundary_biased(ind: Individual, mut_rate: float, W: int, H: int):
    tvars = _transpose_variants(W, H)
    boundary_x = (W // 2) - 1

    def biased_x_for_3wide():
        if random.random() < 0.75:
            return max(0, min(W - 3, boundary_x - 1 + random.randint(0, 2)))
        return random.randint(0, W - 3)

    def biased_x_for_2wide():
        if random.random() < 0.75:
            return max(0, min(W - 2, boundary_x + random.randint(0, 1)))
        return random.randint(0, W - 2)

    for i in range(len(ind.ops)):
        if random.random() < mut_rate:
            r = random.random()

            if r < 0.10:
                ind.ops[i] = Op("N", 0, 0, "noop")

            elif r < 0.55:
                x = biased_x_for_3wide()
                y = random.randint(0, H - 3)
                ind.ops[i] = Op("T", x, y, random.choice(tvars))

            else:
                v = random.choice(["n", "s", "e", "w"])
                if v in ("n", "s"):
                    x = biased_x_for_3wide()
                    y = random.randint(0, H - 2)
                else:
                    x = biased_x_for_2wide()
                    y = random.randint(0, H - 3)
                ind.ops[i] = Op("F", x, y, v)


def population_summary(pop: List[Individual], gen: int):
    xs = [-p.fitness for p in pop if p.fitness is not None]
    print(f"\n=== POPULATION SUMMARY (Gen {gen}) ===")
    print(f"Population size: {len(pop)}")
    print(f"Best crossings : {min(xs):.0f}")
    print(f"Worst crossings: {max(xs):.0f}")

    uniq = sorted(set(int(-p.fitness) for p in pop if p.fitness is not None))  # ✅
    print(f"Unique crossings values: {len(uniq)} -> {uniq[:10]}{' ...' if len(uniq) > 10 else ''}")  # ✅


# Plotting
def plot_individual_from_base_newfig(
    ind: Individual,
    base_edges,
    W: int,
    H: int,
    title: str,
    save_path: Optional[str] = None,
    show_best_prefix: bool = True,
):
    zones = zones_left_right(W, H)
    h = HamiltonianSTL(W, H)
    restore_edges(h, base_edges)

    best_cross = compute_crossings(h, zones)
    best_edges = snapshot_edges(h)
    applied = 0

    for op in ind.ops:
        if apply_op(h, op):
            applied += 1
        c = compute_crossings(h, zones)
        if c < best_cross:
            best_cross = c
            best_edges = snapshot_edges(h)

    if show_best_prefix:
        restore_edges(h, best_edges)

    plt.figure()

    for y in range(H):
        for x in range(W):
            color = "lightblue" if zones[(x, y)] == 1 else "lightgreen"
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))

    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                a, b = (x, y), (x + 1, y)
                col = "red" if zones[a] != zones[b] else "black"
                plt.plot([x, x + 1], [y, y], color=col, linewidth=2)

    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                a, b = (x, y), (x, y + 1)
                col = "red" if zones[a] != zones[b] else "black"
                plt.plot([x, x], [y, y + 1], color=col, linewidth=2)

    crossings = compute_crossings(h, zones)
    plt.title(f"{title} | crossings={crossings} | applied={applied}/{len(ind.ops)}")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


# Main GA
def run_ga_sequences_sa_init(
    W: int = 30,
    H: int = 30,
    pop_size: int = 20,
    generations: int = 200,
    elite_k: int = 2,
    cx_rate: float = 0.8,
    mut_rate: float = 0.20,
    tourn_k: int = 3,
    sa_iters: int = 3000,
    sa_seed0: int = 0,
    sa_Tmax: float = 80.0,
    sa_Tmin: float = 0.5,
    genome_len: int = GENOME_LEN,
    ratio: float = 0.80,
):
    zones = zones_left_right(W, H)

    # SA init
    base_edges, pop = build_population_from_sa(
        W=W, H=H,
        pop_size=pop_size,
        sa_iters=sa_iters,
        seed0=sa_seed0,
        Tmax=sa_Tmax,
        Tmin=sa_Tmin,
        genome_len=genome_len,
    )

    base_h = HamiltonianSTL(W, H)
    base_cross = compute_crossings(base_h, zones)
    print(f"\nBase Left/Right crossings (zigzag init) = {base_cross}")

    # Evaluate gen 0
    for i, ind in enumerate(pop):
        evaluate_individual(ind, base_edges, W, H, zones)
        print(i, "GA best_seen =", ind.best_seen, "| applied =", ind.applied)

    # GA loop
    for gen in range(1, generations + 1):
        pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)

        # elites
        elites = [
            Individual(ops=pop[i].ops[:], fitness=pop[i].fitness, applied=pop[i].applied, best_seen=pop[i].best_seen)
            for i in range(min(elite_k, len(pop)))
        ]

        if PRINT_ELITES and PRINT_SELECTION:  
            elite_str = ", ".join([f"{_sid(e)}:cross={_cross(e)}" for e in elites])
            print(f"\n--- GEN {gen} --- elites: {elite_str}")  

        new_pop: List[Individual] = elites[:]

        while len(new_pop) < pop_size:
            placed = False

            slot = len(new_pop)  
            if DEBUG_SELECTION:  
                print(f"\n  [SLOT {slot}] trying to create a child...")  

            for _try in range(MAX_TRIES_PER_SLOT):
                p1 = tournament_select_debug(pop, k=tourn_k, tag=f"g{gen}/slot{slot}/try{_try} P1") 
                p2 = tournament_select_debug(pop, k=tourn_k, tag=f"g{gen}/slot{slot}/try{_try} P2")  

                #child = one_point_crossover(p1, p2, cx_rate=cx_rate)
                child = crossover_ratio_debug(p1, p2, cx_rate=cx_rate, ratio=ratio, tag=f"g{gen}/slot{slot}/try{_try}")
                mutate_boundary_biased(child, mut_rate, W, H)

                evaluate_individual(child, base_edges, W, H, zones)

                parent_best = min(_cross(p1), _cross(p2))  
                child_best = _cross(child) 

                valid = (child.applied >= MIN_APPLIED_VALID)
                EPS = 2   # allow +2 crossings
                not_too_worse = (child_best <= parent_best + EPS)   

                if DEBUG_SELECTION:  
                    print(
                        f"    [CHILD g{gen}/slot{slot}/try{_try}] "
                        f"p1={_cross(p1)} p2={_cross(p2)} -> child_best={child_best} "
                        f"| applied={child.applied} | valid={valid} | not_too_worse={not_too_worse}"
                    ) 

                if valid and not_too_worse:
                    if DEBUG_SELECTION:  
                        print(
                            f"[GEN {gen:03d} | slot {slot:02d}] ✅ ACCEPT  "
                            f"P1({_sid(p1)}) cross={_cross(p1)}  "
                            f"P2({_sid(p2)}) cross={_cross(p2)}  "
                            f"-> Child({_sid(child)}) cross={child_best} applied={child.applied}"
                        )
                    new_pop.append(child)
                    placed = True
                    break

            if not placed:
                p = tournament_select(pop, k=tourn_k)
                clone = Individual(ops=p.ops[:], fitness=p.fitness, applied=p.applied, best_seen=p.best_seen)
                if DEBUG_SELECTION:  
                    print(
                        f"[GEN {gen:03d} | slot {slot:02d}] ✅ CLONE   "
                        f"P({_sid(p)}) cross={_cross(p)} -> Clone({_sid(clone)}) cross={_cross(clone)}"
                    )
                    new_pop.append(clone)
        pop = new_pop

        if DEBUG_SUMMARY_EVERY and (gen % DEBUG_SUMMARY_EVERY == 0 or gen == generations):
            population_summary(pop, gen)

    # END: Top 10 print + plots
    pop.sort(key=lambda z: z.fitness if z.fitness is not None else -1e18, reverse=True)

    top_k = min(10, len(pop))
    top10 = pop[:top_k]

    print("\n" + "=" * 60)
    print(f"TOP {top_k} INDIVIDUALS (best_seen crossings, Left/Right)")
    for rank, ind in enumerate(top10, start=1):
        crossings = int(-ind.fitness)
        print(f"#{rank:02d}: crossings={crossings} | applied={ind.applied}/{len(ind.ops)}")
    print("=" * 60)

    if SAVE_TOP10_PNGS:
        os.makedirs(TOP10_DIR, exist_ok=True)

    for rank, ind in enumerate(top10, start=1):
        crossings = int(-ind.fitness)
        save_path = f"{TOP10_DIR}/top_{rank:02d}_cross_{crossings}.png" if SAVE_TOP10_PNGS else None
        plot_individual_from_base_newfig(
            ind,
            base_edges,
            W,
            H,
            title=f"TOP {rank:02d}",
            save_path=save_path,
            show_best_prefix=True,
        )

    best_final = top10[0]
    print(f"\nFINAL best crossings = {-best_final.fitness:.0f} | applied={best_final.applied}/{len(best_final.ops)}")

    plt.show()
    return best_final


if __name__ == "__main__":
    run_ga_sequences_sa_init(
        W=30, H=30,
        pop_size=20,
        generations=200,
        elite_k=2,
        cx_rate=0.8,
        mut_rate=0.20,
        tourn_k=3,
        sa_iters=3000,
        sa_seed0=0,
        sa_Tmax=80.0,
        sa_Tmin=0.5,
        genome_len=100,
        ratio=0.40
    )
