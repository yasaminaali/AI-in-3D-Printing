# FusionNet System Documentation

## 1. Overview

FusionNet is a neural network-based system for optimizing Hamiltonian toolpaths in multi-material 3D printing. Given a grid divided into material zones, the system modifies a Hamiltonian cycle to control how many times the toolpath crosses zone boundaries ("crossings"). The goal is to reach a specified target range of crossings while distributing them uniformly across boundaries.

The system uses two strategies depending on the zone pattern:

| Zone Pattern | Strategy | Uses Model? | Uses SA? |
|-------------|----------|-------------|----------|
| left_right | Constructive | No | No |
| stripes | Constructive | No | No |
| voronoi | Model-guided | Yes | Light (perturbation only) |
| islands | Model-guided | Yes | Light (perturbation only) |

---

## 2. Why Constructive for left_right and stripes

### The Problem with Using the Model for Regular Patterns

Left_right and stripes patterns have perfectly regular, axis-aligned zone boundaries (vertical or horizontal lines). The optimal approach for these patterns is mathematically deterministic — start from a zigzag path that has the minimum crossings (k-1 for k zones), then systematically add crossings at boundary positions until the target range is reached.

Using the model for these patterns would be wasteful because:

1. **The optimal solution is known analytically.** For vertical stripes with k zones, the maximum crossings is `H * (k-1)` where H is the grid height. The target range is 60-80% of this maximum.
2. **The constructive approach is deterministic and fast.** It completes in under 3 seconds even on 100x100 grids, using 11-39 operations.
3. **The constructive approach always hits the target range.** There is zero failure rate.
4. **Training the model on regular patterns wastes capacity.** The model's learning capacity is better spent on irregular patterns (voronoi, islands) where the optimal solution is NOT known analytically.

### How the Constructive Approach Works

Two-phase algorithm:

1. **Phase 1 (Propagate):** Start from the zigzag path (k-1 crossings). Greedily add crossings by applying 3x3 transpose/flip operations at boundary positions. Each operation is chosen to maximize the number of crossings added. This phase does NOT stop at the target — it covers the full boundary length.

2. **Phase 2 (Trim):** Remove crossings using spread ordering (binary subdivision) to bring the count into the target range [60%-80% of max]. This ensures crossings are evenly distributed along boundaries.

### Constructive Results

| Grid Size | Init Crossings | Final Crossings | Operations | Time |
|-----------|---------------|-----------------|------------|------|
| 30x30 | 1 | 23 | 11 | 0.05s |
| 50x50 | 1 | 39 | 19 | 0.23s |
| 60x60 | 1 | 47 | 23 | 0.58s |
| 80x80 | 1 | 63 | 31 | 1.44s |
| 100x100 | 1 | 79 | 39 | 2.77s |

100% hit the target range. CV (coefficient of variation of crossing distribution) < 0.3 for all.

---

## 3. Why the Model for voronoi and islands

Voronoi and islands patterns have irregular, non-axis-aligned zone boundaries. The zone boundaries curve, branch, and intersect unpredictably. There is no analytical formula for the optimal set of operations — it depends on the specific zone layout.

The model (FusionNet) is a 4-level ResU-Net with GRU history encoding and FiLM conditioning. It takes the current grid state as a 9-channel 128x128 image and predicts:
- **Position map:** Where to apply the next 3x3 operation (scored per-pixel)
- **Action map:** Which of 12 operations (8 transposes + 4 flips) to apply at each position

The model was trained on SA (Simulated Annealing) trajectories — sequences of operations that SA found to reduce crossings. The model learns to predict SA's moves in a single forward pass, without the thousands of random trials SA needs.

---

## 4. The Model-SA Alternating Cycle

### Why SA is Needed at All

The model sometimes gets stuck in a local minimum — it reaches a state where none of its top predicted positions lead to a crossing reduction. This happens because:

1. **The model is greedy.** It picks the best predicted position each step. Sometimes reaching a better state requires going through a temporarily worse state (uphill move) that the greedy model won't take.
2. **Escaping local minima cannot be learned from the current dataset.** The training data consists of SA trajectories where SA already found the optimal path. The model learns to imitate SA's final moves, but SA's ability to escape local minima comes from its temperature-based acceptance of worse moves — a stochastic search property that cannot be captured in a supervised position-prediction dataset. You would need reinforcement learning or a fundamentally different training approach to teach escape behavior.
3. **Random sampling has limited coverage.** Even with boundary-biased random sampling, the model only tries ~50 random positions per step. On a 60x60 grid with hundreds of valid positions, the specific position+variant combination that would reduce crossings might never be sampled.

### How the Alternating Cycle Works

The model-SA cycle runs up to 5 iterations:

```
for each cycle (up to 5):
    1. MODEL PHASE: Model predicts positions + actions, applies the best
       reducing operation each step. Runs until stagnation (150 steps
       with no improvement to global best) or target reached.

    2. SA PERTURBATION PHASE: Light SA (3000 steps, 15s time limit)
       runs from the current best state. Critically, SA does NOT restore
       to its best state — it keeps its final explored state, even if
       it's worse. This CHANGES the grid state so the model sees a
       different configuration next cycle.

    3. BACK TO MODEL: The model runs again from the SA-perturbed state.
       Because the state is different, the model may find reduction
       paths that were inaccessible from the previous local minimum.

    Stop conditions:
    - Target reached (global best <= trim_target)
    - Two consecutive cycles with no global improvement
    - Maximum 5 cycles
```

After all cycles, the global best state (tracked separately from SA perturbation) is restored.

### SA is NOT the Main Optimizer

SA's role is strictly perturbation — it shakes the state out of the model's local minimum. The numbers confirm this:

- SA typically runs 300 steps with 17-50 accepted moves
- SA's "best" is almost always equal to the starting state (SA finds no improvement)
- SA's "now" (explored state) is usually WORSE than the starting state (e.g., 60 -> 63)
- The MODEL then finds reductions from the perturbed state on the next cycle

Example from a real run:
```
Cycle 1: Model -0 (stuck at 60). SA perturbed to 60.
Cycle 2: Model found -10 from the perturbed state! (best=50)
```

The model did ALL the reduction. SA just provided the state change.

### Why Not Teach the Model to Escape?

The current training dataset contains successful SA trajectories — sequences of operations that monotonically reduce crossings (with occasional uphill steps). To teach the model to escape local minima, you would need:

1. **Trajectories that include escape sequences:** Multi-step uphill paths followed by a bigger downhill. SA does this internally but the saved trajectories only record the net-positive moves.
2. **Reinforcement learning:** Train the model to maximize long-term crossing reduction, not just imitate the next SA move. This is a fundamentally different training paradigm.
3. **State-value estimation:** Teach the model that "this state looks bad now but leads to a better state in 5 moves." The current position-prediction head has no concept of future value.

None of these are currently implemented. The alternating model-SA cycle is a pragmatic solution that works now without requiring fundamental architecture changes.

### Final Sweep

After the alternating cycle, if the model is within a few crossings of `target_upper` (within 5 or 5%, whichever is larger), a brute-force sweep tries EVERY position in the dilated boundary mask with ALL 12 operation variants. This catches the 1-2 reduction operations that random sampling missed during the model phase. It only runs when close to target, so overhead is minimal for samples that are far away.

---

## 5. Training Data and What's Missing

### Current Dataset

The model is trained on SA trajectories stored in `final_dataset.jsonl` (18,386 trajectories). Each trajectory records:
- The zone pattern and grid layout
- The sequence of operations SA performed
- The crossing count before and after each operation

The dataset builder (`build_fusion_data.py`) replays these trajectories to create (state, action) training pairs — at each step, the current grid state is the input and SA's next operation is the target.

### Data Distribution by Pattern and Grid Size

| Pattern | Grid Size | Trajectories | Model Performance |
|---------|-----------|-------------|-------------------|
| voronoi | 30x30 | 1,079 | Works on most samples |
| voronoi | 30x100 | ~200 | Works well (narrow grid) |
| voronoi | 50x50 | 903 | Partial — often barely reaches target |
| voronoi | 60x60 | 612 | Struggles — often stagnates |
| voronoi | 80x80 | 1,306 | Limited reduction |
| voronoi | 100x100 | 212 | Fails — insufficient data |
| islands | 30x30 | 691 | Works on most samples |
| islands | 50x50 | 528 | Partial |
| islands | 60x60 | 603 | Struggles |
| islands | 80x80 | 502 | Limited reduction |
| islands | 100x100 | 203 | Fails — insufficient data |

### Why Larger Grids Need More Data

1. **Larger state space.** A 30x30 grid has 900 cells. A 100x100 grid has 10,000 cells. The model needs to learn position-action mappings across a 100x larger space.

2. **More zone boundaries.** Larger grids have more boundary cells, more possible operation positions, and more complex interactions between operations at different positions.

3. **Different reduction dynamics.** The operations that reduce crossings on a 30x30 grid are qualitatively different from those on 100x100. The model cannot trivially generalize from small to large grids — it needs to see examples at each scale.

4. **Current data is heavily skewed.** 30x30 and 80x80 voronoi have 1,000+ trajectories each, but 100x100 has only 212. The model simply hasn't seen enough examples at larger sizes.

### What Needs to Be Generated

For the model to work reliably at all grid sizes:
- Generate **~1,000+ SA trajectories** per grid size per pattern for voronoi and islands
- Priority sizes: 50x50, 60x60, 100x100 (currently under-represented)
- Rebuild dataset: `python FusionModel/fusion/build_fusion_data.py`
- Retrain: `sbatch sbatch_train_fusion.sh`

The architecture, loss function, and inference code are all ready. Only data volume is the bottleneck.

---

## 6. How to Read the Inference Results

### Per-Sample Output Line

Each test sample produces one line during inference. The format differs by strategy:

**Constructive (left_right, stripes):**
```
[1/100] left_right 80x80 | 1->63 (max:80) target:[48,64] | constructive: 21%red in 31ops | CV=0.05 | Y | 1.4s
```

**Model-guided (voronoi, islands):**
```
[27/100] voronoi 30x100 | 54->44 (SA:48) target:[43,51] | model: -10 in 109ops | CV=0.10 | Y | 6.0s
```

Breaking down each field:

| Field | Example | Meaning |
|-------|---------|---------|
| `[27/100]` | Sample 27 out of 100 | Progress counter |
| `voronoi` | Zone pattern type | One of: left_right, stripes, voronoi, islands |
| `30x100` | Grid width x height | Physical grid dimensions |
| `54->44` | Initial -> final crossings | FusionNet's starting and ending crossing count |
| `(SA:48)` | SA baseline final crossings | What pure SA achieved on this same sample (for comparison) |
| `target:[43,51]` | [target_lower, target_upper] | The target crossing range. Success = final crossings within this range |
| `model: -10` | Strategy and reduction | How many crossings were reduced (negative = reduction) |
| `109ops` | Number of operations | Total 3x3 operations applied across all phases |
| `CV=0.10` | Coefficient of Variation | How uniformly crossings are distributed across boundaries. Lower = more uniform. < 0.3 is good for constructive, < 0.5 for model |
| `Y` or `N` | In target range? | Whether final crossings fell within [target_lower, target_upper] |
| `6.0s` | Wall-clock time | Total time for this sample |

### Cycle-by-Cycle Output (Model-guided only)

Before each sample's summary line, you may see cycle details:

```
Cycle 1: Model -8 (best=134, need<=119). SA perturbation...
SA: 300 steps, 47 accepted, best=134, now=138, 0.8s
Cycle 1: SA explored to 138 (global best=134)
Cycle 2: Model -10 (best=132, need<=119). SA perturbation...
```

| Field | Meaning |
|-------|---------|
| `Model -8` | Total reduction from initial crossings so far (cumulative, not per-cycle) |
| `best=134` | Global best crossing count found across all cycles |
| `need<=119` | The trim target (lower end of target range the model aims for) |
| `SA: 300 steps` | How many SA steps ran this cycle |
| `47 accepted` | How many SA moves were accepted (includes uphill moves) |
| `best=134` | Best crossing count SA found (usually same as input — SA rarely improves) |
| `now=138` | SA's final explored state (may be worse — this is the perturbation) |
| `global best=134` | Best crossing count found across ALL cycles (this is what's kept) |

### Final Sweep Output

If the model lands close to but above `target_upper`:
```
Final sweep: 55 within 5 of target_upper=54, trying all positions...
Final sweep: -9 total, 2 ops, now=53
```

This means the brute-force sweep of all boundary positions found 2 operations that the random sampling missed.

### Summary Tables

After all samples, the system prints:

1. **Per-Pattern Results:** Average reduction, operations, CV, target hit rate, comparison to SA baseline — broken down by zone pattern.

2. **Overall Summary:** Aggregated metrics across all patterns including efficiency (reduction per operation) and SA comparison.

3. **Per-Sample Detail:** Table showing every sample's init/final crossings, reduction %, SA comparison, CV, and whether it hit the target.

### What "Good" Results Look Like

- **In Target: Y** for most samples (especially constructive patterns should be 100%)
- **CV < 0.3** for constructive, **CV < 0.5** for model-guided
- **model >= SA reduction** indicates the model is matching or exceeding SA performance
- **Low ops count** relative to reduction (efficient operations)
- **Fast time** (constructive < 3s, model < 30s for most grid sizes)

---

## 7. Target Ranges

### Current Target Ranges

Target ranges define the acceptable crossing count as a fraction of the maximum possible crossings. They are configured per-pattern:

| Pattern | Reduction Range | Meaning |
|---------|----------------|---------|
| left_right | 20-40% | Final crossings = 60-80% of maximum |
| stripes | 20-40% | Final crossings = 60-80% of maximum |
| voronoi | 5-20% | Final crossings = 80-95% of initial |
| islands | 10-25% | Final crossings = 75-90% of initial |

For constructive patterns, "maximum" is the theoretical max crossings (`H * (k-1)` for vertical stripes). For model-guided patterns, the range is relative to the initial crossing count.

The model aims for the **lower end** of the range (more reduction) via `trim_target = target_lower + (target_upper - target_lower) // 4`.

### Future: Material Science-Driven Targets

The current target ranges are engineering estimates. In the future, these can be derived from material science factors:

- **Thermal properties:** Materials with higher thermal conductivity may tolerate more crossings (faster heat dissipation at boundaries)
- **Adhesion requirements:** Material pairs with poor inter-layer adhesion may need more crossings (more mechanical interlocking)
- **Print speed constraints:** More crossings mean more direction changes, which may require slower print speeds
- **Surface finish requirements:** Crossing density affects surface quality at zone boundaries
- **Stress distribution:** Uniform crossing distribution (low CV) reduces stress concentrations

The system is designed so that target ranges can be updated without retraining the model. The `TARGET_RANGES` dictionary in `constructive.py` can be modified directly, and the model will aim for the new range. Only the training data needs to cover the desired range (SA trajectories should target similar reduction levels).

---

## 8. Comparison: FusionNet vs Pure SA

### Speed

| Metric | FusionNet | Pure SA |
|--------|-----------|---------|
| left_right/stripes | 0.05-2.8s | 60-416s |
| voronoi 30x30 | 4-10s | 30-120s |
| voronoi 80x80 | 40-70s | 300-600s |
| voronoi 100x100 | 30-65s | 600-1200s |

FusionNet is **10-100x faster** than pure SA on regular patterns (constructive approach) and **3-10x faster** on irregular patterns (model approach).

### Quality

- **Constructive patterns:** FusionNet achieves 100% target hit rate with uniform distribution (CV < 0.3). SA achieves similar quality but takes 100x longer.
- **Model-guided patterns (with sufficient data):** FusionNet matches SA's reduction quality while using far fewer operations. On 30x30 voronoi, the model achieves comparable reductions in ~50-300 operations vs SA's thousands.

### Efficiency

FusionNet uses **targeted operations** — the model predicts exactly where to apply operations for maximum effect. SA uses **random exploration** — it tries thousands of random moves, most of which have zero or negative effect.

| Metric | FusionNet | SA |
|--------|-----------|-----|
| Operations to reach target | 11-39 (constructive), 50-500 (model) | 2,000-50,000 |
| Useful operations (delta < 0) | ~50-80% of total | ~1-5% of total |
| Wasted computation | Minimal (targeted) | Massive (random search) |

### Scalability

SA's runtime scales poorly with grid size because the search space grows quadratically. FusionNet's constructive approach scales linearly (operations proportional to boundary length). The model approach scales with the number of cycles needed, not the grid size directly.

### Reproducibility

- **Constructive:** Fully deterministic. Same input always produces the same output.
- **Model-guided:** Nearly deterministic (small randomness from random sampling and SA perturbation). Results are consistent across runs.
- **Pure SA:** Highly stochastic. Different runs on the same input produce different results. Requires multiple runs to get reliable outcomes.

---

## 9. System Architecture Summary

```
Input: Zone grid + grid dimensions + zone pattern
           |
           v
    select_init_and_strategy()
           |
     +-----+-----+
     |             |
  Constructive   Model-Guided
  (left_right,   (voronoi,
   stripes)       islands)
     |             |
     v             v
  Two-phase:    Alternating Model-SA Loop (up to 5 cycles):
  1. Propagate   1. Model predicts positions + actions
  2. Trim           - Top-N model candidates
                    - Boundary-biased random samples
                    - Greedy best-delta selection
                    - Stagnation: 150 steps no improvement
                 2. SA perturbation (light, 300 stagnation steps)
                    - Does NOT restore to best
                    - Keeps explored (perturbed) state
                 3. Back to step 1 with new state
                 |
                 v
              Final Sweep (if close to target)
                 - Brute-force all positions x all variants
                 |
                 v
              Phase 2: Redistribution
                 - Greedy CV improvement
                 - Model-guided position selection
     |             |
     v             v
  Output: Final Hamiltonian cycle + crossing count + CV + operations log
```

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `FusionModel/fusion/inference_fusion.py` | Main inference: model-guided strategy, alternating loop, SA perturbation, final sweep |
| `FusionModel/fusion/constructive.py` | Constructive strategy for left_right/stripes |
| `FusionModel/fusion/fusion_model.py` | FusionNet architecture (ResU-Net + GRU + FiLM) |
| `FusionModel/fusion/fusion_dataset.py` | Training dataset loading and 9-channel encoding |
| `FusionModel/fusion/train_fusion.py` | DDP training script (4x H100) |
| `FusionModel/fusion/build_fusion_data.py` | Builds compact .pt dataset from JSONL trajectories |
| `SA_generation.py` | Simulated Annealing trajectory generator |
| `sbatch_train_fusion.sh` | SLURM script for training |
| `sbatch_inference_fusion.sh` | SLURM script for inference evaluation |
| `FusionModel/CHANGELOG.md` | Detailed change history |
