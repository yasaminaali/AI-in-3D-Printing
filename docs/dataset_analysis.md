# Dataset Analysis: final_dataset.jsonl

**Total records:** 22,111 (21,863 SA + 248 GA)

---

## Table 1: Per Grid Size Summary

| Grid | Trajs | Seeds | SeqLen min | SeqLen avg | SeqLen max | Runtime min | Runtime avg | Runtime max |
|---------|------:|------:|-----------:|-----------:|-----------:|------------:|------------:|------------:|
| 30x30 | 5,869 | 1,150 | 0 | 408 | 1,738 | 16.3s | 118.0s | 6,762.7s |
| 30x100 | 2,748 | 225 | 0 | 1,221 | 5,897 | 118.1s | 369.4s | 819.8s |
| 50x50 | 4,744 | 875 | 0 | 944 | 2,988 | 39.7s | 90.3s | 199.4s |
| 60x60 | 2,245 | 150 | 0 | 1,086 | 6,303 | 142.9s | 524.0s | 1,184.9s |
| 60x100 | 496 | 150 | 405 | 955 | 7,223 | 375.7s | 982.3s | 2,746.8s |
| 80x80 | 4,322 | 475 | 0 | 872 | 7,405 | 4.4s | 1,104.5s | 3,164.3s |
| 100x100 | 1,687 | 300 | 0 | 970 | 8,159 | 7.1s | 452.4s | 2,108.4s |

---

## Table 2: Trajectories per Grid Size x Pattern

| Grid | Pattern | Trajs | Seeds | SeqLen avg | SeqLen max | Runtime avg | Runtime max |
|---------|------------|------:|------:|-----------:|-----------:|------------:|------------:|
| 30x30 | left_right | 3,457 | 1,150 | 421 | 1,713 | 149.6s | 6,428.7s |
| 30x30 | stripes | 642 | 300 | 229 | 1,568 | 129.0s | 6,762.7s |
| 30x30 | voronoi | 1,079 | 350 | 406 | 1,738 | 53.3s | 166.0s |
| 30x30 | islands | 691 | 225 | 515 | 1,697 | 48.4s | 113.6s |
| | | | | | | | |
| 30x100 | left_right | 601 | 150 | 2,041 | 5,897 | 369.3s | 813.6s |
| 30x100 | stripes | 702 | 175 | 1,109 | 5,806 | 379.8s | 819.8s |
| 30x100 | voronoi | 841 | 225 | 664 | 5,855 | 354.5s | 819.8s |
| 30x100 | islands | 604 | 150 | 1,313 | 5,890 | 378.2s | 818.4s |
| | | | | | | | |
| 50x50 | left_right | 2,631 | 875 | 1,138 | 2,988 | 90.6s | 199.4s |
| 50x50 | stripes | 682 | 225 | 542 | 2,894 | 88.6s | 171.8s |
| 50x50 | voronoi | 903 | 300 | 722 | 2,956 | 90.1s | 170.2s |
| 50x50 | islands | 528 | 175 | 876 | 2,879 | 91.1s | 168.1s |
| | | | | | | | |
| 60x60 | left_right | 412 | 100 | 580 | 1,781 | 528.8s | 1,170.8s |
| 60x60 | stripes | 618 | 150 | 894 | 6,296 | 520.6s | 1,181.0s |
| 60x60 | voronoi | 612 | 150 | 1,179 | 6,242 | 527.9s | 1,184.9s |
| 60x60 | islands | 603 | 150 | 1,535 | 6,303 | 520.4s | 1,176.4s |
| | | | | | | | |
| 60x100 | stripes | 496 | 150 | 955 | 7,223 | 982.3s | 2,746.8s |
| | | | | | | | |
| 80x80 | left_right | 1,912 | 475 | 335 | 3,638 | 1,423.6s | 3,164.3s |
| 80x80 | stripes | 602 | 150 | 1,314 | 7,232 | 1,367.0s | 3,148.0s |
| 80x80 | voronoi | 1,306 | 200 | 1,152 | 7,405 | 414.6s | 1,613.9s |
| 80x80 | islands | 502 | 125 | 1,655 | 7,342 | 1,370.6s | 3,092.3s |
| | | | | | | | |
| 100x100 | left_right | 846 | 300 | 705 | 4,622 | 918.0s | 2,108.4s |
| 100x100 | stripes | 426 | 50 | 1,048 | 7,967 | 29.9s | 102.0s |
| 100x100 | voronoi | 212 | 50 | 1,223 | 8,059 | 30.1s | 105.8s |
| 100x100 | islands | 203 | 50 | 1,651 | 8,159 | 30.0s | 103.6s |

---

## Table 3: Inferred SA Config Distribution

The JSONL does not store which global config was used per trajectory. This table infers the config from `runtime_sec` as a rough proxy (SA records only, 248 GA records excluded).

| Bucket | Runtime |
|------------|-----------|
| short | < 60s |
| medium | 60–300s |
| long | 300–900s |
| extra_long | > 900s |

| Grid | short | medium | long | extra_long | Total |
|---------|------:|-------:|-----:|-----------:|------:|
| 30x30 | 2,348 | 3,379 | 92 | 20 | 5,839 |
| 30x100 | 0 | 1,400 | 1,339 | 0 | 2,739 |
| 50x50 | 1,568 | 3,157 | 0 | 0 | 4,725 |
| 60x60 | 0 | 1,068 | 582 | 550 | 2,200 |
| 60x100 | 0 | 0 | 300 | 196 | 496 |
| 80x80 | 678 | 22 | 1,900 | 1,700 | 4,300 |
| 100x100 | 626 | 194 | 600 | 144 | 1,564 |

---

## Table 4: Source (SA vs GA) per Grid Size

| Grid | SA | GA | Total |
|---------|------:|------:|------:|
| 30x30 | 5,839 | 30 | 5,869 |
| 30x100 | 2,739 | 9 | 2,748 |
| 50x50 | 4,725 | 19 | 4,744 |
| 60x60 | 2,200 | 45 | 2,245 |
| 60x100 | 496 | 0 | 496 |
| 80x80 | 4,300 | 22 | 4,322 |
| 100x100 | 1,564 | 123 | 1,687 |

---

## Table 5: Crossing Reduction per Grid Size x Pattern

| Grid | Pattern | Init avg | Final avg | Red% avg | Red% min | Red% max |
|---------|------------|--------:|---------:|--------:|--------:|--------:|
| 30x30 | left_right | 30.0 | 15.7 | 47.6% | 26.7% | 60.0% |
| 30x30 | stripes | 60.0 | 32.0 | 46.7% | 36.7% | 60.0% |
| 30x30 | voronoi | 33.4 | 26.8 | 17.1% | 0.0% | 54.1% |
| 30x30 | islands | 45.2 | 34.5 | 23.5% | 0.0% | 47.6% |
| | | | | | | |
| 30x100 | left_right | 100.0 | 55.7 | 44.3% | 30.0% | 56.0% |
| 30x100 | stripes | 200.0 | 107.6 | 46.2% | 38.0% | 54.0% |
| 30x100 | voronoi | 37.7 | 31.6 | 9.8% | 0.0% | 47.1% |
| 30x100 | islands | 46.1 | 36.4 | 20.9% | 0.0% | 45.8% |
| | | | | | | |
| 50x50 | left_right | 50.0 | 33.4 | 33.2% | 0.0% | 56.0% |
| 50x50 | stripes | 100.0 | 58.1 | 41.9% | 32.0% | 54.0% |
| 50x50 | voronoi | 55.7 | 47.1 | 13.0% | 0.0% | 46.5% |
| 50x50 | islands | 47.2 | 37.5 | 20.5% | 0.0% | 45.8% |
| | | | | | | |
| 60x60 | left_right | 60.0 | 39.4 | 34.3% | 0.0% | 56.7% |
| 60x60 | stripes | 120.0 | 71.7 | 40.3% | 23.3% | 53.3% |
| 60x60 | voronoi | 65.2 | 56.6 | 11.2% | 0.0% | 46.7% |
| 60x60 | islands | 47.2 | 38.2 | 19.1% | 0.0% | 41.7% |
| | | | | | | |
| 60x100 | stripes | 200.0 | 126.7 | 36.7% | 13.0% | 50.0% |
| | | | | | | |
| 80x80 | left_right | 80.0 | 60.6 | 24.3% | 0.0% | 55.0% |
| 80x80 | stripes | 160.0 | 110.6 | 30.9% | 0.0% | 50.0% |
| 80x80 | voronoi | 89.0 | 80.6 | 7.7% | 0.0% | 46.2% |
| 80x80 | islands | 47.7 | 39.7 | 16.9% | 0.0% | 50.0% |
| | | | | | | |
| 100x100 | left_right | 100.0 | 92.8 | 7.2% | 0.0% | 54.0% |
| 100x100 | stripes | 190.7 | 147.8 | 21.5% | 0.0% | 47.0% |
| 100x100 | voronoi | 102.8 | 95.9 | 5.4% | 0.0% | 33.5% |
| 100x100 | islands | 47.9 | 39.9 | 16.6% | 0.0% | 50.0% |

---

## Global Config Presets

From `config/global_config.yaml`:

| Config | Iterations | Tmax | Tmin |
|------------|----------:|------:|------:|
| short | 3,000 | 60.0 | 0.5 |
| medium | 5,000 | 80.0 | 0.5 |
| long | 10,000 | 100.0 | 0.3 |
| extra_long | 20,000 | 120.0 | 0.2 |

---

## Key Observations

- **60x100 only has stripes** (496 records) — no left_right, voronoi, or islands
- **100x100 stripes/voronoi/islands have only 50 seeds each** — very thin coverage
- **Reduction quality drops sharply at larger grids**: voronoi goes from 17.1% avg reduction at 30x30 to 5.4% at 100x100
- **0% min reduction** appears in most pattern/grid combos, meaning some trajectories made no improvement at all
- **GA contributes only 248 records** (1.1% of the dataset), the rest is SA-generated
- **left_right and stripes have deterministic initial crossings** (equal to grid width or 2× grid width), while voronoi/islands vary by seed
