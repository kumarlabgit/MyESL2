# MyESL2

**My Evolutionary Sparse Learning 2** — a C++20 tool for sparse regression-based classification on genomic data. It supports FASTA alignment inputs (nucleotide/protein) and tabular numeric inputs, designed for comparative genomic analysis across any set of organisms.

## Overview

MyESL2 uses Sparse Group Lasso regression to identify genomic features (alignment positions, genes) that are predictive of a binary phenotype. It operates on multiple sequence alignments or numeric feature matrices and produces per-feature weights, gene significance scores, and visualizations.

## Quick Start
 
### 1. Get a binary

Download the latest release binary for your platform into `./bin/` with the
provided setup script — no build toolchain required:

```bash
# Linux x86_64 / macOS arm64
bash scripts/setup.sh
```

```powershell
# Windows x64 (from a PowerShell prompt in the repo root)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# or: scripts\setup.bat
```

The script queries the latest GitHub release, downloads the platform-matching
executable and its runtime library (`libopenblas.so` / `libopenblas.dylib` /
`openblas.dll` + `lapack.dll`) into `./bin/`, and copies `data_defs.ini`
alongside the binary so it works from any working directory. Pass `--tag vX.Y.Z`
(bash) or `-Tag vX.Y.Z` (PowerShell) to pin a specific release. Prefer to build
from source? See `CLAUDE.md` → *Building the Project* or the `CMakeLists.txt`.

#### Release channels (`main` vs. `dev`)

MyESL2 publishes two release streams:

| Channel | Tags | Platforms | Selected when |
|---|---|---|---|
| `main` (stable) | `v*` (e.g. `v0.2.3`) | Linux x64 (regular + portable), macOS arm64, Windows x64 | Default for any checkout that isn't `dev` |
| `dev` (preview) | `dev-v*` (e.g. `dev-v0.3.0-rc1`) | Linux x64 (regular + portable), Windows x64 (no macOS) | Auto-selected on `git checkout dev` |

The setup scripts auto-detect the channel using this precedence:

1. **`--channel main|dev` / `-Channel main|dev`** — explicit override (highest priority).
2. **`.release-channel` file at the repo root** — committed per branch (`main` on main, `dev` on dev). This is the canonical signal and travels with the source even when you download a GitHub source tarball/zip without using `git clone`.
3. **Current git branch** (`dev` → dev channel, anything else → main) — fallback for older checkouts without the marker file.
4. **`main`** — default when no signal is available.

The scripts also refuse to mix channels with a wrong-prefix `--tag` / `-Tag`,
so you can't accidentally fetch a `dev-v*` tarball with `--channel main`.

### 2. Run

```bash
# Train a model on FASTA alignments
./bin/myesl2 train alignments.txt labels.txt output_dir/

# Evaluate a trained model
./bin/myesl2 evaluate output_dir/lambda_0/weights.txt alignments.txt predictions.txt --hypothesis labels.txt

# Visualize results
./bin/myesl2 visualize eval_gene_predictions.txt heatmap.svg
```

---

## Input File Formats

### Alignment list (`list.txt`)
A plain text file with one file path per line. Each path points to a FASTA alignment file (for genomic mode) or a numeric data file.

```
/data/aln/gene1.fasta
/data/aln/gene2.fasta
/data/aln/gene3.fasta
```

### Hypothesis file (`hypothesis.txt`)
Tab- or space-separated file mapping sample names to response values. Use `1` and `-1` for binary classification. Only samples with non-zero values are included in the model.

```
species_A    1
species_B    1
species_C   -1
species_D   -1
```

### Numeric data files
For `--datatype numeric`, each file is a whitespace-delimited table where the first column is the sample name and remaining columns are float features. All files in the list are concatenated column-wise into one feature matrix.

```
species_A   0.12  3.45  1.00
species_B   0.98  2.11  0.50
```

---

## Commands

### `train` — Train a sparse regression model

```
myesl2 train <list.txt> <hypothesis.txt> <output_dir> [column|row] [options]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `list.txt` | File listing paths to alignment or numeric data files |
| `hypothesis.txt` | Sample labels (non-zero values only) |
| `output_dir` | Output directory (created if it does not exist) |

**Optional orientation (FASTA only):**

| Value | Description |
|-------|-------------|
| `column` | Column-major PFF storage — efficient for position-wise analysis (default) |
| `row` | Row-major PFF storage — efficient for sequence-wise analysis |

#### Data options

| Flag | Description |
|------|-------------|
| `--datatype <type>` | Data type: `universal` (default), `protein`, `nucleotide`, `numeric`, or `vcf` |
| `--het-mode <mode>` | How heterozygous genotypes are weighted (`--datatype vcf` only): `dosage` (default), `presence`, or `alt-dominant`. See [VCF input](#vcf-input) |
| `--cache-dir <dir>` | Cache directory for converted `.pff`/`.pnf` files (default: `pff_cache`) |
| `--min-minor N` | Minimum count of non-major non-indel characters to keep a position (default: 2) |
| `--auto-bit-ct <pct>` | Set `--min-minor` as a percentage of minority class size (overrides `--min-minor`) |
| `--threads N` | Number of worker threads for encoding (default: all available cores) |
| `--dlt` | Use direct lookup table encoder (faster encoding, identical output) |
| `--drop-major-allele` | Exclude the major allele column from FASTA encoding |
| `--minor-column` | Add a per-gene binary column indicating presence of any non-major allele |
| `--tiered-minor-col` | Add per-gene tiered minor allele columns at 0%, 0.1%, 1%, and 5% frequency thresholds (mutually exclusive with `--minor-column`) |
| `--class-bal <mode>` | Class balancing before regression: `up`, `down`, or `weighted` |
| `--feature-normalize <mode>` | Column-wise transform applied to the feature matrix after class balancing and before the solver sees it (numeric and VCF input only): `none`, `center`, `zscore`, or `slep`. Default: `slep` for `--datatype numeric`, otherwise `none` |
| `--dropout <file>` | File listing feature labels to exclude from encoding (one label per line) |

**Class balancing modes:**
- `up` — Upsample minority class to match majority class size
- `down` — Downsample majority class to match minority class size
- `weighted` — Apply inverse class weights; writes `sweights.txt`

**Feature normalization modes (`--feature-normalize`, numeric input only):**
- `none` — No transform; output is byte-identical to disabling normalization
- `center` — Subtract per-column mean (mean-shifting)
- `zscore` — Subtract per-column mean and divide by per-column standard deviation, the standard auto-scaling that brings every column to unit variance
- `slep` — Subtract per-column mean and divide by `sqrt(sum(x²)/N)` per column (RMS denominator computed on the original column before centering); reproduces the SLEP Matlab package's `opts.nFlag=1` normalization from `mcLeastR.m` / `sgLeastR.m` / `overlapping_LeastR.m`

`slep` is the default for `--datatype numeric` because raw numeric feature columns commonly differ in magnitude by orders of magnitude, and L1/L2 regularization on the un-normalized matrix would implicitly weight large-magnitude columns more heavily; pass `--feature-normalize none` to opt out. For non-numeric datatypes the default is `none` and the flag is rejected if set explicitly (centering a one-hot FASTA matrix would silently corrupt categorical semantics).

Statistics are computed in `double` for numerical stability and applied after class balancing on the exact matrix the solver sees. Zero-scale columns (`|scale| ≤ 1e-10`) have their scale clamped to `1.0`. The transform parameters are persisted to `<output_dir>/feature_normalization.txt` (TSV: `Label`, `Mean`, `Scale`, with a `# mode=<mode>` header), once per run — `evaluate` auto-loads this file and replays the identical transform on new data.

#### Feature matrix output

| Flag | Description |
|------|-------------|
| `--write-features <path>` | Write encoded feature matrix to CSV (samples × features) |
| `--write-features-transposed <path>` | Write transposed feature matrix (features × samples) |

#### Regression options

| Flag | Description |
|------|-------------|
| `--method <name>` | Regression method (omit to skip regression and only encode) |
| `--lambda <l1> <l2>` | Single lambda pair: `l1` = site sparsity, `l2` = group sparsity (default: 0.1 0.1) |
| `--lambda-grid <min,max,step> <min,max,step>` | Run Cartesian product of lambda pairs (e.g., `0.1,0.9,0.1 0.0001,0.001,0.0001`) |
| `--lambda-file <path>` | File of lambda pairs, one `l1 l2` per line |
| `--use-logspace` | Replace each `--lambda-grid` linear sweep with a log-spaced grid: `λ_i = vmin × (vmax_eff / vmin)^(i/(N-1))`, where `vmax_eff` is the largest sweep value strictly `< vmax` and `< 1`. Applies only to `--lambda-grid` (not `--lambda` or `--lambda-file`). |
| `--nfolds N` | K-fold cross-validation (N ≥ 2) |
| `--precision fp32\|fp64` | Solver precision (default: `fp32`); `fp64` doubles memory but may improve numerical stability |
| `--max-mem <bytes>` | Abort if estimated feature matrix exceeds this size (default: 8 GB) |
| `--adaptive-sparsification` | On `max-mem` exceeded, automatically split encoding into smaller chunks |
| `--param <key>=<value>` | Pass option to the regression solver (see below) |

> **Lambda range:** every value supplied via `--lambda`, `--lambda-file`, or `--lambda-grid` must lie in the open interval `(0,1)`. Values at or outside the boundary raise an error before any solving begins.

**Available regression methods (`--method`):**

| Method | Description |
|--------|-------------|
| `sg_lasso_logisticr` | Sparse Group Lasso, logistic regression |
| `sg_lasso_leastr` | Sparse Group Lasso, least-squares |
| `olsg_lasso_logisticr` | Overlapping SG Lasso (dual-variable), logistic regression (requires `--param field=<csv>`) |
| `olsg_lasso_leastr` | Overlapping SG Lasso (dual-variable), least-squares (requires `--param field=<csv>`) |
| `ol_sg_lasso_logisticr` | Overlapping SG Lasso (virtual-expansion), logistic regression |
| `ol_sg_lasso_leastr` | Overlapping SG Lasso (virtual-expansion), least-squares |
| `gl_logisticr` | Group Lasso, logistic regression |

**Common `--param` options:**

| Key | Description |
|-----|-------------|
| `intercept=false` | Disable intercept term (default: enabled) |
| `disableEC=1` | Disable equality constraints in SLEP solver |
| `maxIter=N` | Maximum solver iterations |
| `field=<path>` | Group-index CSV for overlapping methods |

#### Group penalty options

Each group's regularization is multiplied by a per-group weight derived from `--group-penalty-type`. When more than one penalty term is requested (via `--initial-gp-value` / `--final-gp-value` / `--gp-step`) the lambda loop is wrapped in a penalty loop, and the per-lambda outputs are nested under `penalty_N/` (see *Output structure* below).

| Flag | Description |
|------|-------------|
| `--group-penalty-type <type>` | One of `std` (default, `sqrt(feature_count)`), `sqrt`, `linear`, or `median`. Default `std` preserves prior behavior. |
| `--initial-gp-value X` | Start of penalty sweep (used by `linear`; default: 1) |
| `--final-gp-value X` | End of penalty sweep (default: 1) |
| `--gp-step X` | Penalty sweep step (default: 1) |

#### Output structure

When regression is run, each lambda pair produces a subdirectory:

```
output_dir/
├── combined.map                      # All encoded features: Position, Label
├── alignment_table.txt               # Gene × sample mapping metadata
├── missing_sequences.txt             # Samples absent from one or more alignments
├── minor_alleles.txt                 # Minor allele labels (only with --minor-column)
├── tiered_minor_alleles.txt          # Minor allele labels with frequencies (only with --tiered-minor-col)
├── feature_normalization.txt        # Per-column Mean+Scale (only with --feature-normalize)
├── lambda_0/
│   ├── weights.txt                   # Feature weights + intercept
│   ├── gss.txt                       # Gene Significance Scores (sum |w| per gene)
│   ├── bss.txt                       # Bit (feature) Significance Scores
│   ├── pss.txt                       # Position Significance Scores (FASTA only)
│   └── eval_gene_predictions.txt     # Per-sample per-gene scores on training data
├── lambda_1/
│   └── ...
├── gss_median.txt                    # Median GSS across all lambdas (multiple lambdas only)
├── bss_median.txt                    # Median BSS across all lambdas (multiple lambdas only)
├── pss_median.txt                    # Median PSS across all lambdas (FASTA only)
└── lambda_list.txt                   # Generated lambda pairs (when using --lambda-grid)
```

When `--group-penalty-type` is anything other than `std`, or when `--initial-gp-value` / `--final-gp-value` / `--gp-step` produce more than one penalty term, the lambda subdirs are nested one level deeper under `penalty_N/`, and the median files are written inside each penalty dir:

```
output_dir/
├── penalty_0/
│   ├── lambda_0/
│   │   └── ...
│   ├── lambda_1/
│   │   └── ...
│   ├── gss_median.txt
│   ├── bss_median.txt
│   └── pss_median.txt
├── penalty_1/
│   └── ...
└── lambda_list.txt
```

When `--nfolds` is used, each `lambda_N/` also contains:
- `cv_predictions.txt` — Cross-validation predictions with TPR/TNR/FPR/FNR metrics
- `weights_fold_k.txt` — Per-fold weights for k = 0…N-1

---

### `evaluate` — Apply a trained model to data

```
myesl2 evaluate <weights.txt> <list.txt> <output_file> [options]
```

Apply a trained model to a set of species and write per-species predictions.
Optionally compare those predictions to known labels for accuracy assessment.

**Required positional arguments** (all three are required; the first two are inputs, the third is an output):

| Argument | Direction | Description |
|----------|-----------|-------------|
| `weights.txt` | **Input** | Trained model weights from a `train` run (e.g. `<train_out>/lambda_0/weights.txt`) |
| `list.txt` | **Input** | List of alignment or numeric data files for the species you want to predict (same format as training's `list.txt`) |
| `output_file` | **Output** | Path where the per-species predictions TSV will be written (you choose the name) |

**Optional flags:**

| Flag | Direction | Description |
|------|-----------|-------------|
| `--hypothesis <file>` | **Input** (optional) | Known class labels for the species in `list.txt`. **Only required if you want accuracy metrics.** When supplied, evaluate also prints TP/TN/FP/FN, TPR/TNR/FPR/FNR, accuracy, and AUC, and writes them to `process_log.txt`. Omit to run prediction-only with no accuracy assessment. |
| `--cache-dir <dir>` | | Cache directory (default: `pff_cache`) |
| `--datatype <type>` | | Must match the datatype used during training |
| `--het-mode <mode>` | | `--datatype vcf` only. Omit to read it from `vcf_encoding.txt` beside `weights.txt`; must match training |
| `--threads N` | | Worker threads (default: all cores) |
| `--no-visualize` | | Skip automatic SVG heatmap generation |
| `--minor-alleles <file>` | Input | Path to `minor_alleles.txt` from training (auto-detected from weights directory if omitted) |
| `--tiered-minor-alleles <file>` | Input | Path to `tiered_minor_alleles.txt` from training (auto-detected if omitted) |
| `--gene-limit N` | | Maximum genes displayed in auto-generated SVG (default: 20) |
| `--species-limit N` | | Maximum species displayed in auto-generated SVG (default: 20) |

If the model was trained with `--feature-normalize`, `evaluate` automatically loads `feature_normalization.txt` (searched first as a sibling of `weights.txt`, then in the parent directory for `lambda_N/weights.txt` layouts) and applies the same column-wise transform to new data before scoring. No CLI flag is required. Models trained without normalization have no such file; the lookup is silently skipped and evaluate runs as before.

**Outputs** (all written next to `<output_file>` in the same directory):
- `<output_file>` — TSV with columns: `SequenceID`, `PredictedValue`, and (only when `--hypothesis` is supplied) `TrueValue`
- `eval_gene_predictions.txt` — Per-sample per-gene scores used by `drphylo` and `aim`
- `eval_SPS_SPP.txt` — Species-level prediction summary
- `<basename>.svg` — Heatmap visualization of per-gene contributions (auto-generated unless `--no-visualize`)
- `process_log.txt` — Run metadata; includes classification metrics when `--hypothesis` was supplied

---

### `visualize` — Generate an SVG heatmap from gene predictions

```
myesl2 visualize <gene_predictions.txt> <output.svg> [options]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `gene_predictions.txt` | Gene predictions table from `train`, `evaluate`, or `drphylo` |
| `output.svg` | Output SVG file path |

**Options:**

| Flag | Description |
|------|-------------|
| `--gene-limit N` | Maximum genes to display, sorted by relevance (default: 20) |
| `--species-limit N` | Maximum species to display, sorted by classification probability (default: 20) |
| `--ssq-threshold <value>` | Exclude genes below this sum-of-squares threshold (default: 0.0) |
| `--m-grid` | DrPhylo mode: show only positive-class samples |

---

### `drphylo` — Per-clade analysis

Runs the training pipeline separately for each clade in a phylogenetic tree (or each group in a hypothesis file), enabling clade-specific feature discovery.

**Direct mode (hypothesis file):**
```
myesl2 drphylo <list.txt> <hypothesis.txt> <output_dir> [options]
```

**Tree mode:**
```
myesl2 drphylo <list.txt> <output_dir> --tree <tree.nwk> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--tree <tree.nwk>` | Newick tree file (activates tree mode) |
| `--clade-list <file>` | File of clade node IDs to analyze, one per line (tree mode only) |
| `--gen-clade-list <lower,upper>` | Auto-generate clade list by leaf count range, e.g., `3,10` (tree mode only) |
| `--class-bal <mode>` | Balancing strategy: `phylo` (default, tree mode only), `up`, `down`, or `weighted` |
| `--datatype <type>` | Data type (default: `universal`) |
| `--het-mode <mode>` | Heterozygote weighting for `--datatype vcf` (see train) |
| `--method <name>` | Regression method (default: `sg_lasso_logisticr`) |
| `--min-groups N` | Minimum number of genes required in the model |
| `--grid-rmse-cutoff <value>` | Maximum RMSE threshold for lambda filtering (default: 100.0) |
| `--grid-acc-cutoff <value>` | Minimum accuracy threshold for lambda filtering (default: 0.0) |
| `--gene-limit N` | Maximum genes displayed in aggregated eval.svg (default: 20) |
| `--species-limit N` | Maximum species displayed in aggregated eval.svg (default: 20) |

All `--lambda`, `--lambda-grid`, `--lambda-file`, `--use-logspace`, `--param`, `--cache-dir`, `--threads`, group-penalty flags (`--group-penalty-type`, `--initial-gp-value`, `--final-gp-value`, `--gp-step`), encoding options (`--auto-bit-ct`, `--drop-major-allele`, `--minor-column`, `--tiered-minor-col`, `--max-mem`), and `--feature-normalize` from `train` are also accepted. Lambda values must be in `(0,1)`.

**Outputs:**

Per-clade subdirectories in `output_dir/`, each containing aggregated results across lambdas that meet the RMSE/accuracy thresholds:
- `eval_gene_predictions.txt` — Median gene scores across qualifying lambdas
- `eval.txt` — Predictions with hypothesis comparison

---

### `aim` — Iterative feature dropout (AIM analysis)

Iteratively trains the model, identifies important features, excludes them, and repeats to discover successive layers of predictive features.

```
myesl2 aim <list.txt> <hypothesis.txt> <output_dir> [options]
```

**AIM-specific options:**

| Flag | Description |
|------|-------------|
| `--aim-acc-cutoff <value>` | Accuracy threshold to stop iteration (default: 0.9) |
| `--aim-max-iter N` | Maximum number of iterations (default: 10) |
| `--aim-max-ft N` | Maximum features to accumulate in dropout list (default: 1000) |
| `--aim-window N` | Top-N features considered per iteration for dropout (default: 100) |

All `train` options are accepted, including `--use-logspace`, `--feature-normalize` (numeric input only), and the group-penalty flags (`--group-penalty-type`, `--initial-gp-value`, `--final-gp-value`, `--gp-step`). Lambda values must be in `(0,1)`. Defaults applied if not specified:
- Method: `sg_lasso_logisticr`
- Lambdas: `--lambda-grid 0.1,0.9,0.1 0.0001,0.0002,0.0001`

**Outputs:**
```
output_dir/
├── aim_iter_0/
│   ├── dropout_0.txt        # Features excluded in this iteration
│   ├── weights.txt
│   └── eval.txt
├── aim_iter_1/
│   └── ...
└── aim_accuracy.svg         # TPR/TNR/Accuracy curves across iterations
```

Iteration stops when the accuracy threshold is met or `--aim-max-iter` is reached.

---

### `psc` — Paired Species Contrast analysis

Detects molecular convergence across species using paired species contrasts. Takes FASTA alignments directly (no PFF conversion), applies gap cancellation, encodes features in memory, runs sparse group lasso across multiple species contrast combinations, and aggregates gene rankings.

```
myesl2 psc <alignments_dir> <output_dir> [options]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `alignments_dir` | Directory containing FASTA alignment files (.fas/.fasta/.fa/.faa) |
| `output_dir` | Output directory (created if needed) |
| `--output-base-name <name>` | Base name for output files |

**Input options:**

| Flag | Description |
|------|-------------|
| `--alignments-list <file>` | List file specifying overlapping groups of alignments — one group per line, comma-separated paths relative to `alignments_dir`. Without this flag, `alignments_dir` is enumerated and each file becomes its own group. Any group with more than one entry requires `--method olsg_lasso_logisticr`, `olsg_lasso_leastr`, `ol_sg_lasso_logisticr`, or `ol_sg_lasso_leastr`; otherwise the run errors out. The resolved field array is written to `<output_dir>/<base>_field.csv`. |

**Species contrast source (exactly one required):**

| Flag | Description |
|------|-------------|
| `--species-groups <file>` | File with alternating convergent/control species lines (one per line, comma-separated for alternates) |
| `--response-file <file>` | Single response matrix (species,value per line) |
| `--response-dir <dir>` | Directory of response matrices (one combo per file) |

**Lambda grid options:**

| Flag | Description |
|------|-------------|
| `--initial-lambda1 X` | Start of lambda1 range (default: 0.01) |
| `--final-lambda1 X` | End of lambda1 range (default: 0.99) |
| `--initial-lambda2 X` | Start of lambda2 range (default: 0.01) |
| `--final-lambda2 X` | End of lambda2 range (default: 0.99) |
| `--lambda-step X` | Step size for linear grid (default: 0.05) |
| `--use-logspace` | Use log10-spaced grid instead of linear |
| `--num-log-points N` | Points per dimension in logspace (default: 20) |

**Group penalty options:**

| Flag | Description |
|------|-------------|
| `--group-penalty-type <type>` | `median` (default), `linear`, `sqrt`, or `std` |
| `--initial-gp-value X` | Start of penalty sweep (for `linear` type) |
| `--final-gp-value X` | End of penalty sweep |
| `--gp-step X` | Penalty sweep step |
| `--use-default-gp` | Use sqrt(n_features) regardless of type |

**Gap cancellation options:**

| Flag | Description |
|------|-------------|
| `--use-uncanceled-alignments` | Skip gap cancellation entirely |
| `--cancel-only-partner` | Only cancel the affected pair (not all species) at gapped positions |
| `--cancel-tri-allelic` | Cancel positions with 3+ unique residues (4-species combos only) |
| `--nix-full-deletions` | Skip genes where any combo species is missing |
| `--outgroup-species <name>` | Cancel positions where control species differ from outgroup |
| `--min-pairs N` | Minimum intact pairs required (default: 2) |

**Solver options:**

| Flag | Description |
|------|-------------|
| `--method <name>` | Regression method (default: `sg_lasso_logisticr`) |
| `--precision fp32\|fp64` | Arithmetic precision (default: fp32) |
| `--maxiter N` | Max solver iterations (default: 100) |
| `--threads N` | Worker threads |
| `--param <key>=<value>` | Pass option to solver |

**Prediction/output options:**

| Flag | Description |
|------|-------------|
| `--prediction-alignments-dir <dir>` | Separate alignments for prediction species |
| `--species-pheno-path <file>` | Species phenotype file for prediction evaluation |
| `--no-pred-output` | Skip species prediction output |
| `--no-genes-output` | Skip gene ranks output |
| `--show-selected-sites` | Output selected sites CSV |
| `--dump-weights` | Write the full feature-weight vector from every solver call to disk (see *PSC output files* below). |
| `--top-rank-frac X` | Fraction for "top ranked" threshold (default: 0.01) |
| `--limited-genes-list <file>` | Only process genes in this list |

**Null model options:**

| Flag | Description |
|------|-------------|
| `--make-null-models` | Generate response-flipped null models |
| `--make-pair-randomized-null-models` | Randomize pairs at each position |
| `--num-randomized-alignments N` | Number of randomized replicates (default: 10) |

**PSC output files:**

```
output_dir/
├── <base>_gene_ranks.csv           # Gene rankings (multimatrix: num_combos_ranked, etc.)
├── <base>_species_predictions.csv  # SPS for prediction species
├── <base>_selected_sites.csv       # Selected alignment positions (if --show-selected-sites)
├── <base>_field.csv                # Group-index CSV (only when --alignments-list is used)
└── [combo_N/][penalty_P/]lambda_L/weights.tsv
                                    # Per-run feature weights (only with --dump-weights).
                                    # combo_N/ and penalty_P/ layers are present only when
                                    # there is more than one combo or more than one penalty
                                    # term, mirroring train mode's nesting.
```

**Species groups file format:**
```
ConvergentSpecies1              # Line 1: convergent member(s) of pair 1
ControlSpecies1                 # Line 2: control member(s) of pair 1
ConvergentSpecies2,AltConv2     # Line 3: convergent member(s) of pair 2 (with alternate)
ControlSpecies2                 # Line 4: control member(s) of pair 2
```
Must have an even number of non-empty lines. Multiple species per line (comma-separated) generates Cartesian product combinations.

**Example:**
```bash
# Basic PSC with species groups
myesl2 psc alignments/ output/ \
    --species-groups species_pairs.txt \
    --output-base-name analysis \
    --use-logspace --num-log-points 10 \
    --method sg_lasso_logisticr
```

---

### `info` — Display PFF file metadata

```
myesl2 info <file.pff>
```

Prints metadata from a cached PFF (Parsed FASTA File) binary: number of sequences, alignment length, storage orientation, and sequence IDs.

---

### `taskfile` — Run any task from a YAML control file

```
myesl2 taskfile <control.yaml> [overrides...]
```

Drives any MyESL2 subcommand from a YAML control file instead of a long argv. The YAML must declare a `task_type:` key naming one of `train`, `evaluate`, `drphylo`, `aim`, `psc`, `visualize`, `encode-sizes`, or `info`; every other key maps to a CLI flag with the leading `--` stripped and hyphens preserved (so `feature-normalize: zscore` is the YAML form of `--feature-normalize zscore`).

**Key conventions:**

| YAML form | Maps to |
|-----------|---------|
| `list_path:`, `hypothesis_path:`, `output_dir:`, `alignments_dir:`, etc. | Positional arguments (named per task) |
| `feature-normalize: zscore` | `--feature-normalize zscore` |
| `class-bal: up` | `--class-bal up` |
| `lambda: [0.3, 0.3]` | `--lambda 0.3 0.3` (YAML sequence — one scalar per CLI token) |
| `lambda-grid: ["0.1,0.3,0.1", "0.2,0.4,0.1"]` | `--lambda-grid 0.1,0.3,0.1 0.2,0.4,0.1` (sequence of two comma-triple strings; each string is a `min,max,step` spec) |
| `use-logspace: true` | `--use-logspace` (boolean flag) |
| `param: { intercept: false, maxIter: 100 }` | Repeated `--param key=value` |

**CLI overrides:** any flag passed after the YAML path overrides the matching YAML value and prints `[Warning] taskfile override: key=yaml -> cli` to stderr so the conflict is visible.

**Per-task-type whitelist:** keys that don't apply to the selected `task_type` are rejected before any work begins (e.g. `feature-normalize: zscore` under `task_type: psc` is an error). Validation, defaults, and error paths are identical to direct CLI invocation — `taskfile` synthesizes argv from the merged keys and dispatches into the same per-command handlers.

**Example (train with numeric input, z-score normalization, and a lambda grid):**

```yaml
# train.yaml
task_type: train
list_path: numeric_files.txt
hypothesis_path: labels.txt
output_dir: results/
datatype: numeric
method: sg_lasso_logisticr

# --lambda-grid 0.1,0.3,0.1 0.2,0.4,0.1 (a 3 x 3 = 9-cell Cartesian sweep).
# Each entry is a "min,max,step" comma-triple; quotes keep the comma inside
# one YAML scalar instead of letting it parse as a nested sequence.
lambda-grid:
  - "0.1,0.3,0.1"
  - "0.2,0.4,0.1"

feature-normalize: zscore
threads: 1
cache-dir: pnf_cache/
```

```bash
myesl2 taskfile train.yaml

# CLI override (prints [Warning] taskfile override: threads=1 -> 4):
myesl2 taskfile train.yaml --threads 4
```

To run a **single** lambda pair instead of a grid, replace the `lambda-grid:` block with `lambda: [0.3, 0.3]` (mapping to `--lambda 0.3 0.3`).

---

## Output File Formats

### `weights.txt`
Tab-separated feature weights from regression:
```
Label               Weight
Intercept           0.1234
gene1_42_A          0.5678
gene1_42_G         -0.2345
gene2_100_T         0.0012
```

### `gss.txt` — Gene Significance Scores
Sum of absolute weights per gene:
```
GeneName    Score
gene1       0.8023
gene2       0.0012
```

### `pss.txt` — Position Significance Scores (FASTA only)
Sum of absolute weights per alignment position:
```
GeneName_Position    Score
gene1_42             0.7891
gene1_100            0.0132
```

### `eval_gene_predictions.txt`
Per-sample per-gene score matrix used for visualization and downstream analysis:
```
SeqID       Response    Prediction    Intercept    gene1    gene2    ...
species_A   1.0         0.82          0.10         0.50    -0.20
species_B  -1.0        -0.75          0.10        -0.40     0.05
```

---

## VCF input

`--datatype vcf` reads VCF variant calls directly, without an intermediate FASTA
conversion. Each file in the list is one group ("gene"); sample IDs come from the
`#CHROM` header line and are matched against the hypothesis file by name.

Every (site, allele) pair carried by at least one sample becomes a feature column
labelled `{stem}_{chrom}:{pos}_{allele}`. Multiallelic sites and indels are
supported -- the allele is the literal REF/ALT string, so `AT` and `<DEL>` are
valid alleles. Plain `.vcf` and gzip/bgzip `.vcf.gz` are both accepted (the
latter requires a zlib-enabled build).

### Heterozygote encoding (`--het-mode`)

Because a genotype can carry two different alleles, a heterozygote needs a rule
for how much weight each observed allele receives. Unlike a FASTA intermediate --
which must collapse each call to a single character and therefore loses
heterozygosity entirely -- the feature matrix holds fractional values, so a
heterozygote can contribute partial evidence to both of its alleles.

For a diploid site with `REF=A`, `ALT=G`:

| Genotype | `dosage` (default) | `presence` | `alt-dominant` |
|----------|--------------------|------------|----------------|
| `0/0`    | A=1.0              | A=1.0      | A=1.0          |
| `0/1`    | A=0.5, G=0.5       | A=1.0, G=1.0 | A=0.0, G=1.0 |
| `1/1`    | G=1.0              | G=1.0      | G=1.0          |
| `1/2`    | G=0.5, T=0.5       | G=1.0, T=1.0 | G=1.0, T=1.0 |
| `./.`    | all 0              | all 0      | all 0          |

- `dosage` weights each allele by *copies / called copies*, generalising to
  haploid (`0` -> A=1.0) and polyploid calls without special-casing.
- `presence` gives full weight to every allele present in the genotype.
- `alt-dominant` treats a heterozygote as homozygous-alternate: the reference
  allele gets 0.0 and each observed alternate 1.0.

Homozygous calls encode identically under all three modes. Phased (`|`) and
unphased (`/`) separators are equivalent. Missing calls (`./.`) are all-zero,
matching the convention used for a sample absent from a gene. A partial call
(`./1`) counts only its called copies, so it encodes as G=1.0.

The mode is recorded in `vcf_encoding.txt` in the output directory; `evaluate`
reads it automatically so a model is re-encoded the same way it was trained.
Passing a different `--het-mode` to `evaluate` silently changes every score, so
override it only deliberately.

### Filtering and outputs

`--min-minor` and `--drop-major-allele` apply to VCF the same way they do to
FASTA, using per-allele carrier counts over the samples selected by the
hypothesis file: monomorphic sites are dropped, a site whose non-major carrier
total is below `--min-minor` is dropped, and individual allele columns below
`--min-minor` are dropped. `--minor-column` and `--tiered-minor-col` are
FASTA-only and are rejected for VCF input rather than silently ignored.

VCF input produces per-site scores (`pss.txt` / `pss_median.txt`) in addition to
per-gene scores, since variant positions have real genomic meaning.

### Limitations

- Multiallelic sites split across several records (one row per ALT, as produced
  by `bcftools norm -m-`) are rejected, because the per-record genotypes cannot
  be recombined unambiguously. Merge them first with `bcftools norm -m+any`.
- The feature matrix is dense, so a whole-genome VCF expands to a very large
  column count. Split large VCFs by region and list them as separate files.
- INFO/QUAL/FILTER-based site filtering is not performed; pre-filter with
  `bcftools view` if needed.

---

## Data Types

| Type | Description |
|------|-------------|
| `universal` | Any characters; treats 0/1 as presence/absence (default) |
| `protein` | IUPAC amino acids (case-insensitive); X, -, ? treated as missing |
| `nucleotide` | DNA/RNA: A/T/C/G/U (case-insensitive); all other characters treated as missing |
| `numeric` | Whitespace-delimited tabular data with float features |
| `vcf` | VCF variant calls (`.vcf` / `.vcf.gz`); genotypes are one-hot encoded per allele with configurable heterozygote weighting (see [VCF input](#vcf-input)) |

---

## Examples

### Basic training run
```bash
myesl2 train Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt results/ \
    --method sg_lasso_logisticr --lambda 0.1 0.1
```

### Lambda grid search with cross-validation
```bash
myesl2 train Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt results/ \
    --method sg_lasso_logisticr_leastr \
    --lambda-grid 0.1,0.9,0.1 0.0001,0.001,0.0001 \
    --nfolds 5 \
    --threads 8
```

### Encode only (no regression), write feature matrix
```bash
myesl2 train Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt results/ \
    --write-features features.csv
```

### Evaluate on training data
```bash
myesl2 evaluate results/lambda_0/weights.txt Fungi_data/aln.txt predictions.txt \
    --hypothesis Fungi_data/A_B_Hyp.txt
```

### Numeric data
```bash
myesl2 train numeric_files.txt labels.txt results/ \
    --datatype numeric \
    --method sg_lasso_logisticr \
    --lambda 0.05 0.1
```

### DrPhylo clade analysis
```bash
myesl2 drphylo Fungi_data/aln.txt Fungi_data/tree.nwk drphylo_out/ \
    --gen-clade-list 3,15 \
    --method sg_lasso_logisticr \
    --lambda-grid 0.1,0.5,0.1 0.0001,0.001,0.0001
```

### AIM iterative analysis
```bash
myesl2 aim Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt aim_out/ \
    --aim-acc-cutoff 0.85 \
    --aim-max-iter 15
```
