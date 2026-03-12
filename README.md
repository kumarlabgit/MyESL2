# MyESL2

**My Evolutionary Sparse Learning 2** — a C++20 tool for sparse regression-based classification on genomic data. It supports FASTA alignment inputs (nucleotide/protein) and tabular numeric inputs, designed for comparative genomic analysis across any set of organisms.

## Overview

MyESL2 uses Sparse Group Lasso regression to identify genomic features (alignment positions, genes) that are predictive of a binary phenotype. It operates on multiple sequence alignments or numeric feature matrices and produces per-feature weights, gene significance scores, and visualizations.

## Quick Start

```bash
# Train a model on FASTA alignments
myesl2 train alignments.txt labels.txt output_dir/

# Evaluate a trained model
myesl2 evaluate output_dir/lambda_0/weights.txt alignments.txt predictions.txt --hypothesis labels.txt

# Visualize results
myesl2 visualize eval_gene_predictions.txt heatmap.svg
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
| `--datatype <type>` | Data type: `universal` (default), `protein`, `nucleotide`, or `numeric` |
| `--cache-dir <dir>` | Cache directory for converted `.pff`/`.pnf` files (default: `pff_cache`) |
| `--min-minor N` | Minimum count of non-major non-indel characters to keep a position (default: 2) |
| `--auto-bit-ct <pct>` | Set `--min-minor` as a percentage of minority class size (overrides `--min-minor`) |
| `--threads N` | Number of worker threads for encoding (default: all available cores) |
| `--dlt` | Use direct lookup table encoder (faster encoding, identical output) |
| `--drop-major-allele` | Exclude the major allele column from FASTA encoding |
| `--class-bal <mode>` | Class balancing before regression: `up`, `down`, or `weighted` |
| `--dropout <file>` | File listing feature labels to exclude from encoding (one label per line) |

**Class balancing modes:**
- `up` — Upsample minority class to match majority class size
- `down` — Downsample majority class to match minority class size
- `weighted` — Apply inverse class weights; writes `sweights.txt`

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
| `--nfolds N` | K-fold cross-validation (N ≥ 2) |
| `--param <key>=<value>` | Pass option to the regression solver (see below) |

**Available regression methods (`--method`):**

| Method | Description |
|--------|-------------|
| `sg_lasso` | Sparse Group Lasso (float precision) |
| `sg_lasso_leastr` | Sparse Group Lasso with least-squares refinement (double precision) |
| `olsg_lasso_leastr` | Overlapping SG Lasso, least-squares (requires `--param field=<csv>`) |
| `olsg_lasso_logisticr` | Overlapping SG Lasso, logistic regression (requires `--param field=<csv>`) |

**Common `--param` options:**

| Key | Description |
|-----|-------------|
| `intercept=false` | Disable intercept term (default: enabled) |
| `disableEC=1` | Disable equality constraints in SLEP solver |
| `maxIter=N` | Maximum solver iterations |
| `field=<path>` | Group-index CSV for overlapping methods |

#### Output structure

When regression is run, each lambda pair produces a subdirectory:

```
output_dir/
├── combined.map                      # All encoded features: Position, Label
├── alignment_table.txt               # Gene × sample mapping metadata
├── missing_sequences.txt             # Samples absent from one or more alignments
├── lambda_0/
│   ├── weights.txt                   # Feature weights + intercept
│   ├── gss.txt                       # Gene Significance Scores (sum |w| per gene)
│   ├── pss.txt                       # Position Significance Scores (FASTA only)
│   └── eval_gene_predictions.txt     # Per-sample per-gene scores on training data
├── lambda_1/
│   └── ...
├── gss_median.txt                    # Median GSS across all lambdas (multiple lambdas only)
├── pss_median.txt                    # Median PSS across all lambdas (FASTA only)
└── lambda_list.txt                   # Generated lambda pairs (when using --lambda-grid)
```

When `--nfolds` is used, each `lambda_N/` also contains:
- `cv_predictions.txt` — Cross-validation predictions with TPR/TNR/FPR/FNR metrics
- `weights_fold_k.txt` — Per-fold weights for k = 0…N-1

---

### `evaluate` — Apply a trained model to data

```
myesl2 evaluate <weights.txt> <list.txt> <output_file> [options]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `weights.txt` | Weights file from a `train` run |
| `list.txt` | Alignment or numeric data file list (same format as training) |
| `output_file` | Path for the output predictions file |

**Options:**

| Flag | Description |
|------|-------------|
| `--hypothesis <file>` | Compare predictions to known labels; enables TPR/TNR/FPR/FNR metrics |
| `--cache-dir <dir>` | Cache directory (default: `pff_cache`) |
| `--datatype <type>` | Must match the datatype used during training |
| `--threads N` | Worker threads (default: all cores) |
| `--no-visualize` | Skip automatic SVG heatmap generation |

**Outputs:**
- `<output_file>` — TSV with columns: `SeqID`, `Prediction`, `Probability`, `ClassPrediction` (plus accuracy metrics if `--hypothesis` provided)
- `<output_file>.svg` — Heatmap visualization of per-gene contributions (auto-generated unless `--no-visualize`)
- `eval_gene_predictions.txt` — Per-sample per-gene scores used by `drphylo` and `aim`

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
| `--gene-limit N` | Maximum genes to display, sorted by relevance (default: 100) |
| `--species-limit N` | Maximum species to display, sorted by classification probability (default: 100) |
| `--ssq-threshold <value>` | Exclude genes below this sum-of-squares threshold (default: 0.0) |
| `--m-grid` | DrPhylo mode: show only positive-class samples |

---

### `drphylo` — Per-clade analysis

Runs the training pipeline separately for each clade in a phylogenetic tree (or each group in a hypothesis file), enabling clade-specific feature discovery.

**Tree mode:**
```
myesl2 drphylo <list.txt> <tree.nwk> <output_dir> [options]
```

**Hypothesis mode:**
```
myesl2 drphylo <list.txt> <output_dir> --hypothesis <file> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--hypothesis <file>` | Direct hypothesis file (activates hypothesis mode instead of tree mode) |
| `--clade-list <file>` | File of clade node IDs to analyze, one per line (tree mode only) |
| `--gen-clade-list <lower,upper>` | Auto-generate clade list by leaf count range, e.g., `3,10` (tree mode only) |
| `--class-bal <mode>` | Balancing strategy: `phylo` (default, tree mode only), `up`, `down`, or `weighted` |
| `--datatype <type>` | Data type (default: `universal`) |
| `--method <name>` | Regression method (default: `sg_lasso`) |
| `--min-groups N` | Minimum number of genes required in the model |
| `--grid-rmse-cutoff <value>` | Maximum RMSE threshold for lambda filtering (default: 100.0) |
| `--grid-acc-cutoff <value>` | Minimum accuracy threshold for lambda filtering (default: 0.0) |

All `--lambda`, `--lambda-grid`, `--lambda-file`, `--param`, `--cache-dir`, and `--threads` options from `train` are also accepted and passed through to each clade run.

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

All `train` options are accepted. Defaults applied if not specified:
- Method: `sg_lasso`
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

### `info` — Display PFF file metadata

```
myesl2 info <file.pff>
```

Prints metadata from a cached PFF (Parsed FASTA File) binary: number of sequences, alignment length, storage orientation, and sequence IDs.

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

## Data Types

| Type | Description |
|------|-------------|
| `universal` | Any characters; treats 0/1 as presence/absence (default) |
| `protein` | IUPAC amino acids (case-insensitive); X, -, ? treated as missing |
| `nucleotide` | DNA/RNA: A/T/C/G/U (case-insensitive); all other characters treated as missing |
| `numeric` | Whitespace-delimited tabular data with float features |

---

## Examples

### Basic training run
```bash
myesl2 train Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt results/ \
    --method sg_lasso --lambda 0.1 0.1
```

### Lambda grid search with cross-validation
```bash
myesl2 train Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt results/ \
    --method sg_lasso_leastr \
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
    --method sg_lasso \
    --lambda 0.05 0.1
```

### DrPhylo clade analysis
```bash
myesl2 drphylo Fungi_data/aln.txt Fungi_data/tree.nwk drphylo_out/ \
    --gen-clade-list 3,15 \
    --method sg_lasso \
    --lambda-grid 0.1,0.5,0.1 0.0001,0.001,0.0001
```

### AIM iterative analysis
```bash
myesl2 aim Fungi_data/aln.txt Fungi_data/A_B_Hyp.txt aim_out/ \
    --aim-acc-cutoff 0.85 \
    --aim-max-iter 15
```
