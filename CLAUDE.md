# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MyESL2 (My Evolutionary Sparse Learning 2) is a C++20 tool for sparse regression-based classification on genomic data. It supports FASTA alignment inputs (nucleotide/protein) and tabular numeric inputs, and is designed for comparative genomic analysis across any set of organisms.

## Building the Project

### Windows with Visual Studio

First, set up the Visual Studio environment (required for CMake):

```cmd
"D:\programs\Microsoft Visual Studio\VC\Auxiliary\Build\vcvars64.bat"
```

Then build:

```cmd
mkdir build
cd build
cmake ..
cmake --build .
```

Or use the Visual Studio generator:

```cmd
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
```

The executable will be in `build\bin\myesl2.exe`.

### Running

```cmd
.\build\bin\myesl2.exe
```

Or from the build directory:

```cmd
cd build\bin
myesl2.exe
```

## Project Structure

```
.
├── src/              Source files
├── include/          Header files
├── Fungi_data/       Example data (fungal FASTA alignments and labels)
├── build/            Build output (generated, in .gitignore)
├── CMakeLists.txt    CMake build configuration
└── CLAUDE.md         This file
```

## Data Structure

### Fungi_data/

The main data directory containing:

- **A_B_Hyp.txt**: Species labels file mapping 50 fungal species (example dataset) to binary classifications (-1 or 1). First 40 species are labeled -1, last 10 are labeled 1.

- **aln.txt**: List of paths to all ~960 BUSCO protein alignment files

- **aln_200.txt**: Subset listing 200 alignment files (likely for testing or quick analysis)

- **aln/**: Directory containing ~960 BUSCO (Benchmarking Universal Single-Copy Orthologs) protein multiple sequence alignment files in FASTA format. Each file:
  - Named as `BUSCOfEOG[ID].domain.aa.aln.trimmed.fasta`
  - Contains aligned protein sequences from multiple species (fungal example dataset)
  - Sequences are conserved single-copy orthologs across the species
  - Pre-aligned and trimmed for quality

## Working with the Data

The example alignment files are protein sequences in FASTA format where each species has one sequence per file. Species names match those in A_B_Hyp.txt. MyESL2 supports any genomic dataset in this format (not limited to fungi).

## Modes

MyESL2 has six main commands: `train`, `evaluate`, `drphylo`, `aim`, `psc`, and `visualize`.

### PSC (Paired Species Contrast)

The `psc` mode detects molecular convergence across species using paired species contrasts. It takes FASTA alignments directly (not PFF) and performs gap cancellation, in-memory feature encoding, sparse group lasso training, and cross-combo gene ranking aggregation.

Key files:
- `include/pipeline_psc.hpp` - Types and `run_psc()` declaration
- `src/pipeline_psc.cpp` - Full PSC pipeline implementation
- `include/encoder.hpp` - `encode_raw_sequences()` for in-memory encoding (used by PSC)

PSC reuses the existing solver (`regression::createRegressionAnalysis`) and newick tree parser (`newick.hpp`) without modifications.

### VCF input (`--datatype vcf`)

VCF genotypes are read directly into the feature matrix, skipping any FASTA
intermediate — which matters because a FASTA character cannot represent a
heterozygote, while the `arma::fmat` feature matrix holds fractional values.

Architecture: VCF reuses the **numeric** (float-matrix) path, not the FASTA path.

- `src/vcf_parser.cpp` / `include/vcf_parser.hpp` — parses `.vcf`/`.vcf.gz`
  (zlib `gzopen`, which also reads uncompressed input), applies the het-mode
  dosage rule, and writes a `.vnf` cache.
- `.vnf` uses the **PNF on-disk layout** (`include/pnf_format.hpp`) so
  `numeric::read_pnf_metadata` / `read_pnf_data` are reused verbatim; the
  distinct extension only prevents cache collisions with `.pnf`. Both formats
  are written through the shared `numeric::write_pnf()`.
- `pipeline_encode.cpp` — the numeric branch condition is
  `datatype == "numeric" || datatype == "vcf"`; VCF adds an allele-column filter
  (carrier counts, monomorphic sites, `--drop-major-allele`) that runs *after*
  sample selection, mirroring `encoder.cpp:364-387`.
- `evaluate` works because its numeric branch resolves features **by label name**
  (`pipeline_evaluate.cpp:210-228`), unlike the FASTA branch's single-char allele
  comparison — which is also why multi-character alleles (indels, `<DEL>`) work.

Feature labels are `{stem}_{chrom}:{pos}_{allele}`. **Do not parse these with the
FASTA `rfind('_')`-twice parser** — chromosome names contain underscores
(`chr1_KI270706v1_random`) and it would mis-split the stem. Use the
longest-stem-prefix match plus a split at the final `_` (`split_vcf_label` in
`pipeline_train.cpp`). Position tokens are sorted via `pss_pos_key()`, since
`chr1:100` is not parseable by `stoul`.

`--het-mode dosage|presence|alt-dominant` (default `dosage`) controls heterozygote
weighting. It is persisted in three places, all of which must stay in sync:
`preprocess_config` (so encode agrees), the `.vnf` header's `het_mode` key (so a
stale cache is re-converted), and `vcf_encoding.txt` beside the model (so evaluate
re-encodes identically). `pipeline_utils::input_stem()` strips a trailing `.gz`
so `sample.vcf.gz` and `sample.vcf` share the gene name `sample`.

Not supported for VCF: `--minor-column` / `--tiered-minor-col` (rejected with an
error), `psc` mode, and multiallelic sites split across records (rejected;
merge with `bcftools norm -m+any`).

### Encoding Options

- `--minor-column`: Adds a per-gene binary summary column (`{stem}_minor`) indicating whether each species has any non-major allele in that gene. Writes `minor_alleles.txt` listing which feature labels are minor alleles, used by evaluate to reconstitute the column for new species.
- `--tiered-minor-col`: Adds up to 4 per-gene binary columns (`{stem}_tminor_Xpct`) at frequency thresholds 0%, 0.1%, 1%, and 5%. A tier column is 1 if the species has any minor allele at any position in the gene whose per-position frequency exceeds the threshold. Writes `tiered_minor_alleles.txt` (stem, pos, allele, freq). Mutually exclusive with `--minor-column`. Evaluate uses `--tiered-minor-alleles <file>` to locate the auxiliary file.
- `--drop-major-allele`: Excludes the most frequent allele column from FASTA encoding.
- `--auto-bit-ct X`: Sets `min_minor = ceil(X% × minority_class_size)`.

### Grid-Loop Threading (train / drphylo / aim / adaptive)

`--threads N` parallelises the lambda grid solver loop in addition to preprocessing/encoding (previously only the preprocess phase was threaded). The `--min-groups` skip-ahead ratchet is disabled while the parallel loop runs (the ratchet relies on sequential gene-count ordering). Pass `--prune-skipped-lambda` to replay the skip-ahead logic after the parallel loop finishes, deleting the contents of `lambda_<idx>/` directories that a single-threaded run would have skipped (the dirs are preserved as sentinels for drphylo aggregation). Grid-loop parallelism is disabled when `--nfolds > 0`. When running `--threads > 1`, set `OPENBLAS_NUM_THREADS=1` to avoid nested parallelism between worker threads and OpenBLAS kernels. `psc` has its own independent grid-loop threading (unchanged by this).

### Lambda grid (train / drphylo / aim / adaptive)

- Every lambda value reaching these paths must lie in the open interval `(0,1)`. Values from `--lambda`, `--lambda-file`, or any `--lambda-grid` sweep that fall outside (including the boundary values `0` and `1`) raise an error before any solving happens.
- `--use-logspace` replaces a `--lambda-grid` linear sweep with `λ_i = vmin × (vmax_eff/vmin)^(i/(N-1))`, where `vmax_eff` is the largest sweep value strictly `< vmax` and `< 1` (i.e. anchored to the largest *valid* point the linear sweep would have produced, not to `vmax` itself). Applies only to `--lambda-grid` — `--lambda-file` and single-`--lambda` values pass through verbatim. PSC has its own independent `--use-logspace` with different anchor semantics (uses `--initial-lambda*` / `--final-lambda*`).

### Group penalty (train / drphylo / aim)

PSC's group-penalty system (`std`/`sqrt`/`linear`/`median`) is shared with these commands via `include/group_penalty.hpp`. Flags: `--group-penalty-type` (default `std` = `sqrt(feature_count)`, preserves prior behavior), `--initial-gp-value` / `--final-gp-value` / `--gp-step` (used by the `linear` mode to drive a penalty sweep). When the type is anything other than `std`, or when more than one penalty term is produced, the lambda subdirs are nested one level deeper under `penalty_<N>/` and `gss_median.txt` / `bss_median.txt` / `pss_median.txt` are written inside each penalty dir. Adaptive sparsification forwards these flags through `TrainOptions` automatically.

### PSC inputs and outputs (additions)

- `--alignments-list <file>`: list file specifying overlapping groups of alignments (one group per line, comma-separated paths relative to `alignments_dir`). Required for the olsg solvers when groups have >1 entry; resolved field array is written to `<output_dir>/<base>_field.csv` and injected via `params["field"]` (does not overwrite a user-supplied `--param field=`). Without the flag, the legacy directory-enumeration path is preserved byte-for-byte.
- `--dump-weights`: opt-in TSV dump of the full feature-weight vector from every solver call to `<output_dir>/[combo_N/][penalty_P/]lambda_L/weights.tsv`. The `combo_N/` and `penalty_P/` layers are skipped when there is only one combo or only one penalty term. Each worker writes a unique leaf file, so the parallel grid path needs no additional synchronization.

### Virtual-expansion overlap methods (ol_sg_lasso_*)

The `ol_sg_lasso_logisticr` / `ol_sg_lasso_leastr` solvers return an F-length parameter vector where F = expanded-field length (one entry per (feature, group) pair). The original feature count n ≤ F whenever any feature appears in more than one group.

- **PSC** (`pipeline_psc.cpp:1799-1855`): keeps the un-aggregated `params` vector for gene_gss / selected_sites computation (so reporting matches train's `sum_j |beta_j|`), and separately sums the expanded copies into an original-feature `result.beta` for predictions and dump-weights. Sum (not max) preserves the model's prediction since each expanded copy multiplies the same input column: `(sum_j beta_j) * x[i]`.
- **Train/drphylo/aim/adaptive** (`pipeline_train.cpp:738-745`): write `weights.txt` (one row per expanded copy via `expanded.map`) plus `weights_grouped.txt` (same rows with a Group column). No in-place aggregation — duplicate-label rows are emitted and downstream consumers sum at read time.
- **Evaluate** (`pipeline_evaluate.cpp`): FASTA branch prefers `weights_grouped.txt` and accumulates via `+=` at scoring time (line 577-578). Numeric branch uses `raw_weights[label] += w` to accumulate (line 119).

Per-gene significance reporting (`gene_gss`) is `sum_j |beta_j|` over expanded copies in both PSC and train — same value in both paths.

## Code Architecture

### PFF (Parsed FASTA File) Format

The project implements a custom binary format for efficient access to aligned sequences:

- **include/pff_format.hpp**: Complete PFF format specification with extensive documentation
  - Metadata structure (`PFFMetadata`)
  - Two orientations: `COLUMN_MAJOR` (position-wise) and `ROW_MAJOR` (sequence-wise)
  - Efficient offset calculation for random access

- **include/fasta_parser.hpp**: API for FASTA/PFF conversions
  - `fasta_to_pff()`: Convert FASTA to PFF format
  - `read_pff_metadata()`: Read metadata without loading sequence data
  - `read_pff_sequence()`: Extract individual sequences
  - `read_pff_position()`: Extract alignment columns

- **src/fasta_parser.cpp**: Implementation of FASTA parsing and PFF I/O

### Usage Examples

Convert FASTA to column-major PFF (efficient for position-wise analysis):
```bash
myesl2 convert input.fasta output.pff column
```

Convert to row-major PFF (efficient for sequence-wise analysis):
```bash
myesl2 convert input.fasta output.pff row
```

Display PFF file information:
```bash
myesl2 info output.pff
```

### Documentation

Complete format specification: `docs/PFF_FORMAT_SPEC.md`

Key implementation files with extensive inline documentation:
- `include/pff_format.hpp` - 200+ lines of format documentation
- `include/fasta_parser.hpp` - API documentation with examples
- `src/fasta_parser.cpp` - Implementation details

## Disabled features

The PSC auto-pairs feature is **disabled at the CLI surface only**. All implementation logic in `src/pipeline_psc.cpp` (`run_auto_pairs`, the activator branch, helper functions) and the `PscOptions` fields in `include/pipeline_psc.hpp` are intact and still compile — only the `main.cpp` argv parsing, help text, and contrast-source validation are commented out. Re-enabling does not require any code changes, just uncommenting.

### Gated flags (PSC mode)

- `--auto-pairs-tree <file>` — activator (one of the four contrast sources)
- `--auto-pairs-method <name>` — tuning sub-flag
- `--auto-pairs-num-alternates N` — tuning sub-flag
- `--auto-pairs-max-combinations N` — tuning sub-flag

### Where the gates live

All edits are in `src/main.cpp`. Each commented block carries an `AUTO-PAIRS DISABLED` breadcrumb comment.

1. CLI parsing — the `else if (arg == "--auto-pairs-tree" ...)` line in the PSC contrast-source block, and the four-line Auto-pairs subsection (label comment + three tuning flags) at the end of the PSC argv loop.
2. Help text — the `--auto-pairs-tree` line under the PSC "Species contrast source" subsection, and the entire `Auto-pairs:` subsection at the end of the PSC help block.
3. Contrast-source validation — the `if (!psc_opts.auto_pairs_tree.empty()) ++sources;` line; the two `throw std::runtime_error(...)` messages no longer mention `--auto-pairs-tree`.

### Re-enable recipe

In `src/main.cpp`:

1. Search for `AUTO-PAIRS DISABLED` (5 occurrences). At each site, uncomment the lines immediately following the breadcrumb.
2. Restore `--auto-pairs-tree` in the two `runtime_error` strings under the contrast-source validation.

Rebuild. No changes to `pipeline_psc.{hpp,cpp}` are required.
