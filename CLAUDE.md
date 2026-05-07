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

### Encoding Options

- `--minor-column`: Adds a per-gene binary summary column (`{stem}_minor`) indicating whether each species has any non-major allele in that gene. Writes `minor_alleles.txt` listing which feature labels are minor alleles, used by evaluate to reconstitute the column for new species.
- `--tiered-minor-col`: Adds up to 4 per-gene binary columns (`{stem}_tminor_Xpct`) at frequency thresholds 0%, 0.1%, 1%, and 5%. A tier column is 1 if the species has any minor allele at any position in the gene whose per-position frequency exceeds the threshold. Writes `tiered_minor_alleles.txt` (stem, pos, allele, freq). Mutually exclusive with `--minor-column`. Evaluate uses `--tiered-minor-alleles <file>` to locate the auxiliary file.
- `--drop-major-allele`: Excludes the most frequent allele column from FASTA encoding.
- `--auto-bit-ct X`: Sets `min_minor = ceil(X% × minority_class_size)`.

### Grid-Loop Threading (train / drphylo / aim / adaptive)

`--threads N` parallelises the lambda grid solver loop in addition to preprocessing/encoding (previously only the preprocess phase was threaded). The `--min-groups` skip-ahead ratchet is disabled while the parallel loop runs (the ratchet relies on sequential gene-count ordering). Pass `--prune-skipped-lambda` to replay the skip-ahead logic after the parallel loop finishes, deleting the contents of `lambda_<idx>/` directories that a single-threaded run would have skipped (the dirs are preserved as sentinels for drphylo aggregation). Grid-loop parallelism is disabled when `--nfolds > 0`. When running `--threads > 1`, set `OPENBLAS_NUM_THREADS=1` to avoid nested parallelism between worker threads and OpenBLAS kernels. `psc` has its own independent grid-loop threading (unchanged by this).

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
