# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fungi genomics project containing protein sequence alignment data for comparative analysis and classification. All development is done in C++ using the C++20 standard.

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

The executable will be in `build\bin\fungi_analysis.exe`.

### Running

```cmd
.\build\bin\fungi_analysis.exe
```

Or from the build directory:

```cmd
cd build\bin
fungi_analysis.exe
```

## Project Structure

```
.
├── src/              Source files
├── include/          Header files
├── Fungi_data/       Input data (FASTA alignments and labels)
├── build/            Build output (generated, in .gitignore)
├── CMakeLists.txt    CMake build configuration
└── CLAUDE.md         This file
```

## Data Structure

### Fungi_data/

The main data directory containing:

- **A_B_Hyp.txt**: Species labels file mapping 50 fungal species to binary classifications (-1 or 1). First 40 species are labeled -1, last 10 are labeled 1.

- **aln.txt**: List of paths to all ~960 BUSCO protein alignment files

- **aln_200.txt**: Subset listing 200 alignment files (likely for testing or quick analysis)

- **aln/**: Directory containing ~960 BUSCO (Benchmarking Universal Single-Copy Orthologs) protein multiple sequence alignment files in FASTA format. Each file:
  - Named as `BUSCOfEOG[ID].domain.aa.aln.trimmed.fasta`
  - Contains aligned protein sequences from multiple fungal species
  - Sequences are conserved single-copy orthologs across the species
  - Pre-aligned and trimmed for quality

## Working with the Data

The alignment files are protein sequences in FASTA format where each species has one sequence per file. Species names match those in A_B_Hyp.txt. This data is typically used for:
- Phylogenetic analysis
- Protein evolution studies
- Machine learning classification tasks
- Comparative genomics

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
fungi_analysis convert input.fasta output.pff column
```

Convert to row-major PFF (efficient for sequence-wise analysis):
```bash
fungi_analysis convert input.fasta output.pff row
```

Display PFF file information:
```bash
fungi_analysis info output.pff
```

### Documentation

Complete format specification: `docs/PFF_FORMAT_SPEC.md`

Key implementation files with extensive inline documentation:
- `include/pff_format.hpp` - 200+ lines of format documentation
- `include/fasta_parser.hpp` - API documentation with examples
- `src/fasta_parser.cpp` - Implementation details
