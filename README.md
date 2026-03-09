# Fungi Genomics Analysis

C++20 project for processing and analyzing fungal protein sequence alignment data.

## Building

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Run
./bin/fungi_analysis     # Linux/Mac
.\bin\fungi_analysis.exe # Windows
```

### Using CMake with Visual Studio on Windows

```bash
# Generate Visual Studio solution
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build

# Build
cmake --build build --config Release
```

## Project Structure

- `src/` - Source files
  - `main.cpp` - Command-line interface
  - `fasta_parser.cpp` - FASTA parsing and PFF conversion implementation
- `include/` - Header files
  - `pff_format.hpp` - PFF format specification (extensively documented)
  - `fasta_parser.hpp` - FASTA/PFF API declarations
- `Fungi_data/` - Input data (FASTA alignments and species labels)
- `build/` - Build output directory (generated)

## Features

### Parsed FASTA File (PFF) Format

The project implements a custom file format optimized for efficient access to aligned sequences:

- **Column-major orientation**: All sequences organized by alignment position
  - Ideal for: conservation analysis, phylogenetics, position-specific scoring
  - Access pattern: Read all sequences at position *i* in one contiguous block

- **Row-major orientation**: Each complete sequence stored contiguously
  - Ideal for: sequence extraction, k-mer analysis, per-sequence statistics
  - Access pattern: Read entire sequence *i* in one contiguous block

See `include/pff_format.hpp` for complete format specification.

## Usage

### Convert FASTA to PFF

```bash
# Column-major (position-wise, good for phylogenetic analysis)
fungi_analysis convert alignment.fasta alignment.pff column

# Row-major (sequence-wise, good for extracting sequences)
fungi_analysis convert alignment.fasta alignment.pff row
```

### Display PFF Information

```bash
fungi_analysis info alignment.pff
```

## Requirements

- C++20 compatible compiler
- CMake 3.20 or higher
