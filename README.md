# MyESL2 - My Evolutionary Sparse Learning 2

C++20 tool for sparse regression-based classification on genomic data. Supports FASTA alignment inputs (nucleotide/protein) and tabular numeric inputs for comparative analysis across any set of organisms.

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
./bin/myesl2     # Linux/Mac
.\bin\myesl2.exe # Windows
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
  - `numeric_parser.cpp` - Tabular numeric parsing and PNF conversion
- `include/` - Header files
  - `pff_format.hpp` - PFF format specification
  - `fasta_parser.hpp` - FASTA/PFF API declarations
  - `pnf_format.hpp` - PNF format specification
  - `numeric_parser.hpp` - Tabular/PNF API declarations
- `Fungi_data/` - Example data (fungal FASTA alignments and species labels)
- `build/` - Build output directory (generated)

## Features

### Parsed FASTA File (PFF) Format

Custom binary format optimized for efficient access to aligned sequences:

- **Column-major orientation**: All sequences organized by alignment position
- **Row-major orientation**: Each complete sequence stored contiguously

See `include/pff_format.hpp` for complete format specification.

### Parsed Numeric File (PNF) Format

Custom binary cache format for tabular numeric data (sample × float features).

See `include/pnf_format.hpp` for complete format specification.

## Usage

### Convert FASTA to PFF

```bash
myesl2 convert alignment.fasta alignment.pff column
myesl2 convert alignment.fasta alignment.pff row
```

### Display PFF Information

```bash
myesl2 info alignment.pff
```

### Train a Model

```bash
myesl2 train list.txt hypothesis.txt output_dir --datatype nucleotide --method sg_lasso --lambda 0.1 0.1
myesl2 train list.txt hypothesis.txt output_dir --datatype numeric --method sg_lasso --lambda-file lambdas.txt --nfolds 5
```

## Requirements

- C++20 compatible compiler
- CMake 3.20 or higher
