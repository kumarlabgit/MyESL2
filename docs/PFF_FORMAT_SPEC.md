# Parsed FASTA File (PFF) Format Specification

## Version 1.0

## Overview

The Parsed FASTA File (PFF) format is a specialized file format designed for efficient random access to aligned biological sequences. It supports two data layouts optimized for different access patterns in sequence analysis.

## File Structure

```
┌─────────────────────────────────────┐
│      METADATA SECTION               │
│  (Text-based, newline-delimited)    │
├─────────────────────────────────────┤
│      SEQUENCE DATA SECTION          │
│  (Binary, orientation-dependent)    │
└─────────────────────────────────────┘
```

## Metadata Section

The metadata section consists of key-value pairs, one per line, in the format `KEY=VALUE\n`.

### Required Metadata Tags (in order)

1. **DATA_OFFSET**
   - Type: unsigned 64-bit integer
   - Description: Byte offset from file start where sequence data begins
   - Purpose: Enables direct seeking to sequence data

2. **NUM_SEQUENCES**
   - Type: unsigned 32-bit integer
   - Description: Total number of sequences in the alignment
   - Constraints: Must be > 0

3. **SEQUENCE_IDS**
   - Type: pipe-delimited string list
   - Format: `id1|id2|id3|...|idN`
   - Description: Sequence identifiers in original FASTA order
   - Constraints: Number of IDs must equal NUM_SEQUENCES

4. **ALIGNMENT_LENGTH**
   - Type: unsigned 32-bit integer
   - Description: Number of positions in the alignment
   - Constraints: Must be > 0; all sequences must have this exact length

5. **ORIENTATION**
   - Type: enum string
   - Values: `COLUMN_MAJOR` or `ROW_MAJOR`
   - Description: Data layout strategy (see below)

6. **END_METADATA**
   - Marker indicating end of metadata section
   - Sequence data begins immediately after the newline

### Example Metadata Section

```
DATA_OFFSET=152
NUM_SEQUENCES=3
SEQUENCE_IDS=Species_A|Species_B|Species_C
ALIGNMENT_LENGTH=100
ORIENTATION=COLUMN_MAJOR
END_METADATA
```

## Sequence Data Section

The sequence data section contains alignment data in one of two orientations:

### COLUMN_MAJOR Orientation

**Layout:** Position-wise storage

```
[Seq0,Pos0][Seq1,Pos0]...[SeqN,Pos0][Seq0,Pos1][Seq1,Pos1]...[SeqN,Pos1]...[Seq0,PosM]...[SeqN,PosM]
```

**Access Formula:**
```
offset(seq_i, pos_j) = DATA_OFFSET + (j × NUM_SEQUENCES) + i
```

**Use Cases:**
- Computing conservation scores across positions
- Identifying variable sites
- Phylogenetic analysis
- Multiple sequence alignment visualization
- Position-specific scoring matrices

**Example:** 3 sequences, length 5
```
Position 0: [A][C][G]
Position 1: [T][T][A]
Position 2: [G][C][C]
Position 3: [C][A][T]
Position 4: [A][G][A]

File bytes: ACGTTACCCAT...
            └─┬─┘└─┬─┘└─┬─┘
             P0   P1   P2
```

### ROW_MAJOR Orientation

**Layout:** Sequence-wise storage

```
[Seq0,Pos0][Seq0,Pos1]...[Seq0,PosM][Seq1,Pos0][Seq1,Pos1]...[Seq1,PosM]...[SeqN,Pos0]...[SeqN,PosM]
```

**Access Formula:**
```
offset(seq_i, pos_j) = DATA_OFFSET + (i × ALIGNMENT_LENGTH) + j
```

**Use Cases:**
- Extracting individual sequences
- Per-sequence statistics (GC content, entropy, etc.)
- K-mer analysis
- Sequence comparison
- Exporting subsets of sequences

**Example:** 3 sequences, length 5
```
Sequence 0: [A][T][G][C][A]
Sequence 1: [C][T][C][A][G]
Sequence 2: [G][A][C][T][A]

File bytes: ATGCACTCAGGACTA
            └──┬──┘└──┬──┘└──┬──┘
              S0     S1     S2
```

## Performance Characteristics

| Operation | COLUMN_MAJOR | ROW_MAJOR |
|-----------|--------------|-----------|
| Read position *j* (all sequences) | **O(1)** single read | O(n) seeks |
| Read sequence *i* (all positions) | O(m) seeks | **O(1)** single read |
| Random access to (i, j) | **O(1)** | **O(1)** |
| Cache efficiency for columns | **High** | Low |
| Cache efficiency for rows | Low | **High** |

Where:
- n = NUM_SEQUENCES
- m = ALIGNMENT_LENGTH

## File Size

Both orientations produce identical file sizes:

```
total_size = DATA_OFFSET + (NUM_SEQUENCES × ALIGNMENT_LENGTH)
```

## Validation Rules

A valid PFF file must satisfy:

1. All required metadata tags present in specified order
2. `NUM_SEQUENCES > 0`
3. `ALIGNMENT_LENGTH > 0`
4. `len(SEQUENCE_IDS) == NUM_SEQUENCES`
5. `DATA_OFFSET` accurately points to first byte of sequence data
6. Sequence data size = `NUM_SEQUENCES × ALIGNMENT_LENGTH` bytes

## Character Encoding

- Metadata: UTF-8 text
- Sequence data: ASCII (typically `ACGT-` for nucleotides, `ACDEFGHIKLMNPQRSTVWY-` for proteins)

## Reference Implementation

See the C++20 reference implementation:
- `include/pff_format.hpp` - Format specification and utilities
- `include/fasta_parser.hpp` - Conversion and I/O API
- `src/fasta_parser.cpp` - Implementation

## Version History

- **1.0** (2025-03-01): Initial specification
