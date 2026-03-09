/**
 * @file pff_format.hpp
 * @brief Parsed FASTA File (PFF) Format Specification and Data Structures
 *
 * PFF Format Overview:
 * ===================
 * The Parsed FASTA File format is designed for efficient access to aligned sequence data
 * in both row-major and column-major orientations. The format consists of two sections:
 *
 * 1. METADATA SECTION (Text-based, newline-delimited tags)
 * 2. SEQUENCE DATA SECTION (Binary or text, depending on orientation)
 *
 * METADATA SECTION FORMAT:
 * -----------------------
 * All metadata tags are key-value pairs in the format: "KEY=VALUE\n"
 * The metadata section MUST include the following tags in this order:
 *
 * 1. DATA_OFFSET=<byte_index>
 *    - Byte offset from start of file where sequence data begins
 *    - Allows direct seeking to sequence data
 *    - Type: unsigned 64-bit integer
 *
 * 2. NUM_SEQUENCES=<count>
 *    - Total number of sequences in the alignment
 *    - Type: unsigned 32-bit integer
 *
 * 3. SEQUENCE_IDS=<id1>|<id2>|...|<idN>
 *    - Pipe-delimited list of sequence identifiers
 *    - Order matches the order in the original FASTA file
 *    - IDs are taken from FASTA header lines (after '>')
 *
 * 4. ALIGNMENT_LENGTH=<positions>
 *    - Number of positions in the multiple sequence alignment
 *    - All sequences must have this exact length
 *    - Type: unsigned 32-bit integer
 *
 * 5. ORIENTATION=<mode>
 *    - Data layout orientation, one of:
 *      * "COLUMN_MAJOR" - Data organized by alignment position
 *      * "ROW_MAJOR"    - Data organized by sequence
 *
 * 6. END_METADATA
 *    - Marker indicating end of metadata section
 *    - Sequence data begins on the next byte after the newline
 *
 * SEQUENCE DATA SECTION:
 * ---------------------
 *
 * COLUMN_MAJOR Orientation (Position-wise):
 * -----------------------------------------
 * Data is organized so that all sequences' character at position i are stored together.
 * Layout: [S0_P0][S1_P0]...[SN_P0][S0_P1][S1_P1]...[SN_P1]...[S0_PM][S1_PM]...[SN_PM]
 *
 * Where:
 *   - N = NUM_SEQUENCES - 1
 *   - M = ALIGNMENT_LENGTH - 1
 *   - S<i>_P<j> = character at position j of sequence i
 *
 * Byte offset to access position j of sequence i:
 *   offset = DATA_OFFSET + (j * NUM_SEQUENCES) + i
 *
 * Use case: Ideal for column-wise analysis (e.g., computing conservation scores,
 *           identifying variable positions, phylogenetic analysis)
 *
 * ROW_MAJOR Orientation (Sequence-wise):
 * --------------------------------------
 * Data is organized so that each complete sequence is stored contiguously.
 * Layout: [S0_P0][S0_P1]...[S0_PM][S1_P0][S1_P1]...[S1_PM]...[SN_P0][SN_P1]...[SN_PM]
 *
 * Where:
 *   - N = NUM_SEQUENCES - 1
 *   - M = ALIGNMENT_LENGTH - 1
 *   - S<i>_P<j> = character at position j of sequence i
 *
 * Byte offset to access position j of sequence i:
 *   offset = DATA_OFFSET + (i * ALIGNMENT_LENGTH) + j
 *
 * Use case: Ideal for sequence-wise analysis (e.g., extracting individual sequences,
 *           computing per-sequence statistics, k-mer analysis)
 *
 * EXAMPLE FILE:
 * -------------
 * DATA_OFFSET=156
 * NUM_SEQUENCES=3
 * SEQUENCE_IDS=Species_A|Species_B|Species_C
 * ALIGNMENT_LENGTH=5
 * ORIENTATION=COLUMN_MAJOR
 * END_METADATA
 * ACGTA[sequence data continues...]
 *
 * For COLUMN_MAJOR with 3 sequences of length 5:
 * Bytes 0-2:   Position 0 of all sequences (Seq0[0], Seq1[0], Seq2[0])
 * Bytes 3-5:   Position 1 of all sequences (Seq0[1], Seq1[1], Seq2[1])
 * Bytes 6-8:   Position 2 of all sequences (Seq0[2], Seq1[2], Seq2[2])
 * Bytes 9-11:  Position 3 of all sequences (Seq0[3], Seq1[3], Seq2[3])
 * Bytes 12-14: Position 4 of all sequences (Seq0[4], Seq1[4], Seq2[4])
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace pff {

/**
 * @enum Orientation
 * @brief Defines the data layout orientation in PFF files
 */
enum class Orientation {
    COLUMN_MAJOR,  ///< Position-wise storage (all sequences at position i, then i+1, etc.)
    ROW_MAJOR      ///< Sequence-wise storage (all positions of sequence i, then i+1, etc.)
};

/**
 * @brief Convert Orientation enum to string representation
 */
inline std::string to_string(Orientation orient) {
    switch (orient) {
        case Orientation::COLUMN_MAJOR: return "COLUMN_MAJOR";
        case Orientation::ROW_MAJOR:    return "ROW_MAJOR";
        default: throw std::invalid_argument("Invalid orientation");
    }
}

/**
 * @brief Convert string to Orientation enum
 */
inline Orientation orientation_from_string(const std::string& str) {
    if (str == "COLUMN_MAJOR") return Orientation::COLUMN_MAJOR;
    if (str == "ROW_MAJOR")    return Orientation::ROW_MAJOR;
    throw std::invalid_argument("Invalid orientation string: " + str);
}

/**
 * @struct PFFMetadata
 * @brief Metadata structure for Parsed FASTA Files
 */
struct PFFMetadata {
    uint64_t data_offset;              ///< Byte offset where sequence data begins
    uint32_t num_sequences;            ///< Number of sequences in the alignment
    std::vector<std::string> seq_ids;  ///< Sequence identifiers in order
    uint32_t alignment_length;         ///< Number of positions in alignment
    Orientation orientation;           ///< Data layout orientation
    std::string datatype = "universal"; ///< Character validation class
    std::string source_path;           ///< Absolute path of the original FASTA file

    /**
     * @brief Validate metadata consistency
     * @throws std::runtime_error if metadata is inconsistent
     */
    void validate() const {
        if (seq_ids.size() != num_sequences) {
            throw std::runtime_error(
                "Metadata inconsistency: num_sequences=" + std::to_string(num_sequences) +
                " but seq_ids.size()=" + std::to_string(seq_ids.size())
            );
        }
        if (num_sequences == 0) {
            throw std::runtime_error("Invalid metadata: num_sequences must be > 0");
        }
        if (alignment_length == 0) {
            throw std::runtime_error("Invalid metadata: alignment_length must be > 0");
        }
    }

    /**
     * @brief Calculate expected sequence data size in bytes
     */
    uint64_t get_data_size() const {
        return static_cast<uint64_t>(num_sequences) * static_cast<uint64_t>(alignment_length);
    }

    /**
     * @brief Calculate byte offset for a specific sequence position
     * @param seq_index Index of the sequence (0-based)
     * @param pos_index Position in the alignment (0-based)
     * @return Absolute byte offset from start of file
     */
    uint64_t get_offset(uint32_t seq_index, uint32_t pos_index) const {
        if (seq_index >= num_sequences) {
            throw std::out_of_range("Sequence index out of range");
        }
        if (pos_index >= alignment_length) {
            throw std::out_of_range("Position index out of range");
        }

        uint64_t relative_offset;
        if (orientation == Orientation::COLUMN_MAJOR) {
            // offset = (position * num_sequences) + sequence_index
            relative_offset = static_cast<uint64_t>(pos_index) * num_sequences + seq_index;
        } else {
            // offset = (sequence_index * alignment_length) + position
            relative_offset = static_cast<uint64_t>(seq_index) * alignment_length + pos_index;
        }

        return data_offset + relative_offset;
    }
};

} // namespace pff
