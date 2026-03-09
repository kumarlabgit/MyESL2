/**
 * @file fasta_parser.hpp
 * @brief FASTA file parsing and PFF conversion utilities
 */

#pragma once

#include "pff_format.hpp"
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace fasta {

/**
 * @struct FastaSequence
 * @brief Represents a single sequence from a FASTA file
 */
struct FastaSequence {
    std::string id;        ///< Sequence identifier (from header line)
    std::string sequence;  ///< Sequence data
};

/**
 * @brief Parse a FASTA file into memory
 * @param filepath Path to the FASTA file
 * @return Vector of sequences in the order they appear in the file
 * @throws std::runtime_error if file cannot be read or is malformed
 */
std::vector<FastaSequence> parse_fasta(const std::filesystem::path& filepath);

/**
 * @brief Convert FASTA file to PFF format
 *
 * This function reads a FASTA alignment file and writes it in the Parsed FASTA File
 * format with the specified orientation. The function validates that all sequences
 * in the alignment have the same length.
 *
 * @param input_fasta Path to input FASTA file
 * @param output_pff Path to output PFF file
 * @param orientation Desired data layout (COLUMN_MAJOR or ROW_MAJOR)
 * @throws std::runtime_error if sequences have different lengths or file I/O fails
 *
 * @example
 * ```cpp
 * // Convert to column-major format (position-wise access)
 * fasta::fasta_to_pff("alignment.fasta", "alignment.pff", pff::Orientation::COLUMN_MAJOR);
 *
 * // Convert to row-major format (sequence-wise access)
 * fasta::fasta_to_pff("alignment.fasta", "alignment_row.pff", pff::Orientation::ROW_MAJOR);
 * ```
 */
void fasta_to_pff(
    const std::filesystem::path& input_fasta,
    const std::filesystem::path& output_pff,
    pff::Orientation orientation,
    const std::string& datatype = "universal",
    const std::unordered_set<char>& allowed_chars = {}
);

/**
 * @brief Read PFF metadata without loading sequence data
 *
 * Efficiently reads only the metadata section of a PFF file, allowing quick
 * access to file information without loading the entire sequence data into memory.
 *
 * @param pff_path Path to PFF file
 * @return PFFMetadata structure with file information
 * @throws std::runtime_error if file cannot be read or metadata is malformed
 */
pff::PFFMetadata read_pff_metadata(const std::filesystem::path& pff_path);

/**
 * @brief Read a single sequence from a PFF file
 *
 * Extracts one complete sequence from a PFF file without loading all sequences.
 * Works efficiently with both COLUMN_MAJOR and ROW_MAJOR orientations.
 *
 * @param pff_path Path to PFF file
 * @param seq_index Index of sequence to read (0-based)
 * @return String containing the sequence
 * @throws std::out_of_range if seq_index is invalid
 * @throws std::runtime_error if file I/O fails
 */
std::string read_pff_sequence(const std::filesystem::path& pff_path, uint32_t seq_index);

/**
 * @brief Read a single position (column) from a PFF file
 *
 * Extracts all sequences' characters at a specific alignment position.
 * Works efficiently with both COLUMN_MAJOR and ROW_MAJOR orientations.
 *
 * @param pff_path Path to PFF file
 * @param pos_index Position in alignment to read (0-based)
 * @return String containing characters from all sequences at this position
 * @throws std::out_of_range if pos_index is invalid
 * @throws std::runtime_error if file I/O fails
 */
std::string read_pff_position(const std::filesystem::path& pff_path, uint32_t pos_index);

} // namespace fasta
