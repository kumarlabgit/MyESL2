#pragma once

#include "pnf_format.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace numeric {

/**
 * @brief Parse a whitespace-delimited tabular file and write a .pnf binary cache.
 *
 * Input format:
 *   - First row: column headers (first token ignored as sample-name label)
 *   - Remaining rows: first token = sample/species name, rest = float values
 *
 * @param input  Path to the tabular input file
 * @param output Path to the output .pnf file
 * @throws std::runtime_error on parse or I/O failure
 */
void tabular_to_pnf(
    const std::filesystem::path& input,
    const std::filesystem::path& output);

/**
 * @brief Read PNF metadata without loading float data.
 * @param pnf_path Path to .pnf file
 * @return PNFMetadata populated from the text header
 * @throws std::runtime_error on I/O or parse failure
 */
pnf::PNFMetadata read_pnf_metadata(const std::filesystem::path& pnf_path);

/**
 * @brief Read all float data from a .pnf file.
 * @param pnf_path Path to .pnf file
 * @param meta     Metadata previously read by read_pnf_metadata()
 * @return data[seq_idx][feat_idx]
 * @throws std::runtime_error on I/O failure
 */
std::vector<std::vector<float>> read_pnf_data(
    const std::filesystem::path& pnf_path,
    const pnf::PNFMetadata& meta);

} // namespace numeric
