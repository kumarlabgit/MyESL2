#include "encoder.hpp"
#include "fasta_parser.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>

namespace encoder {

AlignmentResult encode_pff(
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major,
    const std::unordered_set<std::string>& dropout_labels)
{
    AlignmentResult result;
    result.stem = pff_path.stem().string();

    auto metadata = fasta::read_pff_metadata(pff_path);

    // Map each PFF sequence ID to its index
    std::unordered_map<std::string, uint32_t> pff_id_to_idx;
    pff_id_to_idx.reserve(metadata.seq_ids.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(metadata.seq_ids.size()); ++i)
        pff_id_to_idx[metadata.seq_ids[i]] = i;

    // seq_mapping[i] = index in PFF for hypothesis sequence i, or -1 if missing
    uint32_t N = static_cast<uint32_t>(hyp_seq_names.size());
    std::vector<int> seq_mapping(N, -1);
    for (uint32_t i = 0; i < N; ++i) {
        auto it = pff_id_to_idx.find(hyp_seq_names[i]);
        if (it != pff_id_to_idx.end())
            seq_mapping[i] = static_cast<int>(it->second);
        else
            result.missing_sequences.push_back(result.stem + "\t" + hyp_seq_names[i]);
    }

    // Read all binary sequence data in one shot
    uint32_t S = metadata.num_sequences;
    uint32_t L = metadata.alignment_length;
    std::vector<char> raw(metadata.get_data_size());
    {
        std::ifstream file(pff_path, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(metadata.data_offset));
        file.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    }

    // Encode each alignment position
    for (uint32_t p = 0; p < L; ++p) {
        // Count alleles among hypothesis sequences, ignoring '-' and '?'
        std::map<char, int> counts;
        for (uint32_t i = 0; i < N; ++i) {
            if (seq_mapping[i] < 0) continue;
            char c = raw[static_cast<size_t>(p) * S + static_cast<uint32_t>(seq_mapping[i])];
            if (c == '-' || c == '?') continue;
            counts[c]++;
        }

        if (counts.size() < 2) continue; // monomorphic or no valid data

        // Major allele = highest count (ties broken by map order, i.e. lowest char)
        char major = std::max_element(counts.begin(), counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;

        // Total non-major, non-indel characters
        int non_major = 0;
        for (const auto& [c, cnt] : counts)
            if (c != major) non_major += cnt;

        if (non_major < min_minor) continue;

        // One column per allele; skip major if drop_major=true (map iteration = sorted by char)
        for (const auto& [allele, cnt] : counts) {
            if (drop_major && allele == major) continue;

            // Check dropout
            if (!dropout_labels.empty()) {
                std::string lbl = result.stem + "_" + std::to_string(p) + "_" + allele;
                if (dropout_labels.count(lbl)) continue;
            }

            std::vector<uint8_t> col(N, 0);
            for (uint32_t i = 0; i < N; ++i) {
                if (seq_mapping[i] < 0) continue;
                char c = raw[static_cast<size_t>(p) * S + static_cast<uint32_t>(seq_mapping[i])];
                if (c == allele) col[i] = 1;
            }
            result.columns.push_back(std::move(col));
            result.map.push_back({p, allele});
        }
    }

    return result;
}

AlignmentResult encode_pff_dlt(
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major,
    const std::unordered_set<std::string>& dropout_labels)
{
    AlignmentResult result;
    result.stem = pff_path.stem().string();

    auto metadata = fasta::read_pff_metadata(pff_path);

    std::unordered_map<std::string, uint32_t> pff_id_to_idx;
    pff_id_to_idx.reserve(metadata.seq_ids.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(metadata.seq_ids.size()); ++i)
        pff_id_to_idx[metadata.seq_ids[i]] = i;

    uint32_t N = static_cast<uint32_t>(hyp_seq_names.size());
    std::vector<int> seq_mapping(N, -1);
    for (uint32_t i = 0; i < N; ++i) {
        auto it = pff_id_to_idx.find(hyp_seq_names[i]);
        if (it != pff_id_to_idx.end())
            seq_mapping[i] = static_cast<int>(it->second);
        else
            result.missing_sequences.push_back(result.stem + "\t" + hyp_seq_names[i]);
    }

    uint32_t S = metadata.num_sequences;
    uint32_t L = metadata.alignment_length;
    std::vector<char> raw(metadata.get_data_size());
    {
        std::ifstream file(pff_path, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(metadata.data_offset));
        file.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    }

    // Lookup table: characters to skip when counting alleles
    static const auto is_skip = []() {
        std::array<bool, 256> t{};
        t[static_cast<uint8_t>('-')] = true;
        t[static_cast<uint8_t>('?')] = true;
        return t;
    }();

    // Per-position working arrays declared outside the loop
    int     counts[256];   // allele occurrence counts
    int8_t  col_idx[256];  // char -> column index, or -1

    for (uint32_t p = 0; p < L; ++p) {
        // Count alleles via direct array — no character equality comparisons
        std::memset(counts, 0, sizeof(counts));
        for (uint32_t i = 0; i < N; ++i) {
            if (seq_mapping[i] < 0) continue;
            auto c = static_cast<uint8_t>(raw[static_cast<size_t>(p) * S + static_cast<uint32_t>(seq_mapping[i])]);
            if (is_skip[c]) continue;
            ++counts[c];
        }

        // Find major allele (highest count; ties broken by lowest ASCII value)
        int     major_count = 0;
        uint8_t major       = 0;
        int     num_alleles = 0;
        for (int c = 0; c < 256; ++c) {
            if (counts[c] == 0) continue;
            ++num_alleles;
            if (counts[c] > major_count) {
                major_count = counts[c];
                major = static_cast<uint8_t>(c);
            }
        }

        if (num_alleles < 2) continue;

        // Total non-major occurrences
        int non_major = 0;
        for (int c = 0; c < 256; ++c)
            if (c != static_cast<int>(major)) non_major += counts[c];

        if (non_major < min_minor) continue;

        // Build col_idx: assign a column index to each allele to include
        std::memset(col_idx, -1, sizeof(col_idx));
        int8_t next_col = 0;
        for (int c = 0; c < 256; ++c) {
            if (counts[c] == 0) continue;
            if (drop_major && c == static_cast<int>(major)) continue;
            // Check dropout
            if (!dropout_labels.empty()) {
                std::string lbl = result.stem + "_" + std::to_string(p) + "_" + static_cast<char>(c);
                if (dropout_labels.count(lbl)) continue;
            }
            col_idx[c] = next_col++;
        }

        int num_cols = static_cast<int>(next_col);
        if (num_cols == 0) continue;

        // Allocate output columns (all zeros)
        std::vector<std::vector<uint8_t>> new_cols(num_cols, std::vector<uint8_t>(N, 0));

        // Single pass over sequences — lookup replaces per-allele equality comparison
        for (uint32_t i = 0; i < N; ++i) {
            if (seq_mapping[i] < 0) continue;
            auto c  = static_cast<uint8_t>(raw[static_cast<size_t>(p) * S + static_cast<uint32_t>(seq_mapping[i])]);
            int8_t ci = col_idx[c];
            if (ci >= 0) new_cols[ci][i] = 1;
        }

        // Append columns and map entries in column-index order
        for (int j = 0; j < num_cols; ++j)
            result.columns.push_back(std::move(new_cols[j]));
        for (int c = 0; c < 256; ++c)
            if (col_idx[c] >= 0)
                result.map.push_back({p, static_cast<char>(c)});
    }

    return result;
}

} // namespace encoder
