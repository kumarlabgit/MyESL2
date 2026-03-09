#include "numeric_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace numeric {

namespace {

// Split a string by a delimiter character
std::vector<std::string> split_char(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::string tok;
    for (char c : s) {
        if (c == delim) {
            tokens.push_back(tok);
            tok.clear();
        } else {
            tok += c;
        }
    }
    tokens.push_back(tok);
    return tokens;
}

} // anonymous namespace

void tabular_to_pnf(
    const std::filesystem::path& input,
    const std::filesystem::path& output)
{
    std::ifstream in(input);
    if (!in)
        throw std::runtime_error("Cannot open tabular file: " + input.string());

    std::string line;

    // Read header row
    if (!std::getline(in, line))
        throw std::runtime_error("Tabular file is empty: " + input.string());

    // Remove trailing \r
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::vector<std::string> feature_labels;
    {
        std::istringstream ss(line);
        std::string tok;
        ss >> tok; // skip first column header (sample name label)
        while (ss >> tok)
            feature_labels.push_back(tok);
    }
    if (feature_labels.empty())
        throw std::runtime_error("No feature columns in header: " + input.string());

    uint32_t num_features = static_cast<uint32_t>(feature_labels.size());

    // Read data rows
    std::vector<std::string> seq_ids;
    std::vector<float> flat_data; // row-major

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string seq_id;
        if (!(ss >> seq_id)) continue;

        seq_ids.push_back(seq_id);
        for (uint32_t j = 0; j < num_features; ++j) {
            float val = 0.0f;
            if (!(ss >> val))
                throw std::runtime_error(
                    "Row '" + seq_id + "': expected " + std::to_string(num_features) +
                    " features but got fewer in " + input.string());
            flat_data.push_back(val);
        }
    }

    if (seq_ids.empty())
        throw std::runtime_error("No data rows found in: " + input.string());

    uint32_t num_sequences = static_cast<uint32_t>(seq_ids.size());

    // Build text metadata
    std::string source_abs = std::filesystem::absolute(input).string();

    // Build seq_ids and feature_labels semicolon-joined strings
    std::string seq_ids_str;
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        if (i > 0) seq_ids_str += ';';
        seq_ids_str += seq_ids[i];
    }
    std::string feat_labels_str;
    for (size_t i = 0; i < feature_labels.size(); ++i) {
        if (i > 0) feat_labels_str += ';';
        feat_labels_str += feature_labels[i];
    }

    std::string metadata;
    metadata += "VERSION=1\n";
    metadata += "source_path=" + source_abs + "\n";
    metadata += "num_sequences=" + std::to_string(num_sequences) + "\n";
    metadata += "num_features=" + std::to_string(num_features) + "\n";
    metadata += "seq_ids=" + seq_ids_str + "\n";
    metadata += "feature_labels=" + feat_labels_str + "\n";
    metadata += "END_METADATA\n";

    // Write output
    std::ofstream out(output, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open output file for writing: " + output.string());

    out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
    out.write(reinterpret_cast<const char*>(flat_data.data()),
              static_cast<std::streamsize>(flat_data.size() * sizeof(float)));

    if (!out)
        throw std::runtime_error("Write failed for: " + output.string());
}

pnf::PNFMetadata read_pnf_metadata(const std::filesystem::path& pnf_path) {
    std::ifstream in(pnf_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open PNF file: " + pnf_path.string());

    pnf::PNFMetadata meta;
    std::string line;
    uint64_t bytes_read = 0;

    while (std::getline(in, line)) {
        bytes_read += line.size() + 1; // +1 for '\n'
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
            // bytes_read already counts the \n; \r was part of the line content
        }
        if (line == "END_METADATA") {
            meta.data_offset = bytes_read;
            break;
        }
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key   = line.substr(0, eq);
        std::string value = line.substr(eq + 1);

        if (key == "source_path") {
            meta.source_path = value;
        } else if (key == "num_sequences") {
            meta.num_sequences = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "num_features") {
            meta.num_features = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "seq_ids") {
            meta.seq_ids = split_char(value, ';');
        } else if (key == "feature_labels") {
            meta.feature_labels = split_char(value, ';');
        }
        // VERSION is silently ignored
    }

    if (meta.data_offset == 0)
        throw std::runtime_error("END_METADATA not found in: " + pnf_path.string());

    return meta;
}

std::vector<std::vector<float>> read_pnf_data(
    const std::filesystem::path& pnf_path,
    const pnf::PNFMetadata& meta)
{
    std::ifstream in(pnf_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open PNF file: " + pnf_path.string());

    in.seekg(static_cast<std::streamoff>(meta.data_offset));
    if (!in)
        throw std::runtime_error("Seek failed in: " + pnf_path.string());

    uint64_t total = static_cast<uint64_t>(meta.num_sequences) * meta.num_features;
    std::vector<float> flat(total);
    in.read(reinterpret_cast<char*>(flat.data()),
            static_cast<std::streamsize>(total * sizeof(float)));
    if (!in)
        throw std::runtime_error("Read failed in: " + pnf_path.string());

    std::vector<std::vector<float>> data(meta.num_sequences,
                                         std::vector<float>(meta.num_features));
    for (uint32_t s = 0; s < meta.num_sequences; ++s)
        for (uint32_t f = 0; f < meta.num_features; ++f)
            data[s][f] = flat[static_cast<uint64_t>(s) * meta.num_features + f];

    return data;
}

} // namespace numeric
