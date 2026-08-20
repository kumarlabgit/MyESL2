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

void write_pnf(
    const std::filesystem::path& output,
    const std::vector<std::string>& seq_ids,
    const std::vector<std::string>& feature_labels,
    const std::vector<float>& row_major_data,
    const std::string& source_path,
    const std::string& het_mode)
{
    const uint32_t num_sequences = static_cast<uint32_t>(seq_ids.size());
    const uint32_t num_features  = static_cast<uint32_t>(feature_labels.size());

    const uint64_t expected =
        static_cast<uint64_t>(num_sequences) * static_cast<uint64_t>(num_features);
    if (row_major_data.size() != expected)
        throw std::runtime_error(
            "write_pnf: payload has " + std::to_string(row_major_data.size()) +
            " values but " + std::to_string(num_sequences) + " x " +
            std::to_string(num_features) + " = " + std::to_string(expected) +
            " were expected for " + output.string());

    auto join_semicolons = [](const std::vector<std::string>& items) {
        std::string joined;
        for (size_t i = 0; i < items.size(); ++i) {
            if (i > 0) joined += ';';
            joined += items[i];
        }
        return joined;
    };

    std::string metadata;
    metadata += "VERSION=1\n";
    metadata += "source_path=" + source_path + "\n";
    metadata += "num_sequences=" + std::to_string(num_sequences) + "\n";
    metadata += "num_features=" + std::to_string(num_features) + "\n";
    metadata += "seq_ids=" + join_semicolons(seq_ids) + "\n";
    metadata += "feature_labels=" + join_semicolons(feature_labels) + "\n";
    // Written only for VCF-derived caches; absent keys are ignored by the reader,
    // so .pnf files stay byte-identical to those produced before this field existed.
    if (!het_mode.empty())
        metadata += "het_mode=" + het_mode + "\n";
    metadata += "END_METADATA\n";

    std::ofstream out(output, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open output file for writing: " + output.string());

    out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
    out.write(reinterpret_cast<const char*>(row_major_data.data()),
              static_cast<std::streamsize>(row_major_data.size() * sizeof(float)));

    if (!out)
        throw std::runtime_error("Write failed for: " + output.string());
}

void tabular_to_pnf(
    const std::filesystem::path& input,
    const std::filesystem::path& output)
{
    std::ifstream in(input);
    if (!in)
        throw std::runtime_error("Cannot open tabular file: " + input.string());

    std::string line;

    // Read first row
    if (!std::getline(in, line))
        throw std::runtime_error("Tabular file is empty: " + input.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();

    // Auto-detect header: try to parse the second token as a float.
    // If it succeeds the first row is data; if it fails the first row is a header.
    std::vector<std::string> feature_labels;
    std::string first_data_line; // non-empty only when the first row is data

    {
        std::istringstream ss(line);
        std::string first_tok, second_tok;
        ss >> first_tok >> second_tok;
        float probe;
        std::istringstream probe_ss(second_tok);
        if (second_tok.empty() || !(probe_ss >> probe)) {
            // First row is a header: collect remaining tokens as feature labels
            std::istringstream hdr(line);
            std::string tok;
            hdr >> tok; // skip sample-name column label
            while (hdr >> tok)
                feature_labels.push_back(tok);
        } else {
            // First row is data: generate synthetic labels and keep the line for parsing
            first_data_line = line;
        }
    }

    if (feature_labels.empty() && first_data_line.empty())
        throw std::runtime_error("No feature columns in: " + input.string());

    uint32_t num_features = 0;

    // Helper lambda: parse one data line into seq_ids / flat_data
    std::vector<std::string> seq_ids;
    std::vector<float> flat_data; // row-major

    auto parse_data_line = [&](const std::string& l) {
        std::istringstream ss(l);
        std::string seq_id;
        if (!(ss >> seq_id)) return;

        // On first data row, finalise num_features and (if no header) generate labels
        if (num_features == 0) {
            std::istringstream counter(l);
            std::string skip; counter >> skip;
            uint32_t cnt = 0; std::string tmp;
            while (counter >> tmp) ++cnt;
            num_features = cnt;
            if (feature_labels.empty()) {
                // No header: generate synthetic labels
                feature_labels.reserve(num_features);
                for (uint32_t k = 0; k < num_features; ++k)
                    feature_labels.push_back("col_" + std::to_string(k));
            }
        }

        seq_ids.push_back(seq_id);
        for (uint32_t j = 0; j < num_features; ++j) {
            float val = 0.0f;
            if (!(ss >> val))
                throw std::runtime_error(
                    "Row '" + seq_id + "': expected " + std::to_string(num_features) +
                    " features but got fewer in " + input.string());
            flat_data.push_back(val);
        }
    };

    if (!first_data_line.empty())
        parse_data_line(first_data_line);

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        parse_data_line(line);
    }

    if (seq_ids.empty())
        throw std::runtime_error("No data rows found in: " + input.string());

    // The data rows, not the header, are authoritative for the column count (a
    // header may carry a stray extra/missing token). Reconcile the label list to
    // num_features so the emitted header stays self-consistent; for well-formed
    // input the two already agree and this is a no-op.
    feature_labels.resize(num_features);
    for (uint32_t k = 0; k < num_features; ++k)
        if (feature_labels[k].empty())
            feature_labels[k] = "col_" + std::to_string(k);

    write_pnf(output, seq_ids, feature_labels, flat_data,
              std::filesystem::absolute(input).string());
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
        } else if (key == "het_mode") {
            meta.het_mode = value;
        }
        // VERSION is silently ignored
    }

    if (meta.data_offset == 0)
        throw std::runtime_error("END_METADATA not found in: " + pnf_path.string());

    // See the matching check in read_pff_metadata: a short payload means the
    // conversion was interrupted, and silently reusing it corrupts the matrix.
    {
        std::error_code ec;
        auto actual = std::filesystem::file_size(pnf_path, ec);
        auto needed = meta.data_offset + meta.get_data_size();
        if (ec || actual < needed)
            throw std::runtime_error(
                "Truncated PNF file (need " + std::to_string(needed) +
                " bytes, found " + std::to_string(ec ? 0 : actual) + "): " + pnf_path.string());
    }

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
