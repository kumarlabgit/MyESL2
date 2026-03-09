/**
 * @file fasta_parser.cpp
 * @brief Implementation of FASTA parsing and PFF conversion
 */

#include "fasta_parser.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <cctype>

namespace fasta {

std::vector<FastaSequence> parse_fasta(const std::filesystem::path& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open FASTA file: " + filepath.string());
    }

    std::vector<FastaSequence> sequences;
    FastaSequence current_seq;
    std::string line;
    bool in_sequence = false;

    while (std::getline(file, line)) {
        // Remove trailing whitespace
        while (!line.empty() && std::isspace(line.back())) {
            line.pop_back();
        }

        if (line.empty()) {
            continue; // Skip empty lines
        }

        if (line[0] == '>') {
            // Header line - save previous sequence if exists
            if (in_sequence) {
                sequences.push_back(std::move(current_seq));
                current_seq = FastaSequence();
            }

            // Extract sequence ID (everything after '>', trimmed)
            current_seq.id = line.substr(1);
            // Remove leading whitespace from ID
            size_t start = current_seq.id.find_first_not_of(" \t");
            if (start != std::string::npos) {
                current_seq.id = current_seq.id.substr(start);
            }

            in_sequence = true;
        } else if (in_sequence) {
            // Sequence data line
            current_seq.sequence += line;
        } else {
            throw std::runtime_error("FASTA format error: sequence data before header");
        }
    }

    // Don't forget the last sequence
    if (in_sequence) {
        sequences.push_back(std::move(current_seq));
    }

    if (sequences.empty()) {
        throw std::runtime_error("No sequences found in FASTA file");
    }

    return sequences;
}

void fasta_to_pff(
    const std::filesystem::path& input_fasta,
    const std::filesystem::path& output_pff,
    pff::Orientation orientation,
    const std::string& datatype,
    const std::unordered_set<char>& allowed_chars
) {
    // Parse input FASTA file
    auto sequences = parse_fasta(input_fasta);

    // Validate alignment - all sequences must have same length
    uint32_t alignment_length = static_cast<uint32_t>(sequences[0].sequence.length());
    for (size_t i = 1; i < sequences.size(); ++i) {
        if (sequences[i].sequence.length() != alignment_length) {
            throw std::runtime_error(
                "Sequence length mismatch: " + sequences[0].id + " has length " +
                std::to_string(alignment_length) + " but " + sequences[i].id +
                " has length " + std::to_string(sequences[i].sequence.length())
            );
        }
    }

    // Build metadata
    pff::PFFMetadata metadata;
    metadata.num_sequences = static_cast<uint32_t>(sequences.size());
    metadata.alignment_length = alignment_length;
    metadata.orientation = orientation;
    metadata.datatype = datatype;
    metadata.source_path = std::filesystem::absolute(input_fasta).string();

    for (const auto& seq : sequences) {
        metadata.seq_ids.push_back(seq.id);
    }

    // Write metadata to string stream to calculate offset
    std::ostringstream metadata_stream;
    metadata_stream << "DATA_OFFSET=";
    size_t offset_pos = metadata_stream.tellp();
    metadata_stream << std::string(20, ' ') << '\n'; // Placeholder for offset

    metadata_stream << "NUM_SEQUENCES=" << metadata.num_sequences << '\n';

    metadata_stream << "SEQUENCE_IDS=";
    for (size_t i = 0; i < metadata.seq_ids.size(); ++i) {
        if (i > 0) metadata_stream << '|';
        metadata_stream << metadata.seq_ids[i];
    }
    metadata_stream << '\n';

    metadata_stream << "ALIGNMENT_LENGTH=" << metadata.alignment_length << '\n';
    metadata_stream << "ORIENTATION=" << pff::to_string(metadata.orientation) << '\n';
    metadata_stream << "DATATYPE=" << metadata.datatype << '\n';
    metadata_stream << "SOURCE_PATH=" << metadata.source_path << '\n';
    metadata_stream << "END_METADATA\n";

    std::string metadata_str = metadata_stream.str();
    metadata.data_offset = metadata_str.length();

    // Update the offset placeholder
    std::string offset_str = std::to_string(metadata.data_offset);
    size_t spaces_needed = 20 - offset_str.length();
    metadata_str.replace(offset_pos, 20, offset_str + std::string(spaces_needed, ' '));

    metadata.validate();

    // Open output file in binary mode
    std::ofstream outfile(output_pff, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create output file: " + output_pff.string());
    }

    // Write metadata
    outfile.write(metadata_str.data(), metadata_str.size());

    // Write sequence data in the specified orientation
    if (orientation == pff::Orientation::COLUMN_MAJOR) {
        // Write position by position (column-wise)
        for (uint32_t pos = 0; pos < alignment_length; ++pos) {
            for (uint32_t seq = 0; seq < metadata.num_sequences; ++seq) {
                char c = sequences[seq].sequence[pos];
                if (!allowed_chars.empty()) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    if (!allowed_chars.count(c))
                        throw std::runtime_error(
                            input_fasta.filename().string() + ": invalid char '" + c +
                            "' in sequence '" + sequences[seq].id +
                            "' at position " + std::to_string(pos));
                }
                outfile.write(&c, 1);
            }
        }
    } else {
        // Write sequence by sequence (row-wise)
        for (uint32_t seq = 0; seq < metadata.num_sequences; ++seq) {
            if (allowed_chars.empty()) {
                outfile.write(sequences[seq].sequence.data(), alignment_length);
            } else {
                std::string buf = sequences[seq].sequence;
                for (uint32_t pos = 0; pos < alignment_length; ++pos) {
                    buf[pos] = static_cast<char>(std::toupper(static_cast<unsigned char>(buf[pos])));
                    if (!allowed_chars.count(buf[pos]))
                        throw std::runtime_error(
                            input_fasta.filename().string() + ": invalid char '" + buf[pos] +
                            "' in sequence '" + sequences[seq].id +
                            "' at position " + std::to_string(pos));
                }
                outfile.write(buf.data(), alignment_length);
            }
        }
    }

    if (!outfile.good()) {
        throw std::runtime_error("Error writing to output file: " + output_pff.string());
    }
}

pff::PFFMetadata read_pff_metadata(const std::filesystem::path& pff_path) {
    std::ifstream file(pff_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PFF file: " + pff_path.string());
    }

    pff::PFFMetadata metadata;
    std::string line;

    while (std::getline(file, line)) {
        if (line == "END_METADATA") {
            break;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // Trim whitespace from value
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "DATA_OFFSET") {
            metadata.data_offset = std::stoull(value);
        } else if (key == "NUM_SEQUENCES") {
            metadata.num_sequences = std::stoul(value);
        } else if (key == "SEQUENCE_IDS") {
            // Split by pipe character
            std::istringstream ss(value);
            std::string id;
            while (std::getline(ss, id, '|')) {
                metadata.seq_ids.push_back(id);
            }
        } else if (key == "ALIGNMENT_LENGTH") {
            metadata.alignment_length = std::stoul(value);
        } else if (key == "ORIENTATION") {
            metadata.orientation = pff::orientation_from_string(value);
        } else if (key == "DATATYPE") {
            metadata.datatype = value;
        } else if (key == "SOURCE_PATH") {
            metadata.source_path = value;
        }
    }

    metadata.validate();
    return metadata;
}

std::string read_pff_sequence(const std::filesystem::path& pff_path, uint32_t seq_index) {
    auto metadata = read_pff_metadata(pff_path);

    if (seq_index >= metadata.num_sequences) {
        throw std::out_of_range(
            "Sequence index " + std::to_string(seq_index) +
            " out of range [0, " + std::to_string(metadata.num_sequences - 1) + "]"
        );
    }

    std::ifstream file(pff_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PFF file: " + pff_path.string());
    }

    std::string sequence;
    sequence.resize(metadata.alignment_length);

    if (metadata.orientation == pff::Orientation::ROW_MAJOR) {
        // Easy case: sequence is contiguous
        uint64_t offset = metadata.get_offset(seq_index, 0);
        file.seekg(offset);
        file.read(sequence.data(), metadata.alignment_length);
    } else {
        // COLUMN_MAJOR: need to read from different positions
        for (uint32_t pos = 0; pos < metadata.alignment_length; ++pos) {
            uint64_t offset = metadata.get_offset(seq_index, pos);
            file.seekg(offset);
            file.read(&sequence[pos], 1);
        }
    }

    if (!file.good()) {
        throw std::runtime_error("Error reading sequence from PFF file");
    }

    return sequence;
}

std::string read_pff_position(const std::filesystem::path& pff_path, uint32_t pos_index) {
    auto metadata = read_pff_metadata(pff_path);

    if (pos_index >= metadata.alignment_length) {
        throw std::out_of_range(
            "Position index " + std::to_string(pos_index) +
            " out of range [0, " + std::to_string(metadata.alignment_length - 1) + "]"
        );
    }

    std::ifstream file(pff_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PFF file: " + pff_path.string());
    }

    std::string position;
    position.resize(metadata.num_sequences);

    if (metadata.orientation == pff::Orientation::COLUMN_MAJOR) {
        // Easy case: position is contiguous
        uint64_t offset = metadata.get_offset(0, pos_index);
        file.seekg(offset);
        file.read(position.data(), metadata.num_sequences);
    } else {
        // ROW_MAJOR: need to read from different sequences
        for (uint32_t seq = 0; seq < metadata.num_sequences; ++seq) {
            uint64_t offset = metadata.get_offset(seq, pos_index);
            file.seekg(offset);
            file.read(&position[seq], 1);
        }
    }

    if (!file.good()) {
        throw std::runtime_error("Error reading position from PFF file");
    }

    return position;
}

} // namespace fasta
