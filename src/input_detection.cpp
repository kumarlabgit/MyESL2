#include "input_detection.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

// Read up to max_bytes from p into a string. Returns empty on any I/O failure.
std::string read_head(const fs::path& p, std::size_t max_bytes = 4096) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return {};
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::string buf(max_bytes, '\0');
    f.read(buf.data(), static_cast<std::streamsize>(max_bytes));
    buf.resize(static_cast<std::size_t>(f.gcount()));
    return buf;
}

} // namespace

namespace input_detection {

bool is_single_fasta(const fs::path& p) {
    std::string head = read_head(p);
    for (char c : head) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc)) continue;
        return c == '>';
    }
    return false;
}

bool is_single_numeric_table(const fs::path& p) {
    std::string head = read_head(p);
    // Walk the first non-empty line; report true if it contains a tab.
    bool on_line = false;
    bool any_content = false;
    for (char c : head) {
        if (c == '\n' || c == '\r') {
            if (any_content) return false; // first non-empty line had no tab
            on_line = false;
            continue;
        }
        if (!on_line && std::isspace(static_cast<unsigned char>(c))) continue;
        on_line = true;
        any_content = true;
        if (c == '\t') return true;
    }
    return false;
}

bool is_single_vcf(const fs::path& p) {
    // A gzipped VCF is opaque here (read_head does not decompress), so fall back
    // to the extension for those; an uncompressed VCF must start with the
    // mandatory ##fileformat=VCF line.
    std::string s = p.string();
    if (s.size() >= 7 && s.compare(s.size() - 7, 7, ".vcf.gz") == 0) return true;
    return read_head(p, 32).rfind("##fileformat=VCF", 0) == 0;
}

std::string detect_single_file(const fs::path& p, const std::string& datatype) {
    if (is_single_fasta(p)) return "FASTA";
    if (datatype == "numeric" && is_single_numeric_table(p)) return "numeric";
    if (datatype == "vcf" && is_single_vcf(p)) return "VCF";
    return "";
}

std::string maybe_warn_single_file(const fs::path& p, const std::string& datatype) {
    std::string detected = detect_single_file(p, datatype);
    if (detected.empty()) return detected;
    static std::mutex mu;
    static std::unordered_set<std::string> warned;
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(p, ec);
    std::string key = ec ? p.string() : canon.string();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!warned.insert(key).second) return detected;
    }
    std::cerr << "[Warning] " << p
              << " looks like a single " << detected
              << " file rather than a list of paths. "
              << "Treating it as a 1-entry list.\n";
    return detected;
}

} // namespace input_detection
