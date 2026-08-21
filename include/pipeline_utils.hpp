#pragma once
#include "atomic_out.hpp"
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef __linux__
#include <malloc.h>          // malloc_trim
#elif defined(__APPLE__)
#include <malloc/malloc.h>   // malloc_zone_pressure_relief
#elif defined(_WIN32)
#include <malloc.h>          // _heapmin
#endif

namespace fs = std::filesystem;

namespace pipeline_utils {

// Stem of an input file, used as its gene identity everywhere downstream (cache
// filename, feature-label prefix, GSS keys). Strips a trailing .gz first so
// "sample.vcf.gz" and "sample.vcf" both yield "sample" -- otherwise compressed
// input would carry ".vcf" through every gene name and a model trained on one
// form could not be evaluated against the other. A no-op for .fasta/.txt/.vcf.
inline std::string input_stem(const std::filesystem::path& p) {
    std::filesystem::path q = p;
    if (q.extension() == ".gz") q = q.stem();
    return q.stem().string();
}


// Read entire text file contents, transparently decoding UTF-16 LE/BE BOM-prefixed
// inputs to UTF-8 and stripping a UTF-8 BOM when present. Without a BOM, the file
// is returned verbatim (assumed already UTF-8/ASCII). Throws on open failure,
// odd-byte-count UTF-16, or malformed surrogate pairs.
//
// Motivation: Windows PowerShell's default redirection (`>`, `Out-File`) writes
// UTF-16 LE with BOM. When such a file is fed to a byte-oriented `getline` loop,
// every other byte is `\x00`, which then gets embedded into `fs::path` strings.
// OS-level path ops then truncate at the first null, producing very confusing
// errors (e.g. "END_METADATA not found in: <cache_dir>/"). This helper lets
// callers feed PowerShell-generated list files directly.
inline std::string read_text_file_utf8(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + path.string());

    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    auto u8 = [&](size_t i) -> unsigned char {
        return static_cast<unsigned char>(bytes[i]);
    };

    if (bytes.size() >= 3 && u8(0) == 0xEF && u8(1) == 0xBB && u8(2) == 0xBF)
        return std::string(bytes.begin() + 3, bytes.end());

    const bool utf16_le = bytes.size() >= 2 && u8(0) == 0xFF && u8(1) == 0xFE;
    const bool utf16_be = bytes.size() >= 2 && u8(0) == 0xFE && u8(1) == 0xFF;
    if (!utf16_le && !utf16_be)
        return std::string(bytes.begin(), bytes.end());

    if ((bytes.size() - 2) % 2 != 0)
        throw std::runtime_error(
            "File has UTF-16 BOM but odd byte count: " + path.string());

    auto read_u16 = [&](size_t pos) -> uint16_t {
        return utf16_le
            ? static_cast<uint16_t>(u8(pos) | (u8(pos + 1) << 8))
            : static_cast<uint16_t>((u8(pos) << 8) | u8(pos + 1));
    };

    std::string out;
    out.reserve(bytes.size());
    for (size_t i = 2; i < bytes.size(); i += 2) {
        uint32_t cp = read_u16(i);
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (i + 4 > bytes.size())
                throw std::runtime_error(
                    "Truncated UTF-16 surrogate pair in: " + path.string());
            uint32_t low = read_u16(i + 2);
            if (low < 0xDC00 || low > 0xDFFF)
                throw std::runtime_error(
                    "Invalid UTF-16 surrogate pair in: " + path.string());
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            i += 2;
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            throw std::runtime_error(
                "Unpaired UTF-16 low surrogate in: " + path.string());
        }
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// Read current VmRSS from /proc/self/status (Linux only).
// Returns RSS in bytes; 0 on failure or non-Linux.
inline uint64_t current_rss_bytes()
{
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // Format: "VmRSS:    1234 kB"
            uint64_t kb = 0;
            for (char c : line) if (c >= '0' && c <= '9') kb = kb * 10 + (c - '0');
            return kb * 1024;
        }
    }
#endif
    return 0;
}

inline std::string fmt_rss(uint64_t bytes)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << bytes / double(uint64_t(1) << 30) << " GiB";
    return oss.str();
}

inline void log_rss(const char* label)
{
    uint64_t rss = current_rss_bytes();
    if (rss > 0)
        std::cout << "  [RSS] " << label << ": " << fmt_rss(rss) << "\n";
}

// Ask the platform allocator to return freed pages to the OS.
// glibc (Linux) retains freed heap pages by default; without this call,
// RSS stays high even after large allocations are freed, which can cause
// OOM when the next large allocation lands on top of the retained pages.
inline void release_freed_heap()
{
#ifdef __linux__
    malloc_trim(0);
#elif defined(__APPLE__)
    malloc_zone_pressure_relief(NULL, 0);
#elif defined(_WIN32)
    _heapmin();
#endif
}

// ---- INI reader: returns chars= value for [section] ----
inline std::unordered_set<char> load_datatype_chars(
    const fs::path& ini_path, const std::string& section)
{
    std::ifstream f(ini_path);
    std::string line, cur;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { cur = line.substr(1, line.find(']') - 1); continue; }
        if (cur == section) {
            auto eq = line.find('=');
            if (eq != std::string::npos && line.substr(0, eq) == "chars") {
                std::unordered_set<char> s;
                for (char c : line.substr(eq + 1)) s.insert(c);
                return s;
            }
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Content fingerprints (FNV-1a, 64-bit)
//
// Used by run_manifest to decide whether a resumed run is continuing the same
// work. Not a security hash -- it only has to notice that a file or an input
// set changed between two runs on the same machine.
// ---------------------------------------------------------------------------

inline uint64_t fnv1a(const void* data, size_t len, uint64_t h = 1469598103934665603ULL)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fnv1a_str(const std::string& s, uint64_t h = 1469598103934665603ULL)
{
    return fnv1a(s.data(), s.size(), h);
}

/// Hash a file's contents. Returns 0 when the file cannot be read, so a missing
/// file compares unequal to a present one rather than silently matching.
inline uint64_t hash_file(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return 0;
    uint64_t h = 1469598103934665603ULL;
    char buf[64 * 1024];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0)
        h = fnv1a(buf, static_cast<size_t>(f.gcount()), h);
    return h;
}

inline std::string hex64(uint64_t v)
{
    static const char* d = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) { out[i] = d[v & 0xF]; v >>= 4; }
    return out;
}

// ---------------------------------------------------------------------------
// Conversion failure sidecars (<cache_dir>/<stem>.err)
//
// A failed conversion drops a sidecar so the file is not retried on every run.
// The sidecar is honoured only while it still describes the CURRENT input: its
// first line records the absolute source path, and it is ignored when that path
// differs or the source has been modified since the failure.
//
// Without that check a sidecar outlives whatever caused it -- move the inputs,
// or fix the file it complained about, and the affected gene stays suppressed
// forever, with no cure but deleting the sidecar by hand.
// ---------------------------------------------------------------------------

inline void write_err(const fs::path& err_path, const fs::path& src,
                      const std::string& message)
{
    std::ofstream ef(err_path);
    if (!ef) return;
    ef << "source_path=" << fs::absolute(src).string() << "\n" << message << "\n";
}

/// Drop a sidecar judged not to block: it describes an input that no longer
/// matches, so it has no further purpose and would otherwise be re-read forever.
inline void clear_err(const fs::path& err_path)
{
    std::error_code ec;
    fs::remove(err_path, ec);
}

/// True when err_path is a live failure record for src, so src should be skipped.
inline bool err_blocks(const fs::path& err_path, const fs::path& src)
{
    std::error_code ec;
    if (!fs::exists(err_path, ec)) return false;

    std::ifstream ef(err_path);
    std::string first;
    if (!ef || !std::getline(ef, first)) return false;      // unreadable: retry
    if (!first.empty() && first.back() == '\r') first.pop_back();

    // Sidecars written before this format carry no header. Treat them as stale
    // so existing ones self-heal on the next run instead of needing cleanup.
    const std::string key = "source_path=";
    if (first.rfind(key, 0) != 0) return false;
    if (first.substr(key.size()) != fs::absolute(src).string()) return false;

    // Source touched since the failure: the user may have fixed it.
    auto err_t = fs::last_write_time(err_path, ec);
    if (ec) return true;
    auto src_t = fs::last_write_time(src, ec);
    if (ec) return true;
    return src_t <= err_t;
}

// ---- Parallel conversion worker ----
// conv_fn(src_path, dst_path): the actual conversion; may throw.
// Writes .err sidecar on failure. Prints [N/total] OK/FAIL per item.
// Returns {converted, failed}. Queue must be non-empty.
template<typename ConvFn>
inline std::pair<int,int> run_parallel_conversions(
    std::queue<fs::path> queue,       // pass by value (move at call site)
    const fs::path& cache_dir,
    const std::string& cache_ext,     // ".pff", ".pnf" or ".vnf"
    unsigned int num_threads,
    ConvFn conv_fn)
{
    int converted = 0, failed = 0;
    const int total = static_cast<int>(queue.size());
    if (total == 0) return {0, 0};

    std::mutex queue_mutex, print_mutex;

    auto worker = [&]() {
        while (true) {
            fs::path src;
            { std::lock_guard<std::mutex> lk(queue_mutex);
              if (queue.empty()) break;
              src = queue.front(); queue.pop(); }
            // input_stem, not stem(): callers locate these files by input_stem, so
            // a .vcf.gz source must land on "sample.vnf", not "sample.vcf.vnf".
            fs::path dst = cache_dir / (input_stem(src) + cache_ext);
            fs::path err = cache_dir / (input_stem(src) + ".err");
            // Convert into a temporary and rename into place, so a crash
            // mid-conversion leaves no cache file at all rather than a partial
            // one. read_pff_metadata/read_pnf_metadata reject a short payload,
            // so a partial file would already be re-converted rather than
            // trusted -- this just stops it existing in the first place, which
            // also keeps a concurrent reader from seeing a half-written cache.
            fs::path tmp = dst;
            tmp += ".tmp";
            try {
                conv_fn(src, tmp);
                std::error_code mv_ec;
                fs::rename(tmp, dst, mv_ec);
                if (mv_ec) {
                    fs::copy_file(tmp, dst, fs::copy_options::overwrite_existing, mv_ec);
                    fs::remove(tmp, mv_ec);
                    if (mv_ec)
                        throw std::runtime_error("Cannot publish cache file: " + dst.string());
                }
                // Clear any sidecar that was ignored as stale, so it does not
                // block this file again on the next run.
                std::error_code rm_ec;
                fs::remove(err, rm_ec);
                std::lock_guard<std::mutex> lk(print_mutex);
                ++converted;
                std::cout << "[" << converted + failed << "/" << total
                          << "] OK: " << src.filename() << "\n";
            } catch (const std::exception& e) {
                std::error_code rm_ec;
                fs::remove(tmp, rm_ec);      // never leave the partial behind
                write_err(err, src, e.what());
                std::lock_guard<std::mutex> lk(print_mutex);
                ++failed;
                std::cerr << "[" << converted + failed << "/" << total
                          << "] FAIL: " << src.filename() << " -> " << e.what() << "\n";
            }
        }
    };

    unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total));
    if (tc == 0) tc = 1;
    std::vector<std::thread> workers;
    workers.reserve(tc);
    for (unsigned int i = 0; i < tc; ++i) workers.emplace_back(worker);
    for (auto& t : workers) t.join();
    return {converted, failed};
}

// ---- NaN-aware median of non-zero values; returns 0 if all zero/NaN ----
inline double median_nonzero(std::vector<double> v) {  // by value intentional
    std::vector<double> nz;
    for (double x : v) if (!std::isnan(x) && x != 0.0) nz.push_back(x);
    if (nz.empty()) return 0.0;
    std::sort(nz.begin(), nz.end());
    size_t m = nz.size() / 2;
    return (nz.size() % 2 == 0) ? (nz[m-1] + nz[m]) * 0.5 : nz[m];
}

// Map each expanded position to its group index using alg_table ranges.
// alg_table: 3 x n_groups (row 0 = 1-based start, row 1 = 1-based end).
// Returns vector of size F where [j] = group index for expanded position j.
inline std::vector<int> build_group_assignment(const arma::mat& alg_table, size_t F) {
    std::vector<int> group_of(F, -1);
    for (arma::uword gi = 0; gi < alg_table.n_cols; ++gi) {
        int start = static_cast<int>(alg_table(0, gi)) - 1;
        int end   = static_cast<int>(alg_table(1, gi)) - 1;
        for (int j = start; j <= end && j < static_cast<int>(F); ++j)
            group_of[j] = static_cast<int>(gi);
    }
    return group_of;
}

} // namespace pipeline_utils
