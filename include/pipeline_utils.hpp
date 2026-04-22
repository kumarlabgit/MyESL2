#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
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

// ---- Parallel conversion worker ----
// conv_fn(src_path, dst_path): the actual conversion; may throw.
// Writes .err sidecar on failure. Prints [N/total] OK/FAIL per item.
// Returns {converted, failed}. Queue must be non-empty.
template<typename ConvFn>
inline std::pair<int,int> run_parallel_conversions(
    std::queue<fs::path> queue,       // pass by value (move at call site)
    const fs::path& cache_dir,
    const std::string& cache_ext,     // ".pff" or ".pnf"
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
            fs::path dst = cache_dir / (src.stem().string() + cache_ext);
            fs::path err = cache_dir / (src.stem().string() + ".err");
            try {
                conv_fn(src, dst);
                std::lock_guard<std::mutex> lk(print_mutex);
                ++converted;
                std::cout << "[" << converted + failed << "/" << total
                          << "] OK: " << src.filename() << "\n";
            } catch (const std::exception& e) {
                std::ofstream ef(err);
                if (ef) ef << e.what() << "\n";
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

} // namespace pipeline_utils
