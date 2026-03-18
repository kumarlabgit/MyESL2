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

namespace fs = std::filesystem;

namespace pipeline_utils {

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
