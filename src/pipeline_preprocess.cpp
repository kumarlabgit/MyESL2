#include "pipeline_preprocess.hpp"
#include "fasta_parser.hpp"
#include "numeric_parser.hpp"
#include "newick.hpp"
#include "process_log.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <limits>
#include <chrono>
#include <unordered_map>

namespace pipeline {

// ---------------------------------------------------------------------------
// Static helper: parse an INI file for chars= under [section]
// ---------------------------------------------------------------------------
static std::unordered_set<char> load_datatype_chars(
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
// write_preprocess_config
// ---------------------------------------------------------------------------
void write_preprocess_config(const fs::path& output_dir, const PreprocessOptions& opts)
{
    fs::path cfg = output_dir / "preprocess_config";
    std::ofstream f(cfg);
    if (!f) throw std::runtime_error("Cannot write preprocess_config: " + cfg.string());

    f << "list_path=" << fs::absolute(opts.list_path).string() << "\n";
    f << "cache_dir=" << fs::absolute(opts.cache_dir).string() << "\n";
    f << "orientation=" << (opts.orientation == pff::Orientation::COLUMN_MAJOR ? "column" : "row") << "\n";
    f << "datatype=" << opts.datatype << "\n";
    f << "num_threads=" << opts.num_threads << "\n";
    f << "min_minor=" << opts.min_minor << "\n";
    f << "use_dlt=" << (opts.use_dlt ? "true" : "false") << "\n";
}

// ---------------------------------------------------------------------------
// read_preprocess_config
// ---------------------------------------------------------------------------
PreprocessOptions read_preprocess_config(const fs::path& output_dir)
{
    fs::path cfg = output_dir / "preprocess_config";
    std::ifstream f(cfg);
    if (!f) throw std::runtime_error("Cannot read preprocess_config: " + cfg.string());

    PreprocessOptions opts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key   = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        if (!value.empty() && value.back() == '\r') value.pop_back();

        if      (key == "list_path")   opts.list_path   = value;
        else if (key == "cache_dir")   opts.cache_dir   = value;
        else if (key == "orientation") {
            if (value == "row")
                opts.orientation = pff::Orientation::ROW_MAJOR;
            else
                opts.orientation = pff::Orientation::COLUMN_MAJOR;
        }
        else if (key == "datatype")    opts.datatype    = value;
        else if (key == "num_threads") opts.num_threads = static_cast<unsigned int>(std::stoul(value));
        else if (key == "min_minor")   opts.min_minor   = std::stoi(value);
        else if (key == "use_dlt")     opts.use_dlt     = (value == "true");
    }
    return opts;
}

// ---------------------------------------------------------------------------
// preprocess
// ---------------------------------------------------------------------------
std::vector<fs::path> preprocess(const PreprocessOptions& opts)
{
    // --- Resolve mutable working copy of options ---
    PreprocessOptions resolved = opts;

    if (resolved.num_threads == 0) {
        resolved.num_threads = std::thread::hardware_concurrency();
        if (resolved.num_threads == 0) resolved.num_threads = 1;
    }

    if (resolved.cache_dir.empty()) {
        resolved.cache_dir = fs::current_path() / "pff_cache";
    }

    fs::create_directories(resolved.cache_dir);
    fs::create_directories(resolved.output_dir);

    // --- Load allowed chars if needed ---
    std::unordered_set<char> allowed_chars;
    if (resolved.datatype != "universal" && resolved.datatype != "numeric") {
        fs::path ini = resolved.binary_dir / "data_defs.ini";
        if (!resolved.binary_dir.empty() && !fs::exists(ini))
            ini = fs::current_path() / "data_defs.ini";
        if (resolved.binary_dir.empty())
            ini = fs::current_path() / "data_defs.ini";
        if (!fs::exists(ini))
            throw std::runtime_error("data_defs.ini not found");
        allowed_chars = load_datatype_chars(ini, resolved.datatype);
        if (allowed_chars.empty())
            throw std::runtime_error(
                "No chars defined for '" + resolved.datatype + "' in data_defs.ini");
    }

    // --- Read list file — supports overlapping groups (comma-separated files per line) ---
    std::vector<fs::path> all_fasta_paths;
    std::vector<std::vector<fs::path>> groups;
    std::unordered_map<std::string, size_t> stem_to_unique_idx;
    {
        std::ifstream list_file(resolved.list_path);
        if (!list_file)
            throw std::runtime_error(
                "Cannot open list file: " + resolved.list_path.string());
        fs::path list_dir = resolved.list_path.parent_path();
        std::string line;
        while (std::getline(list_file, line)) {
            if (line.empty()) continue;
            std::vector<fs::path> group;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                while (!token.empty() && (token.front() == ' ' || token.front() == '\t'))
                    token.erase(token.begin());
                while (!token.empty() &&
                       (token.back() == ' ' || token.back() == '\t' || token.back() == '\r'))
                    token.pop_back();
                if (token.empty()) continue;
                for (char& c : token) if (c == '\\') c = '/';
                fs::path p = list_dir / token;
                group.push_back(p);
                std::string stem = p.stem().string();
                if (stem_to_unique_idx.find(stem) == stem_to_unique_idx.end()) {
                    stem_to_unique_idx[stem] = all_fasta_paths.size();
                    all_fasta_paths.push_back(p);
                }
            }
            if (!group.empty()) groups.push_back(std::move(group));
        }
    }

    // --- Phase 1: Conversion ---
    int conv_converted = 0, conv_failed = 0;
    int skipped_done = 0, skipped_error = 0, skipped_mismatch = 0, total_to_convert = 0;

    process_log::Section plog(resolved.output_dir / "process_log.txt", "preprocess");
    plog.param("list_path",   resolved.list_path)
        .param("cache_dir",   resolved.cache_dir)
        .param("datatype",    resolved.datatype)
        .param("num_threads", (int)resolved.num_threads)
        .param("min_minor",   resolved.min_minor)
        .param("use_dlt",     resolved.use_dlt);
    if (!resolved.tree_path.empty()) plog.param("tree_path", resolved.tree_path);

    try {

    if (resolved.datatype == "numeric") {
        // Numeric branch: list entries are tabular files; cache as .pnf
        std::queue<fs::path> convert_queue;
        skipped_done = 0; skipped_error = 0;
        for (auto& tab_path : all_fasta_paths) {
            fs::path pnf_path = resolved.cache_dir / (tab_path.stem().string() + ".pnf");
            fs::path err_path = resolved.cache_dir / (tab_path.stem().string() + ".err");
            if (fs::exists(err_path)) { ++skipped_error; continue; }
            if (fs::exists(pnf_path)) {
                try {
                    auto meta = numeric::read_pnf_metadata(pnf_path);
                    if (meta.source_path == fs::absolute(tab_path).string())
                        { ++skipped_done; continue; }
                } catch (...) {}
            }
            convert_queue.push(tab_path);
        }

        total_to_convert = static_cast<int>(convert_queue.size());
        std::cout << "\n--- Phase 1: Conversion (numeric) ---\n";
        std::cout << "  Cache directory:       " << resolved.cache_dir << "\n";
        std::cout << "  To convert:            " << total_to_convert << "\n";
        std::cout << "  Skipped (done):        " << skipped_done << "\n";
        std::cout << "  Skipped (prior error): " << skipped_error << "\n";
        std::cout << "  Worker threads:        " << resolved.num_threads << "\n\n";

        if (total_to_convert > 0) {
            std::mutex queue_mutex, print_mutex;
            auto conv_start = std::chrono::steady_clock::now();

            auto conv_worker = [&]() {
                while (true) {
                    fs::path tab_path;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (convert_queue.empty()) break;
                        tab_path = convert_queue.front();
                        convert_queue.pop();
                    }
                    fs::path pnf_path = resolved.cache_dir / (tab_path.stem().string() + ".pnf");
                    fs::path err_path = resolved.cache_dir / (tab_path.stem().string() + ".err");
                    try {
                        numeric::tabular_to_pnf(tab_path, pnf_path);
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++conv_converted;
                        std::cout << "[" << conv_converted + conv_failed << "/" << total_to_convert
                                  << "] OK: " << tab_path.filename() << "\n";
                    } catch (const std::exception& e) {
                        std::ofstream err_file(err_path);
                        if (err_file) err_file << e.what() << "\n";
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++conv_failed;
                        std::cerr << "[" << conv_converted + conv_failed << "/" << total_to_convert
                                  << "] FAIL: " << tab_path.filename() << " -> " << e.what() << "\n";
                    }
                }
            };

            unsigned int tc = std::min(resolved.num_threads,
                                       static_cast<unsigned int>(total_to_convert));
            std::vector<std::thread> threads;
            threads.reserve(tc);
            for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(conv_worker);
            for (auto& t : threads) t.join();

            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - conv_start).count();
            std::cout << "\nConversion done. Converted: " << conv_converted
                      << ", Failed: " << conv_failed
                      << " (" << elapsed << "s)\n";
        } else {
            std::cout << "Nothing to convert.\n";
        }
    } else {
        // FASTA branch
        std::queue<fs::path> convert_queue;
        skipped_done = 0; skipped_error = 0; skipped_mismatch = 0;
        for (auto& fasta_path : all_fasta_paths) {
            fs::path pff_path = resolved.cache_dir / (fasta_path.stem().string() + ".pff");
            fs::path err_path = resolved.cache_dir / (fasta_path.stem().string() + ".err");
            if (fs::exists(err_path)) { ++skipped_error; continue; }
            if (fs::exists(pff_path)) {
                try {
                    auto meta = fasta::read_pff_metadata(pff_path);
                    if (meta.orientation == resolved.orientation &&
                        meta.datatype    == resolved.datatype &&
                        meta.source_path == fs::absolute(fasta_path).string())
                        { ++skipped_done; continue; }
                    ++skipped_mismatch;
                } catch (...) {}
            }
            convert_queue.push(fasta_path);
        }

        total_to_convert = static_cast<int>(convert_queue.size());
        std::cout << "\n--- Phase 1: Conversion ---\n";
        std::cout << "  Cache directory:       " << resolved.cache_dir << "\n";
        std::cout << "  To convert:            " << total_to_convert << "\n";
        std::cout << "  Skipped (done):        " << skipped_done << "\n";
        std::cout << "  Skipped (prior error): " << skipped_error << "\n";
        std::cout << "  Re-converting (orientation mismatch): " << skipped_mismatch << "\n";
        std::cout << "  Worker threads:        " << resolved.num_threads << "\n\n";

        if (total_to_convert > 0) {
            std::mutex queue_mutex, print_mutex;
            auto conv_start = std::chrono::steady_clock::now();

            auto conv_worker = [&]() {
                while (true) {
                    fs::path fasta_path;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (convert_queue.empty()) break;
                        fasta_path = convert_queue.front();
                        convert_queue.pop();
                    }
                    fs::path pff_path = resolved.cache_dir / (fasta_path.stem().string() + ".pff");
                    fs::path err_path = resolved.cache_dir / (fasta_path.stem().string() + ".err");
                    try {
                        fasta::fasta_to_pff(fasta_path, pff_path, resolved.orientation,
                                            resolved.datatype, allowed_chars);
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++conv_converted;
                        std::cout << "[" << conv_converted + conv_failed << "/" << total_to_convert
                                  << "] OK: " << fasta_path.filename() << "\n";
                    } catch (const std::exception& e) {
                        std::ofstream err_file(err_path);
                        if (err_file) err_file << e.what() << "\n";
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++conv_failed;
                        std::cerr << "[" << conv_converted + conv_failed << "/" << total_to_convert
                                  << "] FAIL: " << fasta_path.filename() << " -> " << e.what() << "\n";
                    }
                }
            };

            unsigned int tc = std::min(resolved.num_threads,
                                       static_cast<unsigned int>(total_to_convert));
            std::vector<std::thread> threads;
            threads.reserve(tc);
            for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(conv_worker);
            for (auto& t : threads) t.join();

            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - conv_start).count();
            std::cout << "\nConversion done. Converted: " << conv_converted
                      << ", Failed: " << conv_failed
                      << " (" << elapsed << "s)\n";
        } else {
            std::cout << "Nothing to convert.\n";
        }
    }

    // --- Write preprocess_config ---
    write_preprocess_config(resolved.output_dir, resolved);

    // --- Write output_dir/aln_list.txt ---
    {
        fs::path aln_list_out = resolved.output_dir / "aln_list.txt";
        std::ofstream af(aln_list_out);
        if (!af) throw std::runtime_error("Cannot write aln_list.txt: " + aln_list_out.string());
        for (auto& p : all_fasta_paths)
            af << fs::absolute(p).string() << "\n";
    }

    // --- If no tree_path: done ---
    if (resolved.tree_path.empty()) {
        std::ostringstream plog_m;
        plog_m << "files_to_convert = "       << total_to_convert  << "\n"
               << "files_converted = "        << conv_converted    << "\n"
               << "files_skipped_done = "     << skipped_done      << "\n"
               << "files_skipped_error = "    << skipped_error     << "\n"
               << "files_skipped_mismatch = " << skipped_mismatch  << "\n"
               << "files_failed = "           << conv_failed        << "\n";
        plog.finish(plog_m.str());
        return {};
    }

    // =========================================================================
    // DrPhylo hypothesis generation
    // =========================================================================

    auto tree = newick::read_newick_file(resolved.tree_path);

    // Build clade list
    std::vector<std::string> clade_names;
    if (!resolved.clade_list_file.empty()) {
        std::ifstream clf(resolved.clade_list_file);
        if (!clf)
            throw std::runtime_error("Cannot open clade-list: " + resolved.clade_list_file);
        std::string line;
        while (std::getline(clf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) clade_names.push_back(line);
        }
    }
    if (!resolved.gen_clade_spec.empty()) {
        auto comma = resolved.gen_clade_spec.find(',');
        if (comma == std::string::npos)
            throw std::runtime_error("gen_clade_spec must be 'lower,upper'");
        int lower = std::stoi(resolved.gen_clade_spec.substr(0, comma));
        int upper = std::stoi(resolved.gen_clade_spec.substr(comma + 1));
        auto gen_names = newick::auto_name_clades(tree, lower, upper);
        clade_names.insert(clade_names.end(), gen_names.begin(), gen_names.end());
        std::cout << "Auto-named " << gen_names.size()
                  << " clades in [" << lower << "," << upper << "]\n";
    }
    if (clade_names.empty())
        throw std::runtime_error(
            "No clades specified. Use clade_list_file or gen_clade_spec.");

    auto all_tree_leaves = newick::get_leaves(tree);
    std::cout << "Tree has " << all_tree_leaves.size() << " leaves\n";

    // Pre-compute shared helpers for phylo sampling modes
    auto parent_map = newick::build_parent_map(tree);

    // Reversed DFS leaf order for tie-breaking
    auto taxa_list_rev = all_tree_leaves;
    std::reverse(taxa_list_rev.begin(), taxa_list_rev.end());

    // Leaf branch lengths
    std::unordered_map<std::string, double> leaf_bl_map;
    std::function<void(const newick::NewickNode&)> collect_bl =
        [&](const newick::NewickNode& n) {
            if (n.is_leaf()) leaf_bl_map[n.name] = n.branch_length;
            for (auto& c : n.children) collect_bl(c);
        };
    collect_bl(tree);

    // Total branch length (modes 1 and 2 skip trimming when == 0)
    double tree_tbl = 0.0;
    std::function<void(const newick::NewickNode&)> sum_bl =
        [&](const newick::NewickNode& n) {
            tree_tbl += n.branch_length;
            for (auto& c : n.children) sum_bl(c);
        };
    sum_bl(tree);

    // BFS find_node helper
    auto find_node = [](const newick::NewickNode& root, const std::string& name)
            -> const newick::NewickNode* {
        std::vector<const newick::NewickNode*> queue = {&root};
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            auto* n = queue[qi];
            if (n->name == name) return n;
            for (auto& c : n->children) queue.push_back(&c);
        }
        return nullptr;
    };

    const std::string& class_bal_dp = resolved.class_bal_phylo;

    std::vector<fs::path> hyp_files;

    for (auto& clade_name : clade_names) {
        std::cout << "\n=== Preprocess clade: " << clade_name << " ===\n";
        fs::path clade_dir = resolved.output_dir / clade_name;
        fs::create_directories(clade_dir);

        const newick::NewickNode* clade_node = find_node(tree, clade_name);
        if (!clade_node) {
            std::cerr << "Warning: clade '" << clade_name
                      << "' not found in tree, skipping\n";
            continue;
        }

        auto pos_leaves = newick::get_leaves(*clade_node);
        std::set<std::string> pos_set(pos_leaves.begin(), pos_leaves.end());

        std::vector<std::string> neg_leaves;
        for (auto& leaf : all_tree_leaves)
            if (!pos_set.count(leaf)) neg_leaves.push_back(leaf);

        if (class_bal_dp == "phylo" || class_bal_dp == "phylo_1" || class_bal_dp == "phylo_2") {
            std::unordered_map<std::string, int> resp;
            for (auto& leaf : all_tree_leaves) resp[leaf] = 0;
            for (auto& leaf : pos_leaves)      resp[leaf] = 1;
            int pos_count = static_cast<int>(pos_leaves.size());

            // Recompute response_sum
            auto recompute = [&]() {
                int s = 0;
                for (auto& [l, r] : resp) s += r;
                return s;
            };

            // Trim one leaf from set: smallest branch_length, reversed-DFS tie-break
            auto trim_one = [&](std::vector<std::string>& set) {
                double min_bl = std::numeric_limits<double>::infinity();
                for (auto& leaf : set) min_bl = std::min(min_bl, leaf_bl_map[leaf]);
                std::set<std::string> ties;
                for (auto& leaf : set) if (leaf_bl_map[leaf] == min_bl) ties.insert(leaf);
                for (auto& taxa : taxa_list_rev) {
                    if (ties.count(taxa)) {
                        resp[taxa] = 0;
                        set.erase(std::find(set.begin(), set.end(), taxa));
                        return;
                    }
                }
            };

            // Assign all unassigned leaves under a node as -1
            auto assign_neg = [&](const newick::NewickNode& n) {
                for (auto& leaf : newick::get_leaves(n))
                    if (resp[leaf] == 0) resp[leaf] = -1;
            };

            // Recursive walk up tree assigning negatives
            auto recursive_walk = [&]() {
                const newick::NewickNode* target = clade_node;
                int response_sum = pos_count;
                while (response_sum > 0) {
                    auto pit = parent_map.find(target);
                    if (pit == parent_map.end()) {
                        // at root
                        for (auto& [l, r] : resp) if (r == 0) r = -1;
                        response_sum = 0;
                        break;
                    }
                    const newick::NewickNode* par = pit->second;
                    assign_neg(*par);
                    response_sum = recompute();
                    target = par;

                    if (response_sum < static_cast<int>(0.1 * pos_count)) {
                        auto git = parent_map.find(target);
                        const newick::NewickNode* gpar =
                            (git != parent_map.end()) ? git->second : &tree;
                        int next_sum = 0;
                        for (auto& leaf : newick::get_leaves(*gpar))
                            next_sum += (resp[leaf] == 1) ? 1 : -1;
                        if (std::abs(next_sum) > response_sum * 4)
                            response_sum = 0;
                    }
                }
            };

            if (class_bal_dp == "phylo_1") {
                // Mode 1: immediate parent only
                auto pit = parent_map.find(clade_node);
                const newick::NewickNode* par =
                    (pit != parent_map.end()) ? pit->second : &tree;
                assign_neg(*par);
                int response_sum = recompute();
                if (tree_tbl > 0 && response_sum <= -1) {
                    std::vector<std::string> neg_set;
                    for (auto& leaf : all_tree_leaves)
                        if (resp[leaf] == -1) neg_set.push_back(leaf);
                    for (int i = 0; i < -response_sum; ++i) trim_one(neg_set);
                }
            } else if (class_bal_dp == "phylo_2") {
                // Mode 2: recursive walk + trim both directions
                recursive_walk();
                int response_sum = recompute();
                if (tree_tbl > 0 && response_sum < 0) {
                    std::vector<std::string> neg_set;
                    for (auto& leaf : all_tree_leaves)
                        if (resp[leaf] == -1) neg_set.push_back(leaf);
                    for (int i = 0; i < -response_sum; ++i) trim_one(neg_set);
                    response_sum = recompute();
                }
                if (tree_tbl > 0 && response_sum > 0) {
                    std::vector<std::string> pos_set_trim;
                    for (auto& leaf : all_tree_leaves)
                        if (resp[leaf] == 1) pos_set_trim.push_back(leaf);
                    for (int i = 0; i < response_sum; ++i) trim_one(pos_set_trim);
                }
            } else { // "phylo" → Mode 3 (default)
                // Mode 3: recursive walk + trim negatives only (no tree_tbl guard)
                recursive_walk();
                int response_sum = recompute();
                if (response_sum <= -1) {
                    std::vector<std::string> neg_set;
                    for (auto& leaf : all_tree_leaves)
                        if (resp[leaf] == -1) neg_set.push_back(leaf);
                    for (int i = 0; i < -response_sum; ++i) trim_one(neg_set);
                }
            }

            // Rebuild neg_leaves (and pos_leaves for mode 2 which may trim positives)
            neg_leaves.clear();
            for (auto& leaf : all_tree_leaves)
                if (resp[leaf] == -1) neg_leaves.push_back(leaf);
            if (class_bal_dp == "phylo_2") {
                pos_leaves.clear();
                for (auto& leaf : all_tree_leaves)
                    if (resp[leaf] == 1) pos_leaves.push_back(leaf);
            }
        }

        fs::path hyp_file = clade_dir / "hypothesis.txt";
        {
            std::ofstream hf(hyp_file);
            if (!hf) throw std::runtime_error(
                "Cannot write hypothesis file: " + hyp_file.string());
            for (auto& s : pos_leaves) hf << s << "\t1\n";
            for (auto& s : neg_leaves) hf << s << "\t-1\n";
        }
        std::cout << "  Written: " << hyp_file
                  << "  (pos=" << pos_leaves.size()
                  << ", neg=" << neg_leaves.size() << ")\n";

        hyp_files.push_back(hyp_file);
    }

    std::ostringstream plog_m;
    plog_m << "files_to_convert = "       << total_to_convert  << "\n"
           << "files_converted = "        << conv_converted    << "\n"
           << "files_skipped_done = "     << skipped_done      << "\n"
           << "files_skipped_error = "    << skipped_error     << "\n"
           << "files_skipped_mismatch = " << skipped_mismatch  << "\n"
           << "files_failed = "           << conv_failed        << "\n";
    plog.finish(plog_m.str());
    return hyp_files;

    } catch (const std::exception& e) {
        plog.fail(e.what());
        throw;
    }
}

} // namespace pipeline
