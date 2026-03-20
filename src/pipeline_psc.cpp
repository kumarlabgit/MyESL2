#include "pipeline_psc.hpp"
#include "encoder.hpp"
#include "newick.hpp"
#include "regression.hpp"

#include <algorithm>
#include <armadillo>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_set>

namespace fs = std::filesystem;

namespace pipeline::psc {

// ── Utility helpers ───────────────────────────────────────────────────────────

static std::string format_float_trim(double x) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(15) << x;
    std::string s = ss.str();
    auto dot = s.find('.');
    if (dot != std::string::npos) {
        size_t last = s.size() - 1;
        while (last > dot && s[last] == '0') --last;
        if (last == dot) --last;
        s.resize(last + 1);
    }
    return s;
}

static double round_to(double x, int digits) {
    double p = std::pow(10.0, digits);
    return std::round(x * p) / p;
}

static bool is_fasta_ext(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".fas" || ext == ".fasta" || ext == ".fa" || ext == ".faa";
}

// ── FASTA loading ─────────────────────────────────────────────────────────────

static std::pair<std::unordered_map<std::string, std::vector<uint8_t>>, uint32_t>
read_fasta_map(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open FASTA: " + path.string());

    std::unordered_map<std::string, std::vector<uint8_t>> seq_map;
    std::string current_id;
    std::vector<uint8_t> current_seq;
    uint32_t seq_len = 0;
    bool first = true;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto t = line;
        // trim
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.erase(t.begin());
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
        if (t.empty()) continue;

        if (t[0] == '>') {
            if (!current_id.empty()) {
                uint32_t len = static_cast<uint32_t>(current_seq.size());
                if (first) { seq_len = len; first = false; }
                else if (len != seq_len)
                    throw std::runtime_error("Sequence length mismatch in " + path.string()
                        + " for species " + current_id);
                seq_map[std::move(current_id)] = std::move(current_seq);
                current_seq.clear();
            }
            current_id = t.substr(1);
            // trim id
            while (!current_id.empty() && std::isspace(static_cast<unsigned char>(current_id.front())))
                current_id.erase(current_id.begin());
            while (!current_id.empty() && std::isspace(static_cast<unsigned char>(current_id.back())))
                current_id.pop_back();
        } else if (!current_id.empty()) {
            for (char c : t) {
                if (c == '?') current_seq.push_back('-');
                else current_seq.push_back(static_cast<uint8_t>(std::toupper(static_cast<unsigned char>(c))));
            }
        }
    }

    if (!current_id.empty()) {
        uint32_t len = static_cast<uint32_t>(current_seq.size());
        if (first) { seq_len = len; first = false; }
        else if (len != seq_len)
            throw std::runtime_error("Sequence length mismatch in " + path.string()
                + " for species " + current_id);
        seq_map[std::move(current_id)] = std::move(current_seq);
    }

    if (seq_map.empty())
        throw std::runtime_error("No sequences parsed from " + path.string());

    return {std::move(seq_map), seq_len};
}

static std::vector<GeneAlignment> load_alignments(
    const fs::path& dir,
    const std::unordered_set<std::string>& limited_genes)
{
    std::vector<fs::path> fasta_files;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_fasta_ext(entry.path()))
            fasta_files.push_back(entry.path());
    }
    // Sort for deterministic order
    std::sort(fasta_files.begin(), fasta_files.end());

    std::vector<GeneAlignment> out;
    out.reserve(fasta_files.size());

    for (auto& fasta : fasta_files) {
        std::string gene_name = fasta.stem().string();
        if (!limited_genes.empty() && !limited_genes.count(gene_name))
            continue;

        auto [seq_map, seq_len] = read_fasta_map(fasta);
        out.push_back({std::move(gene_name), seq_len, std::move(seq_map)});
    }

    return out;
}

// ── Species groups & Cartesian product ────────────────────────────────────────

static std::vector<std::vector<std::string>> parse_species_groups(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open species groups: " + path.string());

    std::vector<std::vector<std::string>> groups;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // trim
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (line.empty()) continue;

        std::vector<std::string> members;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // trim token
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) token.erase(token.begin());
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.pop_back();
            if (token.empty())
                throw std::runtime_error("Invalid species groups at line " + std::to_string(line_num));
            members.push_back(std::move(token));
        }
        groups.push_back(std::move(members));
    }

    if (groups.empty())
        throw std::runtime_error("Species groups file is empty: " + path.string());
    if (groups.size() % 2 != 0)
        throw std::runtime_error("Species groups file must have an even number of lines: " + path.string());

    return groups;
}

static std::vector<std::vector<std::string>> expand_species_groups(
    const std::vector<std::vector<std::string>>& groups)
{
    std::vector<std::vector<std::string>> combos = {{}};
    for (auto& group : groups) {
        std::vector<std::vector<std::string>> next;
        for (auto& combo : combos)
            for (auto& member : group) {
                auto c = combo;
                c.push_back(member);
                next.push_back(std::move(c));
            }
        combos = std::move(next);
    }
    return combos;
}

static std::string make_combo_tag(const std::vector<std::string>& species, const std::string& label) {
    if (species.size() > 12) return label;
    bool use_underscore = !species.empty() && species[0].find('_') != std::string::npos;

    std::string tag;
    for (size_t i = 0; i < species.size(); ++i) {
        if (i > 0) tag += '.';
        if (use_underscore) {
            auto pos = species[i].find('_');
            std::string part = (pos != std::string::npos) ? species[i].substr(pos + 1) : species[i];
            tag += part.substr(0, 3);
        } else {
            tag += species[i].substr(0, 3);
        }
    }
    return tag;
}

// ── Response matrix reading ───────────────────────────────────────────────────

static std::pair<std::vector<std::string>, std::vector<double>>
read_response_matrix(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open response matrix: " + path.string());

    std::vector<std::string> species;
    std::vector<double> values;
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (line.empty()) continue;

        // Split by tab or comma
        char delim = (line.find('\t') != std::string::npos) ? '\t' : ',';
        auto dpos = line.find(delim);
        if (dpos == std::string::npos)
            throw std::runtime_error("Bad response line (no delimiter): " + line);

        std::string sp = line.substr(0, dpos);
        std::string val_str = line.substr(dpos + 1);
        // trim
        while (!sp.empty() && std::isspace(static_cast<unsigned char>(sp.front()))) sp.erase(sp.begin());
        while (!sp.empty() && std::isspace(static_cast<unsigned char>(sp.back()))) sp.pop_back();
        while (!val_str.empty() && std::isspace(static_cast<unsigned char>(val_str.front()))) val_str.erase(val_str.begin());
        while (!val_str.empty() && std::isspace(static_cast<unsigned char>(val_str.back()))) val_str.pop_back();

        if (sp.empty()) continue;
        try {
            double v = std::stod(val_str);
            species.push_back(std::move(sp));
            values.push_back(v);
        } catch (...) {
            continue; // skip unparseable
        }
    }

    if (species.empty())
        throw std::runtime_error("No valid entries in response matrix: " + path.string());

    return {std::move(species), std::move(values)};
}

// ── Species phenotypes ────────────────────────────────────────────────────────

static std::unordered_map<std::string, double>
read_species_phenotypes(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open phenotypes: " + path.string());

    std::unordered_map<std::string, double> values;
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (line.empty()) continue;

        // CSV: species,value
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;

        std::string sp = line.substr(0, comma);
        std::string val_str = line.substr(comma + 1);
        while (!sp.empty() && std::isspace(static_cast<unsigned char>(sp.front()))) sp.erase(sp.begin());
        while (!sp.empty() && std::isspace(static_cast<unsigned char>(sp.back()))) sp.pop_back();
        while (!val_str.empty() && std::isspace(static_cast<unsigned char>(val_str.front()))) val_str.erase(val_str.begin());
        while (!val_str.empty() && std::isspace(static_cast<unsigned char>(val_str.back()))) val_str.pop_back();

        if (sp.empty()) continue;
        // Skip header
        {
            std::string sp_lower = sp;
            for (auto& c : sp_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (sp_lower == "species") continue;
        }
        try {
            values[sp] = std::stod(val_str);
        } catch (...) {
            continue;
        }
    }

    return values;
}

// ── Combo job building ────────────────────────────────────────────────────────

static std::vector<double> binary_pair_labels(size_t n) {
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = (i % 2 == 0) ? 1.0 : -1.0;
    return y;
}

static std::vector<ComboJob> build_combo_jobs(const PscOptions& opts) {
    // Mode 1: response file
    if (!opts.response_file.empty()) {
        auto [species, y_raw] = read_response_matrix(opts.response_file);
        std::string label = "combo_0";
        return {{0, label, make_combo_tag(species, label), std::move(species), std::move(y_raw)}};
    }

    // Mode 2: response dir
    if (!opts.response_dir.empty()) {
        std::vector<fs::path> files;
        for (auto& entry : fs::directory_iterator(opts.response_dir))
            if (entry.is_regular_file()) files.push_back(entry.path());
        std::sort(files.begin(), files.end());

        std::vector<ComboJob> combos;
        for (size_t i = 0; i < files.size(); ++i) {
            auto [species, y_raw] = read_response_matrix(files[i]);
            std::string label = "combo_" + std::to_string(i);
            combos.push_back({i, label, make_combo_tag(species, label),
                              std::move(species), std::move(y_raw)});
        }
        return combos;
    }

    // Mode 3: species groups file
    if (!opts.species_groups_file.empty()) {
        auto groups = parse_species_groups(opts.species_groups_file);
        auto expanded = expand_species_groups(groups);

        std::vector<ComboJob> combos;
        for (size_t i = 0; i < expanded.size(); ++i) {
            std::string label = "combo_" + std::to_string(i);
            auto y = binary_pair_labels(expanded[i].size());
            combos.push_back({i, label, make_combo_tag(expanded[i], label),
                              std::move(expanded[i]), std::move(y)});
        }
        return combos;
    }

    throw std::runtime_error("No species contrast source specified");
}

// ── Null model generation ─────────────────────────────────────────────────────

static std::vector<std::vector<size_t>> choose_k(const std::vector<size_t>& items, size_t k) {
    std::vector<std::vector<size_t>> result;
    std::vector<size_t> current;
    std::function<void(size_t)> rec = [&](size_t start) {
        if (current.size() == k) { result.push_back(current); return; }
        for (size_t i = start; i < items.size(); ++i) {
            current.push_back(items[i]);
            rec(i + 1);
            current.pop_back();
        }
    };
    rec(0);
    return result;
}

static std::vector<ComboJob> make_null_combo_jobs(const std::vector<ComboJob>& combos) {
    std::vector<ComboJob> out;

    for (size_t combo_num = 0; combo_num < combos.size(); ++combo_num) {
        auto& combo = combos[combo_num];
        if (combo.species.size() % 4 != 0)
            throw std::runtime_error("--make-null-models requires an even number of pairs per combo");

        size_t num_pairs = combo.species.size() / 2;
        std::vector<size_t> pair_indices(num_pairs);
        std::iota(pair_indices.begin(), pair_indices.end(), 0);
        size_t k = num_pairs / 2;

        auto combos_idx = choose_k(pair_indices, k);
        if (combo_num % 2 == 1)
            std::reverse(combos_idx.begin(), combos_idx.end());

        // Build species -> y mapping from original combo
        std::unordered_map<std::string, double> sp_to_y;
        for (size_t i = 0; i < combo.species.size(); ++i)
            sp_to_y[combo.species[i]] = combo.y_raw[i];

        std::set<std::vector<size_t>> seen;
        for (auto& subset : combos_idx) {
            auto subset_sorted = subset;
            std::sort(subset_sorted.begin(), subset_sorted.end());

            // Compute mirror
            std::vector<size_t> mirror;
            for (auto& idx : pair_indices) {
                if (!std::binary_search(subset_sorted.begin(), subset_sorted.end(), idx))
                    mirror.push_back(idx);
            }
            std::sort(mirror.begin(), mirror.end());

            if (seen.count(mirror)) continue;
            seen.insert(subset_sorted);

            // Swap pairs in subset
            auto new_species = combo.species;
            for (auto pidx : subset_sorted) {
                size_t a = 2 * pidx;
                size_t b = a + 1;
                std::swap(new_species[a], new_species[b]);
            }

            // Remap y values
            std::vector<double> new_y;
            new_y.reserve(new_species.size());
            for (auto& sp : new_species)
                new_y.push_back(sp_to_y.count(sp) ? sp_to_y[sp] : 0.0);

            out.push_back({0, "", "", std::move(new_species), std::move(new_y)});
        }
    }

    // Renumber
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].index = i;
        out[i].combo_label = "combo_" + std::to_string(i);
        out[i].combo_tag = make_combo_tag(out[i].species, out[i].combo_label);
    }

    return out;
}

// ── Gap cancellation ──────────────────────────────────────────────────────────

static void apply_gap_cancellation(
    std::vector<std::vector<uint8_t>>& seqs,
    const std::vector<bool>& missing_flags,
    const std::vector<uint8_t>* outgroup,
    size_t min_pairs,
    bool cancel_only_partner,
    bool cancel_tri_allelic)
{
    if (seqs.empty()) return;
    size_t seq_len = seqs[0].size();
    size_t n = seqs.size();

    // Check intact pairs
    if (n % 2 == 0) {
        size_t intact_pairs = 0;
        for (size_t p = 0; p + 1 < n; p += 2) {
            if (!missing_flags[p] && !missing_flags[p + 1])
                ++intact_pairs;
        }
        if (intact_pairs < min_pairs) {
            for (auto& seq : seqs)
                std::fill(seq.begin(), seq.end(), '-');
            return;
        }
    }

    for (size_t pos = 0; pos < seq_len; ++pos) {
        bool has_gap = false;
        for (auto& seq : seqs) {
            if (seq[pos] == '-') { has_gap = true; break; }
        }

        if (cancel_only_partner && has_gap) {
            size_t pairs_left = n / 2;
            for (size_t pi = 0; pi < n / 2; ++pi) {
                uint8_t a = seqs[2 * pi][pos];
                uint8_t b = seqs[2 * pi + 1][pos];
                if (a == '-' || b == '-') {
                    seqs[2 * pi][pos] = '-';
                    seqs[2 * pi + 1][pos] = '-';
                    if (pairs_left > 0) --pairs_left;
                }
            }
            if (pairs_left < min_pairs) {
                for (auto& seq : seqs) seq[pos] = '-';
                continue;
            }
        } else if (has_gap) {
            for (auto& seq : seqs) seq[pos] = '-';
            continue;
        }

        // Outgroup check
        if (outgroup && pos < outgroup->size()) {
            uint8_t out_res = (*outgroup)[pos];
            if (out_res != '-') {
                bool mismatch = false;
                for (size_t ci = 1; ci < n; ci += 2) {
                    if (seqs[ci][pos] != out_res) { mismatch = true; break; }
                }
                if (mismatch) {
                    for (auto& seq : seqs) seq[pos] = '-';
                    continue;
                }
            }
        }

        // Tri-allelic check
        if (cancel_tri_allelic && n == 4) {
            std::set<uint8_t> uniq;
            for (auto& seq : seqs) uniq.insert(seq[pos]);
            if (uniq.size() == 3) {
                for (auto& seq : seqs) seq[pos] = '-';
            }
        }
    }
}

// ── Pair randomization ────────────────────────────────────────────────────────

static void apply_pair_randomization(std::vector<std::vector<uint8_t>>& seqs) {
    if (seqs.size() < 2 || seqs.size() % 2 != 0) return;
    size_t seq_len = seqs[0].size();
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution coin(0.5);

    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t pi = 0; pi < seqs.size() / 2; ++pi) {
            if (coin(rng)) {
                size_t a = 2 * pi, b = a + 1;
                std::swap(seqs[a][pos], seqs[b][pos]);
            }
        }
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

static void validate_combo_species_against_alignments(
    const std::vector<ComboJob>& combos,
    const std::vector<GeneAlignment>& alignments)
{
    std::unordered_set<std::string> present;
    for (auto& gene : alignments)
        for (auto& [sp, _] : gene.seqs)
            present.insert(sp);

    std::set<std::string> missing;
    for (auto& combo : combos)
        for (auto& sp : combo.species)
            if (!present.count(sp))
                missing.insert(sp);

    if (!missing.empty()) {
        std::string msg = "Species not found in alignments: ";
        bool first = true;
        for (auto& sp : missing) {
            if (!first) msg += ", ";
            msg += sp;
            first = false;
        }
        throw std::runtime_error(msg);
    }
}

// ── Lambda grid ───────────────────────────────────────────────────────────────

static std::vector<double> linear_space(double start, double end, double step) {
    if (step <= 0) throw std::runtime_error("step must be > 0");
    std::vector<double> out;
    double val = start;
    while (val <= end + 1e-12) {
        out.push_back(round_to(val, 6));
        val += step;
    }
    return out;
}

static std::vector<double> logspace(double start, double end, size_t points) {
    if (start <= 0 || end <= 0) throw std::runtime_error("logspace bounds must be > 0");
    if (points < 2) throw std::runtime_error("logspace requires at least 2 points");

    double log_start = std::log10(start);
    double log_end = std::log10(end);
    double step = (log_end - log_start) / (static_cast<double>(points) - 1.0);

    std::vector<double> out;
    out.reserve(points);
    for (size_t i = 0; i < points; ++i)
        out.push_back(std::pow(10.0, log_start + static_cast<double>(i) * step));
    return out;
}

static std::vector<std::array<double, 2>> build_lambda_grid(const PscOptions& opts) {
    std::vector<std::array<double, 2>> grid;

    if (opts.use_logspace) {
        auto l1_vals = logspace(opts.initial_lambda1, opts.final_lambda1, opts.num_log_points);
        auto l2_vals = logspace(opts.initial_lambda2, opts.final_lambda2, opts.num_log_points);
        int digits = static_cast<int>(std::abs(std::log10(opts.initial_lambda1))) + 5;
        for (double l1 : l1_vals)
            for (double l2 : l2_vals)
                grid.push_back({round_to(l1, digits), round_to(l2, digits)});
    } else {
        double l1 = opts.initial_lambda1;
        while (l1 <= opts.final_lambda1 + 1e-12) {
            double l2 = opts.initial_lambda2;
            while (l2 <= opts.final_lambda2 + 1e-12) {
                grid.push_back({l1, l2});
                l2 = round_to(l2 + opts.lambda_step, 3);
            }
            l1 = round_to(l1 + opts.lambda_step, 3);
        }
    }

    return grid;
}

// ── Group penalty ─────────────────────────────────────────────────────────────

static std::vector<double> build_penalty_terms(
    const PscOptions& opts,
    const std::vector<GeneMeta>& genes)
{
    std::string kind = opts.group_penalty_type;
    for (auto& c : kind) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (kind == "median") {
        std::vector<size_t> vars;
        for (auto& g : genes)
            if (g.var_site_count > 0) vars.push_back(g.var_site_count);
        if (vars.empty()) vars.push_back(1);
        std::sort(vars.begin(), vars.end());
        double median;
        if (vars.size() % 2 == 1) {
            median = static_cast<double>(vars[vars.size() / 2]);
        } else {
            size_t hi = vars.size() / 2;
            median = static_cast<double>(vars[hi - 1] + vars[hi]) / 2.0;
        }
        return {std::floor(median)};
    }

    return linear_space(opts.initial_gp_value, opts.final_gp_value, opts.gp_step);
}

static std::vector<double> compute_group_weights(
    const std::string& kind,
    double penalty_term,
    const std::vector<GeneMeta>& genes)
{
    std::string k = kind;
    for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    std::vector<double> weights;
    weights.reserve(genes.size());
    for (auto& g : genes) {
        size_t feature_len = (g.feature_end >= g.feature_start) ?
            g.feature_end - g.feature_start + 1 : 0;
        double w;
        if (k == "std")         w = std::sqrt(static_cast<double>(feature_len));
        else if (k == "linear" || k == "median") w = static_cast<double>(g.var_site_count) + penalty_term;
        else if (k == "sqrt")   w = std::sqrt(static_cast<double>(g.var_site_count));
        else                    w = std::sqrt(static_cast<double>(feature_len));
        weights.push_back(w);
    }
    return weights;
}

// ── Preprocessing (per-combo, per-gene) ───────────────────────────────────────

struct PreprocessedCombo {
    std::vector<std::string> species;
    std::vector<double> y_model;
    arma::fmat features;
    arma::frowvec responses;
    arma::mat alg_table;     // 3 x n_genes
    std::vector<FeatureMeta> feature_metas;
    std::vector<GeneMeta> gene_metas;
    std::vector<std::string> stems_ordered;
    fs::path combined_map_path;
};

static GeneEncoded preprocess_one_gene(
    const GeneAlignment& gene,
    const ComboJob& combo,
    bool apply_gap_cancel,
    bool randomize_pairs,
    const PscOptions& opts)
{
    // Extract sequences for combo species
    std::vector<std::vector<uint8_t>> seqs;
    std::vector<bool> missing_flags;
    seqs.reserve(combo.species.size());
    missing_flags.reserve(combo.species.size());

    for (auto& sp : combo.species) {
        auto it = gene.seqs.find(sp);
        if (it != gene.seqs.end()) {
            seqs.push_back(it->second);
            missing_flags.push_back(false);
        } else {
            seqs.push_back(std::vector<uint8_t>(gene.seq_len, '-'));
            missing_flags.push_back(true);
        }
    }

    // nix_full_deletions: skip gene entirely if any species missing
    if (opts.nix_full_deletions) {
        for (bool m : missing_flags) {
            if (m) return {gene.name, 0, {}, {}};
        }
    }

    // Gap cancellation
    if (apply_gap_cancel) {
        const std::vector<uint8_t>* outgroup = nullptr;
        std::vector<uint8_t> outgroup_seq;
        if (!opts.outgroup_species.empty()) {
            auto it = gene.seqs.find(opts.outgroup_species);
            if (it != gene.seqs.end()) {
                outgroup_seq = it->second;
                outgroup = &outgroup_seq;
            }
        }
        apply_gap_cancellation(seqs, missing_flags, outgroup,
            opts.min_pairs, opts.cancel_only_partner, opts.cancel_tri_allelic);
    }

    // Pair randomization
    if (randomize_pairs)
        apply_pair_randomization(seqs);

    // Encode
    auto result = encoder::encode_raw_sequences(gene.name, seqs, 2);

    return {gene.name, result.var_site_count, std::move(result.columns), std::move(result.map)};
}

static PreprocessedCombo preprocess_combo(
    const std::vector<GeneAlignment>& alignments,
    const ComboJob& combo,
    bool apply_gap_cancel,
    bool randomize_pairs,
    const PscOptions& opts,
    const fs::path& output_dir)
{
    PreprocessedCombo prep;
    prep.species = combo.species;

    uint32_t N = static_cast<uint32_t>(combo.species.size());

    // Process each gene
    std::vector<GeneEncoded> gene_results;
    gene_results.reserve(alignments.size());
    for (auto& gene : alignments)
        gene_results.push_back(preprocess_one_gene(gene, combo, apply_gap_cancel, randomize_pairs, opts));

    // Count total features
    size_t total_cols = 0;
    for (auto& gr : gene_results) total_cols += gr.columns.size();

    // Build y_model (binary pair labels)
    prep.y_model = binary_pair_labels(N);
    prep.responses.zeros(N);
    for (uint32_t i = 0; i < N; ++i)
        prep.responses(i) = static_cast<float>(prep.y_model[i]);

    // Build features matrix and metadata
    prep.features.zeros(N, static_cast<arma::uword>(total_cols));
    prep.alg_table.zeros(3, static_cast<arma::uword>(gene_results.size()));

    size_t col_offset = 0;
    for (size_t gi = 0; gi < gene_results.size(); ++gi) {
        auto& gr = gene_results[gi];
        size_t n_cols = gr.columns.size();
        size_t feature_start = prep.feature_metas.size();

        for (size_t j = 0; j < n_cols; ++j) {
            for (uint32_t i = 0; i < N; ++i)
                prep.features(i, static_cast<arma::uword>(col_offset + j)) = static_cast<float>(gr.columns[j][i]);

            std::string label = gr.name + "_" + std::to_string(gr.map[j].first) + "_" + gr.map[j].second;
            prep.feature_metas.push_back({label, gi, gr.map[j].first, static_cast<uint8_t>(gr.map[j].second)});
        }

        size_t feature_end = prep.feature_metas.empty() ? 0 : prep.feature_metas.size() - 1;
        prep.gene_metas.push_back({gr.name, feature_start, feature_end, gr.var_site_count});
        prep.stems_ordered.push_back(gr.name);

        // alg_table: 1-based column indices
        prep.alg_table(0, gi) = static_cast<double>(col_offset + 1);
        prep.alg_table(1, gi) = static_cast<double>(col_offset + n_cols);
        // Weight placeholder -- will be updated per penalty term
        prep.alg_table(2, gi) = std::sqrt(static_cast<double>(std::max(n_cols, size_t(1))));

        col_offset += n_cols;
    }

    // Write combined.map
    prep.combined_map_path = output_dir / "combined.map";
    {
        std::ofstream mf(prep.combined_map_path);
        mf << "col_idx\tlabel\n";
        for (size_t j = 0; j < prep.feature_metas.size(); ++j)
            mf << j << "\t" << prep.feature_metas[j].label << "\n";
    }

    return prep;
}

// ── Prediction design ─────────────────────────────────────────────────────────

static PredictionDesign build_prediction_design(
    const std::vector<GeneAlignment>& prediction_alignments,
    const std::vector<FeatureMeta>& features,
    const std::vector<GeneMeta>& genes,
    const std::vector<std::string>& input_species,
    const std::unordered_map<std::string, double>& phenotypes)
{
    std::unordered_set<std::string> input_set(input_species.begin(), input_species.end());

    // Collect all prediction species (not in input, in alphabetical order)
    std::set<std::string> all_species;
    for (auto& gene : prediction_alignments)
        for (auto& [sp, _] : gene.seqs)
            all_species.insert(sp);

    PredictionDesign pd;
    bool have_pheno = !phenotypes.empty();
    for (auto& sp : all_species) {
        if (input_set.count(sp)) continue;
        if (have_pheno && !phenotypes.count(sp)) continue;
        pd.species.push_back(sp);
        if (have_pheno)
            pd.true_values.push_back(phenotypes.at(sp));
        else
            pd.true_values.push_back(std::numeric_limits<double>::quiet_NaN());
    }

    // Build gene_lookup
    std::unordered_map<std::string, const GeneAlignment*> gene_lookup;
    for (auto& gene : prediction_alignments)
        gene_lookup[gene.name] = &gene;

    // Build feature_hit_rows
    pd.feature_hit_rows.resize(features.size());
    for (size_t row_idx = 0; row_idx < pd.species.size(); ++row_idx) {
        auto& sp = pd.species[row_idx];
        for (size_t j = 0; j < features.size(); ++j) {
            auto& fm = features[j];
            auto& gene_name = genes[fm.gene_idx].name;
            auto it = gene_lookup.find(gene_name);
            if (it == gene_lookup.end()) continue;
            auto seq_it = it->second->seqs.find(sp);
            if (seq_it == it->second->seqs.end()) continue;
            auto& seq = seq_it->second;
            if (fm.position < seq.size() && seq[fm.position] == fm.aa)
                pd.feature_hit_rows[j].push_back(row_idx);
        }
    }

    return pd;
}

static std::vector<std::vector<size_t>> build_input_feature_hit_rows(
    const arma::fmat& features)
{
    size_t p = features.n_cols;
    size_t n = features.n_rows;
    std::vector<std::vector<size_t>> rows(p);
    for (size_t j = 0; j < p; ++j)
        for (size_t i = 0; i < n; ++i)
            if (features(i, j) != 0.0f)
                rows[j].push_back(i);
    return rows;
}


// ── Gene stats update ─────────────────────────────────────────────────────────

static void update_best_rank(int& slot, int rank) {
    if (slot < 0 || rank < slot) slot = rank;
}

static void update_gene_stats_for_run(
    const std::vector<double>& gene_gss,  // per-gene GSS
    const std::vector<std::map<size_t, double>>& gene_selected_sites,
    std::vector<double>& combo_highest_gss,
    std::vector<int>& combo_best_rank,
    std::vector<GeneAggregate>& gene_aggregates)
{
    // Build ranked list
    std::vector<std::pair<size_t, double>> ranked;
    for (size_t i = 0; i < gene_gss.size(); ++i)
        if (gene_gss[i] > 0.0)
            ranked.push_back({i, gene_gss[i]});

    // Sort: descending GSS, ascending gene index for ties
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (b.second != a.second) return a.second > b.second;
        return a.first < b.first;
    });

    // Update combo and global stats
    for (auto& [gidx, gss] : ranked) {
        if (gss > combo_highest_gss[gidx]) combo_highest_gss[gidx] = gss;
        if (gss > gene_aggregates[gidx].single_highest_gss)
            gene_aggregates[gidx].single_highest_gss = gss;
    }

    for (size_t rank_idx = 0; rank_idx < ranked.size(); ++rank_idx) {
        int rank = static_cast<int>(rank_idx + 1);
        auto gidx = ranked[rank_idx].first;
        update_best_rank(combo_best_rank[gidx], rank);
        update_best_rank(gene_aggregates[gidx].single_best_rank, rank);
    }

    // Update selected sites
    for (size_t gi = 0; gi < gene_selected_sites.size(); ++gi) {
        for (auto& [pos, pss] : gene_selected_sites[gi]) {
            auto& slot = gene_aggregates[gi].selected_sites[pos];
            if (pss > slot) slot = pss;
        }
    }
}

static void finalize_combo_stats(
    const std::vector<double>& combo_highest_gss,
    const std::vector<int>& combo_best_rank,
    double top_rank_threshold,
    std::vector<GeneAggregate>& gene_aggregates)
{
    for (size_t idx = 0; idx < gene_aggregates.size(); ++idx) {
        if (combo_best_rank[idx] >= 0) {
            gene_aggregates[idx].num_combos_ranked += 1;
            if (static_cast<double>(combo_best_rank[idx]) <= top_rank_threshold)
                gene_aggregates[idx].num_combos_ranked_top += 1;
            if (combo_highest_gss[idx] > gene_aggregates[idx].highest_ever_gss)
                gene_aggregates[idx].highest_ever_gss = combo_highest_gss[idx];
            update_best_rank(gene_aggregates[idx].best_ever_rank, combo_best_rank[idx]);
        }
    }
}

// ── Prediction scoring ────────────────────────────────────────────────────────

struct PredScoreResult {
    std::vector<std::pair<size_t, double>> pred_scores; // (species_idx, score)
    double input_rmse = 0.0;
};

static PredScoreResult compute_prediction_scores(
    const std::vector<double>& y_model,
    const std::vector<std::vector<size_t>>& input_fhr,
    const PredictionDesign& pred,
    const std::vector<double>& beta,
    double intercept)
{
    PredScoreResult result;
    size_t n_input = y_model.size();
    size_t n_pred = pred.species.size();

    std::vector<double> input_scores(n_input, 0.0);
    std::vector<bool> input_touched(n_input, false);
    std::vector<double> pred_scores(n_pred, 0.0);
    std::vector<bool> pred_touched(n_pred, false);

    for (size_t j = 0; j < beta.size(); ++j) {
        if (beta[j] == 0.0) continue;

        if (j < input_fhr.size()) {
            for (auto row : input_fhr[j]) {
                input_scores[row] += beta[j];
                input_touched[row] = true;
            }
        }
        if (j < pred.feature_hit_rows.size()) {
            for (auto row : pred.feature_hit_rows[j]) {
                pred_scores[row] += beta[j];
                pred_touched[row] = true;
            }
        }
    }

    // Add intercept only to touched species
    for (size_t i = 0; i < n_input; ++i)
        if (input_touched[i]) input_scores[i] += intercept;
    for (size_t i = 0; i < n_pred; ++i)
        if (pred_touched[i]) pred_scores[i] += intercept;

    // Compute input RMSE
    double sum_sq = 0.0;
    for (size_t i = 0; i < n_input; ++i) {
        double observed = input_touched[i] ? input_scores[i] : 0.0;
        double diff = y_model[i] - observed;
        sum_sq += diff * diff;
    }
    result.input_rmse = std::sqrt(sum_sq / std::max(n_input, size_t(1)));

    // Collect touched prediction species
    for (size_t i = 0; i < n_pred; ++i)
        if (pred_touched[i])
            result.pred_scores.emplace_back(i, pred_scores[i]);

    return result;
}

// ── CSV writers ───────────────────────────────────────────────────────────────

static void write_gene_ranks_csv(
    const fs::path& output_dir,
    const std::string& base_name,
    const std::vector<GeneAggregate>& genes,
    bool is_multimatrix,
    bool show_selected_sites)
{
    auto path = output_dir / (base_name + "_gene_ranks.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create " + path.string());

    out << std::setprecision(15);

    if (is_multimatrix) {
        // Sort: num_combos_ranked desc, num_combos_ranked_top desc, best_ever_rank asc, highest_ever_gss desc
        std::vector<size_t> order(genes.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (genes[a].num_combos_ranked != genes[b].num_combos_ranked)
                return genes[a].num_combos_ranked > genes[b].num_combos_ranked;
            if (genes[a].num_combos_ranked_top != genes[b].num_combos_ranked_top)
                return genes[a].num_combos_ranked_top > genes[b].num_combos_ranked_top;
            int ar = genes[a].best_ever_rank < 0 ? INT_MAX : genes[a].best_ever_rank;
            int br = genes[b].best_ever_rank < 0 ? INT_MAX : genes[b].best_ever_rank;
            if (ar != br) return ar < br;
            return genes[a].highest_ever_gss > genes[b].highest_ever_gss;
        });

        // Header
        out << "gene_name,num_combos_ranked,num_combos_ranked_top,highest_ever_gss,best_ever_rank";
        if (show_selected_sites) out << ",num_selected_sites";
        out << "\n";

        for (auto idx : order) {
            auto& g = genes[idx];
            out << g.name << ","
                << g.num_combos_ranked << ","
                << g.num_combos_ranked_top << ","
                << format_float_trim(g.highest_ever_gss) << ","
                << (g.best_ever_rank >= 0 ? std::to_string(g.best_ever_rank) : "None");
            if (show_selected_sites) out << "," << g.selected_sites.size();
            out << "\n";
        }
    } else {
        // Single combo mode
        std::vector<size_t> order(genes.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            int ar = genes[a].single_best_rank < 0 ? INT_MAX : genes[a].single_best_rank;
            int br = genes[b].single_best_rank < 0 ? INT_MAX : genes[b].single_best_rank;
            if (ar != br) return ar < br;
            return genes[a].single_highest_gss > genes[b].single_highest_gss;
        });

        out << "gene_name,highest_gss,best_rank";
        if (show_selected_sites) out << ",num_selected_sites";
        out << "\n";

        for (auto idx : order) {
            auto& g = genes[idx];
            out << g.name << ","
                << format_float_trim(g.single_highest_gss) << ","
                << (g.single_best_rank >= 0 ? std::to_string(g.single_best_rank) : "None");
            if (show_selected_sites) out << "," << g.selected_sites.size();
            out << "\n";
        }
    }

    std::cout << "  Gene ranks -> " << path.string() << "\n";
}

static void write_predictions_csv(
    const fs::path& output_dir,
    const std::string& base_name,
    const std::vector<PredictionRow>& rows,
    bool include_true_phenotype)
{
    auto path = output_dir / (base_name + "_species_predictions.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create " + path.string());

    out << "species_combo,lambda1,lambda2,penalty_term,num_genes,input_RMSE,species,SPS";
    if (include_true_phenotype) out << ",true_phenotype";
    out << "\n";

    for (auto& row : rows) {
        out << row.species_combo << ","
            << format_float_trim(row.lambda1) << ","
            << format_float_trim(row.lambda2) << ","
            << format_float_trim(row.penalty_term) << ","
            << row.num_genes << ","
            << format_float_trim(row.input_rmse) << ","
            << row.species << ","
            << format_float_trim(row.sps);
        if (include_true_phenotype) {
            out << "," << row.true_phenotype;
        }
        out << "\n";
    }

    std::cout << "  Predictions -> " << path.string() << "\n";
}

static void write_selected_sites_csv(
    const fs::path& output_dir,
    const std::string& base_name,
    const std::vector<GeneAggregate>& genes)
{
    auto path = output_dir / (base_name + "_selected_sites.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create " + path.string());

    out << "gene_name,position,pss\n";

    // Collect and sort by gene name then position
    std::vector<std::tuple<std::string, size_t, double>> rows;
    for (auto& g : genes)
        for (auto& [pos, pss] : g.selected_sites)
            rows.emplace_back(g.name, pos + 1, pss);  // 1-based

    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    });

    for (auto& [gene, pos, pss] : rows)
        out << gene << "," << pos << "," << format_float_trim(pss) << "\n";

    std::cout << "  Selected sites -> " << path.string() << "\n";
}

// ── Auto-pairs ────────────────────────────────────────────────────────────────


// Helper: find MRCA node for two leaf names
static const newick::NewickNode* find_mrca(
    const newick::NewickNode& root,
    const std::unordered_map<const newick::NewickNode*, const newick::NewickNode*>& parent_map,
    const std::unordered_map<std::string, const newick::NewickNode*>& leaf_map,
    const std::string& a, const std::string& b)
{
    auto it_a = leaf_map.find(a);
    auto it_b = leaf_map.find(b);
    if (it_a == leaf_map.end() || it_b == leaf_map.end()) return &root;

    std::unordered_set<const newick::NewickNode*> ancestors_a;
    const newick::NewickNode* cur = it_a->second;
    while (cur) {
        ancestors_a.insert(cur);
        auto pit = parent_map.find(cur);
        cur = (pit != parent_map.end()) ? pit->second : nullptr;
    }
    cur = it_b->second;
    while (cur) {
        if (ancestors_a.count(cur)) return cur;
        auto pit = parent_map.find(cur);
        cur = (pit != parent_map.end()) ? pit->second : nullptr;
    }
    return &root;
}

// Helper: check if 'ancestor' is an ancestor of 'node' (or they are the same)
static bool is_descendant_of(
    const std::unordered_map<const newick::NewickNode*, const newick::NewickNode*>& parent_map,
    const newick::NewickNode* ancestor, const newick::NewickNode* node)
{
    const newick::NewickNode* cur = node;
    while (cur) {
        if (cur == ancestor) return true;
        auto pit = parent_map.find(cur);
        cur = (pit != parent_map.end()) ? pit->second : nullptr;
    }
    return false;
}

// Helper: compute distance from node to MRCA
static double dist_to_ancestor(
    const std::unordered_map<const newick::NewickNode*, const newick::NewickNode*>& parent_map,
    const newick::NewickNode* node, const newick::NewickNode* ancestor)
{
    double d = 0.0;
    const newick::NewickNode* cur = node;
    while (cur && cur != ancestor) {
        d += cur->branch_length;
        auto pit = parent_map.find(cur);
        cur = (pit != parent_map.end()) ? pit->second : nullptr;
    }
    return d;
}

// Helper: compute Y-positions for tree leaves (for sorting pairs by tree position)
static std::unordered_map<const newick::NewickNode*, double> compute_y_positions(
    const newick::NewickNode& root, int step = 30)
{
    std::unordered_map<const newick::NewickNode*, double> y_pos;
    int idx = 0;
    // Assign Y positions to leaves in DFS order
    std::function<void(const newick::NewickNode&)> assign_leaves = [&](const newick::NewickNode& node) {
        if (node.is_leaf()) {
            y_pos[&node] = idx * step;
            idx++;
            return;
        }
        for (auto& child : node.children) assign_leaves(child);
    };
    assign_leaves(root);

    // Compute internal node Y as mean of children
    std::function<double(const newick::NewickNode&)> set_internal = [&](const newick::NewickNode& node) -> double {
        if (node.is_leaf()) return y_pos[&node];
        double sum = 0.0;
        for (auto& child : node.children) sum += set_internal(child);
        double val = node.children.empty() ? 0.0 : sum / node.children.size();
        y_pos[&node] = val;
        return val;
    };
    set_internal(root);
    return y_pos;
}

static void run_auto_pairs(
    const PscOptions& opts,
    const std::vector<GeneAlignment>& /*alignments*/,
    fs::path& species_groups_out)
{
    // Load tree
    auto tree = newick::read_newick_file(opts.auto_pairs_tree);
    auto leaves = newick::get_leaves(tree); // DFS order

    // Load phenotypes
    auto phenotypes = read_species_phenotypes(opts.species_pheno_path);

    // Build phenotype map for species in the tree: +1 (convergent) or -1 (control)
    std::unordered_map<std::string, int> pheno_for_algo;
    for (auto& leaf : leaves) {
        auto it = phenotypes.find(leaf);
        if (it == phenotypes.end()) continue;
        if (std::abs(it->second - 1.0) < 1e-12) pheno_for_algo[leaf] = 1;
        else if (std::abs(it->second + 1.0) < 1e-12) pheno_for_algo[leaf] = -1;
    }

    int n_labeled = 0;
    for (auto& [_, v] : pheno_for_algo) if (v == 1 || v == -1) n_labeled++;
    if (n_labeled < 4)
        throw std::runtime_error("Auto-pairs: need at least 4 labeled species");

    // Build maps
    auto parent_map = newick::build_parent_map(tree);
    std::unordered_map<std::string, const newick::NewickNode*> leaf_map;
    std::function<void(const newick::NewickNode&)> build_leaf_map = [&](const newick::NewickNode& node) {
        if (node.is_leaf()) leaf_map[node.name] = &node;
        for (auto& child : node.children) build_leaf_map(child);
    };
    build_leaf_map(tree);

    // Step 1: Generate candidate pairs from consecutive leaves in tree order
    // with opposite phenotypes (matches Python's approach)
    struct CandidatePair {
        std::string convergent;
        std::string control;
        double distance;
        const newick::NewickNode* ancestor;
    };
    std::vector<CandidatePair> candidates;

    std::string prev_name;
    int prev_ph = 0;
    for (auto& leaf_name : leaves) {
        auto pit = pheno_for_algo.find(leaf_name);
        if (pit == pheno_for_algo.end()) continue;
        int ph = pit->second;
        if (ph != 1 && ph != -1) continue;

        if (!prev_name.empty() && ph != prev_ph) {
            std::string conv, ctrl;
            if (prev_ph == 1) { conv = prev_name; ctrl = leaf_name; }
            else              { conv = leaf_name; ctrl = prev_name; }

            auto* mrca = find_mrca(tree, parent_map, leaf_map, conv, ctrl);
            auto* la = leaf_map[conv];
            auto* lb = leaf_map[ctrl];
            double dist = dist_to_ancestor(parent_map, la, mrca)
                        + dist_to_ancestor(parent_map, lb, mrca);
            candidates.push_back({conv, ctrl, dist, mrca});
        }
        prev_name = leaf_name;
        prev_ph = ph;
    }

    // Step 2: Sort candidates by distance
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.distance < b.distance; });

    // Step 3: Greedy selection with ancestor-based blocking
    std::vector<const newick::NewickNode*> blocked;
    auto overlaps_blocked = [&](const newick::NewickNode* anc) -> bool {
        for (auto* b : blocked) {
            if (is_descendant_of(parent_map, b, anc) || is_descendant_of(parent_map, anc, b))
                return true;
        }
        return false;
    };

    struct SelectedPair {
        std::string convergent;
        std::string control;
    };
    std::vector<SelectedPair> added;
    std::vector<const newick::NewickNode*> added_ancestors;

    for (auto& cand : candidates) {
        if (cand.ancestor && overlaps_blocked(cand.ancestor)) continue;
        added.push_back({cand.convergent, cand.control});
        added_ancestors.push_back(cand.ancestor);
        if (cand.ancestor) blocked.push_back(cand.ancestor);
    }

    // Step 4: Remove nested pairs (a pair whose ancestor is a descendant of another pair's ancestor)
    std::vector<SelectedPair> keep;
    for (size_t i = 0; i < added.size(); i++) {
        if (!added_ancestors[i]) continue;
        bool nested = false;
        for (size_t j = 0; j < added.size(); j++) {
            if (i == j || !added_ancestors[j]) continue;
            if (is_descendant_of(parent_map, added_ancestors[i], added_ancestors[j])) {
                nested = true;
                break;
            }
        }
        if (!nested) keep.push_back(added[i]);
    }
    added = keep;

    if (added.size() < 2)
        throw std::runtime_error("Auto-pairs: ESL-PSC requires at least 2 valid contrast pairs");

    // Step 5: Sort by ancestor Y-position (tree position)
    auto y_pos = compute_y_positions(tree);
    std::sort(added.begin(), added.end(),
        [&](const SelectedPair& a, const SelectedPair& b) {
            auto* mrca_a = find_mrca(tree, parent_map, leaf_map, a.convergent, a.control);
            auto* mrca_b = find_mrca(tree, parent_map, leaf_map, b.convergent, b.control);
            double ya = y_pos.count(mrca_a) ? y_pos.at(mrca_a) : 0.0;
            double yb = y_pos.count(mrca_b) ? y_pos.at(mrca_b) : 0.0;
            return ya < yb;
        });

    // Write species groups file to output dir
    species_groups_out = opts.output_dir / "auto_species_groups.txt";
    {
        std::ofstream out(species_groups_out);
        for (auto& pair : added) {
            out << pair.convergent << "\n";
            out << pair.control << "\n";
        }
    }

    std::cout << "  Auto-pairs: " << added.size() << " pairs -> " << species_groups_out.string() << "\n";
}

// ── Limited genes list ────────────────────────────────────────────────────────

static std::unordered_set<std::string> load_limited_genes(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open limited genes list: " + path.string());
    std::unordered_set<std::string> genes;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (!line.empty()) genes.insert(line);
    }
    return genes;
}

// ── Main PSC pipeline ─────────────────────────────────────────────────────────

void run_psc(const PscOptions& opts) {
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n=== PSC Mode ===\n";

    // Create output directory
    fs::create_directories(opts.output_dir);

    // Load limited genes if specified
    std::unordered_set<std::string> limited_genes;
    if (!opts.limited_genes_list.empty())
        limited_genes = load_limited_genes(opts.limited_genes_list);

    // Load alignments
    std::cout << "  Loading alignments from " << opts.alignments_dir.string() << "\n";
    auto train_alignments = load_alignments(opts.alignments_dir, limited_genes);
    if (train_alignments.empty())
        throw std::runtime_error("No alignments found in " + opts.alignments_dir.string());
    std::cout << "  Loaded " << train_alignments.size() << " gene alignments\n";

    // Load prediction alignments
    const std::vector<GeneAlignment>* prediction_alignments = &train_alignments;
    std::vector<GeneAlignment> pred_alignments_owned;
    if (!opts.no_pred_output && !opts.prediction_alignments_dir.empty()
        && opts.prediction_alignments_dir != opts.alignments_dir) {
        std::cout << "  Loading prediction alignments from " << opts.prediction_alignments_dir.string() << "\n";
        pred_alignments_owned = load_alignments(opts.prediction_alignments_dir, limited_genes);
        prediction_alignments = &pred_alignments_owned;
    }

    // Load phenotypes
    std::unordered_map<std::string, double> phenotypes;
    if (!opts.species_pheno_path.empty())
        phenotypes = read_species_phenotypes(opts.species_pheno_path);

    // Handle auto-pairs
    PscOptions effective_opts = opts;
    if (!opts.auto_pairs_tree.empty()) {
        if (opts.species_pheno_path.empty())
            throw std::runtime_error("--auto-pairs-tree requires --species-pheno-path");
        fs::path generated_groups;
        run_auto_pairs(opts, train_alignments, generated_groups);
        effective_opts.species_groups_file = generated_groups;
    }

    // Build combo jobs
    auto combos = build_combo_jobs(effective_opts);
    if (effective_opts.make_null_models)
        combos = make_null_combo_jobs(combos);
    if (combos.empty())
        throw std::runtime_error("No combo jobs created");

    std::cout << "  " << combos.size() << " combo(s)\n";

    // Validate
    validate_combo_species_against_alignments(combos, train_alignments);

    // Determine mode
    bool is_multimatrix = !effective_opts.species_groups_file.empty()
                        || !effective_opts.response_dir.empty();
    bool apply_gap_cancel = is_multimatrix && !effective_opts.use_uncanceled_alignments;

    // Build lambda grid
    auto lambda_grid = build_lambda_grid(effective_opts);
    std::cout << "  Lambda grid: " << lambda_grid.size() << " pairs\n";

    // Initialize aggregates
    std::vector<GeneAggregate> gene_aggregates;
    gene_aggregates.reserve(train_alignments.size());
    for (auto& g : train_alignments)
        gene_aggregates.push_back({g.name, 0, -1, 0, -1, 0, 0, {}});

    double top_rank_threshold = std::max(
        static_cast<double>(gene_aggregates.size()) * effective_opts.top_rank_frac, 1.0);

    std::vector<PredictionRow> all_prediction_rows;
    size_t total_model_runs = 0;

    // Determine precision
    regression::Precision precision = regression::Precision::FP32;
    if (effective_opts.precision_str == "fp64")
        precision = regression::Precision::FP64;

    // ── Main combo loop ───────────────────────────────────────────────────
    for (auto& combo : combos) {
        std::cout << "\n--- Combo " << (combo.index + 1) << " of " << combos.size()
                  << " (" << combo.combo_label << ") ---\n";

        size_t random_repeats = effective_opts.make_pair_randomized_null_models
            ? std::max(effective_opts.num_randomized_alignments, size_t(1)) : size_t(1);

        std::vector<double> combo_highest_gss(gene_aggregates.size(), 0.0);
        std::vector<int> combo_best_rank(gene_aggregates.size(), -1);

        for (size_t random_rep = 0; random_rep < random_repeats; ++random_rep) {
            bool randomize = effective_opts.make_pair_randomized_null_models;

            // Preprocess combo
            auto prep = preprocess_combo(train_alignments, combo,
                apply_gap_cancel, randomize, effective_opts, opts.output_dir);

            std::cout << "  Features: " << prep.features.n_cols
                      << " from " << prep.gene_metas.size() << " genes"
                      << " (" << prep.features.n_rows << " species)\n";

            if (prep.features.n_cols == 0) {
                std::cout << "  No features after preprocessing, skipping combo\n";
                continue;
            }

            // Build penalty terms
            auto penalty_terms = build_penalty_terms(effective_opts, prep.gene_metas);
            std::cout << "  Penalty terms: " << penalty_terms.size() << "\n";

            // Build prediction design
            PredictionDesign pred_design;
            std::vector<std::vector<size_t>> input_fhr;
            if (!effective_opts.no_pred_output) {
                pred_design = build_prediction_design(
                    *prediction_alignments, prep.feature_metas, prep.gene_metas,
                    combo.species, phenotypes);
                input_fhr = build_input_feature_hit_rows(prep.features);
            }

            std::string combo_tag = combo.combo_tag;
            if (effective_opts.make_pair_randomized_null_models)
                combo_tag += "_" + std::to_string(random_rep);

            // ── Penalty loop ──────────────────────────────────────────
            for (auto penalty : penalty_terms) {
                std::string gp_kind = effective_opts.use_default_gp ? "std" : effective_opts.group_penalty_type;
                auto group_weights = compute_group_weights(gp_kind, penalty, prep.gene_metas);

                // Update alg_table row 2 with group weights
                for (size_t gi = 0; gi < prep.gene_metas.size(); ++gi) {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(6) << group_weights[gi];
                    prep.alg_table(2, gi) = std::stod(ss.str());
                }

                // ── Lambda loop ───────────────────────────────────────
                struct LambdaResult {
                    bool valid = false;
                    std::vector<double> beta;
                    double intercept = 0.0;
                    std::vector<double> gene_gss;
                    std::vector<std::map<size_t, double>> gene_selected_sites;
                    size_t num_genes_ranked = 0;
                    std::vector<PredictionRow> prediction_rows;
                };

                // Pre-transpose alg_table (shared read-only across threads)
                const arma::mat alg_table_t = prep.alg_table.t();
                const size_t n_genes = prep.gene_metas.size();
                const bool do_predictions = !effective_opts.no_pred_output && !pred_design.species.empty();

                // Determine thread count
                unsigned int n_threads = effective_opts.threads;
                if (n_threads == 0) n_threads = 1;
                n_threads = std::min(n_threads, static_cast<unsigned int>(lambda_grid.size()));
                if (n_threads == 0) n_threads = 1;

                std::vector<LambdaResult> lambda_results(lambda_grid.size());

                // Lambda to process a single lambda index
                auto process_lambda = [&](size_t li) {
                    auto& lam = lambda_grid[li];
                    std::array<double, 2> lambda_pair = {lam[0], lam[1]};

                    std::unique_ptr<regression::RegressionAnalysis> regr;
                    try {
                        regr = regression::createRegressionAnalysis(
                            effective_opts.method,
                            prep.features,
                            prep.responses,
                            alg_table_t,
                            effective_opts.params,
                            lambda_pair,
                            precision);
                    } catch (const std::exception& e) {
                        std::cerr << "  Warning: solver failed for lambda=["
                                  << lam[0] << "," << lam[1] << "]: " << e.what() << "\n";
                        return;
                    }

                    arma::vec params = regr->getParameters();
                    LambdaResult& result = lambda_results[li];
                    result.intercept = regr->getInterceptValue();
                    result.beta.assign(params.begin(), params.end());

                    // Compute per-gene GSS and selected sites
                    result.gene_gss.assign(n_genes, 0.0);
                    result.gene_selected_sites.resize(n_genes);

                    for (size_t j = 0; j < result.beta.size(); ++j) {
                        if (result.beta[j] == 0.0) continue;
                        auto& fm = prep.feature_metas[j];
                        double aw = std::abs(result.beta[j]);
                        result.gene_gss[fm.gene_idx] += aw;
                        auto& slot = result.gene_selected_sites[fm.gene_idx][fm.position];
                        if (aw > slot) slot = aw;
                    }

                    result.num_genes_ranked = 0;
                    for (auto gss : result.gene_gss)
                        if (gss > 0.0) ++result.num_genes_ranked;

                    // Compute predictions
                    if (do_predictions) {
                        auto pred_result = compute_prediction_scores(
                            prep.y_model, input_fhr, pred_design, result.beta, result.intercept);

                        for (auto& [idx, score] : pred_result.pred_scores) {
                            PredictionRow row;
                            row.species_combo = combo_tag;
                            row.lambda1 = lam[0];
                            row.lambda2 = lam[1];
                            row.penalty_term = penalty;
                            row.num_genes = result.num_genes_ranked;
                            row.input_rmse = pred_result.input_rmse;
                            row.species = pred_design.species[idx];
                            row.sps = score;

                            if (!phenotypes.empty()) {
                                row.has_true_phenotype = true;
                                if (!std::isnan(pred_design.true_values[idx])) {
                                    double tv = pred_design.true_values[idx];
                                    if (std::abs(tv) < 1e-12) row.true_phenotype = "";
                                    else row.true_phenotype = format_float_trim(tv);
                                }
                            }
                            result.prediction_rows.push_back(std::move(row));
                        }
                    }

                    result.valid = true;
                };

                if (n_threads <= 1) {
                    // Sequential path
                    for (size_t li = 0; li < lambda_grid.size(); ++li)
                        process_lambda(li);
                } else {
                    // Parallel path — work-stealing with atomic counter
                    std::atomic<size_t> next_lambda{0};

                    auto worker = [&]() {
                        while (true) {
                            size_t li = next_lambda.fetch_add(1);
                            if (li >= lambda_grid.size()) break;
                            process_lambda(li);
                        }
                    };

                    std::vector<std::thread> threads;
                    threads.reserve(n_threads);
                    for (unsigned int i = 0; i < n_threads; ++i)
                        threads.emplace_back(worker);
                    for (auto& t : threads) t.join();
                }

                // Sequential aggregation (order-preserving)
                for (auto& result : lambda_results) {
                    if (!result.valid) continue;
                    update_gene_stats_for_run(result.gene_gss, result.gene_selected_sites,
                        combo_highest_gss, combo_best_rank, gene_aggregates);
                    for (auto& row : result.prediction_rows)
                        all_prediction_rows.push_back(std::move(row));
                    ++total_model_runs;
                }
            }
        }

        // Finalize combo stats (multimatrix mode)
        if (is_multimatrix) {
            finalize_combo_stats(combo_highest_gss, combo_best_rank,
                top_rank_threshold, gene_aggregates);
        }
    }

    // ── Write output CSVs ─────────────────────────────────────────────────
    std::cout << "\n--- Writing outputs ---\n";
    std::cout << "  Total model runs: " << total_model_runs << "\n";

    if (!effective_opts.no_genes_output) {
        write_gene_ranks_csv(opts.output_dir, opts.output_base_name,
            gene_aggregates, is_multimatrix, effective_opts.show_selected_sites);
    }

    if (!effective_opts.no_pred_output && !all_prediction_rows.empty()) {
        write_predictions_csv(opts.output_dir, opts.output_base_name,
            all_prediction_rows, !phenotypes.empty());
    }

    if (effective_opts.show_selected_sites) {
        write_selected_sites_csv(opts.output_dir, opts.output_base_name, gene_aggregates);
    }

    // Cleanup temp files
    fs::remove(opts.output_dir / "temp_weights.txt");
    fs::remove(opts.output_dir / "combined.map");

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  PSC complete in " << elapsed << "s\n";
}

} // namespace pipeline::psc
