#include "taskfile.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "third_party/fkYAML.hpp"

// Forward declarations of the per-command run helpers, defined in main.cpp.
int run_train(int argc, char* argv[]);
int run_evaluate(int argc, char* argv[]);
int run_info(int argc, char* argv[]);
int run_drphylo(int argc, char* argv[]);
int run_aim(int argc, char* argv[]);
int run_psc(int argc, char* argv[]);
int run_encode_sizes(int argc, char* argv[]);
int run_visualize(int argc, char* argv[]);

namespace {

struct TaskSpec {
    std::vector<std::string>        positionals;       // ordered; emit as bare tokens at argv[2..]
    std::unordered_set<std::string> allowed;           // all valid keys including positionals
    std::unordered_set<std::string> bool_flags;        // emit "--flag" only when true
    std::unordered_set<std::string> bare_values;       // emit value as bare token (no --flag prefix)
    std::unordered_set<std::string> mapping_flags;     // YAML mapping → repeated "--flag k=v"
    std::unordered_set<std::string> multivalue_flags;  // YAML sequence → "--flag v1 v2 ..."
};

TaskSpec spec_for(const std::string& task_type) {
    TaskSpec s;

    if (task_type == "train") {
        s.positionals = {"list_path", "hypothesis_path", "output_dir"};
        s.bool_flags = {"dlt", "prune-skipped-lambda", "use-logspace", "cv-scores",
                        "resume", "resume-force", "resume-skip-failed",
                        "drop-major-allele", "minor-column", "tiered-minor-col",
                        "adaptive-sparsification", "no-evaluate"};
        s.bare_values = {"orientation"};
        s.mapping_flags = {"param"};
        s.multivalue_flags = {"lambda", "lambda-grid", "adaptive-lambda-grid"};
        s.allowed = {
            "list_path", "hypothesis_path", "output_dir", "orientation",
            "cache-dir", "min-minor", "dlt", "datatype", "threads",
            "prune-skipped-lambda", "method", "precision",
            "lambda", "lambda-file", "lambda-grid", "use-logspace", "param",
            "nfolds", "cv-seed", "cv-assignments", "cv-scores", "min-groups",
            "resume", "resume-force", "resume-skip-failed",
            "auto-bit-ct", "drop-major-allele", "minor-column", "tiered-minor-col",
            "class-bal", "dropout", "write-features", "write-features-transposed",
            "max-mem", "adaptive-sparsification", "adaptive-lambda-grid", "het-mode",
            "group-penalty-type", "initial-gp-value", "final-gp-value", "gp-step",
            "feature-normalize", "no-evaluate",
        };
    } else if (task_type == "evaluate") {
        // evaluate has two layouts (single-model / from-run). Handled specially below.
        s.positionals = {};
        s.bool_flags = {"no-visualize", "re-evaluate", "visualize"};
        s.allowed = {
            "weights_path", "list_path", "output_file",
            "cache-dir", "hypothesis", "datatype", "threads", "no-visualize",
            "minor-alleles", "tiered-minor-alleles", "gene-limit", "species-limit", "het-mode",
            "from-run", "re-evaluate", "visualize",
        };
    } else if (task_type == "info") {
        s.positionals = {"pff_path"};
        s.allowed = {"pff_path"};
    } else if (task_type == "drphylo") {
        // drphylo has two positional layouts (direct/tree). Handled specially below.
        s.positionals = {};
        s.bool_flags = {"prune-skipped-lambda", "dlt", "use-logspace", "cv-scores",
                        "drop-major-allele", "minor-column", "tiered-minor-col"};
        s.mapping_flags = {"param"};
        s.multivalue_flags = {"lambda", "lambda-grid"};
        s.allowed = {
            "list_path", "hypothesis_path", "output_dir", "tree",
            "clade-list", "gen-clade-list", "class-bal", "datatype", "threads",
            "prune-skipped-lambda", "cache-dir", "dlt", "min-minor",
            "method", "precision",
            "lambda", "lambda-file", "lambda-grid", "use-logspace", "param",
            "nfolds", "cv-seed", "cv-assignments", "cv-scores", "min-groups",
            "resume", "resume-force", "resume-skip-failed",
            "grid-rmse-cutoff", "grid-acc-cutoff", "gene-limit", "species-limit", "het-mode",
            "auto-bit-ct", "drop-major-allele", "minor-column", "tiered-minor-col",
            "max-mem",
            "group-penalty-type", "initial-gp-value", "final-gp-value", "gp-step",
            "feature-normalize",
        };
    } else if (task_type == "aim") {
        s.positionals = {"list_path", "hypothesis_path", "output_dir"};
        s.bool_flags = {"prune-skipped-lambda", "dlt", "use-logspace", "cv-scores",
                        "drop-major-allele", "minor-column", "tiered-minor-col"};
        s.mapping_flags = {"param"};
        s.multivalue_flags = {"lambda", "lambda-grid"};
        s.allowed = {
            "list_path", "hypothesis_path", "output_dir",
            "aim-acc-cutoff", "aim-max-iter", "aim-max-ft", "aim-window",
            "cache-dir", "datatype", "threads",
            "prune-skipped-lambda", "dlt", "min-minor",
            "method", "precision",
            "lambda", "lambda-file", "lambda-grid", "use-logspace", "param",
            "nfolds", "cv-seed", "cv-assignments", "cv-scores", "min-groups",
            "resume", "resume-force", "resume-skip-failed",
            "class-bal", "drop-major-allele", "minor-column", "tiered-minor-col",
            "auto-bit-ct", "max-mem", "het-mode",
            "group-penalty-type", "initial-gp-value", "final-gp-value", "gp-step",
            "feature-normalize",
        };
    } else if (task_type == "psc") {
        s.positionals = {"alignments_dir", "output_dir"};
        s.bool_flags = {"use-logspace", "use-default-gp",
                        "use-uncanceled-alignments", "cancel-only-partner",
                        "cancel-tri-allelic", "nix-full-deletions",
                        "no-pred-output", "no-genes-output", "show-selected-sites",
                        "dump-weights",
                        "make-null-models", "make-pair-randomized-null-models"};
        s.mapping_flags = {"param"};
        s.allowed = {
            "alignments_dir", "output_dir",
            "alignments-list",
            "species-groups", "response-file", "response-dir",
            "initial-lambda1", "final-lambda1",
            "initial-lambda2", "final-lambda2",
            "lambda-step", "use-logspace", "num-log-points",
            "group-penalty-type",
            "initial-gp-value", "final-gp-value", "gp-step", "use-default-gp",
            "use-uncanceled-alignments", "cancel-only-partner",
            "cancel-tri-allelic", "nix-full-deletions",
            "outgroup-species", "min-pairs",
            "method", "precision", "maxiter", "threads", "param",
            "output-base-name", "prediction-alignments-dir", "species-pheno-path",
            "no-pred-output", "no-genes-output", "show-selected-sites", "dump-weights",
            "top-rank-frac", "limited-genes-list",
            "make-null-models", "make-pair-randomized-null-models",
            "num-randomized-alignments",
        };
    } else if (task_type == "encode-sizes") {
        s.positionals = {"output_dir", "hypothesis_path"};
        s.bool_flags = {"drop-major-allele", "minor-column", "tiered-minor-col"};
        s.allowed = {
            "output_dir", "hypothesis_path",
            "min-minor", "auto-bit-ct",
            "drop-major-allele", "minor-column", "tiered-minor-col", "dropout",
        };
    } else if (task_type == "visualize") {
        s.positionals = {"gene_predictions_path", "svg_path"};
        s.bool_flags = {"m-grid"};
        s.allowed = {
            "gene_predictions_path", "svg_path",
            "gene-limit", "species-limit", "ssq-threshold", "m-grid",
        };
    } else {
        throw std::runtime_error("Unknown task_type: '" + task_type + "'");
    }

    return s;
}

std::string node_to_string(const fkyaml::node& n) {
    if (n.is_string())       return n.get_value<std::string>();
    if (n.is_boolean())      return n.get_value<bool>() ? "true" : "false";
    if (n.is_integer())      return std::to_string(n.get_value<int64_t>());
    if (n.is_float_number()) {
        std::ostringstream oss;
        oss.precision(15);
        oss << n.get_value<double>();
        return oss.str();
    }
    if (n.is_null())         return "";
    throw std::runtime_error("YAML value is not a scalar");
}

std::string join_space(const std::vector<std::string>& vs) {
    std::ostringstream oss;
    for (size_t i = 0; i < vs.size(); ++i) {
        if (i) oss << " ";
        oss << vs[i];
    }
    return oss.str();
}

} // anon

int run_taskfile(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: taskfile requires <config.yaml>\n"
                  << "Usage: " << argv[0] << " taskfile <config.yaml> [override-flags...]\n";
        return 1;
    }

    // ---------------- Parse YAML ----------------
    std::string yaml_path = argv[2];
    std::ifstream f(yaml_path);
    if (!f) throw std::runtime_error("Cannot open taskfile: " + yaml_path);

    fkyaml::node root;
    try {
        root = fkyaml::node::deserialize(f);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Malformed YAML in ") + yaml_path + ": " + e.what());
    }
    if (!root.is_mapping())
        throw std::runtime_error("Taskfile must be a YAML mapping at top level");

    if (!root.contains("task_type"))
        throw std::runtime_error("Taskfile is missing required 'task_type' field");
    if (!root["task_type"].is_string())
        throw std::runtime_error("Taskfile 'task_type' must be a string");
    std::string task_type = root["task_type"].get_value<std::string>();

    TaskSpec spec = spec_for(task_type);

    // ---------------- Validate keys ----------------
    for (auto it = root.begin(); it != root.end(); ++it) {
        std::string key = it.key().get_value<std::string>();
        if (key == "task_type") continue;
        if (!spec.allowed.count(key))
            throw std::runtime_error("YAML key '" + key
                + "' is not valid for task_type '" + task_type + "'");
    }

    // ---------------- Collect YAML values ----------------
    std::map<std::string, std::vector<std::string>> yaml_vals;
    for (auto it = root.begin(); it != root.end(); ++it) {
        std::string key = it.key().get_value<std::string>();
        if (key == "task_type") continue;
        const fkyaml::node& v = it.value();
        std::vector<std::string> vals;

        if (spec.mapping_flags.count(key)) {
            if (!v.is_mapping())
                throw std::runtime_error("YAML key '" + key + "' must be a mapping");
            for (auto mit = v.begin(); mit != v.end(); ++mit) {
                std::string mk = mit.key().get_value<std::string>();
                std::string mv = node_to_string(mit.value());
                vals.push_back(mk + "=" + mv);
            }
        } else if (spec.multivalue_flags.count(key)) {
            if (!v.is_sequence())
                throw std::runtime_error("YAML key '" + key + "' must be a sequence");
            for (auto sit = v.begin(); sit != v.end(); ++sit)
                vals.push_back(node_to_string(*sit));
        } else {
            vals.push_back(node_to_string(v));
        }
        yaml_vals[key] = std::move(vals);
    }

    // ---------------- Collect CLI overrides ----------------
    // Anything after argv[2] is an override. Form is --key [value [value]] for non-positionals,
    // or --positional_name value to override a positional.
    std::map<std::string, std::vector<std::string>> cli_vals;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() < 3 || a[0] != '-' || a[1] != '-')
            throw std::runtime_error("Override must be of the form --key [value...]; got '" + a + "'");
        std::string key = a.substr(2);
        if (!spec.allowed.count(key))
            throw std::runtime_error("CLI override --" + key
                + " is not valid for task_type '" + task_type + "'");

        std::vector<std::string> vals;
        if (spec.bool_flags.count(key)) {
            vals.push_back("true");
        } else if (spec.mapping_flags.count(key)) {
            if (i + 1 >= argc)
                throw std::runtime_error("Override --" + key + " requires a k=v value");
            vals.push_back(argv[++i]);
            // Allow multiple --param k=v on the CLI by appending to the existing list.
            auto it = cli_vals.find(key);
            if (it != cli_vals.end()) {
                it->second.insert(it->second.end(), vals.begin(), vals.end());
                continue;
            }
        } else {
            int n = spec.multivalue_flags.count(key) ? 2 : 1;
            for (int k = 0; k < n; ++k) {
                if (i + 1 >= argc)
                    throw std::runtime_error("Override --" + key + " requires "
                        + std::to_string(n) + " value(s)");
                vals.push_back(argv[++i]);
            }
        }
        cli_vals[key] = std::move(vals);
    }

    // ---------------- Conflict warnings ----------------
    for (const auto& [key, cvals] : cli_vals) {
        auto yit = yaml_vals.find(key);
        if (yit == yaml_vals.end()) continue;
        if (join_space(yit->second) == join_space(cvals)) continue;
        std::cerr << "[Warning] taskfile override: " << key
                  << "=" << join_space(yit->second)
                  << " -> " << join_space(cvals) << "\n";
    }

    // ---------------- Build synthetic argv ----------------
    auto resolved = [&](const std::string& key) -> const std::vector<std::string>* {
        auto cit = cli_vals.find(key);
        if (cit != cli_vals.end()) return &cit->second;
        auto yit = yaml_vals.find(key);
        if (yit != yaml_vals.end()) return &yit->second;
        return nullptr;
    };

    std::vector<std::string> tokens;
    tokens.push_back(argv[0]);
    tokens.push_back(task_type);

    // Positional emission.
    std::unordered_set<std::string> emitted_positionals;
    if (task_type == "evaluate") {
        // from-run mode reads everything from the run directory, so it takes no
        // positionals; single-model mode still requires all three.
        if (!resolved("from-run")) {
            for (const char* pname : {"weights_path", "list_path", "output_file"}) {
                const std::vector<std::string>* pv = resolved(pname);
                if (!pv)
                    throw std::runtime_error(
                        std::string("evaluate: missing required key '") + pname +
                        "' (or use 'from-run' to score a whole run directory)");
                tokens.push_back((*pv)[0]);
                emitted_positionals.insert(pname);
            }
        }
    } else if (task_type == "drphylo") {
        // Detect tree mode by presence of `tree` in YAML or CLI.
        const std::vector<std::string>* list_v = resolved("list_path");
        const std::vector<std::string>* tree_v = resolved("tree");
        const std::vector<std::string>* hyp_v  = resolved("hypothesis_path");
        const std::vector<std::string>* out_v  = resolved("output_dir");

        if (!list_v) throw std::runtime_error("drphylo: missing required key 'list_path'");
        if (!out_v)  throw std::runtime_error("drphylo: missing required key 'output_dir'");

        tokens.push_back((*list_v)[0]); emitted_positionals.insert("list_path");

        if (tree_v) {
            // Tree mode: drphylo <list> <output_dir> --tree <tree>
            tokens.push_back((*out_v)[0]); emitted_positionals.insert("output_dir");
        } else {
            // Direct mode: drphylo <list> <hypothesis> <output_dir>
            if (!hyp_v)
                throw std::runtime_error("drphylo: missing 'hypothesis_path' (or use 'tree' for tree mode)");
            tokens.push_back((*hyp_v)[0]); emitted_positionals.insert("hypothesis_path");
            tokens.push_back((*out_v)[0]); emitted_positionals.insert("output_dir");
        }
    } else {
        for (const auto& pname : spec.positionals) {
            const std::vector<std::string>* pv = resolved(pname);
            if (!pv)
                throw std::runtime_error("Taskfile missing required positional '"
                                         + pname + "' for task_type '" + task_type + "'");
            tokens.push_back((*pv)[0]);
            emitted_positionals.insert(pname);
        }
    }

    // Optional/flag emission. Iterate over union of YAML and CLI keys.
    std::unordered_set<std::string> all_keys;
    for (const auto& [k, _] : yaml_vals) all_keys.insert(k);
    for (const auto& [k, _] : cli_vals)  all_keys.insert(k);

    for (const auto& key : all_keys) {
        if (emitted_positionals.count(key)) continue;
        const std::vector<std::string>* vals = resolved(key);
        if (!vals) continue;

        if (spec.bool_flags.count(key)) {
            if (!vals->empty() && (*vals)[0] == "true")
                tokens.push_back("--" + key);
            // false bool: emit nothing
        } else if (spec.bare_values.count(key)) {
            tokens.push_back((*vals)[0]);
        } else if (spec.mapping_flags.count(key)) {
            for (const auto& kv : *vals) {
                tokens.push_back("--" + key);
                tokens.push_back(kv);
            }
        } else {
            tokens.push_back("--" + key);
            for (const auto& v : *vals) tokens.push_back(v);
        }
    }

    // ---------------- Dispatch ----------------
    std::vector<char*> sargv;
    sargv.reserve(tokens.size());
    for (auto& s : tokens) sargv.push_back(const_cast<char*>(s.c_str()));
    int sargc = static_cast<int>(sargv.size());

    if      (task_type == "train")        return run_train(sargc, sargv.data());
    else if (task_type == "evaluate")     return run_evaluate(sargc, sargv.data());
    else if (task_type == "info")         return run_info(sargc, sargv.data());
    else if (task_type == "drphylo")      return run_drphylo(sargc, sargv.data());
    else if (task_type == "aim")          return run_aim(sargc, sargv.data());
    else if (task_type == "psc")          return run_psc(sargc, sargv.data());
    else if (task_type == "encode-sizes") return run_encode_sizes(sargc, sargv.data());
    else if (task_type == "visualize")    return run_visualize(sargc, sargv.data());

    throw std::runtime_error("Unhandled task_type after spec_for: '" + task_type + "'");
}
