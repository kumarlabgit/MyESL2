#pragma once

// ---------------------------------------------------------------------------
// run_manifest — the settings a resumed run must still agree with.
//
// <output_dir>/run_manifest records every option that determines which models a
// run produces, so `--resume` can prove it is continuing the same work rather
// than grafting new models onto ones trained under different settings.
// preprocess_config alone is not enough: it carries eight preprocess fields and
// says nothing about method, lambdas, folds, penalties or encode options.
//
// Keys fall into three classes:
//
//   identity     Compared on resume. A difference means the models would not be
//                mutually consistent, so the resume is refused (overridable
//                with --resume-force, for cases like a bare `touch` that
//                changed an mtime without changing content).
//
//   layout       Fingerprints of the assembled feature matrix (combined.map and
//                alignment_table.txt). A difference means the columns moved, so
//                old and new models index different features. Refused
//                unconditionally -- --resume-force cannot override it, because
//                the result would be silently wrong rather than merely
//                suspicious.
//
//   info         Recorded for provenance but never compared: thread counts,
//                memory ceilings and cache location do not change any model.
//
// Format is the same flat key=value text as preprocess_config, one pair per
// line, sorted, so a diff of two manifests is directly readable.
// ---------------------------------------------------------------------------

#include "pipeline_utils.hpp"
#include "pipeline_preprocess.hpp"
#include "pipeline_encode.hpp"
#include "pipeline_train.hpp"

#include <iomanip>
#include <sstream>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace run_manifest {

namespace fs = std::filesystem;

/// Which comparison class a key belongs to. Encoded as a key-name prefix so the
/// file stays a plain key=value table with no separate schema to keep in sync.
inline bool is_layout_key(const std::string& k) { return k.rfind("layout.", 0) == 0; }
inline bool is_info_key  (const std::string& k) { return k.rfind("info.",   0) == 0; }

using Manifest = std::map<std::string, std::string>;

inline void write(const fs::path& output_dir, const Manifest& m)
{
    pipeline_utils::AtomicOut f(output_dir / "run_manifest");
    if (!f) return;
    for (const auto& [k, v] : m) f << k << "=" << v << "\n";
}

/// Returns an empty manifest when the file is absent or unreadable.
inline Manifest read(const fs::path& output_dir)
{
    Manifest m;
    std::ifstream f(output_dir / "run_manifest");
    if (!f) return m;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

struct Diff {
    std::vector<std::string> identity;   ///< changed identity keys
    std::vector<std::string> layout;     ///< changed layout keys
    bool empty() const { return identity.empty() && layout.empty(); }
};

/// Describe how `now` departs from `stored`, ignoring info.* keys. A key present
/// in one and missing from the other counts as changed.
///
/// `include_layout` is false for the pre-encode check, where `now` does not yet
/// carry layout fingerprints; including them then would report every one as
/// removed. The post-encode check passes true.
inline Diff compare(const Manifest& stored, const Manifest& now, bool include_layout)
{
    Diff d;
    auto note = [&](const std::string& k, const std::string& a, const std::string& b) {
        if (is_info_key(k)) return;
        if (is_layout_key(k) && !include_layout) return;
        std::string line = k + ": " + (a.empty() ? "(unset)" : a)
                             + " -> " + (b.empty() ? "(unset)" : b);
        if (is_layout_key(k)) d.layout.push_back(line);
        else                  d.identity.push_back(line);
    };
    for (const auto& [k, v] : stored) {
        auto it = now.find(k);
        if (it == now.end())      note(k, v, "");
        else if (it->second != v) note(k, v, it->second);
    }
    for (const auto& [k, v] : now)
        if (!stored.count(k)) note(k, "", v);
    return d;
}

/// Refuse to resume when the current invocation departs from the stored run.
///
/// A layout difference is always fatal: the feature columns moved, so models
/// from the two runs index different features and mixing them would produce a
/// plausible-looking but invalid result set. An identity difference is fatal
/// unless the caller passes force.
inline void enforce(const fs::path& output_dir, const Manifest& now,
                    bool force, bool include_layout)
{
    Manifest stored = read(output_dir);
    if (stored.empty())
        throw std::runtime_error(
            "--resume: no run_manifest in " + output_dir.string() +
            " — this directory was not produced by a resumable run. Re-run without --resume.");

    Diff d = compare(stored, now, include_layout);
    if (d.empty()) return;

    std::string msg;
    if (!d.layout.empty()) {
        msg = "--resume refused: the feature matrix layout changed, so existing "
              "models index different features:\n";
        for (const auto& l : d.layout) msg += "  " + l + "\n";
        msg += "This cannot be overridden. Train into a fresh output directory.";
        throw std::runtime_error(msg);
    }
    if (!force) {
        msg = "--resume refused: settings differ from the run being resumed:\n";
        for (const auto& l : d.identity) msg += "  " + l + "\n";
        msg += "Re-run with the original settings, use --resume-force to override, "
               "or train into a fresh output directory.";
        throw std::runtime_error(msg);
    }
    // Announced once, on the pre-encode pass; the post-encode pass re-checks the
    // same identity keys and would otherwise repeat itself verbatim.
    if (!include_layout) {
        std::cerr << "Warning: --resume-force overriding " << d.identity.size()
                  << " changed setting(s):\n";
        for (const auto& l : d.identity) std::cerr << "  " << l << "\n";
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Everything known before encode runs. Deliberately excludes thread counts,
/// memory ceilings and cache location (recorded as info.*), which change no model.
inline Manifest build(const pipeline::PreprocessOptions& pre,
                      const pipeline::EncodeOptions&     enc,
                      const pipeline::TrainOptions&      tr)
{
    Manifest m;
    auto num = [](double v) { std::ostringstream o; o << std::setprecision(10) << v; return o.str(); };
    auto yn  = [](bool b)   { return std::string(b ? "true" : "false"); };
    // preprocess() resolves defaults into a local copy, so the caller's options
    // can still hold empty paths here; fs::absolute() throws on those.
    auto abs = [](const fs::path& p) {
        return p.empty() ? std::string() : fs::absolute(p).string();
    };

    // --- inputs -------------------------------------------------------------
    // The list and hypothesis files are content-hashed: they are small and are
    // what actually changes between runs. Individual alignment contents are not
    // hashed -- the cache layer already keys on path rather than content, so
    // tracking them here would be stricter than the rest of the pipeline.
    m["identity.list_path"]  = abs(pre.list_path);
    m["identity.list_hash"]  = pipeline_utils::hex64(pipeline_utils::hash_file(pre.list_path));
    m["identity.hyp_path"]   = abs(enc.hyp_path);
    m["identity.hyp_hash"]   = pipeline_utils::hex64(pipeline_utils::hash_file(enc.hyp_path));

    // --- preprocess ---------------------------------------------------------
    m["identity.datatype"]    = pre.datatype;
    m["identity.orientation"] = (pre.orientation == pff::Orientation::COLUMN_MAJOR) ? "column" : "row";
    m["identity.min_minor"]   = std::to_string(pre.min_minor);
    m["identity.use_dlt"]     = yn(pre.use_dlt);
    m["identity.het_mode"]    = pre.het_mode;

    // --- encode -------------------------------------------------------------
    m["identity.class_bal"]         = enc.class_bal;
    m["identity.drop_major"]        = yn(enc.drop_major);
    m["identity.minor_column"]      = yn(enc.minor_column);
    m["identity.tiered_minor_col"]  = yn(enc.tiered_minor_col);
    m["identity.auto_bit_ct"]       = num(enc.auto_bit_ct);
    m["identity.feature_normalize"] = enc.feature_normalize;
    m["identity.precision"]         = (enc.precision == regression::Precision::FP64) ? "fp64" : "fp32";
    m["identity.dropout_count"]     = std::to_string(enc.dropout_labels.size());

    // --- train --------------------------------------------------------------
    m["identity.method"]        = tr.method;
    m["identity.lambda"]        = num(tr.lambda[0]) + "," + num(tr.lambda[1]);
    m["identity.lambda_file"]   = tr.lambda_file_path;
    m["identity.lambda_grid"]   = tr.lambda_grid_specs[0] + " / " + tr.lambda_grid_specs[1];
    m["identity.lambda_grid_set"] = yn(tr.lambda_grid_set);
    m["identity.use_logspace"]  = yn(tr.use_logspace);
    m["identity.nfolds"]        = std::to_string(tr.nfolds);
    m["identity.cv_seed"]       = std::to_string(tr.cv_seed);
    m["identity.cv_assignments"]= tr.cv_assignments_path;
    m["identity.cv_scores"]     = yn(tr.cv_scores);
    m["identity.min_groups"]    = std::to_string(tr.min_groups);
    m["identity.group_penalty_type"] = tr.group_penalty_type;
    m["identity.initial_gp_value"]   = num(tr.initial_gp_value);
    m["identity.final_gp_value"]     = num(tr.final_gp_value);
    m["identity.gp_step"]            = num(tr.gp_step);
    {   // --param entries, sorted so spelling order does not matter
        std::string joined;
        for (const auto& [k, v] : tr.params) joined += k + "=" + v + ";";
        m["identity.params"] = joined;
    }

    // --- informational ------------------------------------------------------
    m["info.cache_dir"]   = abs(pre.cache_dir);
    m["info.num_threads"] = std::to_string(pre.num_threads);
    m["info.max_mem"]     = std::to_string(enc.max_mem);
    return m;
}

/// Fingerprints of the assembled matrix. Only meaningful after encode has run.
inline void add_layout(Manifest& m, const pipeline::EncodeResult& enc)
{
    m["layout.combined_map"]     = pipeline_utils::hex64(pipeline_utils::hash_file(enc.combined_map_path));
    m["layout.alignment_table"]  = pipeline_utils::hex64(pipeline_utils::hash_file(enc.alignment_table_path));
    m["layout.samples"]          = std::to_string(enc.N);
    m["layout.total_cols"]       = std::to_string(enc.total_cols);
}

} // namespace run_manifest
