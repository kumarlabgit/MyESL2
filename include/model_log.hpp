#pragma once

// ---------------------------------------------------------------------------
// models.tsv — the model generation log.
//
// Written at <output_dir>/models.tsv at the START of a lambda grid run, listing
// every model the run intends to produce (one row per penalty-term x lambda
// pair). Each row is then updated in place as that model is solved, skipped or
// fails, so a run that is cancelled partway through leaves an accurate record
// of which models actually finished.
//
// Two consumers:
//   * `myesl2 evaluate --from-run <dir>` scores every row marked `complete`.
//   * (future) checkpoint/resume re-solves rows still marked `pending`.
//
// Format: tab-separated, one header line, then one row per model.
//
//   Index  PenaltyIdx  PenaltyValue  LambdaIdx  FoldIdx  Lambda1  Lambda2  WeightsPath  Status  GeneCount  Detail
//
//   Index        global 0-based model index across the whole run (penalty-major)
//   PenaltyIdx   index into the penalty-term list (0 when there is only one)
//   LambdaIdx    index into lambda_list.txt
//   FoldIdx      cross-validation fold, or -1 for an ordinary model. Populated
//                only under --cv-scores, which promotes each fold model to a
//                separately scoreable model with its own row.
//   WeightsPath  path to the weights file RELATIVE to <output_dir>, so a run
//                directory stays valid after being moved. Empty when the run
//                produces no single scoreable model (k-fold CV without
//                --cv-scores writes only weights_fold_N.txt).
//   Status       pending | complete | failed | skipped
//   GeneCount    non-zero gene count the solver reported, or -1 when unknown.
//                Recorded so a resumed run can replay the --min-groups
//                skip-ahead ratchet without re-solving the earlier points it
//                depends on.
//   Detail       failure message or skip reason; empty otherwise
//
// Durability: every status change rewrites the whole file to a sibling .tmp and
// renames it over the original. Grids are at most a few hundred rows, so the
// cost is irrelevant next to a solver call, and rename() is atomic — a run
// killed mid-update leaves either the previous or the next state, never a torn
// file.
// ---------------------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace model_log {

namespace fs = std::filesystem;

enum class Status { Pending, Complete, Failed, Skipped };

inline const char* to_string(Status s) {
    switch (s) {
        case Status::Complete: return "complete";
        case Status::Failed:   return "failed";
        case Status::Skipped:  return "skipped";
        case Status::Pending:  break;
    }
    return "pending";
}

inline Status parse_status(const std::string& v) {
    if (v == "complete") return Status::Complete;
    if (v == "failed")   return Status::Failed;
    if (v == "skipped")  return Status::Skipped;
    return Status::Pending;
}

struct Row {
    size_t      index         = 0;
    size_t      penalty_idx   = 0;
    double      penalty_value = 0.0;
    size_t      lambda_idx    = 0;
    int         fold_idx      = -1;   ///< -1 = not a cross-validation fold model
    double      lambda1       = 0.0;
    double      lambda2       = 0.0;
    std::string weights_path;   ///< relative to the run dir; empty when none
    Status      status        = Status::Pending;
    int         gene_count    = -1;   ///< -1 = unknown / not solved
    std::string detail;
};

inline const char* kHeader =
    "Index\tPenaltyIdx\tPenaltyValue\tLambdaIdx\tFoldIdx\tLambda1\tLambda2\tWeightsPath\tStatus\tGeneCount\tDetail";

// Strip characters that would corrupt a TSV row.
inline std::string sanitize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\t' || c == '\n' || c == '\r') out += ' ';
        else out += c;
    }
    return out;
}

inline std::string fmt_num(double v) {
    std::ostringstream s;
    s << std::setprecision(10) << v;
    return s.str();
}

// ---------------------------------------------------------------------------
// Writer. Thread-safe: the parallel grid loop updates rows from many workers.
// ---------------------------------------------------------------------------
class Log {
public:
    explicit Log(fs::path path) : path_(std::move(path)) {}

    /// Register one intended model. Returns its global index.
    size_t add(size_t penalty_idx, double penalty_value, size_t lambda_idx,
               int fold_idx, double lambda1, double lambda2, std::string rel_weights)
    {
        std::lock_guard<std::mutex> lk(mu_);
        Row r;
        r.index         = rows_.size();
        r.penalty_idx   = penalty_idx;
        r.penalty_value = penalty_value;
        r.lambda_idx    = lambda_idx;
        r.fold_idx      = fold_idx;
        r.lambda1       = lambda1;
        r.lambda2       = lambda2;
        r.weights_path  = std::move(rel_weights);
        rows_.push_back(std::move(r));
        return rows_.size() - 1;
    }

    void set(size_t index, Status st, const std::string& detail = "", int gene_count = -1) {
        std::lock_guard<std::mutex> lk(mu_);
        if (index >= rows_.size()) return;
        rows_[index].status     = st;
        rows_[index].detail     = detail;
        rows_[index].gene_count = gene_count;
        write_locked();
    }

    /// Seed a row from a previous run's record without touching the file.
    /// Used when resuming: every row is adopted first, then one flush() writes
    /// the merged table, instead of one rewrite per adopted row.
    void adopt(size_t index, const Row& prior) {
        std::lock_guard<std::mutex> lk(mu_);
        if (index >= rows_.size()) return;
        rows_[index].status     = prior.status;
        rows_[index].detail     = prior.detail;
        rows_[index].gene_count = prior.gene_count;
    }

    /// Snapshot of the current rows (for resume classification).
    std::vector<Row> rows() const {
        std::lock_guard<std::mutex> lk(mu_);
        return rows_;
    }

    /// Reclassify every still-pending row of one penalty term. Used when the
    /// sequential skip-ahead ratchet breaks out of the grid early: the
    /// remaining points are deliberately skipped, not merely unattempted.
    void set_pending_in_penalty(size_t penalty_idx, Status st, const std::string& detail) {
        std::lock_guard<std::mutex> lk(mu_);
        bool touched = false;
        for (auto& r : rows_)
            if (r.penalty_idx == penalty_idx && r.status == Status::Pending) {
                r.status = st;
                r.detail = detail;
                touched  = true;
            }
        if (touched) write_locked();
    }

    /// Write the initial all-pending table. Call once, after every add().
    void flush() {
        std::lock_guard<std::mutex> lk(mu_);
        write_locked();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return rows_.size();
    }

private:
    void write_locked() {
        fs::path tmp = path_;
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            if (!f) return;  // never let logging failure abort a solve
            f << kHeader << "\n";
            for (const auto& r : rows_) {
                f << r.index << '\t' << r.penalty_idx << '\t'
                  << fmt_num(r.penalty_value) << '\t' << r.lambda_idx << '\t'
                  << r.fold_idx << '\t'
                  << fmt_num(r.lambda1) << '\t' << fmt_num(r.lambda2) << '\t'
                  << r.weights_path << '\t' << to_string(r.status) << '\t'
                  << r.gene_count << '\t' << sanitize(r.detail) << '\n';
            }
        }
        std::error_code ec;
        fs::rename(tmp, path_, ec);
        if (ec) fs::remove(tmp, ec);
    }

    fs::path            path_;
    std::vector<Row>    rows_;
    mutable std::mutex  mu_;
};

// ---------------------------------------------------------------------------
// Reader.
// ---------------------------------------------------------------------------
inline std::vector<Row> read(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open model log: " + path.string());

    std::vector<Row> rows;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first) { first = false; if (line.rfind("Index", 0) == 0) continue; }

        std::vector<std::string> col;
        std::string tok;
        std::istringstream ss(line);
        while (std::getline(ss, tok, '\t')) col.push_back(tok);
        if (col.size() < 10) continue;

        Row r;
        try {
            r.index         = static_cast<size_t>(std::stoull(col[0]));
            r.penalty_idx   = static_cast<size_t>(std::stoull(col[1]));
            r.penalty_value = std::stod(col[2]);
            r.lambda_idx    = static_cast<size_t>(std::stoull(col[3]));
            r.fold_idx      = std::stoi(col[4]);
            r.lambda1       = std::stod(col[5]);
            r.lambda2       = std::stod(col[6]);
        } catch (...) { continue; }
        r.weights_path = col[7];
        r.status       = parse_status(col[8]);
        try { r.gene_count = std::stoi(col[9]); } catch (...) { r.gene_count = -1; }
        r.detail       = col.size() > 10 ? col[10] : "";
        rows.push_back(std::move(r));
    }
    return rows;
}

} // namespace model_log
