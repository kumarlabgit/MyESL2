#pragma once

// AtomicOut — see the class comment below. Kept in its own header (rather than
// pipeline_utils.hpp) so translation units that only need atomic file output do
// not pull in armadillo.

#include <filesystem>
#include <fstream>
#include <ios>
#include <ostream>
#include <system_error>
#include <utility>

namespace pipeline_utils {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// AtomicOut — write-to-temp-then-rename output file.
//
// Data lands in <path>.tmp and is renamed over <path> only once writing has
// succeeded, so:
//   * a reader never observes a half-written file. This matters because
//     `evaluate --from-run` and train's own Phase 4 write the same eval*.txt
//     paths, and may overlap when a run directory is scored while the run is
//     still in progress.
//   * an interrupted or failing write leaves the PREVIOUS version intact
//     rather than a truncated one.
//
// Drop-in for `std::ofstream` at call sites that only use `operator<<`.
// Commits on destruction; call commit() early to rename before the scope ends.
// ---------------------------------------------------------------------------
class AtomicOut {
public:
    explicit AtomicOut(fs::path final_path)
        : final_(std::move(final_path)), tmp_(final_)
    {
        tmp_ += ".tmp";
        std::error_code ec;
        if (!tmp_.parent_path().empty())
            fs::create_directories(tmp_.parent_path(), ec);
        f_.open(tmp_, std::ios::out | std::ios::trunc);
    }

    ~AtomicOut() { commit(); }

    AtomicOut(const AtomicOut&)            = delete;
    AtomicOut& operator=(const AtomicOut&) = delete;

    explicit operator bool() const { return static_cast<bool>(f_); }
    bool operator!()         const { return !f_; }

    template <class T>
    AtomicOut& operator<<(const T& v) { f_ << v; return *this; }
    // Manipulators (std::fixed, std::setprecision, ...) taken by pointer so
    // they are not swallowed by the template above.
    AtomicOut& operator<<(std::ostream&  (*m)(std::ostream&))  { f_ << m; return *this; }
    AtomicOut& operator<<(std::ios_base& (*m)(std::ios_base&)) { f_ << m; return *this; }

    /// Underlying stream, for the rare call site needing more than operator<<.
    std::ofstream& stream() { return f_; }

    /// Close and publish. Idempotent; called automatically by the destructor.
    void commit() {
        if (done_) return;
        done_ = true;
        if (f_.is_open()) f_.close();

        std::error_code ec;
        if (f_.fail()) {           // open or write failed — keep the original
            fs::remove(tmp_, ec);
            return;
        }
        fs::rename(tmp_, final_, ec);
        if (!ec) return;

        // rename can fail if another process holds the destination open
        // (Windows) or the temp landed on a different device. Fall back to a
        // copy so a successful write is never silently discarded.
        std::error_code ec2;
        fs::copy_file(tmp_, final_, fs::copy_options::overwrite_existing, ec2);
        fs::remove(tmp_, ec2);
    }

private:
    fs::path      final_;
    fs::path      tmp_;
    std::ofstream f_;
    bool          done_ = false;
};

} // namespace pipeline_utils
