#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "pff_format.hpp"

namespace fs = std::filesystem;

namespace pipeline {

struct PreprocessOptions {
    fs::path     list_path;
    fs::path     output_dir;
    fs::path     cache_dir;       // if empty: set to cwd/pff_cache
    fs::path     binary_dir;      // directory of the executable (for data_defs.ini lookup)
    pff::Orientation orientation  = pff::Orientation::COLUMN_MAJOR;
    std::string  datatype         = "universal";
    unsigned int num_threads      = 0;   // 0 → hardware_concurrency
    int          min_minor        = 2;
    bool         use_dlt          = false;
    // DrPhylo tree mode (all empty/unset = no tree)
    fs::path     tree_path;
    std::string  clade_list_file;
    std::string  gen_clade_spec;  // "lower,upper" or ""
    std::string  class_bal_phylo  = "phylo";
};

// Run Phase 1 conversion and (optionally) DrPhylo clade hypothesis generation.
// Returns list of per-clade hypothesis files: output_dir/clade_name/hypothesis.txt
// (Empty vector if tree_path is not set.)
// Also writes output_dir/preprocess_config and output_dir/aln_list.txt.
std::vector<fs::path> preprocess(const PreprocessOptions& opts);

// Read/write preprocess_config key=value file.
PreprocessOptions read_preprocess_config(const fs::path& output_dir);
void              write_preprocess_config(const fs::path& output_dir,
                                          const PreprocessOptions& opts);

} // namespace pipeline
