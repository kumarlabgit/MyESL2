#pragma once
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include "regression.hpp"
#include "pipeline_encode.hpp"

namespace fs = std::filesystem;

namespace pipeline {

struct TrainOptions {
    fs::path     output_dir;
    std::string  method;                        // empty = skip regression
    regression::Precision precision = regression::Precision::FP32;
    std::array<double,2>  lambda    = {0.1, 0.1};
    bool         lambda_explicitly_set  = false;
    std::string  lambda_file_path;
    std::array<std::string,2> lambda_grid_specs = {"", ""};
    bool         lambda_grid_set    = false;
    int          nfolds             = 0;
    int          min_groups         = 0;
    std::map<std::string, std::string> params;  // slep opts, merged with enc.extra_params
    bool        adaptive_sparsification = false;
    std::string adaptive_l1_spec = "0.1,0.3,0.1";    // produces 0.1, 0.2, 0.3
    std::string adaptive_l2_spec = "0.1,0.3,0.1"; // produces 0.1, 0.2, 0.3
};

struct TrainResult {
    fs::path output_dir;
    std::vector<fs::path>             weights_paths;  // lambda_N/weights.txt per lambda
    std::vector<std::array<double,2>> lambdas_used;
};

TrainResult train(const EncodeResult& enc, const TrainOptions& opts);

} // namespace pipeline
