#pragma once
#include "pipeline_encode.hpp"
#include "pipeline_train.hpp"

namespace pipeline {

// Called when encode() throws max_mem_exceeded and adaptive_sparsification is enabled.
// Reads preprocess_config from enc_opts.output_dir (written by a prior preprocess() call).
void adaptive_train(const EncodeOptions& enc_opts, const TrainOptions& train_opts);

} // namespace pipeline
