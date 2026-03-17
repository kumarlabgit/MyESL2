#pragma once

#include <armadillo>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <string>

namespace regression {

enum class Precision { FP32, FP64 };

class RegressionAnalysis {
public:
    virtual ~RegressionAnalysis() = default;
    virtual void writeSparseMappedWeightsToStream(std::ofstream& weights_out,
                                                   std::ifstream& feature_map) = 0;
};

// Factory: creates the appropriate RegressionAnalysis subclass for the given method.
//
// Supported methods:
//   "sg_lasso"             - SGLasso           (float arithmetic)
//   "sg_lasso_leastr"      - SGLassoLeastR     (double, least-squares)
//   "olsg_lasso_leastr"    - OLSGLassoLeastR   (double, overlapping groups, least-squares)
//   "olsg_lasso_logisticr" - OLSGLassoLogisticR(double, overlapping groups, logistic)
//
// features:  rows=sequences, cols=encoded positions
// responses: one value per sequence
// alg_table: 3 x alignments weight/index matrix
// params:    key-value options forwarded to the model as slep_opts; recognised keys:
//              "intercept" = "false"  — disable intercept term (default: true)
//              "field"     = <path>   — CSV file of group indices (overlapping methods)
//            all other entries are passed through as slep_opts
// lambda:    {lambda1, lambda2}
// precision: FP32 (default) or FP64
std::unique_ptr<RegressionAnalysis> createRegressionAnalysis(
    const std::string&                        method,
    const arma::fmat&                         features,
    const arma::frowvec&                      responses,
    const arma::mat&                          alg_table,
    const std::map<std::string, std::string>& params,
    const std::array<double, 2>&              lambda,
    Precision                                 precision = Precision::FP32);

std::unique_ptr<RegressionAnalysis> createRegressionAnalysisXVal(
    const std::string&                        method,
    const arma::fmat&                         features,
    const arma::frowvec&                      responses,
    const arma::mat&                          alg_table,
    const std::map<std::string, std::string>& params,
    const std::array<double, 2>&              lambda,
    const arma::rowvec&                       xval_idxs,
    int                                       xval_id,
    Precision                                 precision = Precision::FP32);

} // namespace regression
