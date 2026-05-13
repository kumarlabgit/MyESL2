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

    // Direct parameter access (avoids file-based weight extraction)
    virtual arma::vec getParameters() const = 0;
    virtual double getInterceptValue() const = 0;
};

// Resolve deprecated method aliases.  Returns the canonical name,
// or the input unchanged if it is not a known alias.
// If an alias was resolved, sets *was_alias = true (when non-null).
std::string resolve_method_alias(const std::string& method,
                                 bool* was_alias = nullptr);

// Factory: creates the appropriate RegressionAnalysis subclass for the given method.
//
// Supported methods:
//   "sg_lasso_logisticr"   - SGLassoLogisticR  (logistic, float/double arithmetic)
//   "sg_lasso_leastr"      - SGLassoLeastR     (least-squares, float/double)
//   "olsg_lasso_logisticr" - OLSGLassoLogisticR(dual-variable overlapping, logistic)
//   "olsg_lasso_leastr"    - OLSGLassoLeastR   (dual-variable overlapping, least-squares)
//   "ol_sg_lasso_logisticr"- OLSGLassoLogisticRv(virtual-expansion overlapping, logistic)
//   "ol_sg_lasso_leastr"   - OLSGLassoLeastRv  (virtual-expansion overlapping, least-squares)
//   "gl_logisticr"         - GLLogisticR       (group lasso, logistic)
//
// Deprecated aliases (auto-resolved with warning):
//   "sg_lasso"    -> "sg_lasso_logisticr"
//   "ol_sg_lasso" -> "ol_sg_lasso_logisticr"
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
