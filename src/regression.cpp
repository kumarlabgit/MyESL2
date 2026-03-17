#include "regression.hpp"

#include <stdexcept>

// These headers have no include guards, so they are included only here.
#include <map>
#include <string>
#include "sg_lasso_fp32.hpp"
#include "sg_lasso_fp64.hpp"
#include "sg_lasso_leastr_fp32.hpp"
#include "sg_lasso_leastr_fp64.hpp"
#include "overlapping_sg_lasso_leastr_fp32.hpp"
#include "overlapping_sg_lasso_leastr_fp64.hpp"
#include "overlapping_sg_lasso_logisticr_fp32.hpp"
#include "overlapping_sg_lasso_logisticr_fp64.hpp"

namespace regression {

// ---- Helpers ----------------------------------------------------------------

static std::map<std::string, std::string> slep_opts_from(
    const std::map<std::string, std::string>& params)
{
    std::map<std::string, std::string> opts;
    for (auto& [k, v] : params)
        if (k != "intercept" && k != "field")
            opts[k] = v;
    return opts;
}

static bool intercept_from(const std::map<std::string, std::string>& params)
{
    auto it = params.find("intercept");
    return (it == params.end() || it->second != "false");
}

static arma::rowvec load_field(const std::map<std::string, std::string>& params)
{
    auto it = params.find("field");
    if (it == params.end())
        throw std::runtime_error(
            "This method requires a 'field' parameter (path to feature-group index CSV)");
    arma::rowvec field;
    if (!field.load(it->second, arma::csv_ascii))
        throw std::runtime_error("Failed to load field file: " + it->second);
    return field;
}

// ---- SGLassoFP32 wrapper ----------------------------------------------------

class SGLassoFP32Wrapper : public RegressionAnalysis {
    std::array<double, 2>       lambda_;
    std::unique_ptr<SGLassoFP32> model_;
public:
    SGLassoFP32Wrapper(const arma::fmat&                          features,
                       const arma::frowvec&                       responses,
                       const arma::mat&                           alg_table,
                       const std::map<std::string, std::string>&  params,
                       const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoFP32>(
              features, responses, alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoFP32Wrapper(const arma::fmat&                          features,
                       const arma::frowvec&                       responses,
                       const arma::mat&                           alg_table,
                       const std::map<std::string, std::string>&  params,
                       const std::array<double, 2>&               lambda,
                       const arma::rowvec&                        xval_idxs,
                       int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoFP32>(
              features, responses, alg_table,
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- SGLassoFP64 wrapper ----------------------------------------------------

class SGLassoFP64Wrapper : public RegressionAnalysis {
    std::array<double, 2>       lambda_;
    std::unique_ptr<SGLassoFP64> model_;
public:
    SGLassoFP64Wrapper(const arma::fmat&                          features,
                       const arma::frowvec&                       responses,
                       const arma::mat&                           alg_table,
                       const std::map<std::string, std::string>&  params,
                       const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoFP64Wrapper(const arma::fmat&                          features,
                       const arma::frowvec&                       responses,
                       const arma::mat&                           alg_table,
                       const std::map<std::string, std::string>&  params,
                       const std::array<double, 2>&               lambda,
                       const arma::rowvec&                        xval_idxs,
                       int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table,
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- SGLassoLeastRFP32 wrapper ----------------------------------------------

class SGLassoLeastRFP32Wrapper : public RegressionAnalysis {
    std::array<double, 2>             lambda_;
    std::unique_ptr<SGLassoLeastRFP32> model_;
public:
    SGLassoLeastRFP32Wrapper(const arma::fmat&                          features,
                              const arma::frowvec&                       responses,
                              const arma::mat&                           alg_table,
                              const std::map<std::string, std::string>&  params,
                              const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastRFP32>(
              features, responses, alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoLeastRFP32Wrapper(const arma::fmat&                          features,
                              const arma::frowvec&                       responses,
                              const arma::mat&                           alg_table,
                              const std::map<std::string, std::string>&  params,
                              const std::array<double, 2>&               lambda,
                              const arma::rowvec&                        xval_idxs,
                              int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastRFP32>(
              features, responses, alg_table,
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- SGLassoLeastRFP64 wrapper ----------------------------------------------

class SGLassoLeastRFP64Wrapper : public RegressionAnalysis {
    std::array<double, 2>             lambda_;
    std::unique_ptr<SGLassoLeastRFP64> model_;
public:
    SGLassoLeastRFP64Wrapper(const arma::fmat&                          features,
                              const arma::frowvec&                       responses,
                              const arma::mat&                           alg_table,
                              const std::map<std::string, std::string>&  params,
                              const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoLeastRFP64Wrapper(const arma::fmat&                          features,
                              const arma::frowvec&                       responses,
                              const arma::mat&                           alg_table,
                              const std::map<std::string, std::string>&  params,
                              const std::array<double, 2>&               lambda,
                              const arma::rowvec&                        xval_idxs,
                              int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table,
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- OLSGLassoLeastRFP32 wrapper --------------------------------------------

class OLSGLassoLeastRFP32Wrapper : public RegressionAnalysis {
    std::array<double, 2>               lambda_;
    std::unique_ptr<OLSGLassoLeastRFP32> model_;
public:
    OLSGLassoLeastRFP32Wrapper(const arma::fmat&                          features,
                                const arma::frowvec&                       responses,
                                const arma::mat&                           alg_table,
                                const std::map<std::string, std::string>&  params,
                                const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastRFP32>(
              features, responses,
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLeastRFP32Wrapper(const arma::fmat&                          features,
                                const arma::frowvec&                       responses,
                                const arma::mat&                           alg_table,
                                const std::map<std::string, std::string>&  params,
                                const std::array<double, 2>&               lambda,
                                const arma::rowvec&                        xval_idxs,
                                int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastRFP32>(
              features, responses,
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- OLSGLassoLeastRFP64 wrapper --------------------------------------------

class OLSGLassoLeastRFP64Wrapper : public RegressionAnalysis {
    std::array<double, 2>               lambda_;
    std::unique_ptr<OLSGLassoLeastRFP64> model_;
public:
    OLSGLassoLeastRFP64Wrapper(const arma::fmat&                          features,
                                const arma::frowvec&                       responses,
                                const arma::mat&                           alg_table,
                                const std::map<std::string, std::string>&  params,
                                const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLeastRFP64Wrapper(const arma::fmat&                          features,
                                const arma::frowvec&                       responses,
                                const arma::mat&                           alg_table,
                                const std::map<std::string, std::string>&  params,
                                const std::array<double, 2>&               lambda,
                                const arma::rowvec&                        xval_idxs,
                                int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- OLSGLassoLogisticRFP32 wrapper -----------------------------------------

class OLSGLassoLogisticRFP32Wrapper : public RegressionAnalysis {
    std::array<double, 2>                  lambda_;
    std::unique_ptr<OLSGLassoLogisticRFP32> model_;
public:
    OLSGLassoLogisticRFP32Wrapper(const arma::fmat&                          features,
                                   const arma::frowvec&                       responses,
                                   const arma::mat&                           alg_table,
                                   const std::map<std::string, std::string>&  params,
                                   const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticRFP32>(
              features, responses,
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLogisticRFP32Wrapper(const arma::fmat&                          features,
                                   const arma::frowvec&                       responses,
                                   const arma::mat&                           alg_table,
                                   const std::map<std::string, std::string>&  params,
                                   const std::array<double, 2>&               lambda,
                                   const arma::rowvec&                        xval_idxs,
                                   int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticRFP32>(
              features, responses,
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- OLSGLassoLogisticRFP64 wrapper -----------------------------------------

class OLSGLassoLogisticRFP64Wrapper : public RegressionAnalysis {
    std::array<double, 2>                  lambda_;
    std::unique_ptr<OLSGLassoLogisticRFP64> model_;
public:
    OLSGLassoLogisticRFP64Wrapper(const arma::fmat&                          features,
                                   const arma::frowvec&                       responses,
                                   const arma::mat&                           alg_table,
                                   const std::map<std::string, std::string>&  params,
                                   const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLogisticRFP64Wrapper(const arma::fmat&                          features,
                                   const arma::frowvec&                       responses,
                                   const arma::mat&                           alg_table,
                                   const std::map<std::string, std::string>&  params,
                                   const std::array<double, 2>&               lambda,
                                   const arma::rowvec&                        xval_idxs,
                                   int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticRFP64>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), xval_idxs, xval_id,
              intercept_from(params)))
    {}

    void writeSparseMappedWeightsToStream(std::ofstream& out,
                                          std::ifstream& map_in) override
    {
        model_->writeSparseMappedWeightsToStream(out, map_in);
    }
};

// ---- Factory ----------------------------------------------------------------

std::unique_ptr<RegressionAnalysis> createRegressionAnalysis(
    const std::string&                        method,
    const arma::fmat&                         features,
    const arma::frowvec&                      responses,
    const arma::mat&                          alg_table,
    const std::map<std::string, std::string>& params,
    const std::array<double, 2>&              lambda,
    Precision                                 precision)
{
    if (method == "sg_lasso") {
        if (precision == Precision::FP64)
            return std::make_unique<SGLassoFP64Wrapper>(
                features, responses, alg_table, params, lambda);
        return std::make_unique<SGLassoFP32Wrapper>(
            features, responses, alg_table, params, lambda);
    }
    if (method == "sg_lasso_leastr") {
        if (precision == Precision::FP64)
            return std::make_unique<SGLassoLeastRFP64Wrapper>(
                features, responses, alg_table, params, lambda);
        return std::make_unique<SGLassoLeastRFP32Wrapper>(
            features, responses, alg_table, params, lambda);
    }
    if (method == "olsg_lasso_leastr") {
        if (precision == Precision::FP64)
            return std::make_unique<OLSGLassoLeastRFP64Wrapper>(
                features, responses, alg_table, params, lambda);
        return std::make_unique<OLSGLassoLeastRFP32Wrapper>(
            features, responses, alg_table, params, lambda);
    }
    if (method == "olsg_lasso_logisticr") {
        if (precision == Precision::FP64)
            return std::make_unique<OLSGLassoLogisticRFP64Wrapper>(
                features, responses, alg_table, params, lambda);
        return std::make_unique<OLSGLassoLogisticRFP32Wrapper>(
            features, responses, alg_table, params, lambda);
    }
    throw std::runtime_error(
        "Unknown regression method: '" + method +
        "'. Valid: sg_lasso, sg_lasso_leastr, olsg_lasso_leastr, olsg_lasso_logisticr");
}

std::unique_ptr<RegressionAnalysis> createRegressionAnalysisXVal(
    const std::string&                        method,
    const arma::fmat&                         features,
    const arma::frowvec&                      responses,
    const arma::mat&                          alg_table,
    const std::map<std::string, std::string>& params,
    const std::array<double, 2>&              lambda,
    const arma::rowvec&                       xval_idxs,
    int                                       xval_id,
    Precision                                 precision)
{
    if (method == "sg_lasso") {
        if (precision == Precision::FP64)
            return std::make_unique<SGLassoFP64Wrapper>(
                features, responses, alg_table, params, lambda, xval_idxs, xval_id);
        return std::make_unique<SGLassoFP32Wrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    }
    if (method == "sg_lasso_leastr") {
        if (precision == Precision::FP64)
            return std::make_unique<SGLassoLeastRFP64Wrapper>(
                features, responses, alg_table, params, lambda, xval_idxs, xval_id);
        return std::make_unique<SGLassoLeastRFP32Wrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    }
    if (method == "olsg_lasso_leastr") {
        if (precision == Precision::FP64)
            return std::make_unique<OLSGLassoLeastRFP64Wrapper>(
                features, responses, alg_table, params, lambda, xval_idxs, xval_id);
        return std::make_unique<OLSGLassoLeastRFP32Wrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    }
    if (method == "olsg_lasso_logisticr") {
        if (precision == Precision::FP64)
            return std::make_unique<OLSGLassoLogisticRFP64Wrapper>(
                features, responses, alg_table, params, lambda, xval_idxs, xval_id);
        return std::make_unique<OLSGLassoLogisticRFP32Wrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    }
    throw std::runtime_error(
        "Unknown regression method: '" + method +
        "'. Valid: sg_lasso, sg_lasso_leastr, olsg_lasso_leastr, olsg_lasso_logisticr");
}

} // namespace regression
