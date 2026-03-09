#include "regression.hpp"

#include <stdexcept>

// These headers have no include guards, so they are included only here.
#include <map>
#include <string>
#include "sg_lasso.hpp"
#include "sg_lasso_leastr.hpp"
#include "overlapping_sg_lasso_leastr.hpp"
#include "overlapping_sg_lasso_logisticr.hpp"

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

// ---- SGLasso wrapper --------------------------------------------------------

class SGLassoWrapper : public RegressionAnalysis {
    std::array<double, 2>    lambda_;
    std::unique_ptr<SGLasso> model_;
public:
    SGLassoWrapper(const arma::fmat&                          features,
                   const arma::frowvec&                       responses,
                   const arma::mat&                           alg_table,
                   const std::map<std::string, std::string>&  params,
                   const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLasso>(
              features, responses, alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoWrapper(const arma::fmat&                          features,
                   const arma::frowvec&                       responses,
                   const arma::mat&                           alg_table,
                   const std::map<std::string, std::string>&  params,
                   const std::array<double, 2>&               lambda,
                   const arma::rowvec&                        xval_idxs,
                   int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLasso>(
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

// ---- SGLassoLeastR wrapper --------------------------------------------------

class SGLassoLeastRWrapper : public RegressionAnalysis {
    std::array<double, 2>          lambda_;
    std::unique_ptr<SGLassoLeastR> model_;
public:
    SGLassoLeastRWrapper(const arma::fmat&                          features,
                         const arma::frowvec&                       responses,
                         const arma::mat&                           alg_table,
                         const std::map<std::string, std::string>&  params,
                         const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastR>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table,
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    SGLassoLeastRWrapper(const arma::fmat&                          features,
                         const arma::frowvec&                       responses,
                         const arma::mat&                           alg_table,
                         const std::map<std::string, std::string>&  params,
                         const std::array<double, 2>&               lambda,
                         const arma::rowvec&                        xval_idxs,
                         int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<SGLassoLeastR>(
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

// ---- OLSGLassoLeastR wrapper ------------------------------------------------

class OLSGLassoLeastRWrapper : public RegressionAnalysis {
    std::array<double, 2>            lambda_;
    std::unique_ptr<OLSGLassoLeastR> model_;
public:
    OLSGLassoLeastRWrapper(const arma::fmat&                          features,
                            const arma::frowvec&                       responses,
                            const arma::mat&                           alg_table,
                            const std::map<std::string, std::string>&  params,
                            const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastR>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLeastRWrapper(const arma::fmat&                          features,
                            const arma::frowvec&                       responses,
                            const arma::mat&                           alg_table,
                            const std::map<std::string, std::string>&  params,
                            const std::array<double, 2>&               lambda,
                            const arma::rowvec&                        xval_idxs,
                            int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLeastR>(
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

// ---- OLSGLassoLogisticR wrapper ---------------------------------------------

class OLSGLassoLogisticRWrapper : public RegressionAnalysis {
    std::array<double, 2>               lambda_;
    std::unique_ptr<OLSGLassoLogisticR> model_;
public:
    OLSGLassoLogisticRWrapper(const arma::fmat&                          features,
                               const arma::frowvec&                       responses,
                               const arma::mat&                           alg_table,
                               const std::map<std::string, std::string>&  params,
                               const std::array<double, 2>&               lambda)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticR>(
              arma::conv_to<arma::mat>::from(features),
              arma::conv_to<arma::rowvec>::from(responses),
              alg_table, load_field(params),
              lambda_.data(), slep_opts_from(params), intercept_from(params)))
    {}

    OLSGLassoLogisticRWrapper(const arma::fmat&                          features,
                               const arma::frowvec&                       responses,
                               const arma::mat&                           alg_table,
                               const std::map<std::string, std::string>&  params,
                               const std::array<double, 2>&               lambda,
                               const arma::rowvec&                        xval_idxs,
                               int                                        xval_id)
        : lambda_(lambda),
          model_(std::make_unique<OLSGLassoLogisticR>(
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
    const std::array<double, 2>&              lambda)
{
    if (method == "sg_lasso")
        return std::make_unique<SGLassoWrapper>(
            features, responses, alg_table, params, lambda);
    if (method == "sg_lasso_leastr")
        return std::make_unique<SGLassoLeastRWrapper>(
            features, responses, alg_table, params, lambda);
    if (method == "olsg_lasso_leastr")
        return std::make_unique<OLSGLassoLeastRWrapper>(
            features, responses, alg_table, params, lambda);
    if (method == "olsg_lasso_logisticr")
        return std::make_unique<OLSGLassoLogisticRWrapper>(
            features, responses, alg_table, params, lambda);
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
    int                                       xval_id)
{
    if (method == "sg_lasso")
        return std::make_unique<SGLassoWrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    if (method == "sg_lasso_leastr")
        return std::make_unique<SGLassoLeastRWrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    if (method == "olsg_lasso_leastr")
        return std::make_unique<OLSGLassoLeastRWrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    if (method == "olsg_lasso_logisticr")
        return std::make_unique<OLSGLassoLogisticRWrapper>(
            features, responses, alg_table, params, lambda, xval_idxs, xval_id);
    throw std::runtime_error(
        "Unknown regression method: '" + method +
        "'. Valid: sg_lasso, sg_lasso_leastr, olsg_lasso_leastr, olsg_lasso_logisticr");
}

} // namespace regression
