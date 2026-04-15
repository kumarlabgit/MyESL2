#pragma once

// GLLogisticRFP32 — group lasso logistic regression solver (FP32 precision).
//
// Derived from GLLogisticRFP64 (src/gl_logisticr_fp64.cpp) by substituting
// float / std::vector<float> for double / std::vector<double> in the
// per-element state vectors. Scalar accumulators (c, L, alpha, beta, fun_s,
// fun_x, l_sum, r_sum, gc, lambda) stay in double precision, matching
// src/sg_lasso_fp32.cpp's pattern.
//
// FP32 is NOT expected to match MyESL's reference binary bit-for-bit —
// MyESL is double-only, so there is no FP32 reference to compare against.
// Use GLLogisticRFP64 for bit-exact results against MyESL.

#include <armadillo>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

class GLLogisticRFP32 {
public:
    GLLogisticRFP32(const arma::fmat& features,
                    const arma::frowvec& responses,
                    const arma::mat& weights,
                    double* lambda,
                    std::map<std::string, std::string> slep_opts,
                    const bool intercept = true);

    GLLogisticRFP32(const arma::fmat& features,
                    const arma::frowvec& responses,
                    const arma::mat& weights,
                    double* lambda,
                    std::map<std::string, std::string> slep_opts,
                    const arma::rowvec& xval_idxs,
                    int xval_id,
                    const bool intercept = true);

    arma::frowvec Train(const arma::fmat& features,
                        const arma::frowvec& responses,
                        const arma::mat& weights,
                        std::map<std::string, std::string> slep_opts,
                        const bool intercept = true);

    void writeModelToXMLStream(std::ofstream& XMLFile);
    void writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile,
                                          std::ifstream& FeatureMap);

    const arma::fvec& Parameters() const { return parameters; }
    double InterceptValue() const { return intercept_value; }
    double* Lambda() { return lambda; }
    int NonZeroGeneCount() const { return nz_gene_count; }

private:
    int nz_gene_count = 0;
    arma::fvec parameters;
    double* lambda = nullptr;
    bool intercept = true;
    double intercept_value = 0.0;
};
