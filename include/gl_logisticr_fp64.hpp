#pragma once

// GLLogisticRFP64 — group lasso logistic regression solver (FP64 precision).
//
// Ported from /claude/MyESL/src/gl_logisticr.{hpp,cpp}. Only lambda[0] is
// used; lambda[1] is kept in the interface so this class stays interchangeable
// with the other solvers in the MyESL2 factory but is ignored by the solver.
//
// Phase A: Train() delegates to glLogisticR() in ai_gl_logr.hpp (faithful
// Armadillo port of the MyESL reference). Phase B will replace Train()'s
// body with raw C++ loops matching sg_lasso_fp64.cpp; Phase C derives an
// FP32 variant.

#include <armadillo>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

class GLLogisticRFP64 {
public:
    GLLogisticRFP64(const arma::mat& features,
                    const arma::rowvec& responses,
                    const arma::mat& weights,
                    double* lambda,
                    std::map<std::string, std::string> slep_opts,
                    const bool intercept = true);

    GLLogisticRFP64(const arma::mat& features,
                    const arma::rowvec& responses,
                    const arma::mat& weights,
                    double* lambda,
                    std::map<std::string, std::string> slep_opts,
                    const arma::rowvec& xval_idxs,
                    int xval_id,
                    const bool intercept = true);

    arma::rowvec Train(const arma::mat& features,
                       const arma::rowvec& responses,
                       const arma::mat& weights,
                       std::map<std::string, std::string> slep_opts,
                       const bool intercept = true);

    void writeModelToXMLStream(std::ofstream& XMLFile);
    void writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile,
                                          std::ifstream& FeatureMap);

    const arma::vec& Parameters() const { return parameters; }
    double InterceptValue() const { return intercept_value; }
    double* Lambda() { return lambda; }
    int NonZeroGeneCount() const { return nz_gene_count; }

private:
    int nz_gene_count = 0;
    arma::vec parameters;
    double* lambda = nullptr;
    bool intercept = true;
    double intercept_value = 0.0;
};
