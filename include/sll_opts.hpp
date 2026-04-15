#pragma once

// Port of /claude/MyESL/src/sll_opts.cpp — options struct for the SLEP
// Sparse Learning Library. Used by gl_logisticr in Phase A; Phase B inlines
// option parsing directly into Train() so this header is only referenced by
// ai_gl_logr.hpp (the Phase A reference solver).

#include <armadillo>
#include <cmath>
#include <optional>
#include <string>

struct sll_opts {
    // Starting point
    std::optional<arma::vec> x0;
    std::optional<double> c0;
    int init = 2;

    // Termination
    int maxIter = 100;
    double tol = 1e-3;
    int tFlag = 5;

    // Normalization
    int nFlag = 0;
    std::optional<arma::colvec> mu;
    std::optional<arma::vec> nu;

    // Regularization
    int rFlag = 1;
    double rsL2 = 0.0;

    // Method & Line Search
    int lFlag = 0;
    int mFlag = 0;

    // Group & Others
    std::optional<arma::uvec> ind;
    double q = 2.0;
    std::optional<arma::vec> sWeight;
    std::optional<arma::vec> gWeight;
    std::string fName;
};

// Validate and set default options. Matches the MyESL source exactly.
static inline sll_opts default_sll_opts(sll_opts opts) {
    // Starting point
    if (opts.init != 0 && opts.init != 1 && opts.init != 2) {
        opts.init = 0;
    }

    if (!opts.x0.has_value() && opts.init == 1) {
        opts.init = 0;
    }

    // Termination
    if (opts.maxIter < 1) {
        opts.maxIter = 10000;
    }

    if (opts.tFlag < 0) {
        opts.tFlag = 0;
    } else if (opts.tFlag > 5) {
        opts.tFlag = 5;
    } else {
        opts.tFlag = static_cast<int>(std::floor(opts.tFlag));
    }

    // Normalization
    if (opts.nFlag != 1 && opts.nFlag != 2) {
        opts.nFlag = 0;
    }

    // Regularization
    if (opts.rFlag != 1) {
        opts.rFlag = 0;
    }

    // Method (Line Search)
    if (opts.lFlag != 1) {
        opts.lFlag = 0;
    }

    if (opts.mFlag != 1) {
        opts.mFlag = 0;
    }

    return opts;
}
