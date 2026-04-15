#pragma once

// Port of /claude/MyESL/src/initFactor.cpp.
// Used only by ai_gl_logr.hpp when init!=2 (non-default path).

#include <armadillo>
#include <cmath>
#include <string>

static inline double initFactor(double x_norm, const arma::mat& Ax, const arma::mat& y, double z,
                                const std::string& funName, double rsL2, double x_2norm) {
    double ratio = 0.0;

    if (funName == "LeastC") {
        double ratio_max = z / x_norm;
        double ratio_optimal = arma::as_scalar(Ax.t() * y) / (arma::as_scalar(Ax.t() * Ax) + rsL2 * x_2norm);

        if (std::abs(ratio_optimal) <= ratio_max) {
            ratio = ratio_optimal;
        } else if (ratio_optimal < 0) {
            ratio = -ratio_max;
        } else {
            ratio = ratio_max;
        }
    } else if (funName == "LeastR") {
        ratio = (arma::as_scalar(Ax.t() * y) - z * x_norm) / (arma::as_scalar(Ax.t() * Ax) + rsL2 * x_2norm);
    } else if (funName == "glLeastR") {
        ratio = (arma::as_scalar(Ax.t() * y) - z * x_norm) / arma::as_scalar(Ax.t() * Ax);
    } else if (funName == "mcLeastR") {
        arma::vec Ax_vec = arma::vectorise(Ax);
        arma::vec y_vec = arma::vectorise(y);
        ratio = (arma::as_scalar(Ax_vec.t() * y_vec) - z * x_norm) / std::pow(arma::norm(Ax, "fro"), 2);
    } else if (funName == "mtLeastR") {
        ratio = (arma::as_scalar(Ax.t() * y) - z * x_norm) / arma::as_scalar(Ax.t() * Ax);
    } else if (funName == "nnLeastR") {
        ratio = (arma::as_scalar(Ax.t() * y) - z * x_norm) / (arma::as_scalar(Ax.t() * Ax) + rsL2 * x_2norm);
        ratio = std::max(0.0, ratio);
    } else if (funName == "nnLeastC") {
        double ratio_max = z / x_norm;
        double ratio_optimal = arma::as_scalar(Ax.t() * y) / (arma::as_scalar(Ax.t() * Ax) + rsL2 * x_2norm);

        if (ratio_optimal < 0) {
            ratio = 0;
        } else if (ratio_optimal <= ratio_max) {
            ratio = ratio_optimal;
        } else {
            ratio = ratio_max;
        }
    } else if (funName == "mcLeastC") {
        double ratio_max = z / x_norm;
        arma::vec Ax_vec = arma::vectorise(Ax);
        arma::vec y_vec = arma::vectorise(y);
        double ratio_optimal = arma::as_scalar(Ax_vec.t() * y_vec) / std::pow(arma::norm(Ax.t() * Ax, "fro"), 2);

        if (std::abs(ratio_optimal) <= ratio_max) {
            ratio = ratio_optimal;
        } else if (ratio_optimal < 0) {
            ratio = -ratio_max;
        } else {
            ratio = ratio_max;
        }
    }

    return ratio;
}
