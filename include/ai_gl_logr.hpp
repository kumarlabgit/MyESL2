#pragma once

// Port of /claude/MyESL/src/ai_gl_logr.cpp — group lasso logistic regression
// solver using accelerated proximal gradient descent with Armijo-Goldstein
// line search. Only the mFlag==0 && lFlag==0 branch is implemented (the
// default path that MyESL's wrapper exercises); the other branches throw.
//
// Faithful port from the MyESL source. All `abs(` on double args have been
// qualified as `std::abs` to avoid the silent int-truncation bug seen on
// libstdc++ (see MyESL2 commit 10f7488).

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "eppVector.hpp"
#include "epph.hpp"
#include "initFactor.hpp"
#include "sll_opts.hpp"

struct glLogisticR_result {
    arma::vec x;
    double c;
    arma::vec funVal;
    arma::vec ValueL;
};

static inline glLogisticR_result glLogisticR(const arma::mat& A, const arma::vec& y, double z, const sll_opts& opts) {
    int m = A.n_rows;
    int n = A.n_cols;

    if ((int)y.n_elem != m) {
        throw std::runtime_error("Check the length of y!");
    }

    if (z <= 0) {
        throw std::runtime_error("z should be positive!");
    }

    sll_opts processed_opts = default_sll_opts(opts);

    arma::vec mu, nu;
    arma::uvec ind_zero;

    if (processed_opts.nFlag != 0) {
        if (processed_opts.mu.has_value()) {
            mu = processed_opts.mu.value();
            if ((int)mu.n_elem != n) {
                throw std::runtime_error("Check the input .mu");
            }
        } else {
            mu = arma::mean(A, 0).t();
        }

        if (processed_opts.nFlag == 1) {
            if (processed_opts.nu.has_value()) {
                nu = processed_opts.nu.value();
                if ((int)nu.n_elem != n) {
                    throw std::runtime_error("Check the input .nu!");
                }
            } else {
                nu = arma::sqrt(arma::sum(arma::square(A), 0) / m).t();
            }
        } else {
            if (processed_opts.nu.has_value()) {
                nu = processed_opts.nu.value();
                if ((int)nu.n_elem != m) {
                    throw std::runtime_error("Check the input .nu!");
                }
            } else {
                nu = arma::sqrt(arma::sum(arma::square(A), 1) / n);
            }
        }

        ind_zero = arma::find(arma::abs(nu) <= 1e-10);
        nu.elem(ind_zero).ones();
    }

    // Initialize ind and q
    arma::uvec ind;
    int k;
    if (!processed_opts.ind.has_value()) {
        throw std::runtime_error("In glLogisticR, the fields .ind should be specified");
    } else {
        ind = processed_opts.ind.value();

        k = ind.n_elem - 1;
        if ((int)ind(k) != n) {
            throw std::runtime_error("Check opts.ind");
        }
    }

    // Initialize q
    double q;
    if (processed_opts.q == 0) {
        q = 2;
        processed_opts.q = 2;
    } else {
        q = processed_opts.q;
        if (q < 1) {
            throw std::runtime_error("q should be larger than 1");
        }
    }

    arma::vec weight(m);
    if (processed_opts.sWeight.has_value()) {
        arma::vec sWeight = processed_opts.sWeight.value();

        if (sWeight.n_elem != 2 || sWeight(0) <= 0 || sWeight(1) <= 0) {
            throw std::runtime_error("Check opts.sWeight, which contains two positive values");
        }

        arma::uvec p_flag = arma::find(y == 1);
        double m1_sw = p_flag.n_elem * sWeight(0);
        double m2_sw = (m - p_flag.n_elem) * sWeight(1);

        weight.fill(0);
        weight.elem(p_flag).fill(sWeight(0) / (m1_sw + m2_sw));
        arma::uvec n_flag = arma::find(y != 1);
        weight.elem(n_flag).fill(sWeight(1) / (m1_sw + m2_sw));
    } else {
        weight.ones();
        weight /= m;
    }

    // gWeight
    arma::vec gWeight(k);
    if (processed_opts.gWeight.has_value()) {
        gWeight = processed_opts.gWeight.value();
        if ((int)gWeight.n_elem != k) {
            throw std::runtime_error("opts.gWeight should a " + std::to_string(k) + " x 1 vector");
        }

        if (arma::min(gWeight) <= 0) {
            throw std::runtime_error(".gWeight should be positive");
        }
    } else {
        gWeight.ones();
    }

    // Starting point initialization
    arma::uvec p_flag = arma::find(y == 1);
    double m1 = arma::sum(weight.elem(p_flag));
    double m2 = 1 - m1;

    // Process the regularization parameter
    double lambda;
    if (processed_opts.rFlag == 0) {
        lambda = z;
    } else {
        arma::vec b(m);
        b.elem(p_flag).fill(m2);
        arma::uvec n_flag = arma::find(y != 1);
        b.elem(n_flag).fill(-m1);
        b = b % weight;

        arma::vec ATb;
        if (processed_opts.nFlag == 0) {
            ATb = A.t() * b;
        } else if (processed_opts.nFlag == 1) {
            ATb = A.t() * b - arma::sum(b) * mu;
            ATb = ATb / nu;
        } else {
            arma::vec invNu = b / nu;
            ATb = A.t() * invNu - arma::sum(invNu) * mu;
        }

        double q_bar;
        if (q == 1) {
            q_bar = arma::datum::inf;
        } else if (q >= 1e6) {
            q_bar = 1;
        } else {
            q_bar = q / (q - 1);
        }

        arma::vec norm_ATb(k);
        for (int i = 0; i < k; i++) {
            arma::vec group_ATb = ATb.subvec(ind(i), ind(i+1) - 1);
            norm_ATb(i) = arma::norm(group_ATb, q_bar);
        }

        norm_ATb = norm_ATb / gWeight;

        double lambda_max = arma::max(norm_ATb);

        lambda = z * lambda_max;
    }

    // initialize a starting point
    arma::vec x(n);
    double c;
    if (processed_opts.init == 2) {
        x.zeros();
        c = std::log(m1 / m2);
    } else {
        if (processed_opts.x0.has_value()) {
            x = processed_opts.x0.value();
            if ((int)x.n_elem != n) {
                throw std::runtime_error("Check the input .x0");
            }
        } else {
            x.zeros();
        }

        if (processed_opts.c0.has_value() && processed_opts.c0 != 0) {
            c = processed_opts.c0.value();
        } else {
            c = std::log(m1 / m2);
        }
    }

    // compute A x
    arma::vec Ax(m);
    if (processed_opts.nFlag == 0) {
        Ax = A * x;
    } else if (processed_opts.nFlag == 1) {
        arma::vec invNu = x / nu;
        double mu_invNu = arma::dot(mu, invNu);
        Ax = A * invNu - mu_invNu;
    } else {
        Ax = A * x - arma::dot(mu, x);
        Ax = Ax / nu;
    }

    arma::vec funVal(processed_opts.maxIter);
    arma::vec ValueL(processed_opts.maxIter);
    int iterStep = 0;

    // The main program — only mFlag==0 && lFlag==0 is implemented.
    if (processed_opts.mFlag == 0 && processed_opts.lFlag == 0) {

        bool bFlag = false;

        double L = 1.0 / m;

        arma::vec weighty = weight % y;

        arma::vec xp = x, Axp = Ax, xxp(n, arma::fill::zeros);
        double cp = c, ccp = 0;

        double alphap = 0, alpha = 1;

        for (iterStep = 1; iterStep <= processed_opts.maxIter; iterStep++) {
            double beta = (alphap - 1) / alpha;
            arma::vec s = x + beta * xxp;
            double sc = c + beta * ccp;

            arma::vec As = Ax + beta * (Ax - Axp);

            arma::vec aa = -y % (As + sc);

            arma::vec bb = arma::max(aa, arma::zeros(aa.n_elem));
            double fun_s = arma::dot(weight, arma::log(arma::exp(-bb) + arma::exp(aa - bb)) + bb);

            arma::vec prob = 1.0 / (1 + arma::exp(aa));

            arma::vec b = -weighty % (1 - prob);

            double gc = arma::sum(b);

            arma::vec g;
            if (processed_opts.nFlag == 0) {
                g = A.t() * b;
            } else if (processed_opts.nFlag == 1) {
                g = A.t() * b - arma::sum(b) * mu;
                g = g / nu;
            } else {
                arma::vec invNu = b / nu;
                g = A.t() * invNu - arma::sum(invNu) * mu;
            }

            xp = x;    Axp = Ax;
            cp = c;
            double fun_x;
            while (true) {
                arma::vec v = s - g / L;
                c = sc - gc / L;

                if (q < 1e6) {
                    x = eppVector(v, ind, k, n, lambda / L * gWeight, q);
                } else {
                    x = eppVector(v, ind, k, n, lambda / L * gWeight, 1e6);
                }

                v = x - s;

                if (processed_opts.nFlag == 0) {
                    Ax = A * x;
                } else if (processed_opts.nFlag == 1) {
                    arma::vec invNu = x / nu;
                    double mu_invNu = arma::dot(mu, invNu);
                    Ax = A * invNu - mu_invNu;
                } else {
                    Ax = A * x - arma::dot(mu, x);
                    Ax = Ax / nu;
                }

                aa = -y % (Ax + c);

                bb = arma::max(aa, arma::zeros(aa.n_elem));
                fun_x = arma::dot(weight, arma::log(arma::exp(-bb) + arma::exp(aa - bb)) + bb);

                double r_sum = (arma::dot(v, v) + std::pow(c - sc, 2)) / 2;
                double l_sum = fun_x - fun_s - arma::dot(v, g) - (c - sc) * gc;

                if (r_sum <= 1e-20) {
                    bFlag = true;
                    break;
                }

                if (l_sum <= r_sum * L) {
                    break;
                } else {
                    L = std::max(2 * L, l_sum / r_sum);
                }
            }

            alphap = alpha;
            alpha = (1 + std::sqrt(4 * alpha * alpha + 1)) / 2;

            ValueL(iterStep - 1) = L;

            xxp = x - xp;    ccp = c - cp;

            arma::vec norm_x_k(k);
            for (int i = 0; i < k; i++) {
                arma::vec x_group = x.subvec(ind(i), ind(i+1) - 1);
                norm_x_k(i) = arma::norm(x_group, q);
            }

            funVal(iterStep - 1) = fun_x + lambda * arma::dot(norm_x_k, gWeight);

            if (bFlag) {
                break;
            }
            switch (processed_opts.tFlag) {
                case 0:
                    if (iterStep >= 2) {
                        if (std::abs(funVal(iterStep - 1) - funVal(iterStep - 2)) <= processed_opts.tol) {
                            goto exit_loop_1;
                        }
                    }
                    break;
                case 1:
                    if (iterStep >= 2) {
                        if (std::abs(funVal(iterStep - 1) - funVal(iterStep - 2)) <=
                            processed_opts.tol * funVal(iterStep - 2)) {
                            goto exit_loop_1;
                        }
                    }
                    break;
                case 2:
                    if (funVal(iterStep - 1) <= processed_opts.tol) {
                        goto exit_loop_1;
                    }
                    break;
                case 3:
                    {
                        double norm_xxp = arma::norm(xxp);
                        if (norm_xxp <= processed_opts.tol) {
                            goto exit_loop_1;
                        }
                    }
                    break;
                case 4:
                    {
                        double norm_xp = arma::norm(xp);
                        double norm_xxp = arma::norm(xxp);
                        if (norm_xxp <= processed_opts.tol * std::max(norm_xp, 1.0)) {
                            goto exit_loop_1;
                        }
                    }
                    break;
                case 5:
                    if (iterStep >= processed_opts.maxIter) {
                        goto exit_loop_1;
                    }
                    break;
            }
        }
        exit_loop_1:;
    } else {
        throw std::runtime_error(
            "gl_logisticr: only mFlag=0 and lFlag=0 are implemented in MyESL2. "
            "Set slep_opts.mFlag=0 and slep_opts.lFlag=0."
        );
    }

    funVal.resize(iterStep);
    ValueL.resize(iterStep);

    glLogisticR_result result;
    result.x = x;
    result.c = c;
    result.funVal = funVal;
    result.ValueL = ValueL;

    return result;
}
