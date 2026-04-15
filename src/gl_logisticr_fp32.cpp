// GLLogisticRFP32 — Phase C implementation.
//
// Derived from src/gl_logisticr_fp64.cpp. Per-element state vectors (x, Ax,
// s, v, g, aa, bb, prob, weight, weighty, b_vec, y, xp, Axp, xxp) are stored
// as std::vector<float>. Scalar accumulators (c, L, alpha, beta, sc, gc,
// fun_s, fun_x, l_sum, r_sum, lambda) stay in double precision, matching
// src/sg_lasso_fp32.cpp's pattern — this avoids losing precision in the
// line-search loop while still benefiting from float bandwidth/throughput
// on the per-element updates.
//
// Matvec calls route to cblas_sgemv (the float counterpart to cblas_dgemv
// used by the FP64 variant). Dot products and reductions use double
// accumulators over float inputs to limit FP32 round-off accumulation.
//
// The prox step calls eppVector_inplace(float*, ...) which round-trips
// through double internally (epph.hpp's epp() is double-only — see the
// comment on the overload in include/eppVector.hpp:56).
//
// FP32 is NOT expected to match MyESL's reference binary or the FP64 variant
// bit-for-bit. The V6 verification step checks gene-level numerical proximity
// against FP64 on a real dataset.

#include "gl_logisticr_fp32.hpp"
#include "eppVector.hpp"
#include "sg_lasso_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// Direct OpenBLAS cblas_sgemv declaration. See src/gl_logisticr_fp64.cpp for
// the rationale for calling BLAS directly instead of a hand-written matvec.
extern "C" {
    void cblas_sgemv(int order, int trans, int M, int N, float alpha,
                     const float* A, int lda, const float* X, int incX,
                     float beta, float* Y, int incY);
}
static constexpr int GL32_CBLAS_COL_MAJOR = 102;
static constexpr int GL32_CBLAS_NO_TRANS  = 111;
static constexpr int GL32_CBLAS_TRANS     = 112;


GLLogisticRFP32::GLLogisticRFP32(const arma::fmat& features,
                                 const arma::frowvec& responses,
                                 const arma::mat& weights,
                                 double* lambda,
                                 std::map<std::string, std::string> slep_opts,
                                 const bool intercept)
    : lambda(lambda), intercept(intercept)
{
    Train(features, responses, weights, slep_opts, intercept);
}

GLLogisticRFP32::GLLogisticRFP32(const arma::fmat& features,
                                 const arma::frowvec& responses,
                                 const arma::mat& weights,
                                 double* lambda,
                                 std::map<std::string, std::string> slep_opts,
                                 const arma::rowvec& xval_idxs,
                                 int xval_id,
                                 const bool intercept)
    : lambda(lambda), intercept(intercept)
{
    // Subset features and responses according to xval_id and xval_idxs.
    arma::uvec indices = arma::find(xval_idxs != xval_id);
    Train(features.rows(indices), responses.elem(indices).t(), weights, slep_opts, intercept);
}

void GLLogisticRFP32::writeModelToXMLStream(std::ofstream& XMLFile)
{
    // FP32 variant writes parameters directly (already float) — no float cast
    // is needed here since the FP64 variant's cast was a compatibility shim
    // with MyESL's latent RRLogisticR fvec bug, and the FP32 variant has no
    // corresponding reference to byte-match against.
    int i_level = 0;
    XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
    i_level++;
    XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
    i_level++;
    XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
    for (arma::uword i = 0; i < this->parameters.n_elem; i++) {
        std::ostringstream streamObj;
        streamObj << std::setprecision(17) << std::scientific << this->parameters(i);
        XMLFile << std::string(i_level * 8, ' ') + "<item>" + streamObj.str() + "</item>" + "\n";
    }
    i_level--;
    XMLFile << std::string(i_level * 8, ' ') + "</parameters>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<lambda1>" + std::to_string(this->lambda[0]) + "</lambda1>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<lambda2>" + std::to_string(this->lambda[1]) + "</lambda2>" + "\n";
    XMLFile << std::string(i_level * 8, ' ') + "<intercept_value>" + std::to_string(this->intercept_value) + "</intercept_value>" + "\n";
    i_level--;
    XMLFile << std::string(i_level * 8, ' ') + "</model>" + "\n";
}

void GLLogisticRFP32::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile,
                                                       std::ifstream& FeatureMap)
{
    std::string line;
    std::getline(FeatureMap, line);
    for (arma::uword i = 0; i < this->parameters.n_elem; i++) {
        std::getline(FeatureMap, line);
        if (this->parameters(i) == 0.0f) {
            continue;
        }
        std::istringstream iss(line);
        std::string feature_label;
        std::getline(iss, feature_label, '\t');
        std::getline(iss, feature_label, '\t');
        std::ostringstream streamObj;
        streamObj << std::setprecision(17) << std::scientific << this->parameters(i);
        MappedWeightsFile << feature_label + "\t" + streamObj.str() + "\n";
    }
    MappedWeightsFile << "Intercept\t" + std::to_string(this->intercept_value) + "\n";
    FeatureMap.clear();
    FeatureMap.seekg(0);
}

arma::frowvec GLLogisticRFP32::Train(const arma::fmat& A,
                                     const arma::frowvec& responses,
                                     const arma::mat& weights,
                                     std::map<std::string, std::string> slep_opts,
                                     const bool intercept)
{
    this->intercept = intercept;

    auto trim = [](std::string& s) {
        size_t p = s.find_first_not_of(" \t\r\n");
        s.erase(0, p);
        p = s.find_last_not_of(" \t\r\n");
        if (std::string::npos != p) s.erase(p + 1);
    };

    // Parse slep_opts into plain locals. Defaults match sll_opts.hpp's
    // default_sll_opts for gl_logisticr.
    int opts_maxIter = 100;
    int opts_init = 2;
    int opts_tFlag = 5;
    int opts_nFlag = 0;
    int opts_rFlag = 1;
    int opts_mFlag = 0;
    int opts_lFlag = 0;
    double opts_tol = 1e-3;
    double opts_q = 2.0;

    if (slep_opts.find("maxIter") != slep_opts.end()) opts_maxIter = std::stoi(slep_opts["maxIter"]);
    if (slep_opts.find("init")    != slep_opts.end()) opts_init    = std::stoi(slep_opts["init"]);
    if (slep_opts.find("tFlag")   != slep_opts.end()) opts_tFlag   = std::stoi(slep_opts["tFlag"]);
    if (slep_opts.find("nFlag")   != slep_opts.end()) opts_nFlag   = std::stoi(slep_opts["nFlag"]);
    if (slep_opts.find("rFlag")   != slep_opts.end()) opts_rFlag   = std::stoi(slep_opts["rFlag"]);
    if (slep_opts.find("mFlag")   != slep_opts.end()) opts_mFlag   = std::stoi(slep_opts["mFlag"]);
    if (slep_opts.find("lFlag")   != slep_opts.end()) opts_lFlag   = std::stoi(slep_opts["lFlag"]);
    if (slep_opts.find("q")       != slep_opts.end()) opts_q       = std::stod(slep_opts["q"]);
    if (slep_opts.find("tol")     != slep_opts.end()) opts_tol     = std::stod(slep_opts["tol"]);

    if (opts_init != 0 && opts_init != 1 && opts_init != 2) opts_init = 0;
    if (opts_maxIter < 1) opts_maxIter = 10000;
    if (opts_tFlag < 0) opts_tFlag = 0;
    else if (opts_tFlag > 5) opts_tFlag = 5;
    if (opts_nFlag != 1 && opts_nFlag != 2) opts_nFlag = 0;
    if (opts_rFlag != 1) opts_rFlag = 0;
    if (opts_lFlag != 1) opts_lFlag = 0;
    if (opts_mFlag != 1) opts_mFlag = 0;

    if (opts_nFlag != 0) {
        throw std::runtime_error(
            "gl_logisticr (fp32): only nFlag=0 is implemented in MyESL2 (no normalization)");
    }
    if (opts_mFlag != 0 || opts_lFlag != 0) {
        throw std::runtime_error(
            "gl_logisticr (fp32): only mFlag=0 and lFlag=0 are implemented in MyESL2");
    }
    if (opts_q < 1) {
        throw std::runtime_error("q should be larger than 1");
    }

    int opts_rStartNum = opts_maxIter + 1;  // effectively disables restart

    std::vector<double> opts_sWeight;
    if (slep_opts.find("sWeight") != slep_opts.end()) {
        std::ifstream sWeightFile(slep_opts["sWeight"]);
        std::string line;
        if (sWeightFile.is_open()) {
            while (std::getline(sWeightFile, line)) {
                trim(line);
                if (!line.empty()) opts_sWeight.push_back(std::stod(line));
            }
        }
    }

    const size_t m = A.n_rows;
    const size_t n = A.n_cols;

    const arma::mat& ind_mat = weights;
    if (ind_mat.n_cols < 2) {
        throw std::invalid_argument(
            "gl_logisticr (fp32): weights (alg_table) must have at least 2 columns "
            "(start, end in 1-based inclusive coords)");
    }

    // Build 0-based group boundary array of length k+1 (same layout as FP64).
    const int k = static_cast<int>(ind_mat.n_rows);
    std::vector<int> ind_starts(k + 1);
    ind_starts[0] = 0;
    for (int i = 0; i < k; i++) {
        ind_starts[i + 1] = static_cast<int>(ind_mat(i, 1));
    }
    if (ind_starts[k] != static_cast<int>(n)) {
        throw std::runtime_error(
            "Check opts.ind: final group end (" + std::to_string(ind_starts[k]) +
            ") must equal n (" + std::to_string(n) + ")");
    }

    // Per-group weights default to 1.0 (MyESL's wrapper doesn't populate gWeight).
    std::vector<double> gWeight(k, 1.0);

    // ── Raw-pointer matvec helpers via cblas_sgemv ─────────────────────────
    const float* A_ptr = A.memptr();
    const int M_int = static_cast<int>(m);
    const int N_int = static_cast<int>(n);

    auto matvec = [&](const float* x_in, float* out) {
        cblas_sgemv(GL32_CBLAS_COL_MAJOR, GL32_CBLAS_NO_TRANS,
                    M_int, N_int, 1.0f, A_ptr, M_int,
                    x_in, 1, 0.0f, out, 1);
    };

    auto matvec_t = [&](const float* b_in, float* out) {
        cblas_sgemv(GL32_CBLAS_COL_MAJOR, GL32_CBLAS_TRANS,
                    M_int, N_int, 1.0f, A_ptr, M_int,
                    b_in, 1, 0.0f, out, 1);
    };

    // ── Float reduction helpers with double accumulators ───────────────────
    // Summing floats in a double accumulator keeps FP32 round-off from
    // compounding across hundreds of line-search evaluations. This is the
    // same pattern sg_lasso_fp32.cpp:387-394 uses for fun_s/fun_x.
    auto dotf = [](const float* a, const float* b, size_t len) -> double {
        double sum = 0;
        for (size_t i = 0; i < len; i++) sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        return sum;
    };

    auto sumf = [](const float* a, size_t len) -> double {
        double sum = 0;
        for (size_t i = 0; i < len; i++) sum += static_cast<double>(a[i]);
        return sum;
    };

    auto norm2f = [](const float* a, size_t len) -> double {
        double acc = 0;
        for (size_t i = 0; i < len; i++) {
            double ai = static_cast<double>(a[i]);
            acc += ai * ai;
        }
        return std::sqrt(acc);
    };

    // Convert responses to flat float array.
    std::vector<float> y(m);
    for (size_t i = 0; i < m; i++) y[i] = responses(i);

    double z_input = this->lambda[0];
    if (z_input < 0) {
        throw std::invalid_argument("\n z should be nonnegative!\n");
    }

    // Sample weights (float). Match ai_gl_logr.hpp:108-126 semantics.
    std::vector<float> weight(m);
    if (opts_sWeight.size() == 2) {
        if (opts_sWeight[0] <= 0 || opts_sWeight[1] <= 0) {
            throw std::runtime_error(
                "Check opts.sWeight, which contains two positive values");
        }
        size_t n_pos = 0;
        for (size_t i = 0; i < m; i++) if (y[i] == 1.0f) n_pos++;
        size_t n_neg = m - n_pos;
        double m1_sw = static_cast<double>(n_pos) * opts_sWeight[0];
        double m2_sw = static_cast<double>(n_neg) * opts_sWeight[1];
        double denom = m1_sw + m2_sw;
        for (size_t i = 0; i < m; i++) {
            weight[i] = (y[i] == 1.0f) ? static_cast<float>(opts_sWeight[0] / denom)
                                       : static_cast<float>(opts_sWeight[1] / denom);
        }
    } else {
        float inv_m = 1.0f / static_cast<float>(m);
        for (size_t i = 0; i < m; i++) weight[i] = inv_m;
    }

    // m1 = sum(weight[positive]); m2 = 1 - m1 (in double precision).
    double m1 = 0;
    for (size_t i = 0; i < m; i++) if (y[i] == 1.0f) m1 += static_cast<double>(weight[i]);
    double m2 = 1.0 - m1;

    // ── Regularization parameter ─────────────────────────────────────────
    double lambda;
    std::vector<float> b_vec(m);
    if (opts_rFlag == 0) {
        lambda = z_input;
    } else {
        if (z_input > 1) {
            throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
        }
        // b = weight * (+m2 for positive, -m1 for negative) — computed in double
        // then cast to float to keep the FP32 vector precise at this boundary.
        for (size_t i = 0; i < m; i++) {
            double scale = (y[i] == 1.0f) ? m2 : -m1;
            b_vec[i] = static_cast<float>(scale * static_cast<double>(weight[i]));
        }

        std::vector<float> ATb(n);
        matvec_t(b_vec.data(), ATb.data());

        double q_bar;
        if (opts_q == 1.0) {
            q_bar = std::numeric_limits<double>::infinity();
        } else if (opts_q >= 1e6) {
            q_bar = 1.0;
        } else {
            q_bar = opts_q / (opts_q - 1.0);
        }

        double lambda_max = 0.0;
        for (int i = 0; i < k; i++) {
            int start = ind_starts[i];
            int end = ind_starts[i + 1];
            int glen = end - start;
            double norm_g = 0;
            if (q_bar == 2.0) {
                norm_g = norm2f(ATb.data() + start, glen);
            } else if (std::isinf(q_bar)) {
                for (int j = start; j < end; j++) {
                    double av = std::abs(static_cast<double>(ATb[j]));
                    if (av > norm_g) norm_g = av;
                }
            } else if (q_bar == 1.0) {
                for (int j = start; j < end; j++) norm_g += std::abs(static_cast<double>(ATb[j]));
            } else {
                double acc = 0;
                for (int j = start; j < end; j++) {
                    acc += std::pow(std::abs(static_cast<double>(ATb[j])), q_bar);
                }
                norm_g = std::pow(acc, 1.0 / q_bar);
            }
            norm_g /= gWeight[i];
            if (norm_g > lambda_max) lambda_max = norm_g;
        }
        lambda = z_input * lambda_max;
    }

    // ── Starting point ───────────────────────────────────────────────────
    std::vector<float> x(n, 0.0f);
    double c = std::log(m1 / m2);
    std::vector<float> Ax(m, 0.0f);

    int bFlag = 0;
    double L = 1.0 / static_cast<double>(m);

    std::vector<float> weighty(m);
    for (size_t i = 0; i < m; i++) weighty[i] = weight[i] * y[i];

    std::vector<float> xp(n, 0.0f);
    std::vector<float> Axp(m, 0.0f);
    std::vector<float> xxp(n, 0.0f);
    double cp = c, ccp = 0.0;
    double alphap = 0.0, alpha = 1.0;

    std::vector<float> s(n), v(n), As(m), g(n), aa(m), bb(m), prob(m);
    std::vector<double> rho_per_group(k);
    std::vector<double> ValueL(opts_maxIter);
    std::vector<double> funVal(opts_maxIter);

    double beta = 0, sc = 0, gc = 0, fun_s = 0, fun_x = 0;
    double l_sum = 0, r_sum = 0, reg_norm = 0;

    const double epp_q = (opts_q < 1e6) ? opts_q : 1e6;

    // ── Main accelerated proximal-gradient loop ──────────────────────────
    for (int iterStep = 0; iterStep < opts_maxIter; iterStep++) {
        beta = (alphap - 1.0) / alpha;
        float beta_f = static_cast<float>(beta);

        // s = x + beta * xxp
        for (size_t i = 0; i < n; i++) s[i] = x[i] + xxp[i] * beta_f;
        sc = c + beta * ccp;

        // As = Ax + beta * (Ax - Axp)
        for (size_t i = 0; i < m; i++) As[i] = Ax[i] + (Ax[i] - Axp[i]) * beta_f;

        // aa = -y * (As + sc)
        float sc_f = static_cast<float>(sc);
        for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (As[i] + sc_f);

        // bb = max(aa, 0)
        for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0f);

        // fun_s = weight^T * (log(exp(-bb) + exp(aa - bb)) + bb)
        // Computed in double to preserve precision across many iterations.
        {
            double acc = 0;
            for (size_t i = 0; i < m; i++) {
                double bbd = bb[i];
                double aad = aa[i];
                double val = std::log(std::exp(-bbd) + std::exp(aad - bbd)) + bbd;
                acc += static_cast<double>(weight[i]) * val;
            }
            fun_s = acc;
        }

        // prob = 1 / (1 + exp(aa))
        for (size_t i = 0; i < m; i++) prob[i] = 1.0f / (1.0f + std::exp(aa[i]));

        // b = -weighty * (1 - prob)
        for (size_t i = 0; i < m; i++) b_vec[i] = -weighty[i] * (1.0f - prob[i]);

        // gc = sum(b)
        gc = sumf(b_vec.data(), m);

        // g = A^T * b
        matvec_t(b_vec.data(), g.data());

        // Save xp, Axp, cp before the line search.
        std::memcpy(xp.data(), x.data(), n * sizeof(float));
        std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
        cp = c;

        // ── Line search ────────────────────────────────────────────────
        while (true) {
            double invL = 1.0 / L;
            float invL_f = static_cast<float>(invL);
            // v = s - g/L
            for (size_t i = 0; i < n; i++) v[i] = s[i] - g[i] * invL_f;
            c = sc - gc / L;

            // Proximal step: per-group L1/Lq projection (float overload
            // internally upcasts to double for epp() — see eppVector.hpp:56).
            for (int i = 0; i < k; i++) rho_per_group[i] = (lambda / L) * gWeight[i];
            eppVector_inplace(x.data(), v.data(), ind_starts.data(),
                              k, static_cast<int>(n), rho_per_group.data(), epp_q);

            // v = x - s (reuse v as the step direction for r_sum/l_sum)
            for (size_t i = 0; i < n; i++) v[i] = x[i] - s[i];

            // Ax = A * x
            matvec(x.data(), Ax.data());

            // aa = -y * (Ax + c)
            float c_f = static_cast<float>(c);
            for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (Ax[i] + c_f);

            // bb = max(aa, 0)
            for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0f);

            // fun_x = weight^T * (log(exp(-bb) + exp(aa - bb)) + bb)
            {
                double acc = 0;
                for (size_t i = 0; i < m; i++) {
                    double bbd = bb[i];
                    double aad = aa[i];
                    double val = std::log(std::exp(-bbd) + std::exp(aad - bbd)) + bbd;
                    acc += static_cast<double>(weight[i]) * val;
                }
                fun_x = acc;
            }

            // r_sum = (v^T v + (c - sc)^2) / 2
            r_sum = (dotf(v.data(), v.data(), n) + std::pow(c - sc, 2)) / 2.0;
            // l_sum = fun_x - fun_s - v^T g - (c - sc) * gc
            l_sum = fun_x - fun_s - dotf(v.data(), g.data(), n) - (c - sc) * gc;

            if (r_sum <= 1e-20) {
                bFlag = 1;
                break;
            }

            // Vanilla break condition (matches ai_gl_logr.cpp:467).
            if (l_sum <= r_sum * L) {
                break;
            } else {
                L = std::max(2.0 * L, l_sum / r_sum);
            }
        }

        alphap = alpha;
        alpha = (1.0 + std::sqrt(4.0 * alpha * alpha + 1.0)) / 2.0;

        ValueL[iterStep] = L;

        // xxp = x - xp;  ccp = c - cp
        for (size_t i = 0; i < n; i++) xxp[i] = x[i] - xp[i];
        ccp = c - cp;

        // reg_norm = sum_i gWeight[i] * ||x_i||_q
        reg_norm = 0;
        for (int i = 0; i < k; i++) {
            int start = ind_starts[i];
            int end = ind_starts[i + 1];
            int glen = end - start;
            double norm_g = 0;
            if (opts_q == 2.0) {
                norm_g = norm2f(x.data() + start, glen);
            } else if (opts_q == 1.0) {
                for (int j = start; j < end; j++) norm_g += std::abs(static_cast<double>(x[j]));
            } else if (opts_q >= 1e6) {
                for (int j = start; j < end; j++) {
                    double av = std::abs(static_cast<double>(x[j]));
                    if (av > norm_g) norm_g = av;
                }
            } else {
                double acc = 0;
                for (int j = start; j < end; j++) {
                    acc += std::pow(std::abs(static_cast<double>(x[j])), opts_q);
                }
                norm_g = std::pow(acc, 1.0 / opts_q);
            }
            reg_norm += gWeight[i] * norm_g;
        }
        funVal[iterStep] = fun_x + lambda * reg_norm;

        if (bFlag) break;

        // Termination flags — match ai_gl_logr.hpp:339-382 semantics, 0-indexed.
        switch (opts_tFlag) {
            case 0:
                if (iterStep >= 1) {
                    if (std::abs(funVal[iterStep] - funVal[iterStep - 1]) <= opts_tol) {
                        bFlag = 1;
                    }
                }
                break;
            case 1:
                if (iterStep >= 1) {
                    if (std::abs(funVal[iterStep] - funVal[iterStep - 1]) <=
                        opts_tol * funVal[iterStep - 1]) {
                        bFlag = 1;
                    }
                }
                break;
            case 2:
                if (funVal[iterStep] <= opts_tol) bFlag = 1;
                break;
            case 3: {
                double norm_xxp = norm2f(xxp.data(), n);
                if (norm_xxp <= opts_tol) bFlag = 1;
                break;
            }
            case 4: {
                double norm_xxp = norm2f(xxp.data(), n);
                double norm_xp  = norm2f(xp.data(), n);
                if (norm_xxp <= opts_tol * std::max(norm_xp, 1.0)) bFlag = 1;
                break;
            }
            case 5:
                if (iterStep >= opts_maxIter - 1) bFlag = 1;
                break;
        }
        if (bFlag) break;

        // Restart (disabled by default: opts_rStartNum = opts_maxIter + 1)
        if ((iterStep + 1) % opts_rStartNum == 0) {
            alphap = 0;  alpha = 1;
            std::memcpy(xp.data(), x.data(), n * sizeof(float));
            std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
            std::memset(xxp.data(), 0, n * sizeof(float));
            L = L / 2;
        }
    }

    // Write results back. Unlike the FP64 variant, the FP32 class stores the
    // actual intercept (we're not matching MyESL's latent RRLogisticR bug
    // since FP32 has no byte-identity target).
    this->intercept_value = c;
    (void)opts_init;  // init!=2 paths aren't exercised (no x0/c0 in the API)
    (void)ValueL;     // populated for parity with the FP64 reference structure

    parameters.set_size(n);
    std::memcpy(parameters.memptr(), x.data(), n * sizeof(float));

    this->nz_gene_count = countNonZeroGenes<float>(parameters, weights);

    static thread_local arma::frowvec x_row_ret;
    x_row_ret = arma::frowvec(x.data(), n);
    return x_row_ret;
}
