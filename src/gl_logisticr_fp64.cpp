// GLLogisticRFP64 — Phase B implementation.
//
// Constructors/writers adapted from /claude/MyESL/src/gl_logisticr.cpp lines 8–80.
//
// Phase A used glLogisticR() (ai_gl_logr.hpp, faithful Armadillo port) inside
// Train(). Phase B inlines the solver directly into Train() using raw
// `const double*` / `std::vector<double>` loops, mirroring src/sg_lasso_fp64.cpp's
// refactored structure. This cuts the Armadillo allocation overhead from the hot
// loop and sets up the FP32 variant in Phase C.
//
// Structural differences from sg_lasso_fp64.cpp:
//   - Prox operator is per-group L1/Lq (eppVector_inplace) instead of
//     tree-norm (altra_inplace).
//   - Group indices use a k+1 element starts array (ind_starts[0]=0,
//     ind_starts[i]=weights(i-1,1)), not sg_lasso's 3-col ind_flat layout.
//   - Lambda init uses per-group q_bar-norms of A^T*b / gWeight.
//   - No lambda2, no lambda2_max.
//   - Line-search test is the vanilla `l_sum <= r_sum * L` (no slack, no
//     disableEC) matching /claude/MyESL/src/ai_gl_logr.cpp:467.
//   - Intercept is computed but deliberately not stored into intercept_value,
//     matching MyESL's latent RRLogisticR bug for byte-identical XML output
//     (see comment in writeModelToXMLStream).

#include "gl_logisticr_fp64.hpp"
#include "eppVector.hpp"
#include "sg_lasso_helpers.hpp"
#include "cblas_decl.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>


GLLogisticRFP64::GLLogisticRFP64(const arma::mat& features,
                                 const arma::rowvec& responses,
                                 const arma::mat& weights,
                                 double* lambda,
                                 std::map<std::string, std::string> slep_opts,
                                 const bool intercept)
    : lambda(lambda), intercept(intercept)
{
    Train(features, responses, weights, slep_opts, intercept);
}

GLLogisticRFP64::GLLogisticRFP64(const arma::mat& features,
                                 const arma::rowvec& responses,
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

void GLLogisticRFP64::writeModelToXMLStream(std::ofstream& XMLFile)
{
    // Matches MyESL's byte-for-byte XML format (gl_logisticr.cpp:22–46) so
    // outputs are directly diffable against /claude/MyESL/bin/gl_logisticr.
    //
    // NOTE: MyESL's RRLogisticR stores `parameters` as `arma::fvec` (single
    // precision), silently downcasting solver output before XML serialization.
    // We preserve full double precision internally (parameters is arma::vec)
    // but cast to float here so Phase A XML output matches MyESL byte-for-byte.
    // Phase B removes this cast once refactor bit-identity is verified.
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
        streamObj << std::setprecision(17) << std::scientific
                  << static_cast<double>(static_cast<float>(this->parameters(i)));
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

void GLLogisticRFP64::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile,
                                                       std::ifstream& FeatureMap)
{
    // Matches MyESL's format verbatim (gl_logisticr.cpp:48–80). See the
    // float-cast note in writeModelToXMLStream() above — same lossy cast
    // applied here for Phase A byte-identity with MyESL.
    std::string line;
    std::getline(FeatureMap, line);
    for (arma::uword i = 0; i < this->parameters.n_elem; i++) {
        std::getline(FeatureMap, line);
        if (this->parameters(i) == 0.0) {
            continue;
        }
        std::istringstream iss(line);
        std::string feature_label;
        std::getline(iss, feature_label, '\t');
        std::getline(iss, feature_label, '\t');
        std::ostringstream streamObj;
        streamObj << std::setprecision(17) << std::scientific
                  << static_cast<double>(static_cast<float>(this->parameters(i)));
        MappedWeightsFile << feature_label + "\t" + streamObj.str() + "\n";
    }
    MappedWeightsFile << "Intercept\t" + std::to_string(this->intercept_value) + "\n";
    FeatureMap.clear();
    FeatureMap.seekg(0);
}

arma::rowvec GLLogisticRFP64::Train(const arma::mat& A,
                                    const arma::rowvec& responses,
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
    // default_sll_opts for gl_logisticr: maxIter=100, tol=1e-3, tFlag=5,
    // rFlag=1, q=2, init=2. Only nFlag=0 / mFlag=0 / lFlag=0 is implemented.
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

    // Clamp options the same way default_sll_opts() does, so slep_opts.txt
    // files intended for MyESL produce the same effective settings.
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
            "gl_logisticr: only nFlag=0 is implemented in MyESL2 (no normalization)");
    }
    if (opts_mFlag != 0 || opts_lFlag != 0) {
        throw std::runtime_error(
            "gl_logisticr: only mFlag=0 and lFlag=0 are implemented in MyESL2");
    }
    if (opts_q < 1) {
        throw std::runtime_error("q should be larger than 1");
    }

    int opts_rStartNum = opts_maxIter + 1;  // effectively disables restart

    // Sample weight file loader (two lines: positive-class, negative-class weight).
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
            "gl_logisticr: weights (alg_table) must have at least 2 columns "
            "(start, end in 1-based inclusive coords)");
    }

    // Build 0-based group boundary array of length k+1:
    //   ind_starts[0] = 0
    //   ind_starts[i] = weights(i-1, 1)
    // weights.col(1) holds 1-based inclusive end indices in MyESL2's alg_table,
    // which coincides numerically with the 0-based exclusive ends expected by
    // eppVector_inplace (ind_starts[k] must equal n).
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

    // Per-group weights — MyESL's wrapper doesn't set gWeight, so ai_gl_logr.cpp
    // defaults it to ones(). Match that: ignore ind_mat.col(2) and default to 1.
    std::vector<double> gWeight(k, 1.0);

    // ── Raw-pointer matvec helpers (mirror src/sg_lasso_fp64.cpp:227-256) ────
    // A is column-major (Armadillo default): A(i,j) = A_ptr[j*m + i]
    const double* A_ptr = A.memptr();
    const int M_int = static_cast<int>(m);
    const int N_int = static_cast<int>(n);

    // matvec: out = A * x_in  via cblas_dgemv (y = alpha*A*x + beta*y)
    auto matvec = [&](const double* x_in, double* out) {
        cblas_dgemv(MYESL_CBLAS_COL_MAJOR, MYESL_CBLAS_NO_TRANS,
                    M_int, N_int, 1.0, A_ptr, M_int,
                    x_in, 1, 0.0, out, 1);
    };

    // matvec_t: out = A^T * b_in  via cblas_dgemv with CblasTrans
    auto matvec_t = [&](const double* b_in, double* out) {
        cblas_dgemv(MYESL_CBLAS_COL_MAJOR, MYESL_CBLAS_TRANS,
                    M_int, N_int, 1.0, A_ptr, M_int,
                    b_in, 1, 0.0, out, 1);
    };

    // Dot product / sum helpers mirror Armadillo's 2-way split accumulators
    // (arrayops::accumulate, op_dot::direct_dot_arma, op_norm::vec_norm_2_direct_mem
    // in armadillo_bits/). Strict left-to-right accumulation would produce ~1–2
    // ULP drift per reduction, which compounds across 100 iterations and can
    // cross float rounding boundaries in the XML output at high lambda values.
    auto dotf = [](const double* a, const double* b, size_t len) -> double {
        double val1 = 0, val2 = 0;
        size_t i, j;
        for (i = 0, j = 1; j < len; i += 2, j += 2) {
            val1 += a[i] * b[i];
            val2 += a[j] * b[j];
        }
        if (i < len) val1 += a[i] * b[i];
        return val1 + val2;
    };

    auto sumf = [](const double* a, size_t len) -> double {
        double acc1 = 0, acc2 = 0;
        size_t j;
        for (j = 1; j < len; j += 2) {
            acc1 += a[j - 1];
            acc2 += a[j];
        }
        if ((j - 1) < len) acc1 += a[j - 1];
        return acc1 + acc2;
    };

    // ||v||_2 via 2-way split sum of squares (matches op_norm::vec_norm_2_direct_mem).
    auto norm2f = [](const double* a, size_t len) -> double {
        double acc1 = 0, acc2 = 0;
        size_t j;
        for (j = 1; j < len; j += 2) {
            double ti = a[j - 1];
            double tj = a[j];
            acc1 += ti * ti;
            acc2 += tj * tj;
        }
        if ((j - 1) < len) {
            double ti = a[j - 1];
            acc1 += ti * ti;
        }
        return std::sqrt(acc1 + acc2);
    };

    // Convert responses to flat double array (y)
    std::vector<double> y(m);
    for (size_t i = 0; i < m; i++) y[i] = responses(i);

    double z_input = this->lambda[0];
    if (z_input < 0) {
        throw std::invalid_argument("\n z should be nonnegative!\n");
    }

    // Sample weights (match ai_gl_logr.hpp:108-126). We compute the per-sample
    // `weight` vector exactly as MyESL does: either user-specified class
    // weights or a uniform 1/m.
    std::vector<double> weight(m);
    if (opts_sWeight.size() == 2) {
        if (opts_sWeight[0] <= 0 || opts_sWeight[1] <= 0) {
            throw std::runtime_error(
                "Check opts.sWeight, which contains two positive values");
        }
        size_t n_pos = 0;
        for (size_t i = 0; i < m; i++) if (y[i] == 1.0) n_pos++;
        size_t n_neg = m - n_pos;
        double m1_sw = static_cast<double>(n_pos) * opts_sWeight[0];
        double m2_sw = static_cast<double>(n_neg) * opts_sWeight[1];
        double denom = m1_sw + m2_sw;
        for (size_t i = 0; i < m; i++) {
            weight[i] = (y[i] == 1.0) ? (opts_sWeight[0] / denom)
                                      : (opts_sWeight[1] / denom);
        }
    } else {
        for (size_t i = 0; i < m; i++) weight[i] = 1.0 / m;
    }

    // m1 = sum(weight[positive]); m2 = 1 - m1
    //
    // Match Phase A's `arma::sum(weight.elem(p_flag))` pattern: the .elem(p_flag)
    // materializes a new vec of length n_pos with positive-class weights (in
    // ascending index order), then arma::sum calls arrayops::accumulate which
    // uses a 2-way split accumulator. Mirror that exactly with sumf.
    std::vector<double> pos_weights;
    pos_weights.reserve(m);
    for (size_t i = 0; i < m; i++) if (y[i] == 1.0) pos_weights.push_back(weight[i]);
    double m1 = sumf(pos_weights.data(), pos_weights.size());
    double m2 = 1.0 - m1;

    // ── Regularization parameter ───────────────────────────────────────────
    // When rFlag==1 (default), `z_input` is a ratio in [0,1] and lambda =
    // z_input * lambda_max, where lambda_max = max_i ||ATb_i||_q_bar / gWeight[i].
    double lambda;
    std::vector<double> b_vec(m);
    if (opts_rFlag == 0) {
        lambda = z_input;
    } else {
        if (z_input > 1) {
            throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
        }
        // b = weight * (+m2 for positive, -m1 for negative)
        for (size_t i = 0; i < m; i++) {
            b_vec[i] = (y[i] == 1.0 ? m2 : -m1) * weight[i];
        }

        std::vector<double> ATb(n);
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
                // Match op_norm::vec_norm_2_direct_mem's 2-way split.
                norm_g = norm2f(ATb.data() + start, glen);
            } else if (std::isinf(q_bar)) {
                for (int j = start; j < end; j++) {
                    double av = std::abs(ATb[j]);
                    if (av > norm_g) norm_g = av;
                }
            } else if (q_bar == 1.0) {
                for (int j = start; j < end; j++) norm_g += std::abs(ATb[j]);
            } else {
                double acc = 0;
                for (int j = start; j < end; j++) {
                    acc += std::pow(std::abs(ATb[j]), q_bar);
                }
                norm_g = std::pow(acc, 1.0 / q_bar);
            }
            norm_g /= gWeight[i];
            if (norm_g > lambda_max) lambda_max = norm_g;
        }
        lambda = z_input * lambda_max;
    }

    // ── Starting point ─────────────────────────────────────────────────────
    // init==2 → x=0, c=log(m1/m2). Other init paths currently default to the
    // same since we don't accept an externally-supplied x0.
    std::vector<double> x(n, 0.0);
    double c = std::log(m1 / m2);
    std::vector<double> Ax(m, 0.0);  // A * x = 0 initially

    int bFlag = 0;
    double L = 1.0 / static_cast<double>(m);

    // weighty = weight * y
    std::vector<double> weighty(m);
    for (size_t i = 0; i < m; i++) weighty[i] = weight[i] * y[i];

    std::vector<double> xp(n, 0.0);
    std::vector<double> Axp(m, 0.0);
    std::vector<double> xxp(n, 0.0);
    double cp = c, ccp = 0.0;
    double alphap = 0.0, alpha = 1.0;

    // Working vectors (reused across iterations to avoid reallocations)
    std::vector<double> s(n), v(n), As(m), g(n), aa(m), bb(m), prob(m);
    std::vector<double> rho_per_group(k);
    std::vector<double> ValueL(opts_maxIter);
    std::vector<double> funVal(opts_maxIter);
    // Temporaries for matching Armadillo's 2-way split reductions:
    //   transformed[i] = log(exp(-bb[i]) + exp(aa[i]-bb[i])) + bb[i]
    //   norm_x_k[i]    = ||x_group_i||_q
    // We materialize these first, then feed into dotf / sumf / etc.
    std::vector<double> transformed(m);
    std::vector<double> norm_x_k(k);

    double beta = 0, sc = 0, gc = 0, fun_s = 0, fun_x = 0;
    double l_sum = 0, r_sum = 0, reg_norm = 0;

    // Effective q passed to epp: MyESL clamps q>=1e6 to 1e6 in the eppVector call.
    const double epp_q = (opts_q < 1e6) ? opts_q : 1e6;

    // ── Main accelerated proximal-gradient loop ────────────────────────────
    for (int iterStep = 0; iterStep < opts_maxIter; iterStep++) {
        beta = (alphap - 1.0) / alpha;

        // s = x + beta * xxp
        for (size_t i = 0; i < n; i++) s[i] = x[i] + xxp[i] * beta;
        sc = c + beta * ccp;

        // As = Ax + beta * (Ax - Axp)
        for (size_t i = 0; i < m; i++) As[i] = Ax[i] + (Ax[i] - Axp[i]) * beta;

        // aa = -y * (As + sc)
        for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (As[i] + sc);

        // bb = max(aa, 0)
        for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0);

        // fun_s = weight^T * (log(exp(-bb) + exp(aa - bb)) + bb)
        //
        // Phase A: arma::dot(weight, arma::log(arma::exp(-bb) + arma::exp(aa-bb)) + bb)
        // lands in op_dot::apply_proxy which uses a 2-way split over the eGlue
        // expression. We materialize `transformed` first, then call dotf (which
        // also uses a 2-way split) — same IEEE result as Armadillo's lazy eval.
        for (size_t i = 0; i < m; i++) {
            transformed[i] = std::log(std::exp(-bb[i]) + std::exp(aa[i] - bb[i])) + bb[i];
        }
        fun_s = dotf(weight.data(), transformed.data(), m);

        // prob = 1 / (1 + exp(aa))
        for (size_t i = 0; i < m; i++) prob[i] = 1.0 / (1.0 + std::exp(aa[i]));

        // b = -weighty * (1 - prob)
        for (size_t i = 0; i < m; i++) b_vec[i] = -weighty[i] * (1.0 - prob[i]);

        // gc = sum(b)  (Phase A: arma::sum(b) → arrayops::accumulate, 2-way split)
        gc = sumf(b_vec.data(), m);

        // g = A^T * b
        matvec_t(b_vec.data(), g.data());

        // Save xp, Axp, cp before the line search
        std::memcpy(xp.data(), x.data(), n * sizeof(double));
        std::memcpy(Axp.data(), Ax.data(), m * sizeof(double));
        cp = c;

        // ── Line search ────────────────────────────────────────────────────
        while (true) {
            // v = s - g/L;   c = sc - gc/L
            double invL = 1.0 / L;
            for (size_t i = 0; i < n; i++) v[i] = s[i] - g[i] * invL;
            c = sc - gc / L;

            // Proximal step: per-group L1/Lq projection
            //   rho_i = lambda / L * gWeight[i]
            for (int i = 0; i < k; i++) rho_per_group[i] = (lambda / L) * gWeight[i];
            eppVector_inplace(x.data(), v.data(), ind_starts.data(),
                              k, static_cast<int>(n), rho_per_group.data(), epp_q);

            // v = x - s (reuse v as the step direction for r_sum/l_sum)
            for (size_t i = 0; i < n; i++) v[i] = x[i] - s[i];

            // Ax = A * x
            matvec(x.data(), Ax.data());

            // aa = -y * (Ax + c)
            for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (Ax[i] + c);

            // bb = max(aa, 0)
            for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0);

            // fun_x = weight^T * (log(exp(-bb) + exp(aa - bb)) + bb)
            // Same materialize-then-2-way-split pattern as fun_s above.
            for (size_t i = 0; i < m; i++) {
                transformed[i] = std::log(std::exp(-bb[i]) + std::exp(aa[i] - bb[i])) + bb[i];
            }
            fun_x = dotf(weight.data(), transformed.data(), m);

            // r_sum = (v^T v + (c - sc)^2) / 2
            r_sum = (dotf(v.data(), v.data(), n) + std::pow(c - sc, 2)) / 2.0;
            // l_sum = fun_x - fun_s - v^T g - (c - sc) * gc
            l_sum = fun_x - fun_s - dotf(v.data(), g.data(), n) - (c - sc) * gc;

            if (r_sum <= 1e-20) {
                bFlag = 1;
                break;
            }

            // Vanilla break condition (matches /claude/MyESL/src/ai_gl_logr.cpp:467).
            // gl_logisticr intentionally omits sg_lasso's disableEC / slack test.
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
        //
        // Phase A: first materializes `norm_x_k(k)` via arma::norm(x_group, q),
        // then computes arma::dot(norm_x_k, gWeight). Both materialized vecs,
        // so direct_dot_arma (2-way split) is used. Mirror exactly.
        for (int i = 0; i < k; i++) {
            int start = ind_starts[i];
            int end = ind_starts[i + 1];
            int glen = end - start;
            double norm_g = 0;
            if (opts_q == 2.0) {
                // Match op_norm::vec_norm_2_direct_mem's 2-way split.
                norm_g = norm2f(x.data() + start, glen);
            } else if (opts_q == 1.0) {
                for (int j = start; j < end; j++) norm_g += std::abs(x[j]);
            } else if (opts_q >= 1e6) {
                for (int j = start; j < end; j++) {
                    double av = std::abs(x[j]);
                    if (av > norm_g) norm_g = av;
                }
            } else {
                double acc = 0;
                for (int j = start; j < end; j++) acc += std::pow(std::abs(x[j]), opts_q);
                norm_g = std::pow(acc, 1.0 / opts_q);
            }
            norm_x_k[i] = norm_g;
        }
        reg_norm = dotf(norm_x_k.data(), gWeight.data(), static_cast<size_t>(k));
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
                // Match Phase A: arma::norm(xxp) → vec_norm_2_direct_mem (2-way split).
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
            std::memcpy(xp.data(), x.data(), n * sizeof(double));
            std::memcpy(Axp.data(), Ax.data(), m * sizeof(double));
            std::memset(xxp.data(), 0, n * sizeof(double));
            L = L / 2;
        }
    }

    // Write results back to Armadillo members.
    //
    // NOTE: MyESL's RRLogisticR never stores the solver's intercept into
    // `intercept_value`, so its XML output always prints `0.000000`. We
    // deliberately match that here for byte-identical comparison against
    // /claude/MyESL/bin/gl_logisticr. The actual intercept lives in local `c`.
    // this->intercept_value = c;  // (not set — matches MyESL's latent bug)
    (void)c;
    (void)opts_init;  // init!=2 paths aren't exercised (we don't accept x0/c0)
    (void)ValueL;     // populated for parity with the Phase A reference

    parameters.set_size(n);
    std::memcpy(parameters.memptr(), x.data(), n * sizeof(double));

    this->nz_gene_count = countNonZeroGenes<double>(parameters, weights);

    return arma::rowvec(x.data(), n);
}
