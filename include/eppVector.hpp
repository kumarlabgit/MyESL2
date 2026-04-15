#pragma once

// Port of /claude/MyESL/src/eppVector.c — per-group wrapper that applies
// epp() to each group slice of v. Used by gl_logisticr's proximal step.

#include <armadillo>
#include <cstdlib>
#include "epph.hpp"

// Phase A arma-returning version (matches MyESL source semantics).
static inline arma::vec eppVector(arma::vec& v_in, const arma::uvec& ind, int k, int n,
                                  arma::vec rho, double p) {
    double *x;
    x = (double*) std::malloc(n*sizeof(double));
    double* v = v_in.memptr();
    int i, *iter_step;
    double c0, c;
    double *px, *pv;

    iter_step=(int *)std::malloc(sizeof(int)*2);

    c0=0;
    for(i=0; i<k; i++){

        px=x+(int)ind[i];
        pv=v+(int)ind[i];

        gl_logr_epph::epp(px, &c, iter_step, pv, (int)(ind[i+1]-ind[i]), rho[i], p, c0);
    }
    arma::vec x_vec(&x[0], n);
    std::free(x);
    std::free(iter_step);
    return x_vec;
}

// Phase B raw-pointer inplace overload — writes directly into caller's x buffer.
// ind has k+1 elements (group start/end boundaries).
static inline void eppVector_inplace(double* x, const double* v, const int* ind,
                                     int k, int /*n*/, const double* rho, double p) {
    int iter_step[2] = {0, 0};
    double c = 0;
    double c0 = 0;
    // epp takes non-const double* for v; the MyESL source treats v as read-only
    // in practice (epp1/epp2 don't mutate, eppO temporarily flips signs and restores).
    // We cast away const here to match the C signature.
    double* v_nc = const_cast<double*>(v);
    for (int i = 0; i < k; i++) {
        int start = ind[i];
        int len = ind[i+1] - start;
        gl_logr_epph::epp(x + start, &c, iter_step, v_nc + start, len, rho[i], p, c0);
    }
}

// Phase C float overload — round-trips through double since epp() is double-only.
// The prox operator is called once per iteration so the allocation overhead is minimal.
static inline void eppVector_inplace(float* x, const float* v, const int* ind,
                                     int k, int n, const double* rho, double p) {
    std::vector<double> vd(n), xd(n);
    for (int j = 0; j < n; j++) vd[j] = static_cast<double>(v[j]);
    int iter_step[2] = {0, 0};
    double c = 0;
    double c0 = 0;
    for (int i = 0; i < k; i++) {
        int start = ind[i];
        int len = ind[i+1] - start;
        gl_logr_epph::epp(xd.data() + start, &c, iter_step, vd.data() + start, len, rho[i], p, c0);
    }
    for (int j = 0; j < n; j++) x[j] = static_cast<float>(xd[j]);
}
