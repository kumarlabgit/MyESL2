#pragma once
// Minimal CBLAS declarations for direct OpenBLAS calls.
// OpenBLAS is bundled at deps/{platform}/libopenblas.{so,dylib,lib}
// and already linked — no additional build changes needed.
//
// We declare only the symbols we use rather than including <cblas.h>,
// so there is no dependency on the OpenBLAS header-install path.

extern "C" {
    void cblas_dgemv(int order, int trans, int M, int N, double alpha,
                     const double* A, int lda, const double* X, int incX,
                     double beta, double* Y, int incY);
    void cblas_sgemv(int order, int trans, int M, int N, float alpha,
                     const float* A, int lda, const float* X, int incX,
                     float beta, float* Y, int incY);

    double cblas_ddot(int N, const double* X, int incX, const double* Y, int incY);
    float  cblas_sdot(int N, const float* X, int incX, const float* Y, int incY);
}

// CBLAS enum values (same across OpenBLAS, ATLAS, MKL).
static constexpr int MYESL_CBLAS_COL_MAJOR = 102;
static constexpr int MYESL_CBLAS_NO_TRANS  = 111;
static constexpr int MYESL_CBLAS_TRANS     = 112;
