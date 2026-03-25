
#include <armadillo>
#include <stdexcept>

class SGLassoFP64
{
 public:

  SGLassoFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const bool intercept = true);

  SGLassoFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const arma::rowvec& xval_idxs,
                   int xval_id,
                   const bool intercept = true);

  SGLassoFP64() : lambda(), intercept(true) { }

  arma::rowvec& Train(const arma::mat& features,
               const arma::rowvec& responses,
               const arma::mat& weights,
               std::map<std::string, std::string> slep_opts,
               const bool intercept = true);

  void writeModelToXMLStream(std::ofstream& XMLFile);
  void writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap);

  const arma::colvec altra(const arma::colvec& v_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const;

  const double treeNorm(const arma::rowvec& x,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const;

  const double computeLambda2Max(const arma::rowvec& x,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const;

  void altra_inplace(double* x, const double* v, int n,
                     const double* ind, int nodes) const;
  double treeNorm_flat(const double* x, int n,
                       const double* ind, int nodes) const;
  double computeLambda2Max_flat(const double* x, int n,
                                const double* ind, int nodes) const;

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

  //! Return the intercept value.
  double InterceptValue() const { return intercept_value; }

  //! Return the Tikhonov regularization parameter for ridge regression.
  double* Lambda() { return lambda; }

  //! Return whether or not an intercept term is used in the model.
  bool Intercept() const { return intercept; }

  int NonZeroGeneCount() { return nz_gene_count; }

 private:
  int nz_gene_count = 0;
  arma::vec parameters;
  double* lambda;
  double lambda1 = lambda[0];
  bool intercept;
  double intercept_value;
};

#include "sg_lasso_helpers.hpp"
