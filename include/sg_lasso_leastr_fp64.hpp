
//#ifndef MLPACK_METHODS_SG_LASSO_LEASTR_SG_LASSO_LEASTR_HPP
//#define MLPACK_METHODS_SG_LASSO_LEASTR_SG_LASSO_LEASTR_HPP

//#include <mlpack/prereqs.hpp>
#include <armadillo>
#include <stdexcept>

//namespace mlpack {
//namespace regression /** Regression methods. */ {

/**
 * A simple linear regression algorithm using ordinary least squares.
 * Optionally, this class can perform ridge regression, if the lambda parameter
 * is set to a number greater than zero.
 */
class SGLassoLeastRFP64
{
 public:

  SGLassoLeastRFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const bool intercept = true);

  SGLassoLeastRFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const arma::rowvec& xval_idxs,
                   int xval_id,
                   const bool intercept = true);

  /**
   * Empty constructor.  This gives a non-working model, so make sure Train() is
   * called (or make sure the model parameters are set) before calling
   * Predict()!
   */
  SGLassoLeastRFP64() : lambda(), intercept(true) { }

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

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

  //! Return the Tikhonov regularization parameter for ridge regression.
  double* Lambda() { return lambda; }

  //! Return whether or not an intercept term is used in the model.
  bool Intercept() const { return intercept; }

  int NonZeroGeneCount() { return nz_gene_count; }

 private:
  //Non-zero gene count
  int nz_gene_count = 0;
  /**
   * The calculated B.
   * Initialized and filled by constructor to hold the least squares solution.
   */
  arma::vec parameters;

  /**
   * The Tikhonov regularization parameter for ridge regression (0 for linear
   * regression).
   */
  double* lambda;
  double lambda1 = lambda[0];

  //! Indicates whether first parameter is intercept.
  bool intercept;
  double intercept_value;
};

//} // namespace regression
//} // namespace mlpack

//#endif // MLPACK_METHODS_SG_LASSO_LEASTR_HPP

#include "sg_lasso_helpers.hpp"
