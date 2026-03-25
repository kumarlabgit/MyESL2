
#include <armadillo>
#include <stdexcept>



class OLSGLassoLogisticRFP64
{
 public:

  OLSGLassoLogisticRFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   const arma::rowvec& field,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const bool intercept = true);

  OLSGLassoLogisticRFP64(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   const arma::rowvec& field,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const arma::rowvec& xval_idxs,
                   int xval_id,
                   const bool intercept = true);


  OLSGLassoLogisticRFP64() : lambda(), intercept(true) { }

  arma::rowvec& Train(const arma::mat& features,
               const arma::rowvec& responses,
               const arma::mat& weights,
               std::map<std::string, std::string> slep_opts,
               const arma::rowvec& field,
               const bool intercept = true);

  void writeModelToXMLStream(std::ofstream& XMLFile);
  void writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap);

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


#include "sg_lasso_helpers.hpp"
