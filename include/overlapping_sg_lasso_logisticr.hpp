
#include <armadillo>
#include <stdexcept>



class OLSGLassoLogisticR
{
 public:

  OLSGLassoLogisticR(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   const arma::rowvec& field,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const bool intercept = true);

  OLSGLassoLogisticR(const arma::mat& features,
                   const arma::rowvec& responses,
                   const arma::mat& weights,
                   const arma::rowvec& field,
                   double* lambda,
                   std::map<std::string, std::string> slep_opts,
                   const arma::rowvec& xval_idxs,
                   int xval_id,
                   const bool intercept = true);


  OLSGLassoLogisticR() : lambda(), intercept(true) { }

  arma::rowvec& Train(const arma::mat& features,
               const arma::rowvec& responses,
               const arma::mat& weights,
               std::map<std::string, std::string> slep_opts,
               const arma::rowvec& field,
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
  //! Modify the Tikhonov regularization parameter for ridge regression.
  //double& Lambda() { return lambda; }

  //! Return whether or not an intercept term is used in the model.
  bool Intercept() const { return intercept; }

  int NonZeroGeneCount() { return nz_gene_count; }

  /**
   * Serialize the model.
   */

  //template<typename Archive>
  //void serialize(Archive& ar, const unsigned int /* version */)
  /*
  {
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(lambda1);
    ar & BOOST_SERIALIZATION_NVP(intercept_value);
  }
  */


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


int countNonZeroGenes(const arma::vec& arr, const arma::mat& ranges, const arma::rowvec& field);