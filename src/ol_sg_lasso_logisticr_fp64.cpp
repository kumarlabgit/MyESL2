
#include "ol_sg_lasso_logisticr_fp64.hpp"
#include "sg_lasso_helpers.hpp"
#include "cblas_decl.hpp"
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>


OLSGLassoLogisticRvFP64::OLSGLassoLogisticRvFP64(const arma::mat& features,
                                   const arma::rowvec& responses,
                                   const arma::mat& weights,
                                   const arma::rowvec& field,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(features, responses, weights, field, slep_opts, intercept);
}


OLSGLassoLogisticRvFP64::OLSGLassoLogisticRvFP64(const arma::mat& features,
                                   const arma::rowvec& responses,
                                   const arma::mat& weights,
                                   const arma::rowvec& field,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const arma::rowvec& xval_idxs,
                                   int xval_id,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  arma::uvec indices = arma::find(xval_idxs != xval_id);
  Train(features.rows(indices), responses.elem(indices).t(), weights, field, slep_opts, intercept);
}


void OLSGLassoLogisticRvFP64::writeModelToXMLStream(std::ofstream& XMLFile)
{
  int i_level = 0;
  XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
  for(int i=0; i<this->parameters.n_elem; i++)
  {
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

void OLSGLassoLogisticRvFP64::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap)
{
  std::string line;
  std::getline(FeatureMap, line);
  for(int i=0; i<this->parameters.n_elem; i++)
  {
    std::getline(FeatureMap, line);
    if (this->parameters(i) == 0.0)
    {
	  continue;
    }
    std::istringstream iss(line);
    std::string feature_label;
    std::getline(iss, feature_label, '\t');
    std::getline(iss, feature_label, '\t');
    std::ostringstream streamObj;
    streamObj << std::setprecision(17) << std::scientific << this->parameters(i);
    MappedWeightsFile << feature_label + "	" + streamObj.str() + "\n";
  }
  MappedWeightsFile << "Intercept	" + std::to_string(this->intercept_value) + "\n";
  FeatureMap.clear();
  FeatureMap.seekg(0);
}


arma::rowvec& OLSGLassoLogisticRvFP64::Train(const arma::mat& A,
                               const arma::rowvec& responses,
                               const arma::mat& weights,
                               const arma::rowvec& field,
                               std::map<std::string, std::string> slep_opts,
                               const bool intercept)
{
  this->intercept = intercept;
  auto train_start = std::chrono::steady_clock::now();

  auto trim = [](std::string& s)
  {
     size_t p = s.find_first_not_of(" \t\r\n");
     s.erase(0, p);

     p = s.find_last_not_of(" \t\r\n");
     if (std::string::npos != p)
        s.erase(p+1);
  };

  //Set all optional parameters to defaults
  int opts_maxIter = 100;
  int opts_init = 0;
  int opts_tFlag = 5;
  int opts_nFlag = 0;
  int opts_rFlag = 1;
  int opts_mFlag = 0;
  double opts_tol = 0.0001;
  int opts_disableEC = 0;
  arma::mat opts_ind = weights;

  //Overwrite default options with those found in slep_opts file.
  if ( slep_opts.find("maxIter") != slep_opts.end() ) {
	opts_maxIter = std::stoi(slep_opts["maxIter"]);
  }
  int opts_rStartNum = opts_maxIter;
  if ( slep_opts.find("init") != slep_opts.end() ) {
	opts_init = std::stoi(slep_opts["init"]);
  }
  if ( slep_opts.find("tFlag") != slep_opts.end() ) {
	opts_tFlag = std::stoi(slep_opts["tFlag"]);
  }
  if ( slep_opts.find("nFlag") != slep_opts.end() ) {
	opts_nFlag = std::stoi(slep_opts["nFlag"]);
  }
  if ( slep_opts.find("rFlag") != slep_opts.end() ) {
	opts_rFlag = std::stoi(slep_opts["rFlag"]);
  }
  if ( slep_opts.find("mFlag") != slep_opts.end() ) {
	opts_mFlag = std::stoi(slep_opts["mFlag"]);
  }
  if ( slep_opts.find("tol") != slep_opts.end() ) {
	opts_tol = std::stod(slep_opts["tol"]);
  }
  if ( slep_opts.find("disableEC") != slep_opts.end() ) {
	opts_disableEC = std::stoi(slep_opts["disableEC"]);
  }
  std::string line;
  if ( slep_opts.find("nu") != slep_opts.end() ) {
        std::vector<double> opts_nu;
        std::ifstream nuFile (slep_opts["nu"]);
        if (nuFile.is_open())
        {
          while (getline(nuFile, line))
          {
            trim(line);
            opts_nu.push_back(std::stod(line));
          }
        }
  }
  if ( slep_opts.find("mu") != slep_opts.end() ) {
        std::vector<double> opts_mu;
        std::ifstream muFile (slep_opts["mu"]);
        if (muFile.is_open())
        {
          while (getline(muFile, line))
          {
            trim(line);
            opts_mu.push_back(std::stod(line));
          }
        }
  }
  std::vector<double> opts_sWeight;
  if ( slep_opts.find("sWeight") != slep_opts.end() ) {
        std::ifstream sWeightFile (slep_opts["sWeight"]);
        if (sWeightFile.is_open())
        {
          while (getline(sWeightFile, line))
          {
            trim(line);
            opts_sWeight.push_back(std::stod(line));
          }
        }
  }

  const size_t m = A.n_rows;
  const size_t n = A.n_cols;
  const size_t F = field.n_cols;  // expanded vector size

  // Build field index (0-based)
  std::vector<int> field_idx(F);
  for (size_t j = 0; j < F; ++j)
      field_idx[j] = static_cast<int>(field(j)) - 1;

  arma::mat& ind = opts_ind;

  // Pre-compute flattened group structure
  const int n_groups = static_cast<int>(ind.n_rows);
  std::vector<double> ind_base_weights(n_groups);
  std::vector<double> ind_flat((n_groups + 1) * 3);
  ind_flat[0] = -1.0;  ind_flat[1] = -1.0;  ind_flat[2] = 0.0;
  for (int r = 0; r < n_groups; ++r) {
      ind_flat[(r + 1) * 3 + 0] = ind(r, 0);
      ind_flat[(r + 1) * 3 + 1] = ind(r, 1);
      ind_flat[(r + 1) * 3 + 2] = ind(r, 2);
      ind_base_weights[r] = ind(r, 2);
  }
  const int ind_flat_nodes = n_groups + 1;
  auto update_ind_flat = [&](double l1_val, double l2_val) {
      ind_flat[2] = l1_val;
      for (int r = 0; r < n_groups; ++r)
          ind_flat[(r + 1) * 3 + 2] = ind_base_weights[r] * l2_val;
  };
  std::vector<double> ind_flat_base(n_groups * 3);
  for (int r = 0; r < n_groups; ++r) {
      ind_flat_base[r * 3 + 0] = ind(r, 0);
      ind_flat_base[r * 3 + 1] = ind(r, 1);
      ind_flat_base[r * 3 + 2] = ind(r, 2);
  }

  // Log solver dimensions
  std::cout << "  [ol_sg_lasso_logisticr] m=" << m << " n=" << n << " F=" << F
            << " expansion=" << (F - n) << " ("
            << std::fixed << std::setprecision(1) << (100.0 * (F - n) / n) << "% overlap) (fp64)\n";

  // ── Virtual-expansion matvec ─────────────────────────────────────────────────
  const double* A_ptr = A.memptr();
  const int M_int = static_cast<int>(m);
  const int N_int = static_cast<int>(n);

  std::vector<double> x_physical(n, 0.0);
  std::vector<double> g_physical(n);

  auto matvec = [&](const double* x_exp, double* out) {
    std::memset(x_physical.data(), 0, n * sizeof(double));
    for (size_t j = 0; j < F; ++j)
        x_physical[field_idx[j]] += x_exp[j];
    cblas_dgemv(MYESL_CBLAS_COL_MAJOR, MYESL_CBLAS_NO_TRANS,
                M_int, N_int, 1.0, A_ptr, M_int,
                x_physical.data(), 1, 0.0, out, 1);
  };

  auto matvec_t = [&](const double* b_in, double* out_exp) {
    cblas_dgemv(MYESL_CBLAS_COL_MAJOR, MYESL_CBLAS_TRANS,
                M_int, N_int, 1.0, A_ptr, M_int,
                b_in, 1, 0.0, g_physical.data(), 1);
    for (size_t j = 0; j < F; ++j)
        out_exp[j] = g_physical[field_idx[j]];
  };

  // Helper: double dot product
  auto dotf = [](const double* a, const double* b, size_t len) -> double {
    double sum = 0;
    for (size_t i = 0; i < len; i++) sum += a[i] * b[i];
    return sum;
  };

  // Convert responses to flat array
  std::vector<double> y(m);
  for (size_t i = 0; i < m; i++) y[i] = responses(i);

  double* z = this->Lambda();
  double lambda1 = z[0];
  double lambda2 = z[1];
  double lambda2_max;

  if (lambda1<0 || lambda2<0)
  {
	throw std::invalid_argument("\n z should be nonnegative!\n");
  }

  if(ind.n_cols != 3)
  {
	throw std::invalid_argument("\n Check opts_ind, expected 3 cols\n");
  }

  // Sample weights and class detection
  std::vector<double> sw(m);
  int n_pos = 0, n_neg = 0;
  for (size_t i = 0; i < m; i++) {
    if (y[i] == 1.0) n_pos++; else n_neg++;
  }

  double m1, m2;
  if (opts_sWeight.size() == 2)
  {
    std::cout << "Using sample weights of " << opts_sWeight[0] << "(positive) and " << opts_sWeight[1] << "(negative)" << std::endl;
    double total = n_pos * opts_sWeight[0] + n_neg * opts_sWeight[1];
    for (size_t i = 0; i < m; i++) {
      sw[i] = (y[i] == 1.0) ? (opts_sWeight[0] / total) : (opts_sWeight[1] / total);
    }
  } else if (opts_sWeight.size() != 0) {
    std::cout << "Invalid sample weights specified, defaulting to unweighted samples." << std::endl;
    for (size_t i = 0; i < m; i++) sw[i] = 1.0 / m;
  } else {
    for (size_t i = 0; i < m; i++) sw[i] = 1.0 / m;
  }

  // m1 = sum(sw[positive]) / sum(sw)
  double sw_pos_sum = 0, sw_total = 0;
  for (size_t i = 0; i < m; i++) {
    sw_total += sw[i];
    if (y[i] == 1.0) sw_pos_sum += sw[i];
  }
  m1 = sw_pos_sum / sw_total;
  m2 = 1.0 - m1;

  // Lambda initialization — all F-sized
  double* lambda_ptr;
  std::vector<double> b_vec(m);

  if (opts_rFlag == 0)
  {
	  lambda_ptr = z;
  } else {
	 if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
	 {
		throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
	 }
	 for (size_t i = 0; i < m; i++) {
	   b_vec[i] = (y[i] == 1.0) ? (m2 * sw[i]) : (-m1 * sw[i]);
	 }

	 std::vector<double> ATb(F);
	 matvec_t(b_vec.data(), ATb.data());

	 std::vector<double> temp(F);
	 double lambda1_max_d = 0;
	 for (size_t j = 0; j < F; j++) {
	   temp[j] = std::abs(ATb[j]);
	   if (temp[j] > lambda1_max_d) lambda1_max_d = temp[j];
	 }

	 lambda1 = lambda1 * lambda1_max_d;

	 for (size_t j = 0; j < F; j++) {
	   temp[j] = std::max(temp[j] - lambda1, 0.0);
	 }

	 lambda2_max = computeLambda2Max_flat(temp.data(), F, ind_flat_base.data(), n_groups);
	 lambda2 = lambda2 * lambda2_max;
  }

  // Initial state — F-sized coefficient vectors
  std::vector<double> x(F, 0.0);
  double c = std::log(m1/m2);
  std::vector<double> Ax(m, 0.0);

  int bFlag = 0;
  double L = 1.0/m;

  std::vector<double> weighty(m);
  for (size_t i = 0; i < m; i++) weighty[i] = sw[i] * y[i];

  std::vector<double> xp(F, 0.0);
  std::vector<double> Axp(m, 0.0);
  std::vector<double> xxp(F, 0.0);
  double cp = c, ccp = 0;
  double alphap = 0, alpha = 1;

  double beta, sc, gc, fun_s, fun_x, l_sum, r_sum, tree_norm;
  std::vector<double> s(F), v(F), As(m), g(F), aa(m), bb(m), prob(m);
  std::vector<double> ValueL(opts_maxIter);
  std::vector<double> funVal(opts_maxIter);

  int iterStep = 0;
  for (iterStep = 0; iterStep < opts_maxIter; iterStep++)
  {
    beta = (alphap - 1)/alpha;
    for (size_t i = 0; i < F; i++) s[i] = x[i] + xxp[i] * beta;
    sc = c + beta * ccp;

    for (size_t i = 0; i < m; i++) As[i] = Ax[i] + (Ax[i] - Axp[i]) * beta;

    for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (As[i] + sc);

    for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0);

    {
      double acc = 0;
      for (size_t i = 0; i < m; i++) {
        double val = log(exp(-bb[i]) + exp(aa[i] - bb[i])) + bb[i];
        acc += sw[i] * val;
      }
      fun_s = acc;
    }

    for (size_t i = 0; i < m; i++) prob[i] = 1.0 / (1.0 + exp(aa[i]));

    for (size_t i = 0; i < m; i++) b_vec[i] = -weighty[i] * (1.0 - prob[i]);

    {
      double acc = 0;
      for (size_t i = 0; i < m; i++) acc += b_vec[i];
      gc = acc;
    }

    matvec_t(b_vec.data(), g.data());

    std::memcpy(xp.data(), x.data(), F * sizeof(double));
    std::memcpy(Axp.data(), Ax.data(), m * sizeof(double));
    cp = c;

    while (true)
    {
      double invL = 1.0 / L;
      for (size_t i = 0; i < F; i++) v[i] = s[i] - g[i] * invL;
      c = sc - gc/L;

      update_ind_flat(lambda1/L, lambda2/L);
      altra_inplace(x.data(), v.data(), F, ind_flat.data(), ind_flat_nodes);

      for (size_t i = 0; i < F; i++) v[i] = x[i] - s[i];

      matvec(x.data(), Ax.data());

      for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (Ax[i] + c);

      for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0);

      {
        double acc = 0;
        for (size_t i = 0; i < m; i++) {
          double val = log(exp(-bb[i]) + exp(aa[i] - bb[i])) + bb[i];
          acc += sw[i] * val;
        }
        fun_x = acc;
      }

      r_sum = (dotf(v.data(), v.data(), F) + std::pow(c - sc, 2)) / 2.0;

      l_sum = fun_x - fun_s - dotf(v.data(), g.data(), F) - (c - sc) * gc;

      if (r_sum <= std::pow(0.1, 20))
      {
	     bFlag = 1;
	     break;
	  }

	  if ((opts_disableEC==0 && (l_sum < r_sum * L || std::abs(l_sum - (r_sum * L)) < std::pow(0.1, 12)) || opts_disableEC==1 && l_sum <= r_sum * L))
	  {
	     break;
	  } else {
	     L = std::max(static_cast<double>(2*L), l_sum/r_sum);
	  }
    }

    alphap = alpha;   alpha = (1 + std::pow(4 * alpha * alpha + 1.0, 0.5))/2.0;

    ValueL[iterStep] = L;

    for (size_t i = 0; i < F; i++) xxp[i] = x[i] - xp[i];
    ccp = c - cp;

    funVal[iterStep] = fun_x;

    update_ind_flat(lambda1, lambda2);
    tree_norm = treeNorm_flat(x.data(), F, ind_flat.data(), ind_flat_nodes);

    funVal[iterStep] = fun_x + tree_norm;

    if (bFlag) {break;}

    switch (opts_tFlag)
    {
	  case 0:
        if (iterStep >=1)
        {
	      if (std::abs(funVal[iterStep] - funVal[iterStep - 1]) <= opts_tol * funVal[iterStep - 1])
	      {
	        bFlag = 1;
	      }
	    }
	    break;
	  case 5:
        if (iterStep >= opts_maxIter)
        {
          bFlag = 1;
        }
        break;
    }
    if (bFlag) {break;}

	if ((iterStep+1) % opts_rStartNum == 0)
	{
	  alphap = 0;   alpha = 1;
	  std::memcpy(xp.data(), x.data(), F * sizeof(double));
	  std::memcpy(Axp.data(), Ax.data(), m * sizeof(double));
	  std::memset(xxp.data(), 0, F * sizeof(double));
	  L = L/2;
	}
  }

  // Log timing
  auto train_end = std::chrono::steady_clock::now();
  double train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();
  std::cout << "  [ol_sg_lasso_logisticr] converged in " << iterStep << " iterations, "
            << std::fixed << std::setprecision(1) << train_ms << " ms (fp64)\n";

  // Write results back to Armadillo members
  std::cout << std::defaultfloat << std::setprecision(6) << "Intercept: " << c << std::endl;
  this->intercept_value = c;

  parameters.set_size(F);
  std::memcpy(parameters.memptr(), x.data(), F * sizeof(double));

  this->nz_gene_count = countNonZeroGenes(parameters, weights);

  static thread_local arma::rowvec x_row_ret;
  x_row_ret = arma::rowvec(x.data(), F);
  return x_row_ret;
}




const arma::colvec OLSGLassoLogisticRvFP64::altra(const arma::colvec& v_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	double *x;
	x = (double*) malloc(n*sizeof(double));
	const double* v = v_in.memptr();
    int i, j, m;
    double lambda,twoNorm, ratio;
    std::vector<double> ind_buf(ind_mat.n_cols * ind_mat.n_rows);
    double* ind = ind_buf.data();

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

    if ((int) ind[0]==-1){

        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }

        lambda=ind[2];

        for(j=0;j<n;j++){
            if (v[j]>lambda)
                x[j]=v[j]-lambda;
            else
                if (v[j]<-lambda)
                    x[j]=v[j]+lambda;
                else
                    x[j]=0;
        }

        i=1;
    }
    else{
        memcpy(x, v, sizeof(double) * n);
        i=0;
    }

	for(;i < nodes; i++){
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        lambda=ind[3*i+2];
        if (twoNorm>lambda){
            ratio=(twoNorm-lambda)/twoNorm;

            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]*=ratio;
        }
        else{
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]=0;
        }
	}
	arma::colvec x_col(&x[0], n);
	free(x);
	return x_col;
}


const double OLSGLassoLogisticRvFP64::treeNorm(const arma::rowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	double tree_norm;
	const double* x = x_in.memptr();
    int i, j, m;
    double twoNorm, lambda;
    std::vector<double> ind_buf(ind_mat.n_cols * ind_mat.n_rows);
    double* ind = ind_buf.data();

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

    tree_norm=0;

    if ((int) ind[0]==-1){

        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }

        lambda=ind[2];

        for(j=0;j<n;j++){
            tree_norm+=fabs(x[j]);
        }

        tree_norm=tree_norm * lambda;

        i=1;
    }
    else{
        i=0;
    }

	for(;i < nodes; i++){
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        lambda=ind[3*i+2];

        tree_norm=tree_norm + lambda*twoNorm;
	}

	return tree_norm;
}


const double OLSGLassoLogisticRvFP64::computeLambda2Max(const arma::rowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
    int i, j, m;
    double lambda,twoNorm;
    const double* x = x_in.memptr();
    std::vector<double> ind_buf(ind_mat.n_cols * ind_mat.n_rows);
    double* ind = ind_buf.data();

    double lambda2_max = 0;

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

    for(i=0;i < nodes; i++){
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        twoNorm=twoNorm/ind[3*i+2];

        if (twoNorm >lambda2_max )
            lambda2_max=twoNorm;
	}

	return lambda2_max;
}


void OLSGLassoLogisticRvFP64::altra_inplace(double* x, const double* v, int n,
                                const double* ind, int nodes) const
{
    int i, j;
    double lambda, twoNorm, ratio;

    if ((int) ind[0]==-1){
        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }
        lambda=ind[2];
        for(j=0;j<n;j++){
            if (v[j]>lambda)
                x[j]=v[j]-lambda;
            else if (v[j]<-lambda)
                x[j]=v[j]+lambda;
            else
                x[j]=0;
        }
        i=1;
    }
    else{
        memcpy(x, v, sizeof(double) * n);
        i=0;
    }

    for(;i < nodes; i++){
        twoNorm=0;
        for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
            twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        lambda=ind[3*i+2];
        if (twoNorm>lambda){
            ratio=(twoNorm-lambda)/twoNorm;
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]*=ratio;
        }
        else{
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]=0;
        }
    }
}


double OLSGLassoLogisticRvFP64::treeNorm_flat(const double* x, int n,
                                  const double* ind, int nodes) const
{
    double tree_norm = 0;
    int i, j;
    double twoNorm, lambda;

    if ((int) ind[0]==-1){
        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }
        lambda=ind[2];
        for(j=0;j<n;j++){
            tree_norm+=fabs(x[j]);
        }
        tree_norm=tree_norm * lambda;
        i=1;
    }
    else{
        i=0;
    }

    for(;i < nodes; i++){
        twoNorm=0;
        for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
            twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);
        lambda=ind[3*i+2];
        tree_norm=tree_norm + lambda*twoNorm;
    }

    return tree_norm;
}


double OLSGLassoLogisticRvFP64::computeLambda2Max_flat(const double* x, int n,
                                           const double* ind, int nodes) const
{
    int i, j;
    double twoNorm;
    double lambda2_max = 0;

    for(i=0;i < nodes; i++){
        twoNorm=0;
        for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
            twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);
        twoNorm=twoNorm/ind[3*i+2];
        if (twoNorm > lambda2_max)
            lambda2_max=twoNorm;
    }

    return lambda2_max;
}
