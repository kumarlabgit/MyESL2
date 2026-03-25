
#include "sg_lasso_leastr_fp32.hpp"
#include "sg_lasso_helpers.hpp"
#include <sstream>
#include <iomanip>
#include <cstring>

SGLassoLeastRFP32::SGLassoLeastRFP32(const arma::fmat& features,
                                   const arma::frowvec& responses,
                                   const arma::mat& weights,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(features, responses, weights, slep_opts, intercept);
}


SGLassoLeastRFP32::SGLassoLeastRFP32(const arma::fmat& features,
                                   const arma::frowvec& responses,
                                   const arma::mat& weights,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const arma::rowvec& xval_idxs,
                                   int xval_id,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  //subset features and responses according to xval_id and xval_idxs
  arma::uvec indices = arma::find(xval_idxs != xval_id);
  Train(features.rows(indices), responses.elem(indices).t(), weights, slep_opts, intercept);
}


void SGLassoLeastRFP32::writeModelToXMLStream(std::ofstream& XMLFile)
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

void SGLassoLeastRFP32::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap)
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


arma::frowvec& SGLassoLeastRFP32::Train(const arma::fmat& features,
                               const arma::frowvec& responses,
                               const arma::mat& weights,
                               std::map<std::string, std::string> slep_opts,
                               const bool intercept)
{
  this->intercept = intercept;

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

  const size_t m = features.n_rows;
  const size_t n = features.n_cols;

  const arma::fmat& A = features;
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

  // ── Native flat-array implementation ─────────────────────────────────────────
  const float* A_ptr = A.memptr(); // column-major: A(i,j) = A_ptr[j*m + i]

  // Helper: out = A * x_in  (m×n × n×1 = m×1)
  auto matvec = [&](const float* x_in, float* out) {
    std::memset(out, 0, m * sizeof(float));
    for (size_t j = 0; j < n; j++) {
      float xj = x_in[j];
      const float* col = A_ptr + j * m;
      for (size_t i = 0; i < m; i++) out[i] += col[i] * xj;
    }
  };

  // Helper: out = A^T * b_in  (n×m × m×1 = n×1)
  auto matvec_t = [&](const float* b_in, float* out) {
    for (size_t j = 0; j < n; j++) {
      float sum = 0;
      const float* col = A_ptr + j * m;
      for (size_t i = 0; i < m; i++) sum += col[i] * b_in[i];
      out[j] = sum;
    }
  };

  // Helper: float dot product
  auto dotf = [](const float* a, const float* b, size_t len) -> float {
    float sum = 0;
    for (size_t i = 0; i < len; i++) sum += a[i] * b[i];
    return sum;
  };

  // Convert responses to flat array
  std::vector<float> y(m);
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

  // ATy = A^T * y (computed once)
  std::vector<float> ATy(n);
  matvec_t(y.data(), ATy.data());

  // Lambda initialization
  double* lambda_ptr;

  if (opts_rFlag == 0)
  {
	  lambda_ptr = z;
  } else {
	 if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
	 {
		throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
	 }

	 // temp = abs(ATy), lambda1_max = max(temp)
	 std::vector<float> temp(n);
	 float lambda1_max_f = 0;
	 for (size_t j = 0; j < n; j++) {
	   temp[j] = std::abs(ATy[j]);
	   if (temp[j] > lambda1_max_f) lambda1_max_f = temp[j];
	 }

	 lambda1 = lambda1 * (double)lambda1_max_f;

	 // temp = max(temp - lambda1, 0)
	 for (size_t j = 0; j < n; j++) {
	   temp[j] = std::max(temp[j] - (float)lambda1, 0.0f);
	 }

	 lambda2_max = computeLambda2Max_flat(temp.data(), n, ind_flat_base.data(), n_groups);
	 lambda2 = lambda2 * lambda2_max;
  }

  // Initial state
  std::vector<float> x(n, 0.0f);
  std::vector<float> Ax(m, 0.0f);  // A * x = 0 initially

  int bFlag = 0;
  double L = 1.0;

  std::vector<float> xp(n, 0.0f);
  std::vector<float> Axp(m, 0.0f);
  std::vector<float> xxp(n, 0.0f);

  double alphap = 0, alpha = 1;

  // Working vectors
  double beta, fun_x, l_sum, r_sum, tree_norm;
  std::vector<float> s(n), v(n), As(m), ATAs(n), g(n), Av(m);
  std::vector<double> ValueL(opts_maxIter);
  std::vector<double> funVal(opts_maxIter);

  for (int iterStep = 0; iterStep < opts_maxIter; iterStep++)
  {
    beta = (alphap - 1)/alpha;
    for (size_t i = 0; i < n; i++) s[i] = x[i] + xxp[i] * (float)beta;

    // As = Ax + (Ax - Axp) * beta
    for (size_t i = 0; i < m; i++) As[i] = Ax[i] + (Ax[i] - Axp[i]) * (float)beta;

    // ATAs = A^T * As
    matvec_t(As.data(), ATAs.data());

    // g = ATAs - ATy
    for (size_t j = 0; j < n; j++) g[j] = ATAs[j] - ATy[j];

    // Save
    std::memcpy(xp.data(), x.data(), n * sizeof(float));
    std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));

    // Line search
    while (true)
    {
      // v = s - g/L
      float invL = (float)(1.0 / L);
      for (size_t i = 0; i < n; i++) v[i] = s[i] - g[i] * invL;

      // Proximal operator
      update_ind_flat(lambda1/L, lambda2/L);
      altra_inplace(x.data(), v.data(), n, ind_flat.data(), ind_flat_nodes);

      // v = x - s (reuse v as difference)
      for (size_t i = 0; i < n; i++) v[i] = x[i] - s[i];

      // Ax = A * x
      matvec(x.data(), Ax.data());

      // Av = Ax - As
      for (size_t i = 0; i < m; i++) Av[i] = Ax[i] - As[i];

      // r_sum = v^T * v;  l_sum = Av^T * Av
      r_sum = (double)dotf(v.data(), v.data(), n);
      l_sum = (double)dotf(Av.data(), Av.data(), m);

      if (r_sum <= std::pow(0.1, 20))
      {
	     bFlag = 1;
	     break;
	  }

	  if ((opts_disableEC==0 && (l_sum < r_sum * L || abs(l_sum - (r_sum * L)) < std::pow(0.1, 12)) || opts_disableEC==1 && l_sum <= r_sum * L))
	  {
	     break;
	  } else {
	     L = std::max(2*L, l_sum/r_sum);
	  }
    }

    alphap = alpha;   alpha = (1 + std::pow(4 * alpha * alpha + 1.0, 0.5))/2.0;

    // xxp = x - xp;  Axy = Ax - y
    for (size_t i = 0; i < n; i++) xxp[i] = x[i] - xp[i];

    // Axy for funVal: reuse Av as Ax - y
    for (size_t i = 0; i < m; i++) Av[i] = Ax[i] - y[i];

    ValueL[iterStep] = L;

    update_ind_flat(lambda1, lambda2);
    tree_norm = treeNorm_flat(x.data(), n, ind_flat.data(), ind_flat_nodes);
    funVal[iterStep] = (double)dotf(Av.data(), Av.data(), m) / 2.0 + tree_norm;

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
	  std::memcpy(xp.data(), x.data(), n * sizeof(float));
	  std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
	  std::memset(xxp.data(), 0, n * sizeof(float));
	  L = L/2;
	}
  }

  // Write results back to Armadillo members
  this->intercept_value = 0;

  parameters.set_size(n);
  std::memcpy(parameters.memptr(), x.data(), n * sizeof(float));

  this->nz_gene_count = countNonZeroGenes(parameters, weights);

  // Return value (kept for API compat; callers use Parameters() instead)
  static thread_local arma::frowvec x_row_ret;
  x_row_ret = arma::frowvec(x.data(), n);
  return x_row_ret;
}


const arma::fcolvec SGLassoLeastRFP32::altra(const arma::fcolvec& v_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	float *x;
	x = (float*) malloc(n*sizeof(float));
	const float* v = v_in.memptr();
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
        memcpy(x, v, sizeof(float) * n);
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
	arma::fcolvec x_col(&x[0], n);
	free(x);
	return x_col;
}


const double SGLassoLeastRFP32::treeNorm(const arma::frowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	double tree_norm;
	const float* x = x_in.memptr();
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


const double SGLassoLeastRFP32::computeLambda2Max(const arma::frowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
    int i, j, m;
    double lambda,twoNorm;
    const float* x = x_in.memptr();
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


void SGLassoLeastRFP32::altra_inplace(float* x, const float* v, int n,
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
        memcpy(x, v, sizeof(float) * n);
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


double SGLassoLeastRFP32::treeNorm_flat(const float* x, int n,
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


double SGLassoLeastRFP32::computeLambda2Max_flat(const float* x, int n,
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
