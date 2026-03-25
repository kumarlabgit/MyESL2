
#include "sg_lasso_fp32.hpp"
#include "sg_lasso_helpers.hpp"
#include <sstream>
#include <iomanip>
#include <cstring>


SGLassoFP32::SGLassoFP32(const arma::fmat& features,
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


SGLassoFP32::SGLassoFP32(const arma::fmat& features,
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


void SGLassoFP32::writeModelToXMLStream(std::ofstream& XMLFile)
{
  int i_level = 0;
  //std::string XMLString = "";
  XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
  //for(int i=0; i<this->parameters.n_cols; i++)
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

void SGLassoFP32::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap)
{
  /*
  int i_level = 0;
  //std::string XMLString = "";
  XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
  */
  std::string line;
  std::getline(FeatureMap, line);
  //for(int i=0; i<this->parameters.n_cols; i++)
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


arma::frowvec& SGLassoFP32::Train(const arma::fmat& A,
                               const arma::frowvec& responses,
                               const arma::mat& weights,
                               std::map<std::string, std::string> slep_opts,
                               const bool intercept)
{
  this->intercept = intercept;
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
        std::vector<float> opts_nu;
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
        std::vector<float> opts_mu;
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
  std::vector<float> opts_sWeight;
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
  // A is column-major (Armadillo default): A(i,j) = A_ptr[j*m + i]
  const float* A_ptr = A.memptr();

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

  // Helper: float dot product (matches Armadillo's float accumulation)
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

  // Sample weights and class detection
  std::vector<float> sw(m);
  int n_pos = 0, n_neg = 0;
  for (size_t i = 0; i < m; i++) {
    if (y[i] == 1.0f) n_pos++; else n_neg++;
  }

  double m1, m2;
  if (opts_sWeight.size() == 2)
  {
    std::cout << "Using sample weights of " << opts_sWeight[0] << "(positive) and " << opts_sWeight[1] << "(negative)" << std::endl;
    double total = n_pos * (double)opts_sWeight[0] + n_neg * (double)opts_sWeight[1];
    for (size_t i = 0; i < m; i++) {
      sw[i] = (y[i] == 1.0f) ? (float)(opts_sWeight[0] / total) : (float)(opts_sWeight[1] / total);
    }
  } else if (opts_sWeight.size() != 0) {
    std::cout << "Invalid sample weights specified, defaulting to unweighted samples." << std::endl;
    for (size_t i = 0; i < m; i++) sw[i] = 1.0f / m;
  } else {
    for (size_t i = 0; i < m; i++) sw[i] = 1.0f / m;
  }

  // m1 = sum(sw[positive]) / sum(sw)
  float sw_pos_sum = 0, sw_total = 0;
  for (size_t i = 0; i < m; i++) {
    sw_total += sw[i];
    if (y[i] == 1.0f) sw_pos_sum += sw[i];
  }
  m1 = (double)sw_pos_sum / (double)sw_total;
  m2 = 1.0 - m1;

  // Lambda initialization
  double* lambda_ptr;
  std::vector<float> b_vec(m);

  if (opts_rFlag == 0)
  {
	  lambda_ptr = z;
  } else {
	 if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
	 {
		throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
	 }
	 // b(p_flag) = m2 * sw, b(not_p_flag) = -m1 * sw
	 for (size_t i = 0; i < m; i++) {
	   b_vec[i] = (y[i] == 1.0f) ? (float)(m2 * sw[i]) : (float)(-m1 * sw[i]);
	 }

	 // ATb = A^T * b
	 std::vector<float> ATb(n);
	 matvec_t(b_vec.data(), ATb.data());

	 // temp = abs(ATb), lambda1_max = max(temp)
	 std::vector<float> temp(n);
	 float lambda1_max_f = 0;
	 for (size_t j = 0; j < n; j++) {
	   temp[j] = std::abs(ATb[j]);
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
  double c = std::log(m1/m2);
  std::vector<float> Ax(m, 0.0f);  // A * x = 0 initially

  int bFlag = 0;
  double L = 1.0/m;

  // weighty = sw % y
  std::vector<float> weighty(m);
  for (size_t i = 0; i < m; i++) weighty[i] = sw[i] * y[i];

  std::vector<float> xp(n, 0.0f);
  std::vector<float> Axp(m, 0.0f);
  std::vector<float> xxp(n, 0.0f);
  double cp = c, ccp = 0;
  double alphap = 0, alpha = 1;

  // Working vectors
  double beta, sc, gc, fun_s, fun_x, l_sum, r_sum, tree_norm;
  std::vector<float> s(n), v(n), As(m), g(n), aa(m), bb(m), prob(m);
  std::vector<double> ValueL(opts_maxIter);
  std::vector<double> funVal(opts_maxIter);

  for (int iterStep = 0; iterStep < opts_maxIter; iterStep++)
  {
    beta = (alphap - 1)/alpha;
    for (size_t i = 0; i < n; i++) s[i] = x[i] + xxp[i] * (float)beta;
    sc = c + beta * ccp;

    // As = Ax + (Ax - Axp) * beta
    for (size_t i = 0; i < m; i++) As[i] = Ax[i] + (Ax[i] - Axp[i]) * (float)beta;

    // aa = -y % (As + sc)
    for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (As[i] + (float)sc);

    // bb = max(aa, 0)
    for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0f);

    // fun_s = sw^T * (log(exp(-bb) + exp(aa - bb)) + bb)
    {
      float acc = 0;
      for (size_t i = 0; i < m; i++) {
        float val = logf(expf(-bb[i]) + expf(aa[i] - bb[i])) + bb[i];
        acc += sw[i] * val;
      }
      fun_s = (double)acc;
    }

    // prob = 1 / (1 + exp(aa))
    for (size_t i = 0; i < m; i++) prob[i] = 1.0f / (1.0f + expf(aa[i]));

    // b = -weighty % (1 - prob)
    for (size_t i = 0; i < m; i++) b_vec[i] = -weighty[i] * (1.0f - prob[i]);

    // gc = sum(b)
    {
      float acc = 0;
      for (size_t i = 0; i < m; i++) acc += b_vec[i];
      gc = (double)acc;
    }

    // g = A^T * b
    matvec_t(b_vec.data(), g.data());

    // Save
    std::memcpy(xp.data(), x.data(), n * sizeof(float));
    std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
    cp = c;

    // Line search
    while (true)
    {
      // v = s - g/L;  c = sc - gc/L
      float invL = (float)(1.0 / L);
      for (size_t i = 0; i < n; i++) v[i] = s[i] - g[i] * invL;
      c = sc - gc/L;

      // Proximal operator
      update_ind_flat(lambda1/L, lambda2/L);
      altra_inplace(x.data(), v.data(), n, ind_flat.data(), ind_flat_nodes);

      // v = x - s (reuse v as difference for r_sum/l_sum)
      for (size_t i = 0; i < n; i++) v[i] = x[i] - s[i];

      // Ax = A * x
      matvec(x.data(), Ax.data());

      // aa = -y % (Ax + c)
      for (size_t i = 0; i < m; i++) aa[i] = -y[i] * (Ax[i] + (float)c);

      // bb = max(aa, 0)
      for (size_t i = 0; i < m; i++) bb[i] = std::max(aa[i], 0.0f);

      // fun_x = sw^T * (log(exp(-bb) + exp(aa - bb)) + bb)
      {
        float acc = 0;
        for (size_t i = 0; i < m; i++) {
          float val = logf(expf(-bb[i]) + expf(aa[i] - bb[i])) + bb[i];
          acc += sw[i] * val;
        }
        fun_x = (double)acc;
      }

      // r_sum = (v^T * v + (c - sc)^2) / 2
      r_sum = ((double)dotf(v.data(), v.data(), n) + std::pow(c - sc, 2)) / 2.0;

      // l_sum = fun_x - fun_s - v^T * g - (c - sc) * gc
      l_sum = fun_x - fun_s - (double)dotf(v.data(), g.data(), n) - (c - sc) * gc;

      if (r_sum <= std::pow(0.1, 20))
      {
	     bFlag = 1;
	     break;
	  }

	  if ((opts_disableEC==0 && (l_sum < r_sum * L || abs(l_sum - (r_sum * L)) < std::pow(0.1, 12)) || opts_disableEC==1 && l_sum <= r_sum * L))
	  {
	     break;
	  } else {
	     L = std::max(static_cast<double>(2*L), l_sum/r_sum);
	  }
    }

    alphap = alpha;   alpha = (1 + std::pow(4 * alpha * alpha + 1.0, 0.5))/2.0;

    ValueL[iterStep] = L;

    // xxp = x - xp;  ccp = c - cp
    for (size_t i = 0; i < n; i++) xxp[i] = x[i] - xp[i];
    ccp = c - cp;

    funVal[iterStep] = fun_x;

    update_ind_flat(lambda1, lambda2);
    tree_norm = treeNorm_flat(x.data(), n, ind_flat.data(), ind_flat_nodes);

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
	  std::memcpy(xp.data(), x.data(), n * sizeof(float));
	  std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
	  std::memset(xxp.data(), 0, n * sizeof(float));
	  L = L/2;
	}
  }

  // Write results back to Armadillo members
  std::cout << "Intercept: " << c << std::endl;
  this->intercept_value = c;

  parameters.set_size(n);
  std::memcpy(parameters.memptr(), x.data(), n * sizeof(float));

  this->nz_gene_count = countNonZeroGenes(parameters, weights);

  // Return value (kept for API compat; callers use Parameters() instead)
  static thread_local arma::frowvec x_row_ret;
  x_row_ret = arma::frowvec(x.data(), n);
  return x_row_ret;
}




const arma::fcolvec SGLassoFP32::altra(const arma::fcolvec& v_in,
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


const double SGLassoFP32::treeNorm(const arma::frowvec& x_in,
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


const double SGLassoFP32::computeLambda2Max(const arma::frowvec& x_in,
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


void SGLassoFP32::altra_inplace(float* x, const float* v, int n,
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


double SGLassoFP32::treeNorm_flat(const float* x, int n,
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


double SGLassoFP32::computeLambda2Max_flat(const float* x, int n,
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
