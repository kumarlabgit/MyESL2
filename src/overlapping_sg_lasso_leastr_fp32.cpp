//#include <algorithm>
#include "overlapping_sg_lasso_leastr_fp32.hpp"
#include "overlapping_fp32.hpp"
#include "sg_lasso_helpers.hpp"
#include <sstream>
#include <iomanip>
#include <cstring>


OLSGLassoLeastRFP32::OLSGLassoLeastRFP32(const arma::fmat& features,
                                   const arma::frowvec& responses,
                                   const arma::mat& weights,
                                   const arma::rowvec& field,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(features, responses, weights, slep_opts, field, intercept);
}


OLSGLassoLeastRFP32::OLSGLassoLeastRFP32(const arma::fmat& features,
                                   const arma::frowvec& responses,
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
  //subset features and responses according to xval_id and xval_idxs
  arma::uvec indices = arma::find(xval_idxs != xval_id);
  Train(features.cols(indices), responses.elem(indices).t(), weights, slep_opts, field, intercept);
}


void OLSGLassoLeastRFP32::writeModelToXMLStream(std::ofstream& XMLFile)
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

void OLSGLassoLeastRFP32::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap)
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


arma::frowvec& OLSGLassoLeastRFP32::Train(const arma::fmat& features,
                               const arma::frowvec& responses,
                               const arma::mat& weights,
                               std::map<std::string, std::string> slep_opts,
                               const arma::rowvec& field,
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
  int opts_init = 2;
  int opts_tFlag = 5;
  int opts_nFlag = 0;
  int opts_rFlag = 1;
  int opts_mFlag = 0;
  double opts_tol = 0.0001;
  arma::fmat opts_ind = arma::conv_to<arma::fmat>::from(weights);
  opts_ind.cols(0,1) = opts_ind.cols(0,1) - 1;
  arma::frowvec opts_field = arma::conv_to<arma::frowvec>::from(field) - 1.0f;

  //Set overlapping specific parameters to defaults
  int opts_maxIter2 = 100;
  double opts_tol2 = 0.0001;
  int opts_flag2 = 2;
  int opts_disableEC = 0;
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
  if ( slep_opts.find("tol2") != slep_opts.end() ) {
	opts_tol2 = std::stod(slep_opts["tol2"]);
  }
  if ( slep_opts.find("maxIter2") != slep_opts.end() ) {
	opts_maxIter2 = std::stoi(slep_opts["maxIter2"]);
  }
  if ( slep_opts.find("flag2") != slep_opts.end() ) {
	opts_flag2 = std::stoi(slep_opts["flag2"]);
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

  const arma::fmat& A = features;
  const size_t m = A.n_rows;
  const size_t n = A.n_cols;

  int groupNum = opts_ind.n_rows;

  // Flatten weights for computeLambda2Max_flat (1-based indices, double from original weights)
  std::vector<double> weights_flat(groupNum * 3);
  for (int r = 0; r < groupNum; ++r) {
      weights_flat[r * 3 + 0] = weights(r, 0);
      weights_flat[r * 3 + 1] = weights(r, 1);
      weights_flat[r * 3 + 2] = weights(r, 2);
  }

  // ── Native flat-array implementation (float) ──────────────────────────────────
  const float* A_ptr = A.memptr();

  auto matvec = [&](const float* x_in, float* out) {
    std::memset(out, 0, m * sizeof(float));
    for (size_t j = 0; j < n; j++) {
      float xj = x_in[j];
      const float* col = A_ptr + j * m;
      for (size_t i = 0; i < m; i++) out[i] += col[i] * xj;
    }
  };

  auto matvec_t = [&](const float* b_in, float* out) {
    for (size_t j = 0; j < n; j++) {
      float sum = 0;
      const float* col = A_ptr + j * m;
      for (size_t i = 0; i < m; i++) sum += col[i] * b_in[i];
      out[j] = sum;
    }
  };

  auto dotf = [](const float* a, const float* b, size_t len) -> double {
    double sum = 0;
    for (size_t i = 0; i < len; i++) sum += (double)a[i] * b[i];
    return sum;
  };

  // Convert responses to flat array
  std::vector<float> y(m);
  for (size_t i = 0; i < m; i++) y[i] = responses(i);

  double* z = this->Lambda();
  double lambda1 = z[0];
  double lambda2 = z[1];
  double lambda2_max;

  float *gap;
  gap = (float*) malloc(sizeof(float));
  float penalty2 [5];

  if (lambda1<0 || lambda2<0)
  {
	throw std::invalid_argument("\n z should be nonnegative!\n");
  }

  if(opts_ind.n_cols != 3)
  {
	throw std::invalid_argument("\n Check opts_ind, expected 3 cols\n");
  }

  std::vector<float> Y(opts_field.n_cols, 0.0f);

  // ATy = A^T * y
  std::vector<float> ATy(n);
  matvec_t(y.data(), ATy.data());

  bool estimate_l2 = false;

  if (opts_rFlag == 0)
  {
	  // lambda = z; (unused)
  } else {
	 if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
	 {
		throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
	 }

	 std::vector<float> temp(n);
	 double lambda1_max = 0;
	 for (size_t j = 0; j < n; j++) {
	   temp[j] = std::abs(ATy[j]);
	   if (temp[j] > lambda1_max) lambda1_max = temp[j];
	 }

	 lambda1 = lambda1 * lambda1_max;
	 std::cout << "lambda1_max: " << lambda1_max << " lambda1: " << lambda1 << std::endl;

	 for (size_t j = 0; j < n; j++) {
	   temp[j] = std::max(temp[j] - (float)lambda1, 0.0f);
	 }

	 if (n == field.n_cols)
	 {
		lambda2_max = computeLambda2Max_flat(temp.data(), n, weights_flat.data(), groupNum);
	 }
	 else
	 {
		 lambda2_max = 1;
		 std::cout << "Could not compute Lambda2 max, attempting to estimate instead..." << std::endl;
		 estimate_l2 = true;
	 }
  }

  // Initial state
  std::vector<float> x(n, 0.0f);
  std::vector<float> Ax(m, 0.0f);

  if (opts_init != 2){
	  std::memcpy(x.data(), ATy.data(), n * sizeof(float));
	  matvec(x.data(), Ax.data());
  }

  int bFlag = 0;
  double L = 1.0;

  std::vector<float> xp(n, 0.0f);
  std::vector<float> Axp(m, 0.0f);
  std::vector<float> xxp(n, 0.0f);

  double alphap = 0;   double alpha = 1;

  double beta, fun_x, l_sum, r_sum;
  std::vector<float> s(n), v(n), Av(m);
  std::vector<float> As_pre(m), ATAs_pre(n), g_pre(n);
  std::vector<double> ValueL(opts_maxIter);
  std::vector<double> funVal(opts_maxIter);

  opts_ind = opts_ind.t();

  for (int iterStep = 0; iterStep < opts_maxIter; iterStep++)
  {
    beta = (alphap - 1)/alpha;
    for (size_t i = 0; i < n; i++) s[i] = x[i] + xxp[i] * (float)beta;
    for (size_t i = 0; i < m; i++) As_pre[i] = Ax[i] + (Ax[i] - Axp[i]) * (float)beta;

    matvec_t(As_pre.data(), ATAs_pre.data());
    for (size_t j = 0; j < n; j++) g_pre[j] = ATAs_pre[j] - ATy[j];

    std::memcpy(xp.data(), x.data(), n * sizeof(float));
    std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));

	if (iterStep == 0)
	{
	  if (estimate_l2)
	  {
		  float invL = (float)(1.0 / L);
		  for (size_t i = 0; i < n; i++) v[i] = s[i] - g_pre[i] * invL;
		  double l2_target, l2_low = 0, l2_high = 1;
		  for(int i = 1; i <= 100; i = i + 1)
		  {
			overlapping(x.data(), gap, penalty2, v.data(), n, groupNum, (float)(lambda1/L), (float)(l2_high/L), opts_ind.memptr(), opts_field.memptr(), Y.data(), opts_maxIter2, opts_flag2, (float)opts_tol2);
			float max_abs_x = 0;
			for (size_t j = 0; j < (size_t)n; j++) { float ax = std::abs(x[j]); if (ax > max_abs_x) max_abs_x = ax; }
			if (max_abs_x == 0)
			{
				std::fill(x.begin(), x.end(), 0.0f);
				std::cout << "Lambda2 High set to " << l2_high << std::endl;
				break;
			} else {
				std::fill(x.begin(), x.end(), 0.0f);
				l2_high = l2_high * 2;
			}
		  }
		  for(int i = 1; i <= 100; i = i + 1)
		  {
			l2_target = (l2_high + l2_low) / 2.0;
			overlapping(x.data(), gap, penalty2, v.data(), n, groupNum, (float)(lambda1/L), (float)(l2_target/L), opts_ind.memptr(), opts_field.memptr(), Y.data(), opts_maxIter2, opts_flag2, (float)opts_tol2);
			float max_abs_x = 0;
			for (size_t j = 0; j < (size_t)n; j++) { float ax = std::abs(x[j]); if (ax > max_abs_x) max_abs_x = ax; }
			if (max_abs_x == 0)
			{
				l2_high = l2_target;
			} else {
				l2_low = l2_target;
			}
			std::fill(x.begin(), x.end(), 0.0f);
			if (l2_high - l2_low < 0.0001)
			{
				lambda2_max = l2_high;
				std::cout << "Lambda2 Max set to " << l2_high << std::endl;
				break;
			}
		  }
	  }

	  if (opts_rFlag != 0)
	  {
		  lambda2 = lambda2 * lambda2_max;
		  std::cout << "Lambda2 set to " << lambda2 << std::endl;
	  }
	}
    int firstFlag = 1;

    while (true)
    {
      float invL = (float)(1.0 / L);
      for (size_t i = 0; i < n; i++) v[i] = s[i] - g_pre[i] * invL;

      overlapping(x.data(), gap, penalty2, v.data(), n, groupNum, (float)(lambda1/L), (float)(lambda2/L), opts_ind.memptr(), opts_field.memptr(), Y.data(), opts_maxIter2, opts_flag2, (float)opts_tol2);

      for (size_t i = 0; i < n; i++) v[i] = x[i] - s[i];

      matvec(x.data(), Ax.data());

      for (size_t i = 0; i < m; i++) Av[i] = Ax[i] - As_pre[i];

      r_sum = dotf(v.data(), v.data(), n);
      l_sum = dotf(Av.data(), Av.data(), m);

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

    for (size_t i = 0; i < n; i++) xxp[i] = x[i] - xp[i];
    for (size_t i = 0; i < m; i++) Av[i] = Ax[i] - y[i];

    ValueL[iterStep] = L;

    double l1_norm = 0;
    for (size_t j = 0; j < n; j++) l1_norm += std::abs(x[j]);
    funVal[iterStep] = dotf(Av.data(), Av.data(), m) / 2.0 + lambda1 * l1_norm + lambda2 * penalty2[0];

    if (bFlag) {break;}
    double norm_xp, norm_xxp;
    norm_xxp = std::sqrt(dotf(xxp.data(), xxp.data(), n));

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
	  case 1:
	    if (iterStep >=1)
	    {
		  if (std::abs(funVal[iterStep] - funVal[iterStep - 1]) <= opts_tol * funVal[iterStep - 1])
	      {
			bFlag = 1;
	      }
		}
		break;
	  case 2:
	    if (funVal[iterStep] <= opts_tol)
	    {
		  bFlag = 1;
	    }
	    break;
	  case 3:
	    norm_xxp = std::sqrt(dotf(xxp.data(), xxp.data(), n));
	    if (norm_xxp <= opts_tol)
	    {
		  bFlag = 1;
	    }
	    break;
	  case 4:
	    norm_xp = std::sqrt(dotf(xp.data(), xp.data(), n));
	    norm_xxp = std::sqrt(dotf(xxp.data(), xxp.data(), n));
	    if (norm_xxp <= opts_tol * std::max(norm_xp, 1.0))
	    {
		  bFlag = 1;
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

	if ((iterStep + 1) % opts_rStartNum == 0)
	{
	  alphap = 0;   alpha = 1;
	  std::memcpy(xp.data(), x.data(), n * sizeof(float));
	  std::memcpy(Axp.data(), Ax.data(), m * sizeof(float));
	  std::fill(xxp.begin(), xxp.end(), 0.0f);
	  L = L/2;
	}
  }

  this->intercept_value = 0;

  parameters.set_size(n);
  std::memcpy(parameters.memptr(), x.data(), n * sizeof(float));

  free(gap);

  this->nz_gene_count = countNonZeroGenes(parameters, weights, opts_field);

  static thread_local arma::frowvec x_row_ret;
  x_row_ret = arma::frowvec(x.data(), n);
  return x_row_ret;
}


double OLSGLassoLeastRFP32::computeLambda2Max_flat(const float* x, int n,
                                              const double* ind, int nodes) const
{
    int i, j;
    double twoNorm;
    double lambda2_max = 0;

    for(i=0;i < nodes; i++){
        twoNorm=0;
        for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
            twoNorm += (double)x[j] * x[j];
        twoNorm=sqrt(twoNorm);
        twoNorm=twoNorm/ind[3*i+2];
        if (twoNorm > lambda2_max)
            lambda2_max=twoNorm;
    }

    return lambda2_max;
}

