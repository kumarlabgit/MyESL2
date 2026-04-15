// Standalone comparison test binary for GLLogisticRFP64.
//
// Mirrors /claude/MyESL/src/slep_main.cpp's CSV loader path byte-for-byte
// (load fmat transposed, then .t(), then conv_to<mat>::from) so the
// numerical rounding behaviour matches the MyESL reference. This binary
// bypasses MyESL2's FASTA-encoding pipeline so the solver can be diffed
// in isolation against /claude/MyESL/bin/gl_logisticr.
//
// Usage:
//   gl_logisticr_compare <features.csv> <groups.csv> <response.csv> <out.xml> [lambda1=0.1]
//
// The first three arguments are CSV files in the same shape MyESL's
// slep_main expects (transposed). The output XML is written to <out.xml>
// and can be diffed directly against MyESL's <out>.xml.

#include <armadillo>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "gl_logisticr_fp64.hpp"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <features.csv> <groups.csv> <response.csv> <out.xml>"
                  << " [lambda1=0.1]" << std::endl;
        return 1;
    }

    double lambda[2] = {argc >= 6 ? std::stod(argv[5]) : 0.1, 0.0};

    // Match slep_main.cpp:107–118 byte-for-byte:
    //   fmat features_t; features_t.load(csv_name(features, csv_opts::trans));
    //   fmat features = features_t.t();
    // Then convert to mat for the FP64 solver.
    arma::fmat features_t;
    if (!features_t.load(arma::csv_name(argv[1], arma::csv_opts::trans))) {
        std::cerr << "Failed to load features CSV: " << argv[1] << std::endl;
        return 1;
    }
    arma::fmat features_fmat = features_t.t();
    arma::mat features = arma::conv_to<arma::mat>::from(features_fmat);

    // Match slep_main.cpp:126 — responses loaded transposed as frowvec.
    arma::frowvec responses_fmat;
    if (!responses_fmat.load(arma::csv_name(argv[3], arma::csv_opts::trans))) {
        std::cerr << "Failed to load responses CSV: " << argv[3] << std::endl;
        return 1;
    }
    arma::rowvec responses = arma::conv_to<arma::rowvec>::from(responses_fmat);

    if (responses.n_cols != features.n_rows) {
        std::cerr << "Responses must have same column count as feature row count."
                  << std::endl;
        return 1;
    }

    // Match slep_main.cpp:134 — groups loaded transposed as mat.
    arma::mat opts_ind;
    if (!opts_ind.load(arma::csv_name(argv[2], arma::csv_opts::trans))) {
        std::cerr << "Failed to load groups CSV: " << argv[2] << std::endl;
        return 1;
    }

    std::map<std::string, std::string> slep_opts;  // defaults

    GLLogisticRFP64 model(features, responses, opts_ind, lambda, slep_opts, true);

    std::ofstream out(argv[4]);
    if (!out.is_open()) {
        std::cerr << "Could not open output file: " << argv[4] << std::endl;
        return 1;
    }
    model.writeModelToXMLStream(out);

    std::cout << "Non-zero gene count: " << model.NonZeroGeneCount() << std::endl;
    return 0;
}
