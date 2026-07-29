#pragma once

#include "pnf_format.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace vcf {

/**
 * @brief How a heterozygous genotype is turned into one-hot allele weights.
 *
 * Homozygous calls encode identically under all three modes (the observed
 * allele gets 1.0); the modes differ only for calls carrying more than one
 * distinct allele. Missing calls ("./.") are all-zero in every mode, matching
 * the convention used for a sample absent from a gene elsewhere in MyESL2.
 *
 * For REF=A ALT=G, diploid:
 *
 *   GT     Dosage (default)   Presence        AltDominant
 *   0/0    A=1.0              A=1.0           A=1.0
 *   0/1    A=0.5, G=0.5       A=1.0, G=1.0    A=0.0, G=1.0
 *   1/1    G=1.0              G=1.0           G=1.0
 *   1/2    G=0.5, T=0.5       G=1.0, T=1.0    G=1.0, T=1.0
 *   ./.    all 0              all 0           all 0
 */
enum class HetMode {
    Dosage,       ///< copies of the allele / called copies (half weight for a diploid het)
    Presence,     ///< 1.0 for every allele present in the genotype
    AltDominant,  ///< any non-reference allele present => REF 0.0, each observed ALT 1.0
};

/// Parse a mode name as accepted on the command line ("dosage", "presence",
/// "alt-dominant"). Returns false if the name is unrecognised.
bool parse_het_mode(const std::string& name, HetMode& out);

/// Canonical command-line spelling of a mode; round-trips through parse_het_mode.
const char* het_mode_name(HetMode mode);

struct VcfConvertOptions {
    HetMode het_mode = HetMode::Dosage;
};

/**
 * @brief Convert a VCF into a .vnf dosage-matrix cache (PNF on-disk layout).
 *
 * Reads plain or gzip/bgzip-compressed VCF. Sample IDs come from the #CHROM
 * header line and become the cache's seq_ids. Every (site, allele) pair carried
 * by at least one sample becomes a feature column labelled
 * "{chrom}:{pos}_{allele}"; underscores inside an allele string are replaced
 * with '-' so the label remains splittable at its last underscore.
 *
 * No frequency filtering happens here -- min_minor / drop-major-allele are
 * applied downstream in encode(), where the hypothesis file has already
 * restricted the sample set.
 *
 * @param input  Path to the .vcf or .vcf.gz file
 * @param output Destination .vnf path
 * @param opts   Genotype-encoding options
 * @throws std::runtime_error on malformed input or I/O failure
 */
void vcf_to_vnf(const std::filesystem::path& input,
                const std::filesystem::path& output,
                const VcfConvertOptions& opts);

/**
 * @brief Read just the sample IDs from a VCF's #CHROM header line.
 *
 * Cheap alternative to a full conversion when only the sample universe is
 * needed. Returns an empty vector if the header cannot be read.
 */
std::vector<std::string> read_sample_ids(const std::filesystem::path& input);

} // namespace vcf
