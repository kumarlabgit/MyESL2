#include "vcf_parser.hpp"
#include "numeric_parser.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#ifdef MYESL2_HAVE_ZLIB
#include <zlib.h>
#endif

namespace vcf {

namespace {

namespace fs = std::filesystem;

// Number of fixed VCF columns before the per-sample genotype columns:
// CHROM POS ID REF ALT QUAL FILTER INFO FORMAT
constexpr size_t kFixedColumns = 9;
constexpr size_t kColChrom  = 0;
constexpr size_t kColPos    = 1;
constexpr size_t kColRef    = 3;
constexpr size_t kColAlt    = 4;
constexpr size_t kColFormat = 8;

bool has_gz_suffix(const fs::path& p) {
    std::string s = p.string();
    return s.size() >= 3 && s.compare(s.size() - 3, 3, ".gz") == 0;
}

// Reads lines from a plain or gzip/bgzip-compressed file. zlib's gzopen reads
// uncompressed files transparently, so when zlib is available a single path
// serves both. bgzip output is a multi-member gzip stream, which gzgets also
// handles, so no BGZF-specific code is needed for sequential reading.
class LineReader {
public:
    explicit LineReader(const fs::path& path) {
#ifdef MYESL2_HAVE_ZLIB
        gz_ = gzopen(path.string().c_str(), "rb");
        if (!gz_)
            throw std::runtime_error("Cannot open VCF file: " + path.string());
#else
        if (has_gz_suffix(path))
            throw std::runtime_error(
                "Cannot read compressed VCF '" + path.string() +
                "': this build has no zlib support. Decompress it first "
                "(gunzip/bgzip -d), or rebuild MyESL2 with zlib available.");
        plain_.open(path, std::ios::binary);
        if (!plain_)
            throw std::runtime_error("Cannot open VCF file: " + path.string());
#endif
    }

    ~LineReader() {
#ifdef MYESL2_HAVE_ZLIB
        if (gz_) gzclose(gz_);
#endif
    }

    LineReader(const LineReader&) = delete;
    LineReader& operator=(const LineReader&) = delete;

    // Returns false at end of input. Strips the trailing newline and any CR.
    bool next(std::string& line) {
#ifdef MYESL2_HAVE_ZLIB
        line.clear();
        char buf[65536];
        while (true) {
            if (gzgets(gz_, buf, static_cast<int>(sizeof(buf))) == nullptr)
                return !line.empty();   // trailing line with no final newline
            line.append(buf);
            if (!line.empty() && line.back() == '\n') {
                line.pop_back();
                if (!line.empty() && line.back() == '\r') line.pop_back();
                return true;
            }
            // Line longer than the buffer: keep appending.
        }
#else
        if (!std::getline(plain_, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
#endif
    }

private:
#ifdef MYESL2_HAVE_ZLIB
    gzFile gz_ = nullptr;
#else
    std::ifstream plain_;
#endif
};

std::vector<std::string_view> split_view(std::string_view s, char delim) {
    std::vector<std::string_view> out;
    size_t start = 0;
    while (true) {
        size_t p = s.find(delim, start);
        if (p == std::string_view::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

// Feature labels are split at their last '_' downstream to recover the allele,
// so an allele string must not contain one. Only symbolic alleles realistically
// can (e.g. <INS:ME:ALU>); the original text is preserved nowhere else, so this
// is a display-level substitution only.
std::string sanitize_allele(std::string_view allele) {
    std::string out(allele);
    std::replace(out.begin(), out.end(), '_', '-');
    return out;
}

// Splits a GT value on '/' and '|' (phasing is irrelevant to dosage) and
// resolves each copy to an allele index. '.' copies are dropped, so a partial
// call like "./1" is treated as a single called copy.
void parse_genotype(std::string_view gt, size_t num_alleles,
                    std::vector<size_t>& called_out) {
    called_out.clear();
    size_t start = 0;
    const size_t n = gt.size();
    for (size_t i = 0; i <= n; ++i) {
        if (i == n || gt[i] == '/' || gt[i] == '|') {
            std::string_view tok = gt.substr(start, i - start);
            start = i + 1;
            if (tok.empty() || tok == ".") continue;   // missing copy
            size_t idx = 0;
            bool numeric = true;
            for (char c : tok) {
                if (c < '0' || c > '9') { numeric = false; break; }
                idx = idx * 10 + static_cast<size_t>(c - '0');
            }
            // Out-of-range indices indicate a malformed record; ignore the copy
            // rather than reading past the allele list.
            if (numeric && idx < num_alleles) called_out.push_back(idx);
        }
    }
}

} // anonymous namespace

bool parse_het_mode(const std::string& name, HetMode& out) {
    if (name == "dosage")       { out = HetMode::Dosage;      return true; }
    if (name == "presence")     { out = HetMode::Presence;    return true; }
    if (name == "alt-dominant") { out = HetMode::AltDominant; return true; }
    return false;
}

const char* het_mode_name(HetMode mode) {
    switch (mode) {
        case HetMode::Presence:    return "presence";
        case HetMode::AltDominant: return "alt-dominant";
        case HetMode::Dosage:      break;
    }
    return "dosage";
}

std::vector<std::string> read_sample_ids(const fs::path& input) {
    std::vector<std::string> ids;
    try {
        LineReader reader(input);
        std::string line;
        while (reader.next(line)) {
            if (line.empty() || line.rfind("##", 0) == 0) continue;
            if (line[0] != '#') break;              // data before header: no samples
            auto fields = split_view(line, '\t');
            for (size_t i = kFixedColumns; i < fields.size(); ++i)
                ids.emplace_back(fields[i]);
            break;
        }
    } catch (...) {
        ids.clear();
    }
    return ids;
}

void vcf_to_vnf(const fs::path& input,
                const fs::path& output,
                const VcfConvertOptions& opts)
{
    LineReader reader(input);
    std::string line;

    // --- Header: skip '##' meta lines, take sample IDs from the '#CHROM' line ---
    std::vector<std::string> sample_ids;
    bool saw_chrom_header = false;
    while (reader.next(line)) {
        if (line.empty()) continue;
        if (line.rfind("##", 0) == 0) continue;
        if (line[0] == '#') {
            auto fields = split_view(line, '\t');
            if (fields.size() <= kFixedColumns)
                throw std::runtime_error(
                    "VCF has no genotype columns (sites-only file): " + input.string());
            for (size_t i = kFixedColumns; i < fields.size(); ++i)
                sample_ids.emplace_back(fields[i]);
            saw_chrom_header = true;
            break;
        }
        throw std::runtime_error("VCF data line before the #CHROM header: " + input.string());
    }
    if (!saw_chrom_header)
        throw std::runtime_error("VCF is missing its #CHROM header line: " + input.string());
    if (sample_ids.empty())
        throw std::runtime_error("VCF declares no samples: " + input.string());

    const size_t num_samples = sample_ids.size();

    // Column-major accumulation: columns[j][sample]. Converted to the row-major
    // payload at the end. Memory is (retained alleles x samples) floats.
    std::vector<std::string>        labels;
    std::vector<std::vector<float>> columns;

    std::unordered_set<std::string> seen_sites;
    std::vector<size_t> called;          // reused per genotype
    std::vector<float>  site_values;     // reused per (site, sample)

    while (reader.next(line)) {
        if (line.empty() || line[0] == '#') continue;

        auto f = split_view(line, '\t');
        if (f.size() < kFixedColumns + num_samples)
            throw std::runtime_error(
                "VCF record has " + std::to_string(f.size()) + " columns but " +
                std::to_string(kFixedColumns + num_samples) + " were expected (" +
                std::string(f[kColChrom]) + ":" + std::string(f[kColPos]) + ") in " +
                input.string());

        // A '.' ALT means no alternate allele is called at this site, so it can
        // carry no genotype variation; skip it rather than emit a constant column.
        if (f[kColAlt] == ".") continue;

        std::string site_key = std::string(f[kColChrom]) + ":" + std::string(f[kColPos]);
        if (!seen_sites.insert(site_key).second)
            throw std::runtime_error(
                "VCF contains more than one record for " + site_key + " in " +
                input.string() +
                ". Multiallelic sites split across records cannot be encoded "
                "unambiguously; merge them first (bcftools norm -m+any).");

        // Allele list: REF followed by each comma-separated ALT.
        std::vector<std::string_view> alleles;
        alleles.push_back(f[kColRef]);
        for (auto a : split_view(f[kColAlt], ',')) alleles.push_back(a);
        const size_t num_alleles = alleles.size();

        // Locate GT within FORMAT by name; it is conventionally first but the
        // spec does not require it.
        size_t gt_idx = std::string_view::npos;
        {
            auto fmt = split_view(f[kColFormat], ':');
            for (size_t i = 0; i < fmt.size(); ++i)
                if (fmt[i] == "GT") { gt_idx = i; break; }
        }
        if (gt_idx == std::string_view::npos) continue;   // no genotypes at this site

        // Per-allele column values for this site, all samples.
        std::vector<std::vector<float>> site_cols(num_alleles,
                                                  std::vector<float>(num_samples, 0.0f));

        for (size_t s = 0; s < num_samples; ++s) {
            std::string_view sample_field = f[kFixedColumns + s];
            std::string_view gt;
            {
                auto parts = split_view(sample_field, ':');
                if (gt_idx >= parts.size()) continue;     // truncated => missing
                gt = parts[gt_idx];
            }
            parse_genotype(gt, num_alleles, called);
            if (called.empty()) continue;                 // ./. => all-zero, per convention

            site_values.assign(num_alleles, 0.0f);
            switch (opts.het_mode) {
                case HetMode::Dosage: {
                    const float denom = static_cast<float>(called.size());
                    for (size_t idx : called) site_values[idx] += 1.0f / denom;
                    break;
                }
                case HetMode::Presence: {
                    for (size_t idx : called) site_values[idx] = 1.0f;
                    break;
                }
                case HetMode::AltDominant: {
                    bool any_alt = false;
                    for (size_t idx : called) if (idx != 0) { any_alt = true; break; }
                    if (any_alt) {
                        // REF stays 0.0; every observed alternate gets full weight.
                        for (size_t idx : called) if (idx != 0) site_values[idx] = 1.0f;
                    } else {
                        site_values[0] = 1.0f;
                    }
                    break;
                }
            }
            for (size_t a = 0; a < num_alleles; ++a)
                site_cols[a][s] = site_values[a];
        }

        // Emit one column per allele that any sample actually carries. Dropping
        // all-zero columns keeps the cache compact; encode() would filter them
        // out anyway via the min_minor carrier-count test.
        std::unordered_set<std::string> site_allele_labels;
        for (size_t a = 0; a < num_alleles; ++a) {
            bool carried = false;
            for (size_t s = 0; s < num_samples; ++s)
                if (site_cols[a][s] != 0.0f) { carried = true; break; }
            if (!carried) continue;

            std::string label = site_key + "_" + sanitize_allele(alleles[a]);
            // Guards a malformed ALT list that repeats an allele (e.g. "G,G"),
            // which would otherwise produce duplicate feature labels.
            if (!site_allele_labels.insert(label).second) continue;

            labels.push_back(std::move(label));
            columns.push_back(std::move(site_cols[a]));
        }
    }

    if (labels.empty())
        throw std::runtime_error(
            "No usable variant sites found in " + input.string() +
            " (every site was monomorphic, uncalled, or lacked a GT field).");

    // Flatten column-major accumulation into the row-major payload PNF expects.
    const size_t num_features = labels.size();
    std::vector<float> flat(num_samples * num_features);
    for (size_t s = 0; s < num_samples; ++s)
        for (size_t j = 0; j < num_features; ++j)
            flat[s * num_features + j] = columns[j][s];

    numeric::write_pnf(output, sample_ids, labels, flat,
                       fs::absolute(input).string(),
                       het_mode_name(opts.het_mode));
}

} // namespace vcf
