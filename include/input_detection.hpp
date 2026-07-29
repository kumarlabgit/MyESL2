#pragma once
#include <filesystem>
#include <string>

namespace input_detection {

// True if p is a regular file whose first non-whitespace byte is '>'.
bool is_single_fasta(const std::filesystem::path& p);

// True if p is a regular file whose first non-empty line contains a tab.
// Only meaningful when the caller is in numeric mode (datatype == "numeric").
bool is_single_numeric_table(const std::filesystem::path& p);

// True if p looks like a single VCF: either a .vcf.gz path (contents are
// compressed and not sniffed here) or a file starting with "##fileformat=VCF".
// Only meaningful when the caller is in VCF mode (datatype == "vcf").
bool is_single_vcf(const std::filesystem::path& p);

// Returns "" if p should be treated as a list of paths (existing behavior).
// Returns "FASTA" if p is detected as a single FASTA alignment.
// Returns "numeric" if datatype is "numeric" and p is detected as a single
// tabular numeric file.
// Returns "VCF" if datatype is "vcf" and p is detected as a single VCF.
// Callers are responsible for emitting the warning and synthesizing the
// 1-entry input list when the return value is non-empty.
std::string detect_single_file(
    const std::filesystem::path& p,
    const std::string& datatype);

// Same return semantics as detect_single_file, but additionally emits a
// stderr warning the first time a given path is detected as a single file
// (deduped across call sites within a process so the warning isn't repeated
// when preprocess + encode + train all hit the same path).
std::string maybe_warn_single_file(
    const std::filesystem::path& p,
    const std::string& datatype);

} // namespace input_detection
