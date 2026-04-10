#!/usr/bin/env bash
# compare_golden.sh — Compare drphylo output against golden reference files.
# Usage: bash test/compare_golden.sh <actual_dir> <golden_dir>
# Exits non-zero if any comparison fails.

set -euo pipefail

ACTUAL="${1:?Usage: compare_golden.sh <actual_dir> <golden_dir>}"
GOLDEN="${2:?Usage: compare_golden.sh <actual_dir> <golden_dir>}"
TOL=0.01
FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

# compare_tsv_floats <actual> <golden> <label_cols> <desc>
#   Compares two TSV files. The first <label_cols> columns must match exactly.
#   Remaining columns are compared with floating-point tolerance.
#   NaN values must match (both NaN or both non-NaN).
compare_tsv_floats() {
    local actual="$1" golden="$2" label_cols="$3" desc="$4"

    if [[ ! -f "$actual" ]]; then
        fail "$desc: actual file missing ($actual)"
        return
    fi
    if [[ ! -f "$golden" ]]; then
        fail "$desc: golden file missing ($golden)"
        return
    fi

    local actual_lines golden_lines
    actual_lines=$(wc -l < "$actual")
    golden_lines=$(wc -l < "$golden")
    if [[ "$actual_lines" -ne "$golden_lines" ]]; then
        fail "$desc: line count differs (actual=$actual_lines, golden=$golden_lines)"
        return
    fi

    local result
    result=$(awk -F'\t' -v tol="$TOL" -v lcols="$label_cols" -v desc="$desc" '
    BEGIN { errors = 0 }
    NR == FNR { # golden
        for (i = 1; i <= NF; i++) gold[NR][i] = $i
        gold_nf[NR] = NF
        next
    }
    { # actual
        line = FNR
        if (NF != gold_nf[line]) {
            printf "  %s line %d: column count differs (actual=%d, golden=%d)\n", desc, line, NF, gold_nf[line]
            errors++
            next
        }
        for (i = 1; i <= NF; i++) {
            g = gold[line][i]
            a = $i
            if (i <= lcols) {
                # Label column: exact match
                if (a != g) {
                    printf "  %s line %d col %d: label mismatch (actual=\"%s\", golden=\"%s\")\n", desc, line, i, a, g
                    errors++
                }
            } else {
                # Numeric column: tolerance comparison
                a_nan = (a == "NaN" || a == "nan" || a == "NAN")
                g_nan = (g == "NaN" || g == "nan" || g == "NAN")
                if (a_nan && g_nan) continue
                if (a_nan != g_nan) {
                    printf "  %s line %d col %d: NaN mismatch (actual=\"%s\", golden=\"%s\")\n", desc, line, i, a, g
                    errors++
                    continue
                }
                diff = a - g
                if (diff < 0) diff = -diff
                if (diff > tol) {
                    printf "  %s line %d col %d: value differs by %.6f (actual=%s, golden=%s, tol=%s)\n", desc, line, i, diff, a, g, tol
                    errors++
                }
            }
        }
    }
    END { print errors }
    ' "$golden" "$actual")

    local error_count
    error_count=$(echo "$result" | tail -1)
    if [[ "$error_count" -gt 0 ]]; then
        echo "$result" | head -20
        if [[ "$error_count" -gt 20 ]]; then
            echo "  ... and $((error_count - 20)) more errors"
        fi
        fail "$desc: $error_count comparison errors"
    else
        echo "OK: $desc"
    fi
}

echo "=== Regression test: comparing $ACTUAL against $GOLDEN ==="
echo "  Tolerance: $TOL"
echo ""

# Top-level hss_summary.txt (2 cols: clade_name, HSS)
compare_tsv_floats "$ACTUAL/hss_summary.txt" "$GOLDEN/hss_summary.txt" 1 "hss_summary.txt"

for clade in Control Clade_X1; do
    echo ""
    echo "--- Clade: $clade ---"

    # eval.txt: 3 cols header (SequenceID, PredictedValue, TrueValue) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/eval.txt" "$GOLDEN/$clade/eval.txt" 1 "$clade/eval.txt"

    # eval_SPS_SPP.txt: 4 cols (SeqID, Response, SPS, SPP) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/eval_SPS_SPP.txt" "$GOLDEN/$clade/eval_SPS_SPP.txt" 1 "$clade/eval_SPS_SPP.txt"

    # eval_gene_predictions.txt: header + data (SeqID, Response, Prediction_mean, genes...) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/eval_gene_predictions.txt" "$GOLDEN/$clade/eval_gene_predictions.txt" 1 "$clade/eval_gene_predictions.txt"

    # gss_median.txt: 2 cols (gene, score) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/gss_median.txt" "$GOLDEN/$clade/gss_median.txt" 1 "$clade/gss_median.txt"

    # pss_median.txt: 2 cols (position, score) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/pss_median.txt" "$GOLDEN/$clade/pss_median.txt" 1 "$clade/pss_median.txt"

    # bss_median.txt: 2 cols (weight_label, score) — 1 label col
    compare_tsv_floats "$ACTUAL/$clade/bss_median.txt" "$GOLDEN/$clade/bss_median.txt" 1 "$clade/bss_median.txt"
done

echo ""
if [[ "$FAILURES" -gt 0 ]]; then
    echo "FAILED: $FAILURES comparison(s) failed"
    exit 1
else
    echo "PASSED: All comparisons match within tolerance"
    exit 0
fi
