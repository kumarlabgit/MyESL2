# gl_logisticr reference test dataset

Tiny synthetic dataset for bit-for-bit comparison between MyESL's
`/claude/MyESL/bin/gl_logisticr` and MyESL2's `gl_logisticr_compare`
standalone test binary.

## Shape

- **4 samples, 6 features, 2 groups of 3 features each**
- Labels: first 2 samples are `+1`, last 2 are `-1`
- Samples 0–1 have "low values" in group 2 (features 4–6);
  samples 2–3 have "high values" in group 2 — so group 2 should be
  strongly selected and group 1 should be shrunk toward zero.

## Files

### `features.csv`
4 lines × 6 values — one sample per row.
Loaded by slep_main/gl_logisticr_compare with `csv_opts::trans` into
an fmat, then transposed back, so the final matrix is rows = samples,
cols = features.

### `response.csv`
One value per line, 4 lines total. Loaded with `csv_opts::trans` into
an frowvec of length 4.

### `groups.csv`
Transposed layout: 3 lines × 2 values per line. After
`csv_opts::trans`, Armadillo loads it as a 2×3 mat where:
- `col(0)` = group start (1-based, inclusive)
- `col(1)` = group end   (1-based, inclusive)
- `col(2)` = per-group weight (unused; defaults to 1.0)

So the on-disk shape is `[starts; ends; weights]` transposed, which
becomes `[start end weight]` per gene after loading.

For this dataset:
- Gene 0: features 1..3, weight 1
- Gene 1: features 4..6, weight 1

### `slep_opts.txt`
Key/value SLEP options. Uses the default path
(`mFlag=0 lFlag=0`) which is the only branch implemented in
`ai_gl_logr.hpp`.

## Expected solver behaviour

With `--lambda1 0.5 --lambda2 0.0`, MyESL's reference binary produces:

```
<item>0.00000000000000000e+00</item>  (group 1, feature 0)
<item>0.00000000000000000e+00</item>  (group 1, feature 1)
<item>0.00000000000000000e+00</item>  (group 1, feature 2)
<item>7.51865983009338379e-01</item>  (group 2, feature 3)
<item>-8.71115505695343018e-01</item>  (group 2, feature 4)
<item>-1.18850290775299072e+00</item>  (group 2, feature 5)
```

Group 1 is shrunk to zero, group 2 is non-trivial — a useful
sanity check for the proximal operator and the group-norm
accumulator.

## Running the comparison

From the repository root:

```bash
./test/compare_gl_logisticr.sh 0.5
./test/compare_gl_logisticr.sh 0.1
./test/compare_gl_logisticr.sh 0.9
```

Requires `/claude/MyESL/bin/gl_logisticr` and a MyESL2 build with
`-DBUILD_SOLVER_TESTS=ON` (the default).

## MyESL quirks preserved for Phase A byte-identity

Two latent imprecisions in MyESL's `RRLogisticR` class are deliberately
reproduced by `GLLogisticRFP64` so Phase A XML outputs diff cleanly
against `/claude/MyESL/bin/gl_logisticr`. Both are fixed in Phase B
along with the raw-C++ refactor.

1. **`<intercept_value>` always 0** — MyESL's `RRLogisticR::Train()`
   never assigns the solver's returned `c` to `intercept_value`, so
   the XML always prints `0.000000`. MyESL2's Phase A wrapper leaves
   `intercept_value` at its default 0.0 to match.

2. **`<item>` values lossily truncated to float precision** — MyESL's
   `RRLogisticR` stores `parameters` as `arma::fvec` (single precision),
   silently downcasting the solver's double-precision output before
   XML serialization. With `std::setprecision(17) << std::scientific`,
   the printed value is the float bits expanded back to a double
   decimal — not the original double. MyESL2's `writeModelToXMLStream()`
   preserves full double precision in `parameters` (`arma::vec`) but
   applies `static_cast<double>(static_cast<float>(...))` in the writer
   so the XML output matches MyESL byte-for-byte. The float cast is
   removed in Phase B once refactor bit-identity is verified.

Both discrepancies are cosmetic (XML output only) — the solver math
is bit-identical in Phase A.
