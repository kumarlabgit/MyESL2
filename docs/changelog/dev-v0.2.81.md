# dev-v0.2.81 — 2026-08-20

## New — `models.tsv` records every model a run plans, and how each one turned out
A grid run previously left no machine-readable account of itself. `lambda_list.txt` said which lambda pairs were *intended*, and the presence of a `lambda_<N>/weights.txt` implied one had been fitted, but nothing distinguished a point that was deliberately skipped from one that failed, from one that was never reached because the run died. Recovering that after the fact meant listing directories and guessing.

Every run of `train` — and so of `drphylo`, `aim`, and adaptive sparsification, which all call it — now writes `<output_dir>/models.tsv` **before solving anything**, enumerating every model the run intends to produce, one row per (penalty term × lambda point):

```
Index  PenaltyIdx  PenaltyValue  LambdaIdx  FoldIdx  Lambda1  Lambda2  WeightsPath  Status  Detail
```

Each row is then updated in place as that model completes, is skipped, or fails, so the file is an accurate record at every moment rather than only at the end. `Status` is one of `pending`, `complete`, `failed`, or `skipped`, and the four cover every way a grid point can end: the sequential `--min-groups` skip-ahead ratchet marks `skipped` (including the points after an early break), the parallel post-hoc pruner demotes pruned points to `skipped` because it deletes their output, and both loops record `failed` with the solver's exception message before unwinding. The solver wrapper rethrows, so a failure still aborts the run exactly as before — the log simply survives it.

`WeightsPath` is relative to the output directory, so a run directory stays valid after being moved. `FoldIdx` is `-1` for an ordinary model and identifies the fold under `--cv-scores`. Rows carry the penalty and lambda values as well, which is what a future checkpoint/resume facility will need to pick up the still-`pending` points.

Writes go to a sibling `.tmp` and are renamed over the original, so a run killed mid-update leaves either the previous or the next state, never a torn file. Grids are at most a few hundred rows, so rewriting the whole table on each status change costs nothing next to a solver call.

## New — `evaluate --from-run` scores a whole run directory, including a partial one
Scoring a grid by hand meant one `evaluate` invocation per `lambda_<N>/weights.txt`, restating the list file, datatype, cache directory, and het-mode each time — and after a cancelled run, first working out which models had actually finished.

```
myesl2 evaluate --from-run <run_dir> [--re-evaluate] [--visualize] [--threads N]
```

scores every model `models.tsv` records as `complete`, writing `eval.txt`, `eval_SPS_SPP.txt`, and `eval_gene_predictions.txt` into each lambda directory. No positionals are needed: the list file, datatype, cache directory, het-mode, and thread count are read from the run's `preprocess_config`, and the hypothesis file from its `process_log.txt`. That last one is why the process log is consulted at all — `hyp_path` is the single setting `evaluate` needs that `preprocess_config` does not carry, and it is what enables the accuracy metrics rather than predictions alone.

Models that already carry evaluation output are left alone unless `--re-evaluate` is given. A model that fails to score is reported and the loop continues, since the point of the mode is salvaging a partial run — one bad model must not hide the results of every later one; the exit status is non-zero if any failed. Visualization is off by default (a 9×9 sweep would emit 81 SVGs) and `--visualize` opts in.

A closing summary reports how many models were scored, how many were left alone as already-evaluated, how many carry no single scoreable model (plain k-fold CV, which writes `weights_fold_<N>.txt` and a pooled prediction instead), and how many remain `pending` — worded as *not yet solved*, since a pending row may belong to a run that is still in progress rather than one that was abandoned.

The mode is deliberately safe to run against a directory whose training run is still going. A row only becomes `complete` after its weights file is closed, so `--from-run` never reads a half-written model; and `train`'s own post-processing reads only `weights.txt`, `gss.txt`, `pss.txt`, and `oss.txt`, never evaluation output, so a concurrent partial evaluation cannot feed back into the medians. `taskfile` accepts `from-run` and `re-evaluate`; `evaluate` now has two positional layouts there, as `drphylo` already did.

## New — `--cv-scores` scores each cross-validation fold model individually
A `--nfolds` run reported only the pooled held-out prediction in `cv_predictions.txt`. The individual fold models were written to `weights_fold_<K>.txt` and then never scored, so there was no way to see how any one of them behaved across the full species set.

`--cv-scores` (which requires `--nfolds`) promotes each fold model to a model in its own right: it is registered in `models.tsv` with its own row, returned in the training result, and scored against the **full** dataset exactly as an ordinary model would be. Each fold gets `eval_fold_<K>.txt`, `eval_fold_<K>_SPS_SPP.txt`, and `eval_fold_<K>_gene_predictions.txt`.

**These are in-sample scores.** A fold model is scored on every species, including the ones it was trained on; the held-out numbers remain in `cv_predictions.txt`. That is what "as if it were a normal model" means here, but it is an easy result to misread, so the help text says so explicitly.

Without the flag nothing changes: one pathless row per lambda, no evaluation output, and `cv_predictions.txt` and the fold weights byte-identical either way. Evaluation output names are now derived from the weights filename (`weights.txt` → `eval.txt`, `weights_fold_3.txt` → `eval_fold_3.txt`) in all three scoring loops — `train`'s Phase 4, `drphylo`'s per-lambda loop, and `evaluate --from-run` — so per-fold outputs cannot collide. The grouped-weights companion is derived the same way, so a fold model under `ol_sg_lasso_*` picks up its own `weights_fold_<K>_grouped.txt` rather than silently falling back to ungrouped weights; a `weights_grouped.txt` passed directly to `evaluate` is still recognised.

## Behavior change — `evaluate` and visualizer outputs are published atomically
Every artifact `evaluate` and the visualizer produced was written with a plain `std::ofstream`, which truncates the destination in place. Two exposures followed: a reader could observe a half-written file — `evaluate --from-run` and `train`'s own Phase 4 write the same `eval*.txt` paths and can overlap when a run directory is scored while the run is still in progress — and a write that was interrupted or failed left a truncated file behind rather than the previous, complete one.

These now write to a sibling `.tmp` and rename over the destination, so it only ever holds a complete file. A failed open or write removes the temp and leaves the original intact. `rename` falls back to a copy if it fails — a different device, or a Windows sharing violation from a concurrent reader — rather than silently discarding a successful write.

Covered: `evaluate`'s eight artifacts (`eval.txt`, `eval_SPS_SPP.txt`, `eval_gene_predictions.txt`, the grouped and by-gene variants, and the three `drphylo` aggregate outputs), and the four visualizer SVG writers, which `evaluate` invokes into the same directories. The `.err` conversion sidecar is deliberately excluded — it is a failure marker written immediately before a throw, not an artifact anyone reads concurrently.

Outputs are byte-identical to previous versions; only the manner of publication changed.

## Bug fix — `evaluate` ignored everything after the first comma in a list line
`preprocess` treats a comma-separated list line as one overlapping group, but `evaluate` pushed the whole line as a single path. Every file after the first comma was therefore invisible to `evaluate`, and the model's own alignments looked missing:

```
Error: 4 alignment(s) referenced in the model are missing from the list
```

The parser had been comma-blind for some time, but it only became fatal once `train` began scoring its models automatically: an overlapping-list run now did all of the solving and then exited 1 during Phase 4. Standalone `evaluate` on an overlapping list was equally broken. The `preprocess` tokenizer is now shared by both of `evaluate`'s list readers, so `evaluate`'s view of the list matches the one the model was built from.

**Full Changelog**: https://github.com/kumarlabgit/MyESL2/compare/v0.2.8...dev-v0.2.81
