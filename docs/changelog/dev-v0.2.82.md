# dev-v0.2.82 — 2026-08-20

The headline is `--resume`, which lets a crashed or aborted run pick up where it stopped. Most of the rest of this release is the crash-safety work that had to land first for resume to be trustworthy: if an interrupted run can leave a partial file that a later run silently accepts, then reusing anything it produced is unsound.

## New — `--resume` continues a crashed or aborted run
A large grid is expensive — a DrPhylo tree over ~960 alignments with an 81-point grid runs for hours across many clades — and a crash, an OOM kill, or a Ctrl-C threw all of it away. `--resume` reuses the models the output directory already completed, per `models.tsv`, and solves only the rest. It is accepted by `train`, `drphylo`, and `aim`, and inherited by adaptive sparsification.

What makes this sound is that **encode is deterministic**: class balancing uses a fixed seed and parallel workers fill disjoint, pre-computed column ranges, so two runs produce byte-identical `combined.map`, `alignment_table.txt`, feature dumps, and weights even under `--class-bal down --threads 8`. A resumed run therefore rebuilds the same feature matrix, and models solved before and after the interruption index the same columns. Were that not true, resume could not work this way at all.

A new `<output_dir>/run_manifest` records every setting that determines which models a run produces — method, lambda specification, folds, penalties, encode options, and content hashes of the list and hypothesis files — because `preprocess_config` carries only eight preprocess fields and `process_log.txt` is append-only free text with no stable format. Keys are classed by prefix:

- **`identity.*`** is compared, and a difference refuses the resume while naming exactly which keys changed. `--resume-force` overrides, for cases like a bare `touch` that moved an mtime without changing content.
- **`layout.*`** fingerprints the assembled matrix (`combined.map`, `alignment_table.txt`, sample and column counts). A difference is refused **unconditionally** — the columns moved, so old and new models index different features, and the result would be plausible-looking but invalid rather than merely suspicious. `--resume-force` cannot override it.
- **`info.*`** (thread count, memory ceiling, cache location) is recorded for provenance but never compared, since none of it changes a model.

The identity check runs *before* encode. That ordering is deliberate: encode overwrites `combined.map`, so checking afterwards would leave the directory disagreeing with the models already on disk even as the resume was refused. The layout check necessarily runs after.

Rows are then classified. `complete` with a whole weights file is reused; `complete` with the file missing is re-solved rather than trusted; `skipped` stays skipped, so a resumed run reaches the same shape an uninterrupted one would; `failed` is retried unless `--resume-skip-failed`; `pending` is solved. A lambda is only reused wholesale, so a CV lambda interrupted between folds re-solves all of them — the finished folds are simply rewritten with identical content.

`models.tsv` gains a **`GeneCount`** column, holding the non-zero gene count the solver reported for each model. Both the sequential `--min-groups` skip-ahead ratchet and the parallel post-hoc pruner need the gene counts of *earlier* grid points, and on resume those points are not re-solved; counting lines in `gss.txt` would only approximate it, whereas storing the value the solver actually produced is exact. Both grid loops now advance the ratchet through one shared helper, fed either by a fresh solve or a restored count, so a resumed run makes the same skip decisions.

For `drphylo` and `aim` the unit of reuse is the subdirectory. Both already run each clade or iteration in a self-contained directory with its own `preprocess_config` and `models.tsv`, so a settled one is skipped without entering encode or train at all. Two wrinkles were worth handling explicitly:

- `drphylo` still aggregates a skipped clade, because HSS is computed from that clade's per-lambda `eval_gene_predictions.txt` and `hss_summary.txt` must list every clade. A crash between the grid finishing and the evaluation loop completing leaves a settled `models.tsv` with no evaluation output at all, so a skipped clade regenerates whatever is missing — without re-solving a single model — before aggregating.
- `aim` needed no new state. It carries nothing between iterations, re-deriving each round's feature ranking from `bss_median.txt` or `lambda_0/weights.txt` on disk, so a settled iteration is skipped and the loop continues with the correct ranking.

A clade or iteration the interrupted run never reached has no `models.tsv` and simply runs fresh, rather than the missing manifest aborting the whole tree at the first unstarted clade. A directory that finished under *different* settings is still refused: the manifest check runs before the settled test, so completing is not a way to escape verification. Resuming a finished run is a clean no-op. `taskfile` accepts `resume`, `resume-force`, and `resume-skip-failed`.

## Bug fix — a truncated cache file was silently reused, corrupting the feature matrix
`read_pff_metadata` and `read_pnf_metadata` parsed the text header and computed `data_offset` without ever checking that the file actually held the payload the header advertised. A conversion killed partway leaves exactly that state, and the reuse predicates in `preprocess` and `evaluate` accepted it: the run reported `Skipped (done)`, encode read whatever bytes happened to be present, and the whole thing exited **0**.

Measured on a five-alignment run with one `.pff` truncated to half its size, the feature matrix came out **7820 columns instead of 8269**, with different weights, and nothing anywhere reported a problem. Both readers now validate `file_size >= data_offset + get_data_size()` and throw on a short file. No call site needed changing: every reuse check already sits inside a `try`/`catch` that treats a metadata failure as a cache miss, so a truncated file is simply re-converted.

This is the fix `--resume` most depends on. A resumed run re-runs `preprocess` over a cache that the crash may have left mid-write, so without it a resume would build a corrupted matrix and mix it with correctly-trained models.

## Behavior change — cache files and per-model training outputs are published atomically
dev-v0.2.81 moved `evaluate` and the visualizer onto write-to-temp-then-rename. This release extends that to the two remaining sets of writers.

**Cache conversion** handed the final cache path straight to the converter, so a crash mid-conversion left a partial `.pff` / `.pnf` / `.vnf` sitting at the name every reuse check looks for. Conversion now targets `<dst>.tmp` and renames into place, with a copy fallback if rename fails, and a failed conversion removes its temp so no partial file survives under any name. Together with the size validation above, a partial cache file is both prevented and detected.

**Per-model training outputs** — `weights.txt`, the fold and grouped variants, `gss`/`pss`/`oss` and their grouped forms, and `cv_predictions.txt` — also went out through a plain truncating write. The status ordering in `models.tsv` already kept readers safe, since a row only becomes `complete` after its stream closes, but that made `complete` the sole guarantor of a file's integrity and protected neither a concurrent reader nor a run interrupted between the write and the status flip. These are exactly the files a resumed run reuses without re-deriving, so each must be whole or absent.

The post-loop median files are deliberately left on plain writes: a crash there leaves them missing rather than partial, and a resumed run regenerates them wholesale from the per-lambda files. Outputs are byte-identical to previous versions; only the manner of publication changed.

## Bug fix — `--use-logspace` failed whenever one lambda was held fixed
```
--lambda-grid 0.1,0.3,0.1 0.1,0.1,0.1 --use-logspace
Error: Lambda file contains no valid pairs
```

In the projection step, the erase of values not `< vmax` ran *before* the guard that checks whether at least two values remain. For a single-point specification (`min == max`) nothing survives the filter, so the guard fired on a vector it had already emptied and the Cartesian product came out empty. Sweeping one lambda while pinning the other is ordinary usage, so log-spacing was unusable for a whole class of grids.

The anchorable values are now collected separately and committed only when at least two survive; otherwise the linear sweep passes through untouched. A genuine multi-point sweep that cannot be projected says so on stderr, while a deliberately pinned lambda stays quiet — that is normal usage, not a mistake. The documented anchoring is unchanged: a sweep still drops the value equal to `vmax` and log-spaces the remainder, bit for bit.

## Bug fix — a stale `.err` sidecar blocked its gene permanently
A failed conversion writes `<cache_dir>/<stem>.err`, and five call sites treated that file's mere existence as "never retry". Nothing tied the sidecar to the input that failed, and a successful conversion never removed one, so a sidecar outlived whatever caused it: move the inputs, or fix the file it complained about, and the affected gene stayed suppressed forever with no cure but deleting the sidecar by hand.

Sidecars now record `source_path=` as their first line and are honoured only while they still describe the current input — same path, and the source not modified since the failure. Sidecars predating this format carry no header and are treated as stale, so existing ones self-heal on the next run. A sidecar judged stale is deleted rather than re-read every run, and a successful conversion removes it.

The accompanying message in `evaluate` is also corrected. It printed `has a prior conversion error, skipping` even when a valid `.pff` existed and the gene scored perfectly well — alarming and unactionable, since the scores were in fact correct. The sidecar is now consulted only after the cache is found unusable, and the warning names the actual consequence: that gene will contribute nothing to the score.

## Bug fix — `use_logspace` was always logged as `true`
`process_log.txt` reported `use_logspace = true` on every run regardless of the flag. The call passed `cond ? "true" : "false"`, which is a `const char*`; pointer-to-`bool` is a standard conversion and outranks `const char*`-to-`std::string`, so the call bound to the `bool` overload with the value `(pointer != null)` — always true. Only the log line was affected; the lambda grid itself was always computed correctly.

The flag is now passed as a `bool`, and a `const char*` overload has been added to the logger so the next caller writing `param(k, cond ? "a" : "b")` gets the string it meant. That matters more than it did before: `run_manifest` values are *compared* rather than merely displayed, so a silently-wrong value would cause a spurious refusal to resume.

**Full Changelog**: https://github.com/kumarlabgit/MyESL2/compare/dev-v0.2.81...dev-v0.2.82
