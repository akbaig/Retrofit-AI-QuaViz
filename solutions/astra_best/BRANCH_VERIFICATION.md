# Checks before publishing `astra-best`

Checked on 2026-11-07 in a separate checkout of `akbaig/Retrofit-AI-QuaViz`, based on main commit `596bfdbda8d1cdacfc4534223b20cb69d83f806f`.

The commands used the existing Python 3.12.14 environment. Its installed versions match every pin in `requirements.txt`; `python -m pip check` reported no broken requirements.

| Check | Result |
|---|---|
| `python -m pytest -q --ignore=tests/test_review.py -p no:cacheprovider` | 29 passed in 3.09 seconds |
| Saved-model validation inference against this checkout's data | 17 rows, exactly equal to the reference CSV |
| Official scorer on those new validation predictions | 95.02135167303905 |
| Saved-model public-test inference against this checkout's data | 43 rows, exactly equal to the reference CSV; unscored |
| Official scorer on the saved holdout predictions and labels | 63.50859435861514 |
| Original delivered file checksums | All 233 covered files match |
| Main README, solution README and training-guide relative links | All resolve |
| Independent documentation review | Command, result and portability review completed; two wording/setup corrections applied |

The model SHA256 remains:

```text
0a1884354fa22a4a6de33302a988c99e074b88a59c430f04f64aaa8d5453d983
```

The original challenge scorer, baseline, generator and required source inputs match the files used in the original experiment. No organizer source file was edited on this branch.

**Not rerun for this publication:** full model training, fresh holdout generation/inference, original-layout replay tests, or pip installation into a newly created environment. Those limits are separate from the successful saved-model replay and portable tests. The original experiment's full test and holdout records remain under `provenance/` and `results/`.

The two review corrections were to avoid implying a measured general timing advantage over traditional alarms, and to create `output/` before the standalone holdout scoring command.

Machine-readable results: [branch_verification.json](branch_verification.json).
