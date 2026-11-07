# Add label-informed encoders to the per-pump experiment

User explicitly authorizes this extension to the existing approved structure. Execute inline, using the existing runtime and tests; no separate design approval is required under the session's authorization rules.

- [x] Add tests for training-only annotation, ambiguous overlaps, unlabeled pumps, contrastive state sensitivity and label-free32D inference.
- [x] Implement separate label module and training runner; preserve baseline source/model hashes, use reconstruction +0.1supervised contrastive loss, select checkpoints by validation reconstruction.
- [x] Train eligible pumps and integrate per-pump comparisons, availability, selected embeddings and split exports.
- [x] Update explorer for variable model availability, label-use disclosure, optional fault-label coloring and six-model comparisons.
- [x] Verify all saved artifacts, inference, no label/future leakage; review implementation independently; inspect plots and browser controls.
- [x] Update report/provenance and leave the new model selected in the explorer.

Review focus: original benchmark test pumps have no annotations (never infer healthy); events after training must not change training states; overlapping ambiguous events invalidate supervision; validation labels must not guide checkpoint/model selection; cached unsupervised results remain unchanged.
