# Short persistence Oracle v2

Version 2 is frozen before its revalidation run. It supersedes the v1
implementation only for future screening; existing v1 outputs remain
reproducible historical records.

The only change is the prefix gate. For a short path, the final amplification
must exceed the four-step warmup amplification. The old v1 rule compared the
last two late prefixes, which can reject a path that has already accumulated a
directional component but fluctuates in its final window.

The other gates are unchanged: the observed amplification must exceed the
per-state sign-flip 95th percentile, lag 1 must be positive, and at least two
positive lags must be present. A positive result remains a risk candidate and
requires exact matched F+B confirmation.

The first v1 Llama trajectory run is retained as a protocol-audit observation,
not relabeled as a v2 held-out result. Revalidation must use the frozen
`short_screen_protocol_v2.json` and the real ordered trajectory bank.

