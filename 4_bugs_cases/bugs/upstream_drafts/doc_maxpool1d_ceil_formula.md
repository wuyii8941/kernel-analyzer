### 📚 The doc issue

The output-length formula for `ceil_mode=True` on the **2.10** page of `torch.nn.MaxPool1d` reads

L_out = ⌈ (L_in + 2·padding − dilation·(kernel_size − 1) − 1 + (stride − 1)) / stride ⌉ + 1

which double-counts the rounding (it is the floor formula with `stride - 1` added, then a ceiling on top). The `main` page has
`⌊ ... + (stride − 1) ⌋ + 1`, i.e. `⌈ (L_in + 2·padding − dilation·(kernel_size − 1) − 1) / stride ⌉ + 1`, which is what the
implementation computes. Example: L_in = 6, kernel_size = 2, stride = 2, padding = 0, dilation = 1 -> numerator 4: the 2.10 formula
gives ⌈5/2⌉ + 1 = 4, the implementation (and the main page) give 3.

Suggest backporting the corrected formula to the versioned 2.10 docs (or noting the erratum).

<!-- search record (for the submitter), 2026-10-07: the main page already has the corrected formula; no issue found for the 2.10 page. -->
