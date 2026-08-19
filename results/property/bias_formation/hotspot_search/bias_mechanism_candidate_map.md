# Bias mechanism candidate map

This is an F+B discovery map, not a scientific verdict. Short-screen ratios only
select cases for disjoint confirmation; old T1--T4 and SEUP labels are not inputs.

## Confirmed sensitivity anchor

- Phi seq64 `lm_head dX`: `PARAMETER_GRADIENT`.
- Local ratio: 0.1056 / 0.1067 (calibration / confirmation).
- Gradient ratio: 0.6676 / 0.6754.

## Current candidate funnel

- Screened task-coordinate rows currently present: 751.
- Engineering candidates with gradient ratio >= 0.1: 69.
- Newly confirmed stable bias cases in this rescreen: 0.

## Leading unconfirmed or rejected signals

| model | seq | family | signature | local | gradient | confirmation |
|---|---:|---|---|---:|---:|---|
| mamba | 256 | STATE_SPACE_RECURRENT_BACKWARD | SOURCE_DIRECTIONAL | 0.561 | 0.561 | NOT_PROMOTED |
| deepseek8b | 256 | NORMALIZATION_BACKWARD | TRANSPORT_AMPLIFIED | 0.0309 | 0.536 | NOT_REPRODUCED |
| qwen | 64 | NORMALIZATION_BACKWARD | TRANSPORT_AMPLIFIED | 0.0736 | 0.516 | DIRECTION_REVERSAL |
| deepseek8b | 128 | NORMALIZATION_BACKWARD | SOURCE_DIRECTIONAL | 0.445 | 0.445 | SAME_SIGN_UNRESOLVED_OR_CENTERED |
| mamba | 256 | STATE_SPACE_RECURRENT_BACKWARD | SOURCE_DIRECTIONAL | 0.174 | 0.438 | NOT_PROMOTED |
| qwen | 128 | NORMALIZATION_BACKWARD | TRANSPORT_AMPLIFIED | -0.0101 | 0.425 | SAME_SIGN_UNRESOLVED_OR_CENTERED |
| deepseek8b | 256 | NORMALIZATION_BACKWARD | SOURCE_DIRECTIONAL | 0.138 | 0.425 | SAME_SIGN_UNRESOLVED_OR_CENTERED |
| mamba | 256 | STATE_SPACE_RECURRENT_BACKWARD | TRANSPORT_AMPLIFIED | 0.0228 | 0.41 | NOT_PROMOTED |
| qwen | 64 | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | TRANSPORT_AMPLIFIED | 0.0417 | 0.399 | DIRECTION_REVERSAL |
| mamba | 256 | STATE_SPACE_RECURRENT_BACKWARD | SOURCE_DIRECTIONAL | 0.37 | 0.37 | NOT_PROMOTED |
| phi4 | 128 | NORMALIZATION_BACKWARD | SOURCE_DIRECTIONAL | 0.368 | 0.368 | SAME_SIGN_UNRESOLVED_OR_CENTERED |
| mamba | 64 | STATE_SPACE_RECURRENT_BACKWARD | TRANSPORT_AMPLIFIED | -0.0118 | 0.358 | DIRECTION_REVERSAL |
| qwen | 64 | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | TRANSPORT_AMPLIFIED | -0.0114 | 0.357 | NOT_REPRODUCED |
| phi4 | 256 | ATTENTION_PROJECTION_BACKWARD | TRANSPORT_AMPLIFIED | 0.034 | 0.356 | NOT_REPRODUCED |
| phi4 | 256 | ATTENTION_PROJECTION_BACKWARD | TRANSPORT_AMPLIFIED | 0.033 | 0.356 | NOT_REPRODUCED |
| qwen | 64 | LOSS_CE_BACKWARD | TRANSPORT_AMPLIFIED | 0.0326 | 0.35 | DIRECTION_REVERSAL |
| qwen | 64 | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | TRANSPORT_AMPLIFIED | -0.00549 | 0.341 | NOT_REPRODUCED |
| qwen | 64 | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | TRANSPORT_AMPLIFIED | -0.0105 | 0.339 | NOT_REPRODUCED |
| qwen | 64 | NORMALIZATION_BACKWARD | SOURCE_DIRECTIONAL | 0.328 | 0.328 | NOT_REPRODUCED |
| mamba | 128 | STATE_SPACE_RECURRENT_BACKWARD | TRANSPORT_AMPLIFIED | -0.000419 | 0.326 | NOT_PROMOTED |

## Mechanistic contrast emerging from the screen

- Internal Qwen/DeepSeek normalization reduction differences are erased at the next
  reduction/BF16 writeback boundary.
- Qwen and Mamba loss-head differences reach parameter gradients but remain centered
  across states and shapes.
- Phi seq64, unlike Phi seq128, combines boundary survival with directional backward
  transport. Shape/reduction geometry is therefore part of the case, not merely the
  `mm` operator name.
- A stable feature claim still requires additional confirmed positives.
