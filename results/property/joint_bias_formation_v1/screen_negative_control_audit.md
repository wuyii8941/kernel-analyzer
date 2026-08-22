# Screen-negative control audit

This audit is mechanical and consumes existing backward rescreen artifacts.
It is not a new persistence verdict.

- reachable nonzero pool: **32**
- deterministic sample: **12**
- outcomes in pool: `{'NOT_REPRODUCED': 21, 'SAME_SIGN_UNRESOLVED_OR_CENTERED': 7, 'DIRECTION_REVERSAL': 4}`

| task | model | family | local ratio | gradient status | outcome |
|---|---|---|---:|---|---|
| `backward:1276:out_ptr0` | `deepseek8b` | `NORMALIZATION_BACKWARD` | 0.05803 | `CENTERED` | `NOT_REPRODUCED` |
| `backward:1136:out_ptr0` | `phi4` | `NORMALIZATION_BACKWARD` | 0.002898 | `CENTERED` | `SAME_SIGN_UNRESOLVED_OR_CENTERED` |
| `backward:1303:output_0` | `deepseek8b` | `NORMALIZATION_BACKWARD` | 0.0007788 | `CENTERED` | `DIRECTION_REVERSAL` |
| `backward:699:in_out_ptr0` | `qwen` | `ATTENTION_STATE_OR_TRANSPORT_BACKWARD` | 0.0008541 | `CENTERED` | `NOT_REPRODUCED` |
| `backward:1429:out_ptr0` | `qwen` | `ATTENTION_STATE_OR_TRANSPORT_BACKWARD` | 0.00285 | `UNRESOLVED_INSUFFICIENT_STATES` | `NOT_REPRODUCED` |
| `backward:675:output_0` | `qwen` | `ATTENTION_PROJECTION_BACKWARD` | 0.006476 | `UNRESOLVED_INSUFFICIENT_STATES` | `NOT_REPRODUCED` |
| `backward:868:out_ptr0` | `deepseek8b` | `NORMALIZATION_BACKWARD` | 0.0343 | `UNRESOLVED_INSUFFICIENT_STATES` | `NOT_REPRODUCED` |
| `backward:5602:out_ptr0` | `mamba` | `STATE_SPACE_RECURRENT_BACKWARD` | 1.705e-06 | `CENTERED` | `NOT_REPRODUCED` |
| `backward:606:out_ptr0` | `phi4` | `NORMALIZATION_BACKWARD` | 0.01208 | `CENTERED` | `NOT_REPRODUCED` |
| `backward:1874:out_ptr0` | `deepseek8b` | `ATTENTION_STATE_OR_TRANSPORT_BACKWARD` | 8.021e-05 | `CENTERED` | `DIRECTION_REVERSAL` |
| `backward:497:out_ptr0` | `phi4` | `NORMALIZATION_BACKWARD` | 0.08562 | `UNRESOLVED_INSUFFICIENT_STATES` | `NOT_REPRODUCED` |
| `backward:673:out_ptr0` | `deepseek8b` | `NORMALIZATION_BACKWARD` | 0.01058 | `CENTERED` | `SAME_SIGN_UNRESOLVED_OR_CENTERED` |

This is a deterministic screen-level reachable-negative audit. It is not a new 32-step consequence campaign and cannot estimate end-to-end persistence recall until these sampled rows receive the full trajectory protocol.
