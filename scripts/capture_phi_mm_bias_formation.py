#!/usr/bin/env python3
from bias_capture_adapter import run_adapter

if __name__ == "__main__":
    run_adapter("phi4_lm_head_dx_seq64", "capture_phi_mm_bias_formation")
