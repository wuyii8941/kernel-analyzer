from scripts.run_adamw8bit_full_model_block_probe import draws


def test_followup_uses_new_deterministic_iid_histories():
    assert draws(8192) == draws(8192)
    assert len(draws(8192)) == 16
    assert all(len(unit) == 8 for unit in draws(8192))
