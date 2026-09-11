from scripts import run_training_numerical_analysis as entrypoint


def test_delegated_commands_receive_repository_import_path(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(entrypoint.subprocess, "run", fake_run)
    entrypoint._run("placeholder.py", "--example")

    command, kwargs = calls[0]
    assert command[-1:] == ["--example"]
    assert command[1] == str(entrypoint.ROOT / "scripts" / "placeholder.py")
    assert kwargs["cwd"] == entrypoint.ROOT
    configured = kwargs["env"]["PYTHONPATH"].split(":")
    assert configured[:2] == [str(entrypoint.ROOT / "src"), str(entrypoint.ROOT)]
