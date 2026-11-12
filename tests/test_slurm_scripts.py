"""Check editable scheduler settings without contacting Slurm or loading data."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bash():
    if os.name == "nt":
        candidates = [Path.home() / "scoop/apps/git/current/bin/bash.exe",
                      Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"]
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is unavailable")
    return executable


@pytest.mark.parametrize("script", sorted([*ROOT.glob("slurm/*.sh"), *ROOT.glob("slurm/*.slurm")]))
def test_bash_syntax(bash, script):
    result = subprocess.run([bash, "--noprofile", "--norc", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert b"\r" not in script.read_bytes(), "Cluster scripts require LF line endings"


@pytest.mark.parametrize("script", ["launch.sh", "launch_paper.sh", "preflight.sh", "paper_preflight.sh"])
def test_missing_account_stops_before_other_work(bash, tmp_path, script):
    folder = tmp_path / "slurm"
    folder.mkdir()
    for name in ("env.sh", script):
        shutil.copyfile(ROOT / "slurm" / name, folder / name)
    result = subprocess.run([bash, str(folder / script)], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "fill ACCOUNT and PARTITION in slurm/env.sh" in result.stderr
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("script", ["launch.sh", "launch_paper.sh"])
@pytest.mark.parametrize("optional", [False, True])
def test_dry_run_uses_script_fields_and_optional_flags(bash, tmp_path, script, optional):
    folder = tmp_path / "checkout with spaces" / "slurm"
    folder.mkdir(parents=True)
    text = (ROOT / "slurm/env.sh").read_text()
    text = text.replace('ACCOUNT=""', 'ACCOUNT="example_account"')
    text = text.replace('PARTITION=""', 'PARTITION="example_partition"')
    if optional:
        text = text.replace('QOS=""', 'QOS="example_qos"')
        text = text.replace('MAIL_USER=""', 'MAIL_USER="researcher@example.invalid"')
    (folder / "env.sh").write_text(text, newline="\n")
    shutil.copyfile(ROOT / "slurm" / script, folder / script)
    for name in ("preflight.sh", "paper_preflight.sh"):
        (folder / name).write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
    args = ["--dry-run", "--smoke-only"] if script == "launch.sh" else ["--dry-run", "--no-slurm"]
    result = subprocess.run([bash, str(folder / script), *args], cwd=folder.parent,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    command = result.stdout + result.stderr
    assert "--account=example_account" in command
    assert "--partition=example_partition" in command
    assert ("--qos=example_qos" in command) == optional
    assert ("--mail-user=researcher@example.invalid" in command) == optional
    assert ("--mail-type=FAIL" in command) == optional
    if not optional:
        assert "--qos=" not in command and "--mail-user=" not in command
