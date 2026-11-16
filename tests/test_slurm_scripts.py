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


def test_untracked_env_overrides_blanks_and_omits_empty_gres(bash, tmp_path):
    folder = tmp_path / "slurm"
    folder.mkdir()
    tracked = (ROOT / "slurm/env.sh").read_text()
    assert 'ACCOUNT=""' in tracked and "unicas.it" not in tracked
    shutil.copyfile(ROOT / "slurm/env.sh", folder / "env.sh")
    shutil.copyfile(ROOT / "slurm/launch_paper.sh", folder / "launch_paper.sh")
    (folder / "paper_preflight.sh").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
    (tmp_path / ".env").write_text(
        "\n".join([
            'ACCOUNT="example_account"',
            'PARTITION="example_partition"',
            'QOS="normal"',
            'MAIL_USER="researcher@example.invalid"',
            'MAIL_TYPE="FAIL,END"',
            'PAPER_TIME="12:00:00"',
            'GRES=""',
            'CPUS="8"',
            'MEM="64G"',
            "",
        ]),
        newline="\n",
    )
    result = subprocess.run(
        [bash, str(folder / "launch_paper.sh"), "--dry-run", "--no-slurm"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = result.stdout + result.stderr
    assert "--account=example_account" in command
    assert "--partition=example_partition" in command
    assert "--qos=normal" in command
    assert "--mail-type=FAIL\\,END" in command or "--mail-type=FAIL,END" in command
    assert "--mail-user=researcher@example.invalid" in command
    assert "--time=12:00:00" in command
    assert "--gres" not in command
    assert "GRES=" in command
    assert 'ACCOUNT=""' in (folder / "env.sh").read_text()


@pytest.mark.parametrize("script", ["run_paper.slurm", "run_block.slurm"])
def test_cluster_jobs_require_gpu_only_when_gres_is_set(script):
    text = (ROOT / "slurm" / script).read_text()
    assert 'if [ -n "${GRES:-}" ]; then' in text
    assert "export CISSN_REQUIRE_GPU=1" in text
    assert "unset CISSN_REQUIRE_GPU" in text
    assert "CUDA is not required" in text


@pytest.mark.skipif(os.name == "nt", reason="Uses a POSIX executable stub")
def test_cpu_paper_job_hides_cuda_and_clears_inherited_gpu_requirement(bash, tmp_path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\n'
                      'printf "CUDA_VISIBLE_DEVICES=<%s> CISSN_REQUIRE_GPU=<%s>\\n" '
                      '"${CUDA_VISIBLE_DEVICES-unset}" "${CISSN_REQUIRE_GPU-unset}"\n', newline="\n")
    python.chmod(0o755)
    env = {**os.environ, "ROOT": str(tmp_path), "PAPER_CONFIG": "unused.yaml", "GRES": "",
           "CUDA_VISIBLE_DEVICES": "0", "CISSN_REQUIRE_GPU": "1"}
    result = subprocess.run([bash, str(ROOT / "slurm/run_paper.slurm")], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("CUDA_VISIBLE_DEVICES=<> CISSN_REQUIRE_GPU=<unset>") == 3
    assert "End UTC:" in result.stdout


def test_cells_omit_require_gpu_without_gres():
    import importlib.util
    import sys
    cwd = Path.cwd()
    spec = importlib.util.spec_from_file_location("cissn_cells", ROOT / "slurm/cells.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    os.environ.pop("GRES", None)
    try:
        spec.loader.exec_module(module)
        assert module._gpu_args() == []
        os.environ["GRES"] = "gpu:1"
        assert module._gpu_args() == ["--require_gpu"]
    finally:
        os.environ.pop("GRES", None)
        sys.modules.pop(spec.name, None)
        os.chdir(cwd)
