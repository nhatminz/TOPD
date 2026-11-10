import json
import os
import subprocess
import sys

import pytest

from b200_experiment.config import load_config
from b200_experiment.vllm_rollout import VLLMRolloutEngine
from topd.configuration import ROOT, launch_config


def test_common_pins_torchrun_to_selected_python():
    environment = dict(os.environ, PYTHON_BIN=sys.executable,
                       PYTHON_EXEC="/missing/foreign-venv/python")
    command = 'source "$1"; "$PYTHON_BIN" -c \'import json,os,sys; print(json.dumps([sys.executable,os.environ["PYTHON_EXEC"]]))\''
    result = subprocess.run(["bash", "-c", command, "bash", str(ROOT/"scripts/common_b200.sh")],
                            env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == [sys.executable, sys.executable]


def test_torchrun_workers_keep_selected_venv_with_stale_python_exec(tmp_path):
    worker = tmp_path/"worker.py"
    worker.write_text('import os,sys\nfrom pathlib import Path\n'
                      'Path(os.environ["TEST_OUTPUT_DIR"], os.environ["RANK"]).write_text(sys.executable)\n')
    environment = dict(os.environ, PYTHON_BIN=sys.executable,
                       PYTHON_EXEC="/missing/foreign-venv/python", TEST_OUTPUT_DIR=str(tmp_path))
    command = 'source "$1"; "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 "$2"'
    subprocess.run(["bash", "-c", command, "bash", str(ROOT/"scripts/common_b200.sh"), str(worker)],
                   env=environment, text=True, capture_output=True, check=True, timeout=60)
    assert [(tmp_path/str(rank)).read_text() for rank in range(2)] == [sys.executable]*2


@pytest.fixture
def engine(tmp_path):
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    instance = VLLMRolloutEngine(config, tmp_path, local_rank=1, world_size=2)
    yield instance
    instance.close()


def test_vllm_entrypoint_uses_trainer_python_regardless_of_path(engine, tmp_path, monkeypatch):
    # Execute the launcher against a tiny installed-console-entry-point stand-in.
    # This checks argument forwarding as well as interpreter selection.
    metadata = tmp_path/"vllm-0.17.1.dist-info"
    metadata.mkdir()
    (metadata/"METADATA").write_text("Name: vllm\nVersion: 0.17.1\n")
    (metadata/"entry_points.txt").write_text("[console_scripts]\nvllm = fake_vllm:main\n")
    (tmp_path/"fake_vllm.py").write_text('import json,sys\ndef main(): print(json.dumps([sys.executable,sys.argv[1:]]))\n')
    monkeypatch.setenv("PATH", "/unrelated-venv/bin")
    command = engine._server_command()
    assert command[:2] == [sys.executable, "-c"]
    assert command[3] == "serve"
    child_environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run(command, env=child_environment, text=True, capture_output=True, check=True)
    python, arguments = json.loads(result.stdout)
    assert python == sys.executable
    assert arguments == command[3:]
    assert arguments[arguments.index("--load-format")+1] == "dummy"
    assert json.loads(arguments[arguments.index("--weight-transfer-config")+1]) == {"backend": "ipc"}
    assert "--enable-sleep-mode" in arguments


def test_vllm_child_isolates_rendezvous_and_python_exec(engine, monkeypatch):
    for key, value in dict(CUDA_VISIBLE_DEVICES="2,5", RANK="1", LOCAL_RANK="1",
                           WORLD_SIZE="2", MASTER_PORT="12345", TORCHELASTIC_USE_AGENT_STORE="True",
                           PYTHON_EXEC="/foreign/python").items():
        monkeypatch.setenv(key, value)
    child = engine._server_environment()
    assert child["CUDA_VISIBLE_DEVICES"] == "5"
    assert child["PYTHON_EXEC"] == sys.executable
    assert child["VLLM_SERVER_DEV_MODE"] == "1"
    assert "RANK" not in child and "MASTER_PORT" not in child
    assert not any(name.startswith("TORCHELASTIC_") for name in child)


def test_vllm_failure_keeps_early_engine_cause_and_excludes_old_attempt(engine, tmp_path):
    path = tmp_path/"server.log"
    path.write_text("\nRollout server launch: old\nRuntimeError: obsolete cause\n"
                    "\nRollout server launch: current\n"
                    "(EngineCore) RuntimeError: CUDA out of memory: actual engine cause\n"
                    + "(APIServer) wrapper frame\n"*250
                    + "RuntimeError: Engine core initialization failed.\n")
    engine.process = type("FailedProcess", (), {"returncode": 1})()
    try:
        with pytest.raises(RuntimeError) as error:
            engine._raise_server_failure(path)
        assert "actual engine cause" in str(error.value)
        assert "Engine core initialization failed" in str(error.value)
        assert "obsolete cause" not in str(error.value)
    finally:
        engine.process = None


def test_rollout_diagnostic_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("ROLLOUT_VLLM_LOGGING_LEVEL", "INFO")
    monkeypatch.setenv("ROLLOUT_VLLM_STARTUP_TIMEOUT", "900")
    config = launch_config()
    assert config["rollout"]["vllm"]["logging_level"] == "INFO"
    assert config["rollout"]["vllm"]["startup_timeout"] == 900
