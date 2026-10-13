import pytest

from host_your_own_llm.errors import CliError
from host_your_own_llm.project import find_project_root


def test_project_root_from_subfolder(tmp_path, monkeypatch):
    (tmp_path / "docker-compose.yml").write_text("")
    (tmp_path / "a" / "b").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "a" / "b")
    assert find_project_root() == tmp_path


def test_project_root_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CliError, match="inside the host-your-own-llm"):
        find_project_root()
