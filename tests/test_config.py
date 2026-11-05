"""配置加载单测。"""

from src.common.config import ROOT, get_section, load_config, resolve_config_path


def test_load_default_config():
    cfg = load_config(reload=True)
    assert isinstance(cfg, dict)
    assert "paths" in cfg
    assert "pipeline" in cfg


def test_resolve_default_path():
    path = resolve_config_path()
    assert path == ROOT / "configs" / "default.json"
    assert path.exists()


def test_get_section_pipeline():
    sec = get_section("pipeline")
    assert isinstance(sec, dict)
    assert "lookback_days" in sec or "weight_method" in sec
