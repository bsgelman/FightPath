"""Central path resolver reading from configs/paths.yaml."""
from pathlib import Path
import yaml

_ROOT = Path(__file__).parents[3]  # UFC_Prediction_Model/
_cfg: dict | None = None


def _load() -> dict:
    global _cfg
    if _cfg is None:
        with open(_ROOT / "configs" / "paths.yaml") as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def root() -> Path:
    return _ROOT


def raw_scraper() -> Path:
    return _ROOT / _load()["raw"]["scraper"]


def raw_archive() -> Path:
    return _ROOT / _load()["raw"]["archive"]


def scraper_source() -> Path:
    return _ROOT / _load()["scraper_source"]


def archive_source() -> Path:
    return _ROOT / _load()["archive_source"]


def interim(key: str) -> Path:
    return _ROOT / _load()["interim"][key]


def processed(key: str) -> Path:
    return _ROOT / _load()["processed"][key]


def external_lines() -> Path:
    return _ROOT / _load()["external"]["lines"]


def external_market_lines() -> Path:
    return _ROOT / _load()["external"]["market_lines"]


def outputs_models() -> Path:
    return _ROOT / _load()["outputs"]["models"]


def outputs_models_prod() -> Path:
    p = _ROOT / _load()["outputs"]["models"] / "prod"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_reports() -> Path:
    return _ROOT / _load()["outputs"]["reports"]


def outputs_cv() -> Path:
    return _ROOT / _load()["outputs"]["cv"]


def upcoming_cards() -> Path:
    return _ROOT / _load()["upcoming_cards"]
