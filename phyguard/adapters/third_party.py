from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    commit: str
    expected_files: tuple[str, ...]


REPOS = {
    "fluoresfm": RepoSpec(
        "fluoresfm",
        "https://github.com/qiqi-lu/fluoresfm.git",
        "772ac54b2090d4de343979b8ba221c4de541ff85",
        ("3_0_test_it2i.py", "models/unet_sd_c.py", "1_2_text_embedding.py"),
    ),
    "unimie": RepoSpec(
        "UniMIE",
        "https://github.com/Feynben/UniMIE.git",
        "b46975d7c6f8c2847b382f1d1e801a1060aa40a8",
        ("scripts/sample_x0_enhancement_MI.py", "scripts/guided_diffusion/script_util_x0_enhancement.py"),
    ),
    "caspi": RepoSpec(
        "CASPI",
        "https://github.com/JonghoLee0/CASPI.git",
        "0fe5024e6221f1470f8d4e3e48b346dd38a6fb12",
        ("main.m", "estimate_init_flux_LC.m", "estimate_init_flux_NLC.m"),
    ),
}


def check_repo(root: str | Path, key: str) -> list[str]:
    spec = REPOS[key]
    base = Path(root) / spec.name
    missing = [f for f in spec.expected_files if not (base / f).exists()]
    return missing
