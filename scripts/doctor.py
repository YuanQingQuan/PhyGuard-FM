from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import importlib.util
import json

from phyguard.adapters.third_party import REPOS, check_repo

root = Path(__file__).resolve().parents[1]
print("PhyGuard-FM doctor")
print("root:", root)
for mod in ["torch", "numpy", "scipy", "skimage", "yaml", "tifffile"]:
    print(f"python module {mod:10s}:", "OK" if importlib.util.find_spec(mod) else "MISSING")
for key, spec in REPOS.items():
    missing = check_repo(root / "third_party", key)
    print(f"repo {key:10s}:", "OK" if not missing else f"MISSING ({', '.join(missing[:3])})")
print("\nPinned revisions:")
print(json.dumps({k: v.commit for k, v in REPOS.items()}, indent=2))
