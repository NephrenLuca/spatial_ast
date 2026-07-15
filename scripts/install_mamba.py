"""
Auto-install ``causal-conv1d`` and ``mamba-ssm`` for the current environment.

The script:
  1. Detects your python tag / torch version / CUDA major / C++ ABI.
  2. Queries the GitHub releases API for both projects.
  3. Picks the best-matching prebuilt wheel, downloads it, and pip-installs it
     (``causal-conv1d`` first, then ``mamba-ssm --no-deps``).
  4. Falls back to building from source (with the right env flags) when no
     compatible wheel exists — or always, with ``--build-from-source``.
  5. Verifies the install with a tiny CUDA forward pass.

Stdlib only (urllib/json/re/subprocess); no extra deps required to run it.

Examples
--------
    # auto: download matching wheels, else build from source
    python scripts/install_mamba.py

    # just download wheels into ./wheels, don't install
    python scripts/install_mamba.py --no-install --download-dir wheels

    # force a source build for A100 (+H100)
    python scripts/install_mamba.py --build-from-source --arch "8.0;9.0"

    # pin versions
    python scripts/install_mamba.py --causal-version 1.4.0 --mamba-version 2.2.2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── project metadata ──────────────────────────────────────────────────
CAUSAL_REPO = "Dao-AILab/causal-conv1d"
MAMBA_REPO = "state-spaces/mamba"

# local-version tag inside a wheel name, e.g. "cu122torch2.4cxx11abiFALSE"
_TAG_RE = re.compile(r"cu(\d+)torch(\d+)\.(\d+)cxx11abi(TRUE|FALSE)", re.IGNORECASE)
# full wheel: dist-version+localtag-pytag-abitag-plat.whl
_WHEEL_RE = re.compile(
    r"^(?P<dist>[\w.]+?)-(?P<ver>[\w.]+?)\+(?P<local>[^-]+)-"
    r"(?P<py>cp\d+)-(?P<abi>[\w]+)-(?P<plat>[\w]+)\.whl$"
)


def log(msg: str) -> None:
    print(f"[install-mamba] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[install-mamba] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Environment detection
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Env:
    py_tag: str          # cp310
    torch_major: int
    torch_minor: int
    cuda_major: int      # 12
    cuda_full: int       # 121  (12.1 -> 121), for tie-breaking
    abi: str             # "TRUE" | "FALSE"
    platform_tag: str    # linux_x86_64

    @property
    def torch_mm(self) -> str:
        return f"{self.torch_major}.{self.torch_minor}"


def detect_env() -> Env:
    try:
        import torch
    except ImportError:
        die("PyTorch is not installed in this interpreter. Install torch first.")

    if torch.version.cuda is None:  # type: ignore[union-attr]
        die("This torch build has no CUDA (torch.version.cuda is None). "
            "Install a CUDA build of torch before running this script.")

    tv = torch.__version__.split("+")[0].split(".")
    t_major, t_minor = int(tv[0]), int(tv[1])

    cu = torch.version.cuda  # e.g. "12.1"
    cu_major = int(cu.split(".")[0])
    cu_full = int(cu.replace(".", "")[:3].ljust(3, "0")[:3]) if cu else 0

    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    # normalise to the tag these projects ship (linux_x86_64)
    if "linux" in plat and ("x86_64" in plat or "amd64" in plat):
        plat = "linux_x86_64"

    env = Env(py_tag, t_major, t_minor, cu_major, cu_full, abi, plat)
    log(f"python={env.py_tag} torch={env.torch_mm} cuda={cu} "
        f"(major {env.cuda_major}) cxx11abi={env.abi} platform={env.platform_tag}")
    if env.platform_tag != "linux_x86_64":
        log("WARNING: prebuilt wheels are typically linux_x86_64 only; "
            "you will likely need --build-from-source.")
    return env


# ═══════════════════════════════════════════════════════════════════════
# GitHub releases API
# ═══════════════════════════════════════════════════════════════════════

def _api_get(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "spatial-ast-installer",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            die("GitHub API rate limit hit. Set GITHUB_TOKEN env var and retry.")
        die(f"GitHub API error {e.code} for {url}")
    except urllib.error.URLError as e:
        die(f"Network error contacting GitHub: {e}")
    return []


def list_release_assets(repo: str) -> List[Tuple[str, str, str]]:
    """Return [(release_tag, asset_name, download_url), ...] across all releases."""
    data = _api_get(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    out: List[Tuple[str, str, str]] = []
    for rel in data:  # type: ignore[union-attr]
        tag = rel.get("tag_name", "")
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".whl"):
                out.append((tag, name, asset["browser_download_url"]))
    return out


def latest_release_tag(repo: str) -> Optional[str]:
    try:
        data = _api_get(f"https://api.github.com/repos/{repo}/releases/latest")
        return data.get("tag_name")  # type: ignore[union-attr]
    except SystemExit:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Wheel matching
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WheelCandidate:
    name: str
    url: str
    version: Tuple[int, ...]
    cu: int
    torch_mm: Tuple[int, int]
    abi: str


def _parse_version(v: str) -> Tuple[int, ...]:
    parts = re.split(r"[^\d]+", v)
    return tuple(int(p) for p in parts if p != "")


def find_best_wheel(
    assets: List[Tuple[str, str, str]],
    env: Env,
    pin_version: Optional[str],
    allow_torch_mismatch: bool,
) -> Optional[WheelCandidate]:
    cands: List[WheelCandidate] = []
    for _tag, name, url in assets:
        m = _WHEEL_RE.match(name)
        if not m:
            continue
        if m.group("py") != env.py_tag:
            continue
        if m.group("plat") != env.platform_tag:
            continue
        tm = _TAG_RE.search(m.group("local"))
        if not tm:
            continue
        cu = int(tm.group(1))
        w_tmajor, w_tminor = int(tm.group(2)), int(tm.group(3))
        w_abi = tm.group(4).upper()

        if w_abi != env.abi:
            continue
        if cu // 10 != env.cuda_major:  # cu122 -> 12, cu118 -> 11
            continue
        if not allow_torch_mismatch and (w_tmajor, w_tminor) != (env.torch_major, env.torch_minor):
            continue
        ver = _parse_version(m.group("ver"))
        if pin_version and ver != _parse_version(pin_version):
            continue
        cands.append(WheelCandidate(name, url, ver, cu, (w_tmajor, w_tminor), w_abi))

    if not cands:
        return None

    # Prefer: exact torch minor > highest pkg version > CUDA minor closest to torch's
    def score(c: WheelCandidate):
        exact_torch = (c.torch_mm == (env.torch_major, env.torch_minor))
        cu_close = -abs(c.cu - env.cuda_full)
        return (exact_torch, c.version, cu_close)

    cands.sort(key=score, reverse=True)
    return cands[0]


# ═══════════════════════════════════════════════════════════════════════
# Download / install / build
# ═══════════════════════════════════════════════════════════════════════

def download(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    fname = url.split("/")[-1]
    path = os.path.join(dest_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log(f"already downloaded: {fname}")
        return path
    log(f"downloading {fname} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "spatial-ast-installer"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    log(f"saved -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def pip_install(args: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    log("run: " + " ".join(cmd))
    subprocess.check_call(cmd)


def build_from_source(repo: str, force_env: str, ref: Optional[str],
                      arch: str, max_jobs: int, extra: List[str]) -> None:
    tag = ref or latest_release_tag(repo) or "main"
    url = f"git+https://github.com/{repo}.git@{tag}"
    env = os.environ.copy()
    env[force_env] = "TRUE"
    env["MAX_JOBS"] = str(max_jobs)
    env["TORCH_CUDA_ARCH_LIST"] = arch
    cmd = [sys.executable, "-m", "pip", "install", "-v",
           "--no-build-isolation", url, *extra]
    log(f"building {repo}@{tag} from source (MAX_JOBS={max_jobs}, arch={arch})")
    log("run: " + " ".join(cmd))
    subprocess.check_call(cmd, env=env)


# ═══════════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════════

def verify() -> bool:
    code = (
        "import torch;"
        "from causal_conv1d import causal_conv1d_fn;"
        "from mamba_ssm import Mamba;"
        "m=Mamba(d_model=64,d_state=16,d_conv=4,expand=2).cuda().to(torch.bfloat16);"
        "x=torch.randn(2,32,64,device='cuda',dtype=torch.bfloat16);"
        "assert tuple(m(x).shape)==(2,32,64);"
        "print('mamba forward OK')"
    )
    log("verifying import + CUDA forward ...")
    try:
        subprocess.check_call([sys.executable, "-c", code])
        return True
    except subprocess.CalledProcessError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def handle_package(
    label: str,
    repo: str,
    force_env: str,
    pin_version: Optional[str],
    ref: Optional[str],
    install_extra: List[str],
    env: Env,
    args: argparse.Namespace,
) -> None:
    log(f"=== {label} ({repo}) ===")

    if args.build_from_source:
        build_from_source(repo, force_env, ref, args.arch, args.max_jobs, install_extra)
        return

    assets = list_release_assets(repo)
    wheel = find_best_wheel(assets, env, pin_version, args.allow_torch_mismatch)

    if wheel is None:
        log(f"no matching prebuilt wheel for {label}.")
        if args.no_install:
            die(f"--no-install set but no wheel found for {label}; "
                f"cannot download. Try --allow-torch-mismatch or build from source.")
        log("falling back to building from source ...")
        build_from_source(repo, force_env, ref, args.arch, args.max_jobs, install_extra)
        return

    log(f"selected wheel: {wheel.name}  (torch{wheel.torch_mm[0]}.{wheel.torch_mm[1]}, "
        f"cu{wheel.cu}, cxx11abi{wheel.abi})")
    path = download(wheel.url, args.download_dir)
    if not args.no_install:
        pip_install([path, *install_extra])


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-install causal-conv1d + mamba-ssm")
    ap.add_argument("--download-dir", default="wheels", help="where to save wheels")
    ap.add_argument("--no-install", action="store_true",
                    help="download wheels only, do not pip install")
    ap.add_argument("--build-from-source", action="store_true",
                    help="always build from source instead of using wheels")
    ap.add_argument("--allow-torch-mismatch", action="store_true",
                    help="accept a wheel built for a different torch minor "
                         "(same CUDA major + ABI); risky, verify afterwards")
    ap.add_argument("--arch", default="8.0",
                    help="TORCH_CUDA_ARCH_LIST for source builds (A100=8.0, H100=9.0)")
    ap.add_argument("--max-jobs", type=int, default=4,
                    help="MAX_JOBS for source builds (lower if OOM-killed)")
    ap.add_argument("--causal-version", default=None, help="pin causal-conv1d version")
    ap.add_argument("--mamba-version", default=None, help="pin mamba-ssm version")
    ap.add_argument("--causal-ref", default=None,
                    help="git ref (tag/sha) for causal-conv1d source build")
    ap.add_argument("--mamba-ref", default=None,
                    help="git ref (tag/sha) for mamba-ssm source build")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    env = detect_env()

    # causal-conv1d must be installed before mamba-ssm
    handle_package(
        "causal-conv1d", CAUSAL_REPO, "CAUSAL_CONV1D_FORCE_BUILD",
        args.causal_version, args.causal_ref, [], env, args,
    )
    # --no-deps so mamba doesn't pull a different causal-conv1d
    handle_package(
        "mamba-ssm", MAMBA_REPO, "MAMBA_FORCE_BUILD",
        args.mamba_version, args.mamba_ref, ["--no-deps"], env, args,
    )

    if args.no_install:
        log("done (download-only mode).")
        return

    if args.skip_verify:
        log("done (verification skipped).")
        return

    if verify():
        log("SUCCESS: causal-conv1d + mamba-ssm are installed and working.")
    else:
        die("packages installed but verification failed — likely an ABI/torch "
            "mismatch. Re-run with --build-from-source (needs nvcc), or "
            "--allow-torch-mismatch to try a different wheel.")


if __name__ == "__main__":
    main()
