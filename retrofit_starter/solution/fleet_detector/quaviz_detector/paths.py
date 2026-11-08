"""Paths for the fleet detector.

Layout:
  retrofit_starter/                              PACKAGE_ROOT / challenge root
    data/full/                                   official data
    solution/fleet_detector/                     this solution
      quaviz_detector/                           package
      results/  cache/
  build/sim/                                     simulated fleets (repo root)
"""
import os
from pathlib import Path

FLEET_ROOT = Path(__file__).resolve().parents[1]          # solution/fleet_detector
PACKAGE_ROOT = FLEET_ROOT.parents[1]                      # retrofit_starter
REPO_ROOT = PACKAGE_ROOT.parent
OFFICIAL_DATA = PACKAGE_ROOT / "data" / "full"
RESULTS = FLEET_ROOT / "results"
CACHE = FLEET_ROOT / "cache"
BUILD = Path(os.environ.get("QUAVIZ_BUILD", REPO_ROOT / "build"))
SIM_FLEETS = Path(os.environ.get("QUAVIZ_SIM", BUILD / "sim"))
PUBLIC_FLEET = PACKAGE_ROOT / "configs" / "fleet_public.yaml"
