#!/usr/bin/env python3
"""Prove, empirically qualify, and seal one supported model deployment.

Hardware/compiler facts are observed, not command-line inputs. A sealed bundle
is not engine reachable until the checked deployment assembler accepts it.
"""

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.model_paths import supported_families


def preparation_architecture(family: str):
    if family not in supported_families():
        raise ValueError(f"unsupported model family: {family}")
    from scripts.deployment.common import DeploymentPreparationArchitecture

    candidate = importlib.import_module(f"scripts.deployment.model_families.{family}")
    deployment = importlib.import_module(f"vosti_kernels.model_families.{family}.deployment")
    return DeploymentPreparationArchitecture(
        family_label=candidate.CANDIDATE_ARCHITECTURE.family_label,
        report_schema=deployment.REPORT_SCHEMA,
        prepare_candidate=candidate.prepare_candidate,
        seal_candidate=deployment.seal_candidate,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=supported_families())
    parser.add_argument("model", help="local checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="new or empty output directory")
    args = parser.parse_args(argv)

    from scripts.deployment.common import prepare_and_seal_deployment

    bundle = prepare_and_seal_deployment(preparation_architecture(args.family), args.model, args.output)
    candidate = bundle["candidate"]
    print(f"[QUALIFIED, NOT ENGINE REACHABLE] {candidate['model']['catalog_name']}")
    print(f"[STATIC DEPLOYMENT] {args.output.resolve() / 'deployment.json'}")
    print(f"[BACKEND PROBES] {len(bundle['report']['results'])} passed on "
          f"{candidate['environment']['devices'][0]['name']}")


if __name__ == "__main__":
    main()
