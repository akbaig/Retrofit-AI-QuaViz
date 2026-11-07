"""CLI: python -m explainability --asset PUMP-08 --provider gemini"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))

from paths import SOURCE  # noqa: E402

from explainability.llm.factory import get_llm, load_dotenv  # noqa: E402
from explainability.pipeline import explain_and_advise  # noqa: E402


def main(argv=None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Explain a frozen-best alert with LLM + optional what-if")
    p.add_argument("--data", type=Path, default=SOURCE / "data" / "full")
    p.add_argument("--model", type=Path, default=FINAL / "model.joblib")
    p.add_argument("--policy", type=Path, default=FINAL / "policy.json")
    p.add_argument("--asset", required=True)
    p.add_argument("--alert-index", type=int, default=0)
    p.add_argument("--provider", choices=["gemini", "ollama", "template"], default=None)
    p.add_argument("--no-what-if", action="store_true")
    p.add_argument("--output", type=Path, help="Write full JSON card here")
    a = p.parse_args(argv)

    llm = get_llm(a.provider)
    card = explain_and_advise(
        data_dir=a.data,
        asset_id=a.asset,
        model_path=a.model,
        policy_path=a.policy,
        alert_index=a.alert_index,
        llm=llm,
        run_what_if=not a.no_what_if,
        challenge_root=SOURCE,
    )
    text = json.dumps(card, indent=2, default=str)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {a.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
