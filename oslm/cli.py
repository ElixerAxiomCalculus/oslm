from __future__ import annotations

import argparse
import subprocess
import sys

from oslm.config import ROOT


def _doctor(args: argparse.Namespace) -> int:
    from oslm.doctor import run_doctor

    return run_doctor(as_json=args.json)


def _ingest(args: argparse.Namespace) -> int:
    from oslm.ingest.run import run_ingest

    return run_ingest(force=args.force, web=not args.no_web, course=not args.no_course)


def _index(args: argparse.Namespace) -> int:
    from oslm.index.build import run_index

    return run_index(force=args.force, smoke=args.smoke)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from oslm.config import load_config

    api = load_config().api
    uvicorn.run("oslm.api.main:app", host=api["host"], port=args.port or api["port"], reload=args.reload)
    return 0


def _web(args: argparse.Namespace) -> int:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    return subprocess.call([npm, "run", "dev"], cwd=ROOT / "web")


def _sft(args: argparse.Namespace) -> int:
    from oslm.sft.run import run_sft

    return run_sft(step=args.step, force=args.force, limit=args.limit, until_done=args.until_done, csv_path=args.csv)


def _train(args: argparse.Namespace) -> int:
    from oslm.train.qlora import run_training

    return run_training(resume=args.resume, max_steps=args.max_steps)


def _export(args: argparse.Namespace) -> int:
    from oslm.train.export_gguf import run_export

    return run_export(force=args.force)


def _eval(args: argparse.Namespace) -> int:
    if args.suite == "build-testset":
        from oslm.eval.testset import build_testset

        return build_testset(force=args.force)
    from oslm.eval.run import run_eval

    return run_eval(setups=args.setups, suite=args.suite, force=args.force, limit=args.limit)


def _test(args: argparse.Namespace) -> int:
    from oslm.testing import run_all_tests

    return run_all_tests(integration=args.integration, frontend=not args.no_frontend, e2e=args.e2e)


def _clean(args: argparse.Namespace) -> int:
    from oslm.cleanup import run_clean

    return run_clean(dry_run=not args.yes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oslm", description="OSlm process-synchronization tutor")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="verify the environment")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_doctor)

    p = sub.add_parser("ingest", help="crawl sources.csv and load data/course")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--no-course", action="store_true")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("index", help="clean, chunk, embed and build the vector index")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true", help="only run the retrieval smoke evaluation")
    p.set_defaults(func=_index)

    p = sub.add_parser("serve", help="run the FastAPI backend")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_serve)

    p = sub.add_parser("web", help="run the Vite dev server")
    p.set_defaults(func=_web)

    p = sub.add_parser("sft", help="build the fine-tuning dataset")
    p.add_argument("--step", choices=["generate", "import", "verify", "format", "review", "all"], default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--until-done", action="store_true", help="keep generating across days of free-tier quota")
    p.add_argument(
        "--csv", help="CSV of externally written Q&A for --step import (columns: category,question,answer[,has_bug])"
    )
    p.set_defaults(func=_sft)

    p = sub.add_parser("train", help="QLoRA fine-tune")
    p.add_argument("--resume", help="checkpoint dir, or 'latest'")
    p.add_argument("--max-steps", type=int, help="cap steps (VRAM probing)")
    p.set_defaults(func=_train)

    p = sub.add_parser("export", help="merge adapter and register the fine-tuned model in Ollama")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_export)

    p = sub.add_parser("eval", help="evaluate setups")
    p.add_argument("--setups", nargs="+", default=["base", "base_rag", "ft", "ft_rag"])
    p.add_argument("--suite", choices=["testset", "probes", "all", "build-testset"], default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=_eval)

    p = sub.add_parser("test", help="run the full test suite")
    p.add_argument("--integration", action="store_true")
    p.add_argument("--no-frontend", action="store_true")
    p.add_argument("--e2e", action="store_true")
    p.set_defaults(func=_test)

    p = sub.add_parser("clean", help="remove caches and intermediate artifacts")
    p.add_argument("--yes", action="store_true", help="actually delete (default is dry run)")
    p.set_defaults(func=_clean)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
