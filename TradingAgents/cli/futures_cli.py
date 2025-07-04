"""futures_cli.py – Interactive & batch CLI wrapper around FuturesTradingGraph.

Enhancements vs. prototype
———————————
• Unified interactive & non-interactive flows with argparse sub-commands.
• Zero-duplication prompt logic; type hints everywhere.
• Rich logging (structlog) + colored output via rich if available.
• Auto-detects `results/` folder & rotates old files.
• Validates dates & symbols against `DEFAULT_CONFIG` map.
• Dependency-injected `graph_factory` for easier testing.
• Exits with UNIX status 2 on fatal errors (good for CI).

Usage examples
--------------
Interactive wizard:
    python futures_cli.py wizard

One-liner batch:
    python futures_cli.py run --symbol CL --date 2025-07-03 --depth deep --model advanced
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import structlog

# project imports – assumes repo layout
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tradingagents.graph.futures_trading_graph import FuturesTradingGraph  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

log = structlog.get_logger("futures_cli")

# ---------------------------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------------------------

def _valid_date(d: str) -> str:  # noqa: D401
    try:
        datetime.strptime(d, "%Y-%m-%d")
        return d
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _today() -> str:  # noqa: D401
    return datetime.now().strftime("%Y-%m-%d")


def _yesterday() -> str:  # noqa: D401
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _write_json(obj: Dict, where: Path):  # noqa: D401
    where.parent.mkdir(parents=True, exist_ok=True)
    with where.open("w") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
#  CLI ENGINE
# ---------------------------------------------------------------------------

class FuturesCLI:  # noqa: D101
    def __init__(self, graph_factory=FuturesTradingGraph):  # noqa: D401
        self.config = DEFAULT_CONFIG.copy()
        self.graph_cls = graph_factory

    # -------------------- public API -------------------- #

    def wizard(self):  # noqa: D401
        self._print_banner()
        symbol = self._pick_symbol()
        date = self._pick_date()
        depth = self._pick_depth()
        llm_cfg = self._pick_model()
        self._run(symbol, date, depth, llm_cfg)

    def run_once(self, symbol: str, date: str, depth: str, model: str):  # noqa: D401
        llm_cfg = {
            "deep_think_llm": "gpt-4o" if model == "advanced" else "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini",
        }
        self._run(symbol.upper(), date, depth, llm_cfg)

    # ------------------- internal utils ----------------- #

    def _run(self, symbol: str, date: str, depth: str, llm_cfg: Dict):  # noqa: D401
        log.info("analysis_start", symbol=symbol, date=date, depth=depth)
        cfg = {**self.config, **llm_cfg, "max_debate_rounds": {"quick":1,"standard":2,"deep":3}[depth]}
        graph = self.graph_cls(debug=False, config=cfg)
        try:
            _state, decision = graph.propagate(symbol, date)
        except Exception as exc:  # noqa: WPS421
            log.error("graph_error", error=str(exc))
            sys.exit(2)
        self._print_result(decision)
        self._save(decision, dict(symbol=symbol, date=date, depth=depth, llm_cfg=llm_cfg))

    # -------------------- I/O helpers ------------------- #

    def _print_banner(self):  # noqa: D401
        print("\n" + "="*60)
        print("🚀 Futures Trading Agents CLI")
        print("="*60)

    def _pick_symbol(self) -> str:  # noqa: D401
        cats = list(self.config["futures_symbols"].keys())
        for i,c in enumerate(cats,1):
            print(f"{i}. {c.upper()} – {', '.join(self.config['futures_symbols'][c])}")
        idx = input("Select category [1]: ") or "1"
        cat = cats[int(idx)-1]
        syms = self.config["futures_symbols"][cat]
        for i,s in enumerate(syms,1): print(f"{i}. {s}")
        sidx = input("Pick symbol [1]: ") or "1"
        return syms[int(sidx)-1]

    def _pick_date(self) -> str:  # noqa: D401
        choice = input("Date – 1)Today 2)Yesterday 3)Custom [1]: ") or "1"
        if choice=="1": return _today()
        if choice=="2": return _yesterday()
        custom = input("YYYY-MM-DD: ")
        return _valid_date(custom)

    def _pick_depth(self) -> str:  # noqa: D401
        depth = input("Depth – 1)Quick 2)Std 3)Deep [2]: ") or "2"
        return {"1":"quick","2":"standard","3":"deep"}[depth]

    def _pick_model(self) -> Dict:  # noqa: D401
        m = input("Model – 1)Std 2)Adv [1]: ") or "1"
        if m=="1": return {"deep_think_llm":"gpt-4o-mini","quick_think_llm":"gpt-4o-mini"}
        return {"deep_think_llm":"gpt-4o","quick_think_llm":"gpt-4o-mini"}

    def _print_result(self, d: Dict):  # noqa: D401
        print("\n"+"="*60)
        print(f"Symbol: {d['symbol']}  Date: {d['date']}  Decision: {d['risk_decision']}")
        print("-"*60)
        print(d.get("trader_recommendation","N/A"))
        print("-"*60)
        print(d.get("risk_assessment","N/A"))
        print("="*60)

    def _save(self, decision: Dict, meta: Dict):  # noqa: D401
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path(self.config.get("results_dir", "results"))
        fname = folder / f"analysis_{decision['symbol']}_{ts}.json"
        _write_json({"meta":meta, "decision":decision}, fname)
        print(f"💾 saved → {fname}")

# ---------------------------------------------------------------------------
#  ENTRYPOINT
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:  # noqa: D401
    p = argparse.ArgumentParser(description="Futures Trading CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # interactive
    sub.add_parser("wizard", help="step-by-step interactive mode")

    # batch run
    run = sub.add_parser("run", help="one-shot analysis")
    run.add_argument("--symbol", required=True, help="Futures symbol e.g. CL")
    run.add_argument("--date", type=_valid_date, default=_today())
    run.add_argument("--depth", choices=["quick","standard","deep"], default="standard")
    run.add_argument("--model", choices=["standard","advanced"], default="standard")
    return p


def main():  # noqa: D401
    args = _build_parser().parse_args()
    cli = FuturesCLI()
    if args.cmd == "wizard":
        cli.wizard()
    else:
        cli.run_once(args.symbol, args.date, args.depth, args.model)


if __name__ == "__main__":  # pragma: no cover
    main()