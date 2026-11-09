"""
cli.py – Command-line interface for Portfolio Analytics Agent.

Usage:
    Interactive mode:  python -m src.cli
    One-shot mode:     python -m src.cli "How many portfolios do we have in total?"
"""

from __future__ import annotations

import logging
import sys

from src.agent import AgentResult, PortfolioAgent


def _print_result(result: AgentResult) -> None:
    """Pretty-print an agent result to stdout."""
    print()
    print(f"  Tool: {result.tool_used}")

    if result.tool_used == "sql_query" and result.tool_input:
        print(f"  SQL:  {result.tool_input}")
    elif result.tool_used == "exposure_calculator" and result.tool_input:
        params = result.tool_input
        if isinstance(params, dict):
            resolved = params.get("resolved_name", params.get("portfolio_name", ""))
            print(f"  Portfolio: {resolved}")

    print(f"  Time: {result.latency_ms:.0f}ms")
    print()

    if result.error and result.tool_used == "none":
        print(result.answer or result.error)
    elif result.error:
        print(f"  Error: {result.error}")
    else:
        print(result.answer)

    print()


def interactive() -> None:
    """Run the interactive CLI loop."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  Portfolio Analytics Agent")
    print("  Type your question, or 'exit' to quit.")
    print("=" * 60)
    print()

    agent = PortfolioAgent()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = agent.answer_question(question)
        _print_result(result)


def one_shot(question: str) -> None:
    """Answer a single question and exit."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    agent = PortfolioAgent()
    result = agent.answer_question(question)
    _print_result(result)


def main() -> None:
    """Entry point: interactive or one-shot based on CLI arguments."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        one_shot(question)
    else:
        interactive()


if __name__ == "__main__":
    main()

