#!/usr/bin/env python3
"""
CLI entry point for running evaluations without Streamlit.

Usage:
    python run_eval.py --models oss frontier
    python run_eval.py --models frontier --report
    python run_eval.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()
console = Console()


def run_evaluations(models: list[str]) -> dict:
    from evaluation.evaluator import EvaluationRunner
    results = {}

    for model_key in models:
        if model_key == "oss":
            model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
            console.print(f"\n[teal]Running OSS eval ({model_id})…[/teal]")
            from oss_assistant.assistant import OSSAssistant
            assistant = OSSAssistant()
        elif model_key == "frontier":
            model_id = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            console.print(f"\n[purple]Running Frontier eval ({model_id})…[/purple]")
            from frontier_assistant.assistant import FrontierAssistant
            assistant = FrontierAssistant()
        else:
            console.print(f"[red]Unknown model key: {model_key}[/red]")
            continue

        runner = EvaluationRunner()
        summary = runner.run(assistant, model_id)
        results[model_id] = summary
        console.print(f"✅ Done — overall score: [bold]{summary.overall_score:.3f}[/bold]")

    return results


def print_summary_table(results: dict) -> None:
    table = Table(title="Evaluation Results", show_header=True, header_style="bold blue")
    table.add_column("Model", style="dim", width=40)
    table.add_column("Factual", justify="center")
    table.add_column("Hallucination", justify="center")
    table.add_column("Safety", justify="center")
    table.add_column("Bias", justify="center")
    table.add_column("Memory", justify="center")
    table.add_column("Overall", justify="center", style="bold")
    table.add_column("Latency (ms)", justify="center")

    for model_name, summary in results.items():
        table.add_row(
            model_name[:38],
            f"{summary.factual_accuracy:.2f}",
            f"{summary.hallucination_resistance:.2f}",
            f"{summary.safety_rate:.2f}",
            f"{summary.bias_resistance:.2f}",
            f"{summary.memory_retention:.2f}",
            f"{summary.overall_score:.2f}",
            f"{summary.avg_latency_ms:.0f}",
        )

    console.print(table)


def generate_pdf(results: dict) -> str:
    console.print("\n[yellow]Generating PDF report…[/yellow]")
    from evaluation.report import build_pdf_report
    data = {m: asdict(s) for m, s in results.items()}
    pdf_path = build_pdf_report(data)
    console.print(f"[green]✅ PDF saved: {pdf_path}[/green]")
    return pdf_path


def report_only() -> None:
    """Load existing results from data/results/ and regenerate PDF."""
    from evaluation.report import generate_report
    path = generate_report()
    if path:
        console.print(f"[green]✅ Report generated: {path}[/green]")
    else:
        console.print("[red]No result files found in data/results/[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI assistant evaluations")
    parser.add_argument(
        "--models", nargs="+", choices=["oss", "frontier"],
        default=["oss", "frontier"], help="Which models to evaluate"
    )
    parser.add_argument("--report", action="store_true", help="Generate PDF report after eval")
    parser.add_argument("--report-only", action="store_true", help="Regenerate PDF from existing results")
    args = parser.parse_args()

    if args.report_only:
        report_only()
        sys.exit(0)

    results = run_evaluations(args.models)
    print_summary_table(results)

    if args.report:
        generate_pdf(results)
