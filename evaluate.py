"""
evaluate.py — Run RAGAS evaluation against the built-in Q&A dataset.

Run:
    python evaluate.py
    python evaluate.py --output results/eval_results.csv
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.rule import Rule

import config
from src.vectorstore.indexer  import load_vectorstore
from src.retrieval.retriever  import RerankRetriever
from src.generation.chain     import RAGChain
from src.evaluation.evaluator import run_evaluation

console = Console()


def main(output: str | None = None) -> None:
    console.print(Rule("[bold]Il Giardino dei Finzi-Contini — RAG Evaluation[/bold]"))

    vectorstore = load_vectorstore()
    retriever   = RerankRetriever(vectorstore)
    chain       = RAGChain()

    output_path = Path(output) if output else config.DATA_DIR / "eval_results.csv"

    run_evaluation(
        retriever_fn=retriever.retrieve,
        rag_chain=chain,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save CSV results (default: data/eval_results.csv)")
    args = parser.parse_args()
    main(output=args.output)
