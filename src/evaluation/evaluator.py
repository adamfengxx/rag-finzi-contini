"""
Step 10 — RAG evaluation with RAGAS.

Metrics evaluated:
  ┌─────────────────────────┬────────────────────────────────────────────┐
  │ Metric                  │ What it measures                           │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ faithfulness            │ Answer is grounded in the retrieved context│
  │ answer_relevancy        │ Answer actually addresses the question     │
  │ context_precision       │ Retrieved chunks are relevant to question  │
  │ context_recall          │ Retrieved chunks cover the ground truth    │
  └─────────────────────────┴────────────────────────────────────────────┘

Requires a small evaluation dataset (question + ground_truth pairs).
A starter set focused on the novel is provided below — expand as needed.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    AnswerRelevancy,
)
from ragas.llms import llm_factory
import openai
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

console = Console()

def _configure_ragas_metrics() -> list:
    """
    Instantiate RAGAS metrics using llm_factory (returns InstructorLLM).
    RAGAS 0.4.x collections metrics require InstructorLLM, not LangchainLLMWrapper.
    """
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    # max_tokens=4096 is for RAGAS internal scoring (e.g. Faithfulness statement
    # decomposition), completely separate from our RAG chain's MAX_TOKENS setting.
    llm    = llm_factory("gpt-4o-mini", client=client, max_tokens=4096)

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        LLMContextRecall(llm=llm),
        FactualCorrectness(llm=llm),
    ]

# ── Starter evaluation dataset (Italian Q&A about the novel) ─────────────────
EVAL_DATASET: list[dict[str, Any]] = [
    {
        "question": "Chi è Micol Finzi-Contini e qual è il suo ruolo nel romanzo?",
        "ground_truth":(
            "Micol Finzi-Contini è la figlia della ricca famiglia ebraica ferrarese "
            "dei Finzi-Contini. È il personaggio femminile centrale del romanzo, "
            "amata dal narratore, bella e sfuggente, simbolo di un mondo destinato "
            "a scomparire con le leggi razziali."
        ), 
    },
    {
        "question": "Dove si trova il giardino dei Finzi-Contini?",
        "ground_truth": (
            "Il giardino dei Finzi-Contini si trova a Ferrara, all'interno della "
            "grande proprietà di famiglia, la magna domus, circondata da mura alte "
            "e alberi secolari."
        ),
    },
    {
        "question": "Qual è il significato delle leggi razziali nel romanzo?",
        "ground_truth": (
            "Le leggi razziali del 1938 segnano lo spartiacque del romanzo: "
            "escludono gli ebrei dai club sportivi e dalle scuole pubbliche, "
            "e spingono i Finzi-Contini ad aprire il loro giardino agli amici "
            "ebrei per giocare a tennis."
        ),
    },
    {
        "question": "Che sport praticano i personaggi nel giardino?",
        "ground_truth": (
            "I personaggi giocano principalmente a tennis nel campo privato "
            "del giardino dei Finzi-Contini."
        ),
    },
    {
        "question": "Come si chiama il fratello di Micol?",
        "ground_truth": (
            "Il fratello di Micol si chiama Alberto Finzi-Contini. "
            "È malato e muore prima della deportazione."
        ),
    },
    {
        "question": "Chi è il narratore del romanzo?",
        "ground_truth": (
            "Il narratore è un giovane ebreo ferrarese innominato, voce narrante "
            "in prima persona, innamorato di Micol sin dall'infanzia."
        ),
    },
    {
        "question": "Qual è il destino finale della famiglia Finzi-Contini?",
        "ground_truth": (
            "La famiglia Finzi-Contini viene deportata dai nazisti durante la "
            "Seconda guerra mondiale e muore nei campi di sterminio."
        ),
    },
    {
        "question": "Cosa simboleggia il muro che circonda la proprietà dei Finzi-Contini?",
        "ground_truth": (
            "Il muro simboleggia l'isolamento volontario dei Finzi-Contini dal "
            "mondo esterno, ma anche l'illusione che possano proteggersi dalle "
            "persecuzioni antisemite rinchiudendosi nel loro mondo privilegiato."
        ),
    },
    {
        "question": "In che anno e di quale malattia muore Alberto Finzi Contini?",
        "ground_truth": "Alberto muore nel 1942 di linfogranuloma."
    },
    {
        "question": "Che percentuale di ebrei ferraresi si iscrive al Partito Fascista nel 1933? Il Professor Ermanno accetta la tessera del Partito o la rifiuta?",
        "ground_truth": "Il 90% degli ebrei ferraresi si iscrive al Partito Fascista ma il Professor Ermanno rifiuta di iscriversi."
    },
    {
        "question": "In quale materia viene rimandato a ottobre il protagonista?",
        "ground_truth": "Matematica."
    },
    {
        "question": "Quando Micòl invita il protagonista a scalare il muro per entrare nella proprietà dei Finzi Contini, il protagonista all’inizio rifiuta. Perché?",
        "ground_truth": "Perché non vuole abbandonare in strada la sua bicicletta nuova."
    },
    {
        "question": "Perché Alberto invita il protagonista a giocare a tennis a casa Finzi Contini?",
        "ground_truth": "Perché a causa delle leggi razziali gli ebrei di Ferrara erano stati esclusi dal Circolo del Tennis."
    },
    {
        "question": "Di quale città era originario e che lavoro svolgeva Giampiero Malnate?",
        "ground_truth": "Malnate era di Milano e faceva il chimico."
    },
        
]


def run_evaluation(
    retriever_fn,        # callable(question: str) -> list[Document]
    rag_chain,           # RAGChain instance with rewrite_query()/answer()
    dataset: list[dict] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation over the eval dataset.

    Args:
        retriever_fn: function that takes a question string and returns
                      a list of LangChain Documents (retrieved + reranked).
        rag_chain:    RAGChain instance for generating answers.
        dataset:      override the default EVAL_DATASET if provided.
        output_path:  if given, save results CSV here.

    Returns:
        DataFrame with per-question scores and aggregate means.
    """
    dataset = dataset or EVAL_DATASET
    metrics = _configure_ragas_metrics()
    console.print(
        f"[bold cyan]Running RAGAS evaluation[/bold cyan]  "
        f"({len(dataset)} questions)"
    )

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in dataset:
        q  = sample["question"]
        gt = sample["ground_truth"]

        retrieval_query = rag_chain.rewrite_query(q)
        docs            = retriever_fn(retrieval_query)
        answer          = rag_chain.answer(
            q,
            docs,
            retrieval_query=retrieval_query,
        )
        rag_chain.reset()       # fresh context for each eval question

        questions.append(q)
        answers.append(answer)
        contexts.append([d.page_content for d in docs])
        ground_truths.append(gt)

    hf_dataset = Dataset.from_dict(
        {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        hf_dataset,
        metrics=metrics,
    )

    df = result.to_pandas()

    # ── Pretty-print aggregate scores ─────────────────────────────────────────
    table = Table(title="[bold]RAGAS Evaluation Results[/bold]", show_header=True)
    table.add_column("Metric",    style="cyan")
    table.add_column("Mean",      style="bold green", justify="right")
    table.add_column("Min",       justify="right")
    table.add_column("Max",       justify="right")

    for metric in ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]:
        if metric in df.columns:
            col = df[metric].dropna()
            table.add_row(
                metric,
                f"{col.mean():.3f}",
                f"{col.min():.3f}",
                f"{col.max():.3f}",
            )
    console.print(table)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        console.print(f"[dim]Detailed results saved to {output_path}[/dim]")

    return df
