"""
Step 8+9 — RAG generation chain (LangChain + OpenAI).

Two-stage conversational RAG:

  Stage 1 — Query rewriting
    history (human turns only) + current question
      → LLM rewrites into a self-contained retrieval query
      → resolves pronouns, fills ellipsis, adds literary context

  Stage 2 — Answer generation
    rewritten query → retriever → top-k docs
    system(context) + original question + rewritten query → LLM → answer

  Old assistant answers are NEVER fed back to the model.
  History is used only to understand what the user is referring to,
  never as evidence for the new answer.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

console = Console()

# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """Sei un assistente che riformula domande di ricerca letteraria.

Ti viene fornita una conversazione parziale (solo le domande dell'utente, senza risposte)
e una nuova domanda che può contenere pronomi, riferimenti o ellissi.

Riformula la domanda in modo che sia **autonoma e completa**, adatta a una ricerca
semantica nel romanzo "Il giardino dei Finzi-Contini" di Giorgio Bassani.

Se la domanda è semplice, la risposte deve essere breve.

Regole:
- Risolvi i pronomi ("lui", "lei", "loro", "quella casa") usando il contesto storico.
- Espandi le ellissi ("e la sorella?" → "Chi è la sorella di Alberto Finzi-Contini?").
- Mantieni la domanda concisa (max 2 frasi).
- Restituisci SOLO la domanda riformulata, senza spiegazioni.
- Se la domanda è già autonoma, restituiscila invariata.
"""

_ANSWER_SYSTEM = """Sei un assistente letterario esperto del romanzo "Il giardino dei \
Finzi-Contini" di Giorgio Bassani (1962).

Rispondi **solo** in base ai seguenti brani del romanzo forniti come contesto.
Se la risposta non è ricavabile dai brani, dillo chiaramente senza inventare.
Cita sempre il capitolo di provenienza quando possibile (es. «Parte prima — 2»).
Rispondi in italiano, in modo preciso e letterariamente appropriato.

La cronologia della conversazione NON e` una fonte fattuale.
Se viene fornita una "domanda riformulata per il recupero", usala solo per
risolvere pronomi, ellissi e riferimenti impliciti della domanda corrente.
Non trattarla come una prova in piu': i fatti ammessi devono provenire solo
dai brani nel contesto qui sotto.

=== CONTESTO (brani recuperati per questa domanda) ===
{context}
======================================================
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        chapter = meta.get("chapter", "?")
        part    = meta.get("part")
        label   = chapter + (f" — {part}" if part else "")
        parts.append(f"[{i}] {label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Main class ────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Two-stage conversational RAG chain.

    Usage:
        chain = RAGChain()

        # Stage 1: rewrite for retrieval
        retrieval_query = chain.rewrite_query(question)

        # Stage 2: retrieve externally, then generate answer
        docs   = retriever.retrieve(retrieval_query)
        answer = chain.answer(question, docs, retrieval_query)
    """

    MAX_HISTORY_TURNS = 4   # past human questions kept for rewriting

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=config.CHAT_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            api_key=config.OPENAI_API_KEY,
        )
        self._human_history: list[str] = []
        console.print(f"[green]LLM ready:[/green] {config.CHAT_MODEL}")

    # ── Stage 1: query rewriting ──────────────────────────────────────────────

    def rewrite_query(self, question: str) -> str:
        """
        Rewrite *question* into a self-contained retrieval query.
        Uses recent human history to resolve coreferences.
        Returns the rewritten query string.
        """
        # If no history, no rewriting needed
        if not self._human_history:
            return question

        history_block = "\n".join(
            f"- {q}" for q in self._human_history[-self.MAX_HISTORY_TURNS:]
        )
        user_msg = (
            f"Domande precedenti dell'utente:\n{history_block}\n\n"
            f"Nuova domanda: {question}"
        )

        messages = [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=user_msg),
        ]

        rewritten = self.llm.invoke(messages).content.strip()

        if rewritten != question:
            console.print(
                f"[dim]Query rewritten:[/dim]\n"
                f"  [dim]original :[/dim] {question}\n"
                f"  [dim]rewritten:[/dim] [italic]{rewritten}[/italic]"
            )
        return rewritten

    # ── Stage 2: answer generation ────────────────────────────────────────────

    def answer(
        self,
        question: str,
        docs: list[Document],
        retrieval_query: str | None = None,
    ) -> str:
        """
        Generate an answer grounded strictly in *docs*.
        The model receives the original user question plus an optional rewritten
        retrieval query used only to resolve references.
        No previous assistant answers are included.
        After generating, stores the human question for future rewriting.
        """
        context = _format_context(docs)
        rewritten = retrieval_query or question
        user_msg = (
            f"Domanda utente originale:\n{question}\n\n"
            f"Domanda riformulata per il recupero del contesto:\n{rewritten}\n\n"
            "Rispondi alla domanda originale usando solo i brani forniti sopra. "
            "Usa la domanda riformulata soltanto per chiarire eventuali riferimenti."
        )
        messages = [
            SystemMessage(content=_ANSWER_SYSTEM.format(context=context)),
            HumanMessage(content=user_msg),
        ]

        answer_text = self.llm.invoke(messages).content

        # Store only the human question for coreference in future turns
        self._human_history.append(question)

        return answer_text

    def reset(self) -> None:
        self._human_history = []
        console.print("[dim]Conversation history cleared.[/dim]")
