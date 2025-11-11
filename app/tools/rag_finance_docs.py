# app/tools/rag_finance_docs.py
"""
RAG sur tes documents financiers déjà indexés (FAISS).
Nécessite un retriever exposé par app.retriever.get_retriever().
"""
from langchain.tools import Tool

# On s'appuie sur ton module retriever existant
try:
    from rag.retriever import get_retriever
    _RETRIEVER = get_retriever() 
except Exception as e:
    _RETRIEVER = None
    _ERR = f"[RAG] Retriever indisponible: {e}"

def _rag_search_fn(query: str) -> str:
    if _RETRIEVER is None:
        return _ERR if '_ERR' in globals() else "Retriever non initialisé."
    try:
        docs = _RETRIEVER.invoke(query)  # v0.3: retriever.invoke renvoie list[Document]
        if not docs:
            return "Aucun passage pertinent trouvé dans le corpus."
        lines = []
        for i, d in enumerate(docs[:5], 1):
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_path") or meta.get("path") or "source_inconnue"
            lines.append(f"[{i}] {src}\n{d.page_content[:700]}{'...' if len(d.page_content)>700 else ''}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Erreur RAG: {e}"

search_financial_documents = Tool.from_function(
    func=_rag_search_fn,
    name="search_financial_documents",
    description="Recherche sémantique dans tes PDF/Docs financiers (RAG). Entrée: requête en français."
)
