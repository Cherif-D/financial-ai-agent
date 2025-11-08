"""
Outil RAG (Retrieval-Augmented Generation) pour l'Agent.
Version avec les imports corrigés pour LangChain v0.2+
"""

import os
# --- MODIFICATION ICI ---
# Les outils de base (tool) et les prompts (PromptTemplate)
# ne sont plus dans "langchain", mais dans "langchain_core".
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

# Import de notre "Chercheur" (la fonction que nous avons testée)
from rag.retriever import get_retriever
# Importe le nom du modèle depuis notre configuration centrale
from app.config import MODEL_NAME

# --- 1. Définition des Composants de la Chaîne RAG ---
_prompt = PromptTemplate.from_template(
    "Tu es un analyste financier expert. Réponds **dans la même langue que la Question** en te basant UNIQUEMENT sur les extraits de documents fournis.\n"
    "Si la réponse n'est pas dans les extraits, dis simplement: "
    "'Je n'ai pas trouvé l'information dans les documents fournis.' (ou son équivalent dans la langue de la question).\n\n"
    "Question: {question}\n\n"
    "Extraits des documents:\n{context}\n\n"
    "Réponse sourcée et concise:"
)
_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
retriever = get_retriever(k=4)

def _format_docs(docs: list) -> str:
    """
    Petite fonction "helper" pour mettre en forme la liste
    des documents (chunks) en un seul bloc de texte
    que le LLM peut lire.
    """
    return "\n\n---\n\n".join([d.page_content for d in docs])

# --- 2. La "Chaîne de Montage" RAG (LCEL) LCEL c'est  LangChain Expression Language. ---
rag_chain = (
    RunnableMap({
        "context": (lambda x: x["question"]) | retriever | _format_docs,
        "question": (lambda x: x["question"])
    })
    | _prompt  
    | _llm     
    | StrOutputParser()
)

# --- 3. Définition de l'Outil ---
@tool
def search_financial_documents(query: str) -> str:
    """
    [C'EST LE MODE D'EMPLOI POUR L'AGENT]
    Utilise cet outil EXCLUSIVEMENT pour répondre aux questions
    concernant les rapports financiers (annuels, trimestriels 10-K, 10-Q),
    les résultats d'entreprise (revenus, bénéfices, ...), 
    la direction (CEO, membres du conseil), ou les rapports de 
    durabilité (ESG) qui sont stockés dans la base de données interne.
    
    Ne l'utilise PAS pour des informations en temps réel comme
    le cours de l'action ou les dernières actualités.
    """
    
    print(f"\n--- 🛠️ Outil RAG: Appel de search_financial_documents ---")
    print(f"\n--- 🛠️ Outil RAG: Question reçue: {query} ---")
    
    try:
        answer = rag_chain.invoke({"question": query})
        print(f"--- 🛠️ Outil RAG: Réponse générée: {answer} ---")
        return answer
    except Exception as e:
        print(f"--- 🛠️ Outil RAG: ERREUR: {e} ---")
        return "Erreur lors de l'exécution de la recherche dans les documents."

# --- 4. Testeur ---
if __name__ == "__main__":
    
    # --- MODIFICATION ICI ---
    # Le code de filtrage des avertissements a été supprimé
    # car les avertissements ont déjà été corrigés !
    # --- FIN MODIFICATION ---
    
    print("--- Test de l'Outil RAG (Imports Corrigés, Temp=0) ---")
    
    # On teste la question en Français
    print("\nTest 1 (Français):")
    test_query_fr = "Qui est le directeur général de NVIDIA?"
    # Note: L'appel de l'outil décoré @tool se fait comme ça
    results_fr = search_financial_documents.invoke(test_query_fr)
    print("\n--- Sortie Finale (Français) ---")
    print(results_fr)
    
    # On teste la question en Anglais (la réponse doit être en Anglais)
    print("\nTest 2 (Anglais -> Anglais):")
    test_query_en = "What was NVIDIA's revenue in fiscal 2024?"
    results_en = search_financial_documents.invoke(test_query_en)
    print("\n--- Sortie Finale (Anglais) ---")
    print(results_en)