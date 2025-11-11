"""
Outil de Recherche Web (Tavily).
"""

import os
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Importe le nom du modèle depuis notre configuration centrale
from app.config import MODEL_NAME, TAVILY_API_KEY

# Importe le client de recherche Tavily (version LangChain)
from langchain_community.tools.tavily_search import TavilySearchResults


# --- 1. Définition de l'Outil de Recherche "Brut" ---
# Cet outil renvoie déjà un STRING formaté.
raw_tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)


# --- 2. Définition d'un "Summarizer" (Synthétiseur) ---
TEMP_ANALYSIS = 0.7

_prompt = PromptTemplate.from_template(
    "Tu es un analyste financier expert. Rédige un bref résumé (en 3-4 phrases) "
    "basé sur les résultats de recherche suivants. Réponds **dans la même langue que la Question**.\n\n"
    "Question: {question}\n\n"
    "Résultats de recherche:\n{context}\n\n"
    "Résumé concis:"
)

_llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMP_ANALYSIS)



# --- 3. La "Chaîne de Montage" du Résumé (LCEL) ---
summarize_chain = (
    RunnableMap({
        # 1. Prend la question, l'envoie à Tavily (qui renvoie un string)
        "context": (lambda x: x["question"]) | raw_tavily_tool,
        # 2. Passe la question originale
        "question": (lambda x: x["question"])
    })
    | _prompt  # 3. Envoie le tout au "mode d'emploi"
    | _llm     # 4. Envoie au Cerveau pour résumer
    | StrOutputParser() # 5. Ne garde que le texte final
)

# --- 4. Définition de l'Outil Final (ce que l'Agent verra) ---
@tool
def search_web_tavily(query: str) -> str:
    """
    [C'EST LE MODE D'EMPLOI POUR L'AGENT]
    Utilise cet outil EXCLUSIVEMENT pour rechercher des informations
    en temps réel, des actualités ("news"), ou le cours de l'action ("stock price")
    sur Internet.
    
    Il est parfait pour les questions sur des événements récents,
    des opinions de marché, ou des informations qui ne peuvent
    pas se trouver dans les rapports financiers internes.
    """
    print(f"\n--- 🛠️ Outil Web: Appel de search_web_tavily (v2) ---")
    print(f"--- 🛠️ Outil Web: Question reçue: {query} ---")
    
    try:
        # On appelle notre "chaîne de résumé"
        answer = summarize_chain.invoke({"question": query})
        print(f"--- 🛠️ Outil Web: Résumé généré: {answer} ---")
        return answer
    except Exception as e:
        print(f"--- 🛠️ Outil Web: ERREUR: {e} ---")
        return "Erreur lors de la recherche sur le web."

# --- 5. Testeur (pour nous, les humains) ---
if __name__ == "__main__":
    
    print("--- Test de l'Outil de Recherche Web (v2.3 - Corrigé Bug) ---")
    
    print("\nTest 1 (Français):")
    test_query_fr = "Pourquoi elon musk est l'homme le plus riche au monde ?"
    results_fr = search_web_tavily.invoke(test_query_fr)
    print("\n--- Sortie Finale (Français) ---")
    print(results_fr)