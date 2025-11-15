# app/agent.py
import os
from dotenv import load_dotenv

print("🔧 Agent financier + MÉMOIRE DE SESSION (LangChain v0.3)…")
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent

# Outils
from app.tools.rag_finance_docs import search_financial_documents
from app.tools.recherche_web_tavily import search_web_tavily
from app.tools.stock_data_api import get_stock_data
from app.tools.calculatrice_financiere import calculatrice_financiere
from app.tools.email_tools import draft_email, send_email_smtp

# Routeur
from app.router import build_router, route_query

# Mémoire
from app.memory import with_memory  # get_session_history non requis ici

# Config
from app.config import validate_config, MODEL_NAME, CREATOR_NAME


def _as_tool(obj):
    """Wrap une fonction en Tool si besoin + normalise le nom (snake_case, sans ponctuation)."""
    try:
        from langchain_core.tools import BaseTool
    except Exception:
        BaseTool = tuple()

    if BaseTool and isinstance(obj, BaseTool):
        try:
            obj.name = obj.name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        except Exception:
            pass
        return obj

    if callable(obj):
        from langchain.tools import Tool
        import re
        raw = getattr(obj, "__name__", "custom_tool")
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).lower()
        return Tool.from_function(
            func=obj,
            name=safe,
            description=(obj.__doc__ or "Outil sans description.")
        )
    return obj


SYSTEM_PROMPT_TEMPLATE = """
Tu es un assistant financier utile.

HINT (si présent) : {hint}

Tu as accès aux outils:

{tools}

RÈGLES OUTILS (TRÈS IMPORTANTES) :
- Tu dois produire EXACTEMENT l'un des deux formats à chaque tour :
  (A) Demande d'outil :
      Thought: <raisonnement bref>
      Action: <nom_outil>        # exactement parmi [{tool_names}], sans guillemets ni point
      Action Input: "<texte d'entrée>"
  (B) Réponse finale :
      Thought: <raisonnement bref>
      Final Answer: <ta réponse pour l'utilisateur>
      
- Smalltalk (salutations/“comment tu vas ?”) : **N'utilise PAS d'outil**. Tu dois quand même répondre en utilisant le format "Final Answer:", brièvement et avec un ton naturel.
- **Si l'utilisateur demande qui t'a créé, réponds STRICTEMENT : "{creator_name}". Ne mentionne aucune autre entité.**
- Envoi d'e-mail :
    1) Si les champs to/subject/body ne sont pas fournis, utilise d'abord `draft_email` pour proposer un brouillon.
    2) Une fois confirmé par l'utilisateur ET si tout est fourni, utilise `send_email_smtp`.
    3) N'affirme JAMAIS avoir envoyé un e-mail si l'outil d'envoi renvoie une erreur.

- Quand tu choisis un outil, n'ajoute RIEN après la ligne "Action Input: ...".
  Le système exécutera l'outil et te fournira "Observation:" tout seul au tour suivant.
- Après avoir reçu une Observation, choisis soit un NOUVEL "Action", soit "Final Answer:".
- N'écris JAMAIS de texte libre (explication, chiffres) avant "Final Answer:".

Exemples OK :
Thought: Je vais calculer le CAGR.
Action: calculatrice_financiere
Action Input: "cagr 1000 1300 3"

Thought: J'ai la réponse.
Final Answer: Le CAGR est d'environ 9,14 %.

Exemples FAUX (REFUSÉS) :
Action: Calculatrice_Financiere.
Action Input: 'cagr 1000 1300 3' Le CAGR est...

Processus ReAct :
Thought / Action / Action Input / (Observation fourni par le système) ... puis Final Answer.
"""


def build_agent():
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", MODEL_NAME or "gpt-4o-mini"), temperature=0)

    tools = list(map(_as_tool, [
        search_financial_documents,
        search_web_tavily,
        get_stock_data,
        calculatrice_financiere,
        draft_email,
        send_email_smtp,
        ]))

    # ✅ Mémoire: on insère le placeholder de messages 'chat_history'
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ]).partial(creator_name=CREATOR_NAME)

    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=8,
        handle_parsing_errors=True,
        early_stopping_method="generate",
    )
    return agent_executor


def build_router_llm():
    return build_router(model_name=os.getenv("MODEL_NAME", MODEL_NAME or "gpt-4o-mini"))


def _invoke_with_memory(agent, payload: dict, session_id: str = "local"):
    """
    Enveloppe l'agent avec la mémoire et invoque avec un session_id.
    Le prompt doit contenir MessagesPlaceholder('chat_history').
    """
    runnable = with_memory(agent)  # léger wrapper, store partagé par session_id
    return runnable.invoke(
        payload,
        config={"configurable": {"session_id": session_id}},
    )


def handle_query(agent, router_llm, user_input: str, session_id: str = "local"):
    route = route_query(router_llm, user_input)
    action_to_tool = {
        "calc": "calculatrice_financiere",
        "stock": "stock_data_api",
        "web": "search_web_tavily",
        "RAG": "search_financial_documents",
        "email": "draft_email",
    }
    hint = ""
    if route.action == "smalltalk":
        hint = "C'est du smalltalk. Réponds SANS outil, mais en utilisant OBLIGATOIREMENT le format 'Final Answer:'."
    elif route.action == "email":
        hint = ("Si to/subject/body manquent → Action: draft_email. "
                "Sinon et si l'utilisateur confirme → Action: send_email_smtp.")
    else:
        hint = f"UTILISE d'abord l'outil: {action_to_tool.get(route.action, '')}".strip()

    return _invoke_with_memory(agent, {"input": user_input, "hint": hint}, session_id=session_id)


def handle_query_force(agent, user_input: str, tool_name: str, session_id: str = "local"):
    """Forcer l'utilisation d'un outil via le HINT (bypass routeur)."""
    hint = f"UTILISE d'abord l'outil: {tool_name}"
    return _invoke_with_memory(agent, {"input": user_input, "hint": hint}, session_id=session_id)


if __name__ == "__main__":
    print("🎯 Test de l'agent financier AVEC mémoire…")
    try:
        validate_config()
    except SystemExit as e:
        print(str(e)); raise

    agent = build_agent()
    router_llm = build_router_llm()

    print("✅ Agent créé avec succès !")
    print(f"🛠️ Outils chargés : {[t.name for t in agent.tools]}")

    # =========================
    # TEST 1: Calculatrice
    # =========================
    print("\n" + "="*40)
    print("TEST 1: Calculatrice Financière")
    q1 = "Mon investissement de 1000 est passé à 1300 en 3 ans. Quel est le CAGR ?"
    r1 = handle_query(agent, router_llm, q1, session_id="s1")
    print(f"🤖 AGENT: {r1['output']}")

    # =========================
    # TEST 2: Bourse (via agent)
    # =========================
    print("\n" + "="*40)
    print("TEST 2: Données boursières (via agent)")
    q2 = "Quel est le P/E d'Apple ?"
    r2 = handle_query(agent, router_llm, q2, session_id="s1")
    print(f"🤖 AGENT: {r2['output']}")

    # =========================
    # TEST 3: Web (via agent)
    # =========================
    print("\n" + "="*40)
    print("TEST 3: Recherche web (via agent)")
    q3 = "Quelles sont les dernières nouvelles sur Tesla ?"
    r3 = handle_query(agent, router_llm, q3, session_id="s1")
    print(f"🤖 AGENT: {r3['output']}")

    # =========================
    # TEST 4A: stock_data_api (FORCÉ via agent)
    # =========================
    print("\n" + "="*40)
    print("TEST 4A: stock_data_api (FORCÉ via agent)")
    q4a = "Donne-moi le P/E de NVDA"
    r4a = handle_query_force(agent, q4a, "stock_data_api", session_id="s1")
    print(f"🤖 AGENT: {r4a['output']}")

    # =========================
    # TEST 4B: stock_data_api (APPEL DIRECT DU TOOL)
    # =========================
    print("\n" + "="*40)
    print("TEST 4B: stock_data_api (APPEL DIRECT)")
    try:
        print("➡️ Tool.invoke('pe NVDA') ->", get_stock_data.invoke("pe NVDA"))
        print("➡️ Tool.invoke('close NVDA 1mo 1d') ->", get_stock_data.invoke("close NVDA 1mo 1d"))
    except Exception as e:
        print("❌ Problème d'appel direct du tool stock_data_api:", e)

    # =========================
    # TEST 5A: RAG (FORCÉ via agent)
    # =========================
    print("\n" + "="*40)
    print("TEST 5A: RAG sur documents NVIDIA (FORCÉ via agent)")
    q5a = "Selon mes documents, quels sont les segments de revenus de NVIDIA en 2024 ?"
    r5a = handle_query_force(agent, q5a, "search_financial_documents", session_id="s1")
    print(f"🤖 AGENT: {r5a['output']}")

    # =========================
    # TEST 5B: RAG (APPEL DIRECT DU TOOL)
    # =========================
    print("\n" + "="*40)
    print("TEST 5B: RAG (APPEL DIRECT)")
    try:
        r5b = search_financial_documents.invoke("Liste les segments de revenus de NVIDIA pour 2024 d'après mes PDF.")
        print("➡️ Tool.invoke ->\n", r5b)
    except Exception as e:
        print("❌ Problème d'appel direct du tool RAG:", e)

    # =========================
    # TEST 6: MÉMOIRE (2 tours, même session)
    # =========================
    print("\n" + "="*40)
    print("TEST 6: Mémoire de session (s2)")
    # Tour 1 : on pose un contexte
    r6a = handle_query(agent, router_llm,
                       "Contexte: j'ai 1000€ qui deviennent 1300€ en 3 ans. On en reparle juste après.",
                       session_id="s2")
    print(f"🤖 AGENT (tour 1): {r6a.get('output','')}")
    # Tour 2 : on fait référence au contexte
    r6b = handle_query(agent, router_llm,
                       "Sur cette base, calcule le CAGR.",
                       session_id="s2")
    print(f"🤖 AGENT (tour 2): {r6b.get('output','')}")

    print("\n🎉 Tous les tests terminés !")
    
    print("\n" + "="*40)
    print("TEST 7: Créateur (smalltalk)")
    q7 = "Qui t'a créé ?"
    r7 = handle_query(agent, router_llm, q7, session_id="s1")
    print(f"🤖 AGENT: {r7['output']}")

