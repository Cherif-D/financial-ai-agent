# app/ui/chainlit_app.py
import os
import uuid
import chainlit as cl

from app.agent import (
    build_agent,
    build_router_llm,
    handle_query,
    handle_query_force,
)
from app.config import validate_config

# État global de la session Chainlit
AGENT = None
ROUTER = None
SESSION_ID = None  # utilisé par with_memory()

WELCOME = (
    "🤖 Assistant Financier prêt.\n"
    "Je peux :\n"
    "• Interroger tes docs (RAG)\n"
    "• Chercher l’actu (web)\n"
    "• Donner des chiffres de marché (stock)\n"
    "• Calculer (CAGR, etc.)\n\n"
    "Astuce: tape `!tool:<nom>` pour forcer un outil (ex: `!tool:stock_data_api pe NVDA`)."
)

def _parse_force_tool(msg: str):
    """
    Si l'utilisateur écrit `!tool:<tool_name> ...`, on force cet outil.
    Retourne (tool_name|None, query_sans_prefixe)
    """
    txt = msg.strip()
    if txt.lower().startswith("!tool:"):
        after = txt[6:].strip()
        if " " in after:
            tool, q = after.split(" ", 1)
        else:
            tool, q = after, ""
        return tool.strip(), q.strip() or " "
    return None, msg

@cl.on_chat_start
async def on_start():
    global AGENT, ROUTER, SESSION_ID
    # Génère un session_id unique pour la mémoire
    SESSION_ID = f"chainlit-{uuid.uuid4().hex[:8]}"

    try:
        validate_config()
    except SystemExit as e:
        await cl.Message(content=f"❌ Config invalide : {e}").send()
        return

    AGENT = build_agent()
    ROUTER = build_router_llm()
    await cl.Message(content=WELCOME).send()

@cl.on_message
async def on_message(message: cl.Message):
    global AGENT, ROUTER, SESSION_ID
    if AGENT is None or ROUTER is None:
        await cl.Message(content="❌ L'agent n'est pas initialisé. Relance l'application.").send()
        return

    txt = message.content.strip()

    # Salutations rapides (sans outil)
    if txt.lower() in {"bonjour", "salut", "hello"}:
        await cl.Message(content="👋 Bonjour ! Pose ta question ou utilise `!tool:<nom>` pour forcer un outil.").send()
        return

    # Forçage d'outil
    tool_forced, query = _parse_force_tool(txt)

    try:
        if tool_forced:
            res = await cl.make_async(handle_query_force)(AGENT, query, tool_forced, session_id=SESSION_ID)
        else:
            res = await cl.make_async(handle_query)(AGENT, ROUTER, txt, session_id=SESSION_ID)

        # Réponse finale
        await cl.Message(content=res.get("output", "") or "⚠️ Pas de réponse.").send()

        # Scratchpad (intermediate steps)
        steps = res.get("intermediate_steps", [])
        if steps:
            logs = []
            for (action, observation) in steps:
                tool = getattr(action, "tool", "unknown_tool")
                tool_input = getattr(action, "tool_input", "")
                obs = str(observation)
                logs.append(
                    f"**Tool**: `{tool}`\n"
                    f"**Args**: `{tool_input}`\n"
                    f"**Output**: {obs[:1200]}{'…' if len(obs) > 1200 else ''}"
                )
            await cl.Message(content="\n\n---\n\n".join(logs), author="scratchpad").send()

    except Exception as e:
        await cl.Message(content=f"❌ Erreur: {e}").send()
