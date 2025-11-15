# app/ui/streamlit_app.py
import uuid
import streamlit as st

from app.agent import (
    build_agent,
    build_router_llm,
    handle_query,
    handle_query_force,
)
from app.config import validate_config

st.set_page_config(page_title="Assistant Financier", layout="wide")
st.title("Assistant Financier (RAG + Agents)")

# --- Sidebar: état & configuration ---
with st.sidebar:
    st.subheader("État & Configuration")
    try:
        validate_config()
        st.success("Configuration OK ✅")
    except SystemExit as e:
        st.error(f"Configuration invalide ❌ : {e}")

    st.caption("Astuce: utilise le forçage d'outil pour tester un tool précis.")

# --- Init session ---
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "router" not in st.session_state:
    st.session_state.router = build_router_llm()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_res" not in st.session_state:
    st.session_state.last_res = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"

# --- UI: choix optionnel de forçage d'outil ---
with st.sidebar:
    st.subheader("Forçage d'outil (optionnel)")
    force_tool = st.selectbox(
        "Choisir un outil à forcer :",
        options=["", "search_financial_documents", "search_web_tavily", "stock_data_api", "calculatrice_financiere"],
        index=0
    )
    st.caption("Laisse vide pour laisser le routeur décider.")

# --- Chat input ---
user = st.chat_input("Pose ta question… (ex: 'P/E NVDA', 'CAGR 1000→1300 en 3 ans')")
if user:
    sid = st.session_state.session_id
    if force_tool:
        res = handle_query_force(st.session_state.agent, user, force_tool, session_id=sid)
    else:
        res = handle_query(st.session_state.agent, st.session_state.router, user, session_id=sid)

    st.session_state.chat.append(("user", user))
    st.session_state.chat.append(("ai", res.get("output", "")))
    st.session_state.last_res = res

# --- Historique ---
for role, msg in st.session_state.chat:
    with st.chat_message("assistant" if role == "ai" else "user"):
        st.write(msg)

# --- Scratchpad / Steps ---
with st.expander("Afficher le scratchpad (intermediate steps)"):
    res = st.session_state.last_res
    if res is None:
        st.info("Envoie un message pour voir le scratchpad.")
    else:
        steps = res.get("intermediate_steps", [])
        if not steps:
            st.write("Aucun appel d'outil sur la dernière requête.")
        else:
            for (action, observation) in steps:
                tool = getattr(action, "tool", "unknown_tool")
                tool_input = getattr(action, "tool_input", "")
                obs = str(observation)
                st.markdown(
                    f"**Tool**: `{tool}`\n\n"
                    f"**Args**: `{tool_input}`\n\n"
                    f"**Output**: {obs[:1500]}{'…' if len(obs) > 1500 else ''}\n\n---"
                )

st.caption("Conseil: pour tester le RAG sur NVIDIA, lance d'abord `python -m app.ingest`, puis pose une question du type “Selon mes documents NVIDIA, …”.")
