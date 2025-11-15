# app/memory.py (v0.3-compatible)

from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Petit store en mémoire par session_id
_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retourne (ou crée) l'historique pour une session donnée."""
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]

def with_memory(runnable):
    """
    Enveloppe un agent/chaine avec l'historique de messages.
    - input_messages_key: clé d'entrée (ton prompt attend 'input')
    - history_messages_key: placeholder dans le prompt (MessagesPlaceholder('chat_history'))
    """
    return RunnableWithMessageHistory(
        runnable=runnable,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
