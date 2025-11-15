# app/router.py
# ce fichier router définit la logique de routage des requêtes utilisateur vers les outils appropriés
# il permet de dire "pour cette requête, l'outil le plus adapté est tel outil"
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Tuple
import re
from langchain_openai import ChatOpenAI

Action = Literal[
    "smalltalk","RAG","web","stock","calc","portfolio","valuation","fx",
    "events","kpi","risk","statements","esg","options","bonds","parity",
    "rebalance","auto", "email"
]

class Route(BaseModel):
    action: Action = Field(description="Catégorie d'outil la plus pertinente")
    query: str

_PATTERNS: List[Tuple[str, Action]] = [
    (r"^\s*(bonjour|salut|hello)\b", "smalltalk"),
    (r"\b(qui\s*t['’]?a\s*cr(é|e)é|ton\s*cr(é|e)ateur|cr(é|e)é\s*par\s*qui|who\s*created\s*you)\b", "smalltalk"),
    (r"\b(selon\s+(le|la)\s+(rapport|document)|dans\s+mes\s+docs|corpus)\b", "RAG"),
    (r"\b(actu|actualités|news|dernières nouvelles|latest\s+news)\b", "web"),
    (r"\b(pe\b|p/?e|close\s+[A-Z]{1,6}\b|\bticker\b)\b", "stock"),
    (r"\b(cagr|cag\b|rendement|roi|npv|van|irr|calcul|%)\b", "calc"),
    (r"\b(email|mail|courriel|envoie( r)? un (mail|email)|écris un mail|rédige un mail)\b", "email"), 
]

def fastpath_route(user_input: str) -> Optional[Action]:
    text = user_input.lower()
    for pat, act in _PATTERNS:
        if re.search(pat, text):
            return act
    return None

def build_router(model_name="gpt-4o-mini", temperature=0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)

def route_query(llm: ChatOpenAI, user_input: str) -> Route:
    act = fastpath_route(user_input)
    if act is not None:
        return Route(action=act, query=user_input)

    schema = Route.model_json_schema()
    prompt = (
        "Tu es un routeur d'intentions pour un assistant financier.\n"
        "Choisis la meilleure catégorie parmi: smalltalk,RAG,web,stock,calc,portfolio,valuation,fx,events,"
        "kpi,risk,statements,esg,options,bonds,parity,rebalance,auto.\n"
        "Rends STRICTEMENT un JSON respectant le schéma suivant.\n"
        f"Schéma JSON: {schema}\n\n"
        f"Texte: {user_input}"
    )
    return llm.with_structured_output(Route).invoke(prompt)
