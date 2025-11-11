# app/tools/calculatrice_financiere.py
"""
Calculatrice financière simple (CAGR, alias 'cag').
Exemples:
  - cagr 1000 1300 3
  - cag 1000 1300 3
"""
import re
from langchain.tools import Tool

def _calc_fin_fn(query: str) -> str:
    q = re.sub(r'[\"\'“”’]', "", query).strip().rstrip(".:;")
    parts = q.split()
    if not parts:
        return "Commande vide. Ex: 'cagr 1000 1300 3'"

    cmd = parts[0].lower()
    if cmd in ("cag", "cagr"):
        if len(parts) != 4:
            return "Usage: cagr <val_init> <val_fin> <années> (ex: cagr 1000 1300 3)"
        try:
            v0 = float(parts[1]); v1 = float(parts[2]); n = float(parts[3])
            if v0 <= 0 or v1 <= 0 or n <= 0:
                return "Les valeurs et la durée doivent être strictement positives."
            cagr = (v1 / v0) ** (1.0 / n) - 1.0
            return f"CAGR = {cagr:.4%} (de {v0:g} à {v1:g} sur {n:g} ans)"
        except ValueError:
            return "Paramètres invalides. Ex: cagr 1000 1300 3"
    else:
        return "Commande inconnue. Utilise: cagr <val_init> <val_fin> <années>"

calculatrice_financiere = Tool.from_function(
    func=_calc_fin_fn,
    name="calculatrice_financiere",
    description="Calculs financiers de base. Commandes: 'cagr <v0> <v1> <années>' (alias: 'cag')."
)
