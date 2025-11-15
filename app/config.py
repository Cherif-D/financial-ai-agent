"""
Fichier Central de Configuration

Ce module centralise toutes les variables de configuration nécessaires
au fonctionnement de l'application. Il utilise `python-dotenv` pour
charger les variables depuis un fichier .env à la racine du projet.

Variables gérées :
- Clés API (OpenAI, Tavily)
- Nom du modèle LLM utilisé pour la génération)
- Chemins vers les répertoires de données (data/)
- Chemin vers le stockage du vector store (vectorstore/)
- Choix du backend pour le vector store (faiss ou chroma)
"""

# Importe les modules nécessaires
import os  # Pour interagir avec le système d'exploitation (lire les variables d'env)
from dotenv import load_dotenv  # Pour charger les variables depuis le fichier .env

# --- Chargement Initial ---
# Exécute la fonction load_dotenv() qui va chercher un fichier .env
# dans le répertoire courant (ou parent) et charger les variables
# qu'il contient dans l'environnement système (os.environ).
load_dotenv()

# === Section 1: Configuration du Modèle (LLM) ===
# Récupère la clé API OpenAI.
# Tente d'abord de lire "OPENAI_API_KEY".
# Si elle n'est pas trouvée, tente par sécurité "openai_key" (utilisé dans certains TP).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")

# Vérification de sécurité : si aucune clé n'est trouvée, l'application ne
# peut pas fonctionner. On arrête le programme avec un message d'erreur clair.
if not OPENAI_API_KEY:
    raise SystemExit(
        "ERREUR: OPENAI_API_KEY (ou openai_key) non définie dans le fichier .env. "
        "Veuillez créer un compte sur platform.openai.com pour en obtenir une et le mettre dans le .env."
    )

# Définit le modèle LLM à utiliser pour la génération.
# Utilise "gpt-4o-mini" par défaut si la variable MODEL_NAME n'est pas
# définie dans le .env. C'est un bon équilibre coût/performance.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# creator 
CREATOR_NAME = os.getenv("CREATOR_NAME", "Diallo Mamadou Cherif")


# === Section 2: Configuration du RAG (Retrieval-Augmented Generation) ===
# 
# Définit le chemin vers le répertoire contenant les documents bruts
# (PDF, DOCX,) que le RAG doit ingérer.
# Par défaut, utilise un dossier nommé "data".
DOCS_DIR = os.getenv("DOCS_DIR", "data")

# Définit le chemin où la base de données vectorielle (Vector Store)
# sera sauvegardée après l'ingestion.
# Par défaut, utilise un dossier nommé "vectorstore".
PERSIST_DIR = os.getenv("PERSIST_PATH", "vectorstore")

# Définit le "moteur" de la base de données vectorielle à utiliser.
# 'faiss' : rapide, en mémoire, idéal pour le développement.
# 'chroma' : persistant sur disque, plus robuste pour la production.
# Le code d'ingestion (ingest.py) devra lire cette variable pour
# savoir quelle logique de sauvegarde utiliser.
# faiss est développé par Facebook(plus rapide, comme bloc-notes), chroma(petite base de données) est développé par ChromaDB.
VS_BACKEND = os.getenv("VECTORSTORE_BACKEND", "faiss")  # Options: 'faiss' ou 'chroma'


# === Section 3: Configuration des Outils (Tools) ===

# 
# Récupère la clé API pour le service de recherche web Tavily.
# Cette clé est nécessaire pour l'outil `recherche_web_tavily.py`.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Vérification de sécurité : Si la recherche web est considérée comme
# --- AU LIEU DE lever SystemExit directement, fais ceci ---
def validate_config():
    missing = []
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    if not TAVILY_API_KEY: missing.append("TAVILY_API_KEY")
    if missing:
        raise SystemExit(f"ERREUR: clés manquantes: {', '.join(missing)}")
