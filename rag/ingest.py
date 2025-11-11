"""
Script d'Ingestion pour le RAG (Retrieval-Augmented Generation)

Ce script est responsable de la "mémoire" de l'assistant. Il fait les opérations suivantes :
1.  Il lit les documents (PDF, DOCX) depuis le dossier `data/` (défini dans config.py).
2.  Il les découpe en petits morceaux ("chunks") pour qu'ils soient digestes.
3.  Il les transforme en "vecteurs" (embeddings) via l'API OpenAI.
4.  Il stocke ces vecteurs dans une base de données locale (FAISS ou Chroma)
    dans le dossier `vectorstore/`.

Pour l'exécuter :
1.  Placez vos fichiers PDF/DOCX dans le dossier `data/`.
2.  Assurez-vous que votre .env est configuré (OPENAI_API_KEY).
3.  Exécutez `python rag/ingest.py` depuis la racine du projet.
"""

import os, glob
from app.config import DOCS_DIR, PERSIST_DIR, VS_BACKEND     # <= pas app.config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  
# --- Fonctions ---

def load_docs(data_dir=DOCS_DIR):
    """
    Charge tous les documents PDF et DOCX depuis le répertoire de données spécifié.
    
    Args:
        data_dir (str): Le chemin vers le dossier 'data/'.
    
    Returns:
        list: Une liste d'objets 'Document' chargés par LangChain.
    """
    print(f"Chargement des documents depuis {data_dir}...")
    docs = []
    
    # 1. Trouver et charger tous les fichiers PDF
    # os.path.join crée un chemin compatible (ex: "data/*.pdf")
    # glob.glob trouve tous les fichiers qui correspondent à ce modèle
    for path in glob.glob(os.path.join(data_dir, "*.pdf")):
        print(f"  -> Chargement de {path}")
        # Crée un chargeur PDF pour ce chemin et charge le fichier
        docs += PyPDFLoader(path).load()
        
    # 2. Trouver et charger tous les fichiers DOCX
    for path in glob.glob(os.path.join(data_dir, "*.docx")):
        print(f"  -> Chargement de {path}")
        # Crée un chargeur DOCX et charge le fichier
        docs += Docx2txtLoader(path).load()
        
    return docs

def build_index():
    """
    Fonction principale qui construit l'index vectoriel.
    - Charge les documents
    - Les découpe (chunking)
    - Crée les embeddings
    - Sauvegarde l'index sur le disque.
    """
    
    # --- 1. Chargement ---
    docs = load_docs()
    
    # Vérification de sécurité : si data/ est vide, on arrête.
    if not docs:
        raise SystemExit(f"ERREUR: Aucun document .pdf ou .docx trouvé dans {DOCS_DIR}/. "
                         "Veuillez ajouter des fichiers avant de lancer l'ingestion.")
    
    print(f"Chargé {len(docs)} pages/documents.")

    # --- 2. Découpage (Chunking) ---
    # 
    # Nous découpons les longs documents en morceaux plus petits.
    # C'est essentiel pour que le RAG trouve des passages spécifiques.
    # - chunk_size=1000 : taille de chaque morceau (en caractères).
    # - chunk_overlap=150 : chevauchement entre les morceaux. Quand tu coupes ton morceau n°1, et que tu commences ton morceau n°2, recommence 200 caractères plus tôt."
    #                       ne pas perdre le contexte entre deux chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    
    print(f"Documents découpés en {len(splits)} morceaux (chunks).")

    # --- 3. Embeddings (Vectorisation) ---
    # Initialise le modèle d'embedding d'OpenAI.
    # C'est lui qui va lire chaque "chunk" et le transformer en
    # une liste de chiffres (vecteur) qui représente son "sens".
    # Il utilise OPENAI_API_KEY automatiquement.
    embeddings = OpenAIEmbeddings()
    
    print("Modèle d'embeddings OpenAI initialisé.")

    # --- 4. Stockage (Vector Store) ---
    # Lit la variable VS_BACKEND de notre config pour décider
    # quelle base de données utiliser.
    
    print("Construction de l'index FAISS...")
    vectordb = FAISS.from_documents(splits, embeddings)
    print(f"Sauvegarde de l'index FAISS dans {PERSIST_DIR}...")
    vectordb.save_local(PERSIST_DIR)


    print("\n--- Ingestion Terminée ---")
    print(f"✅ Index construit et sauvegardé dans {PERSIST_DIR}")
    print(f"   (Backend utilisé: {VS_BACKEND})")
    print(f"   Total chunks indexés: {len(splits)}")

# --- Point d'Entrée du Script ---
# Cette convention Python signifie:
# "Si j'exécute ce fichier directement (python rag/ingest.py),
#  alors exécute la fonction build_index()."
if __name__ == "__main__":
    build_index()