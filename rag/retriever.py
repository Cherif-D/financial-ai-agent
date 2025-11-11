"""
Le "Chercheur" (Retriever) du RAG.

Ce module fournit la fonction `get_retriever`, qui est le pont entre
notre base de données vectorielle (créée par ingest.py) et
l'agent LangChain.

Son rôle est de :
1.  Charger l'index vectoriel (Chroma ou FAISS) depuis le disque.
2.  Le transformer en un objet "Retriever" que LangChain peut interroger.
3.  Spécifier *comment* chercher (par exemple, ramener les "K" meilleurs résultats).
"""

from app.config import PERSIST_DIR, VS_BACKEND
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



def get_retriever(k=4):
    """
    Initialise et retourne un objet Retriever configuré.

    Cette fonction lit la configuration (VS_BACKEND) pour savoir
    quelle base de données (Chroma ou FAISS) charger depuis le
    dossier `PERSIST_DIR`.

    Args:
        k (int): Le "TOP_K". C'est le nombre de chunks
                 les plus pertinents à ramener pour une question donnée.
                 Par défaut, 4.

    Returns:
        langchain.schema.vectorstore.VectorStoreRetriever:
            Un objet retriever prêt à être utilisé par un outil ou un agent.
    """
    print(f"Initialisation du retriever (backend: {VS_BACKEND}, k={k})...")

    # Initialise le *même* modèle d'embeddings que celui utilisé
    # lors de l'ingestion (ingest.py).
    embeddings = OpenAIEmbeddings()

    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
    PERSIST_DIR,
    embeddings,
    allow_dangerous_deserialization=True
     )
    return db.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    # Petit test pour vérifier que le retriever fonctionne
    print("--- Test du Retriever ---")
    try:
        retriever = get_retriever(k=2)
        print("Retriever initialisé avec succès.")
        
        # Test de recherche (similaire à une question de l'agent)
        test_query = "Qui est le directeur général de NVIDIA?"
        print(f"Test de recherche pour: '{test_query}'")
        
        # .invoke() est la nouvelle façon d'appeler les objets LangChain
        results = retriever.invoke(test_query)
        
        print(f"\nRésultats trouvés: {len(results)}")
        for i, doc in enumerate(results):
            print(f"\n--- Résultat {i+1} (Source: {doc.metadata.get('source', 'N/A')}) ---")
            print(doc.page_content[:400] + "...") # Affiche les 400 premiers caractères
            
    except Exception as e:
        print(f"\nERREUR: Impossible d'initialiser ou de tester le retriever.")
        print(f"Détail: {e}")
        print("Avez-vous bien lancé 'python -m rag.ingest' d'abord ?")