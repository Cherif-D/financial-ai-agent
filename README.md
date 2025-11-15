# 🧠 Assistant Financier IA (LangChain + RAG + UI)

Projet Python qui joue le rôle d’**assistant financier** :

- tu écris ta question en français,
- l’IA choisit les bons **outils** (docs PDF, web, bourse, calcul, e-mail),
- tu peux discuter via une **interface web** (Streamlit ou Chainlit).

---

## 🎯 Ce que fait le projet

- 📚 **RAG sur tes PDF**  
  Cherche l’info dans tes rapports financiers (ex. rapports NVIDIA).

- 🌐 **Recherche web (Tavily)**  
  Pour les infos récentes : news, contexte marché, etc.

- 📈 **Données boursières (yfinance)**  
  P/E, cours de clôture, séries simples (ex. `NVDA`, `AAPL`…).

- 🧮 **Calculatrice financière**  
  Calcul du CAGR (taux de croissance annuel moyen) et vérifications simples.

- 📧 **Outils e-mail**  
  Génération de brouillons professionnels et envoi par SMTP.

- 💬 **Chat avec mémoire**  
  L’agent garde le contexte dans une même session.

---

## 🧱 Structure du projet

```text
PROJET_GEN_AI/
├─ app/
│  ├─ tools/
│  │   ├─ calculatrice_financiere.py
│  │   ├─ email_tools.py
│  │   ├─ rag_finance_docs.py
│  │   ├─ recherche_web_tavily.py
│  │   └─ stock_data_api.py
│  ├─ ui/
│  │   ├─ chainlit_app.py
│  │   └─ streamlit_app.py
│  │
│  ├─ agent.py        # Construction de l’agent + routeur + tests
│  ├─ config.py       # Lecture .env et paramètres globaux
│  ├─ memory.py       # Mémoire de session
│  └─ router.py       # Routeur d’intentions (web, RAG, bourse, calc, email…)
│
├─ rag/
│  ├─ ingest.py       # Indexation des PDF pour le RAG
│  └─ retriever.py    # Création du retriever (vector store)
│
├─ data/              # PDF / rapports financiers
├─ vectorstore/       # Index vectoriel (créé par ingest.py)
├─ .chainlit/         # Config Chainlit
├─ chainlit.md
├─ .env               # Variables d’environnement (non versionné)
├─ .gitignore
├─ requirements.txt
└─ README.md