ocean_across_assign/
├── data/
│   ├── raw/
│   │   └── Facets Assignment.csv      # The original 399-row file
│   └── processed/
│       └── Enriched_Facets.csv        # The generated output with categories & definitions
├── scripts/
│   └── build_facet_db.py              # Phase 1 script: Automates the data cleaning/enrichment
├── backend/
│   ├── Dockerfile                     # Instructions to containerize the FastAPI app
│   ├── requirements.txt               # Backend dependencies (fastapi, uvicorn, groq, pandas)
│   ├── main.py                        # FastAPI entry point & API route definitions
│   ├── groq_engine.py                 # Core logic: Groq API client, async batching, logprob extraction
│   ├── schemas.py                     # Pydantic models to strictly validate inputs/outputs
│   └── prompt_templates.py            # Centralized Few-Shot and Chain-of-Thought prompts
├── ui/
│   ├── Dockerfile                     # Instructions to containerize the Streamlit app
│   ├── requirements.txt               # UI dependencies (streamlit, requests, pandas)
│   └── app.py                         # The frontend dashboard
├── outputs/
│   └── sample_50_evaluations.jsonl    # The final deliverable file (zipped later)
├── docker-compose.yml                 # Orchestrates the backend and ui containers to talk to each other
├── .env                               # Stores GROQ_API_KEY (Make sure this is gitignored!)
├── .gitignore                         # Ignore venv/, .env, __pycache__/, etc.
└── README.md                          # Crucial documentation explaining your O(1) architecture