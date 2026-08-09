# OmniMarket AI frontend

Static dashboard for the FastAPI paper-trading backend.

## Run

From the repository root, start the API:

```bash
uvicorn api:app --reload --port 8000
```

Then serve this directory with any static HTTP server (opening `index.html` directly may trigger browser CORS/file-origin restrictions):

```bash
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173` and keep the API URL set to `http://localhost:8000`.

The dashboard is intentionally dependency-free during the API-contract validation phase. It uses the existing `/health`, `/ingest`, `/portfolio`, `/risk-log`, and `/learning/status` routes.
