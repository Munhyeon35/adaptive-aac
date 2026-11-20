"""Process entrypoint: read configuration and serve the FastAPI app with uvicorn."""

import os
from pathlib import Path

from .auth import AuthService, GoogleOAuth
from .config import ROOT, load_env
from .llm import OpenAIProvider
from .service import AacService
from .storage import Store


def env_list(name):
    return frozenset(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


def build_app():
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    provider = OpenAIProvider(key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini")) if key else None
    from .local_model import configured_local_provider
    try:
        local_provider = configured_local_provider()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Local model configuration error: {error}") from None
    path = Path(os.environ.get("AAC_DB_PATH", "var/aac.sqlite3"))
    if not path.is_absolute():
        path = ROOT / path
    port = int(os.environ.get("AAC_PORT", "8765"))
    # The public URL decides the OAuth redirect URI and whether cookies are Secure (HTTPS only).
    public_url = os.environ.get("AAC_PUBLIC_URL", "").rstrip("/") or f"http://localhost:{port}"
    store = Store(path)
    client_id, client_secret = os.environ.get("GOOGLE_CLIENT_ID", ""), os.environ.get("GOOGLE_CLIENT_SECRET", "")
    google = GoogleOAuth(client_id, client_secret, public_url + "/auth/google/callback") if client_id and client_secret else None
    from .api import create_app
    app = create_app(AacService(store, provider, local_provider), port,
                     env_list("AAC_ALLOWED_HOSTS"), env_list("AAC_ALLOWED_ORIGINS"),
                     auth=AuthService(store, google), secure_cookies=public_url.startswith("https://"))
    return app, port


def main():
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("FastAPI is not installed. Run: python -m pip install -r requirements.txt") from None
    app, port = build_app()
    # Deployment behind a reverse proxy only: localhost stays the default.
    bind = os.environ.get("AAC_HOST", "127.0.0.1")
    print(f"Adaptive AAC: http://localhost:{port}  (API docs: http://localhost:{port}/docs)", flush=True)
    print("Press Ctrl+C to stop. Cloud recommendations require opt-in in Settings.", flush=True)
    # Access logs stay off: request lines can carry OAuth codes, and communication data never enters logs.
    uvicorn.run(app, host=bind, port=port, access_log=False, log_level="warning")
