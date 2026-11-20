"""FastAPI HTTP layer: request models, security middleware, and error mapping."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, StrictBool, StrictInt, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import ROOT
from .storage import ProfileNotFound

MAX_BODY_BYTES = 20000
APP_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
           "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
# Swagger UI loads its bundle from jsDelivr and boots with an inline script.
DOCS_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
DOCS_PATHS = {"/docs", "/redoc", "/docs/oauth2-redirect"}
BAD_REQUEST = "요청 내용을 확인해 주세요. 그림 선택이나 입력값이 올바르지 않아요."
NOT_FOUND = "요청한 주소를 찾을 수 없어요."
SERVER_ERROR = "저장 또는 연결에 문제가 있어요. 잠시 후 다시 시도해 주세요."
STATIC_ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                 "/style.css": ("style.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}


class ProfileCreate(BaseModel):
    name: StrictStr = "나"


class ProfileUpdate(BaseModel):
    name: StrictStr
    cloud_enabled: StrictBool = False


class RecommendRequest(BaseModel):
    selected: list[StrictStr] = []
    excluded: list[StrictStr] = []
    category: Optional[StrictStr] = None
    context: StrictStr = "any"


class SentenceRequest(BaseModel):
    selected: list[StrictStr]
    context: StrictStr = "any"


class ConfirmRequest(BaseModel):
    id: StrictStr
    selected: list[StrictStr]
    sentence: StrictStr
    mode: StrictStr = "adaptive"
    context: StrictStr = "any"
    selection_count: StrictInt
    elapsed_ms: StrictInt


def error(status, message):
    return JSONResponse({"error": message}, status_code=status)


def create_app(service, port=8765, allowed_hosts=(), allowed_origins=()):
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if service.local_provider:
            service.local_provider.close()

    app = FastAPI(title="Adaptive AAC API", version="0.2.0", lifespan=lifespan)
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}", *allowed_hosts}
    origins = frozenset(allowed_origins)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        response = await _guarded(request, call_next)
        csp = DOCS_CSP if request.url.path in DOCS_PATHS else APP_CSP
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "Content-Security-Policy": csp})
        return response

    async def _guarded(request, call_next):
        host = request.headers.get("host", "")
        if host not in hosts:
            return error(403, "로컬 주소로 접속해 주세요.")
        origin = request.headers.get("origin")
        if origin and origin != f"http://{host}" and origin not in origins:
            return error(403, "다른 사이트의 요청은 허용하지 않아요.")
        if request.method == "POST":
            if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                return error(415, "JSON 형식으로 요청해 주세요.")
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY_BYTES:
                return error(413, "요청 크기를 확인해 주세요.")
        try:
            return await call_next(request)
        except Exception:
            logging.exception("AAC request failed")
            return error(500, SERVER_ERROR)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        return error(400, BAD_REQUEST)

    @app.exception_handler(ValueError)
    async def invalid_value(_request, _exc):
        return error(400, BAD_REQUEST)

    @app.exception_handler(ProfileNotFound)
    async def missing_profile(_request, exc):
        return error(401, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc):
        # Unknown paths and wrong methods look the same, so routes cannot be probed by method.
        return error(404, NOT_FOUND) if exc.status_code in (404, 405) else error(exc.status_code, BAD_REQUEST)

    def profile_id(x_profile_id: str = Header(default="")):
        return x_profile_id

    @app.get("/api/bootstrap", tags=["app"])
    def bootstrap():
        return service.bootstrap()

    @app.post("/api/profiles", tags=["profile"])
    def create_profile(body: ProfileCreate):
        return service.create_profile(body.name)

    @app.get("/api/profile", tags=["profile"])
    def read_profile(pid: str = Depends(profile_id)):
        return service.profile(pid)

    @app.post("/api/profile", tags=["profile"])
    def update_profile(body: ProfileUpdate, pid: str = Depends(profile_id)):
        return service.update_profile(pid, body.name, body.cloud_enabled)

    @app.get("/api/history", tags=["history"])
    def history(pid: str = Depends(profile_id)):
        return service.history(pid)

    @app.post("/api/history/clear", tags=["history"])
    def clear_history(pid: str = Depends(profile_id)):
        return service.clear_history(pid)

    @app.get("/api/export", tags=["history"])
    def export(pid: str = Depends(profile_id)):
        return service.export(pid)

    @app.post("/api/recommend", tags=["communication"])
    def recommend(body: RecommendRequest, pid: str = Depends(profile_id)):
        return service.recommend(pid, body.selected, body.category, body.context, body.excluded)

    @app.post("/api/sentence", tags=["communication"])
    def sentence(body: SentenceRequest, pid: str = Depends(profile_id)):
        return service.sentence(pid, body.selected, body.context)

    @app.post("/api/confirm", tags=["communication"])
    def confirm(body: ConfirmRequest, pid: str = Depends(profile_id)):
        return service.confirm(pid, body.id, body.selected, body.sentence, body.mode, body.context,
                               body.selection_count, body.elapsed_ms)

    for path, (filename, mime) in STATIC_ASSETS.items():
        app.add_api_route(path, _asset(filename, mime), methods=["GET"], include_in_schema=False)

    return app


def _asset(filename, mime):
    def serve():
        return FileResponse(ROOT / "web" / filename, media_type=mime + "; charset=utf-8")
    return serve
