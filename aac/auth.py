"""Google sign-in (OpenID Connect authorization code flow with PKCE). Secrets never reach the browser."""

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .storage import ProfileNotFound

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
SESSION_DAYS = 30


class LoginUnavailable(Exception):
    """Google sign-in is not configured on this server."""


class LoginFailed(Exception):
    """The callback could not be trusted; the browser is sent back without a session."""


def _b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def decode_id_token(id_token):
    try:
        payload = id_token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (AttributeError, IndexError, ValueError) as error:
        raise LoginFailed("Malformed id_token") from error


class GoogleOAuth:
    def __init__(self, client_id, client_secret, redirect_uri, transport=urlopen):
        self.client_id, self.client_secret = client_id, client_secret
        self.redirect_uri, self.transport = redirect_uri, transport

    def authorize_url(self, state, verifier):
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        return AUTHORIZE_URL + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.redirect_uri, "response_type": "code",
            "scope": "openid email profile", "state": state, "code_challenge": challenge,
            "code_challenge_method": "S256", "prompt": "select_account"})

    def exchange(self, code, verifier):
        """Trade the one-time code for verified identity claims."""
        body = urlencode({"code": code, "client_id": self.client_id, "client_secret": self.client_secret,
                          "redirect_uri": self.redirect_uri, "grant_type": "authorization_code",
                          "code_verifier": verifier}).encode()
        request = Request(TOKEN_URL, data=body, method="POST",
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.transport(request, timeout=8) as response:
            raw = response.read(64_001)
        if len(raw) > 64_000:
            raise LoginFailed("Oversized token response")
        tokens = json.loads(raw)
        if not isinstance(tokens, dict) or not isinstance(tokens.get("id_token"), str):
            raise LoginFailed("Token response without id_token")
        # The id_token came straight from Google's token endpoint over TLS, so its claims are
        # validated without a signature check (OpenID Connect Core 3.1.3.7).
        claims = decode_id_token(tokens["id_token"])
        if not isinstance(claims, dict):
            raise LoginFailed("Invalid id_token claims")
        if claims.get("iss") not in ISSUERS or claims.get("aud") != self.client_id:
            raise LoginFailed("id_token issued for another client")
        if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < time.time():
            raise LoginFailed("Expired id_token")
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise LoginFailed("id_token without subject")
        if claims.get("email_verified") is not True or not isinstance(claims.get("email"), str):
            raise LoginFailed("Unverified email")
        return claims


class AuthService:
    def __init__(self, store, google=None):
        self.store, self.google = store, google

    @property
    def google_available(self):
        return self.google is not None

    def start_google(self, anonymous_profile_id=""):
        """Return (authorize_url, state). The state is also bound to the browser by a cookie."""
        if not self.google:
            raise LoginUnavailable()
        linkable = None
        if anonymous_profile_id:
            try:
                profile = self.store.profile(anonymous_profile_id)
                linkable = profile["id"] if profile["user_id"] is None else None
            except ProfileNotFound:
                linkable = None
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        self.store.save_oauth_state(state, verifier, linkable)
        return self.google.authorize_url(state, verifier), state

    def complete_google(self, code, state, cookie_state):
        """Return a new session token, or raise LoginFailed."""
        if not self.google:
            raise LoginUnavailable()
        if not code or not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
            raise LoginFailed("State mismatch")
        pending = self.store.pop_oauth_state(state)
        if not pending:
            raise LoginFailed("Unknown or expired state")
        claims = self.google.exchange(code, pending["verifier"])
        name = claims.get("name") if isinstance(claims.get("name"), str) else ""
        user, _profile = self.store.login_identity("google", claims["sub"], claims["email"], name.strip(),
                                                   pending["profile_id"])
        return self.store.create_session(user["id"], days=SESSION_DAYS)

    def me(self, session_token):
        user_id = self.store.session_user_id(session_token)
        return {"authenticated": bool(user_id), "user": self.store.user(user_id) if user_id else None,
                "google_available": self.google_available}

    def resolve_profile_id(self, session_token, header_profile_id):
        """A session wins; a bare profile ID only reaches profiles that are not linked to an account."""
        user_id = self.store.session_user_id(session_token)
        if user_id:
            return self.store.profile_for_user(user_id)["id"]
        profile = self.store.profile(header_profile_id)
        if profile["user_id"] is not None:
            raise ProfileNotFound("로그인이 필요해요. 다시 로그인해 주세요.")
        return profile["id"]

    def logout(self, session_token):
        self.store.delete_session(session_token)
        return {"logged_out": True}
