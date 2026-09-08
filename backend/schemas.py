"""Request/response models for the Ava chat API."""

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str
    greeting: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class UploadResponse(BaseModel):
    ok: bool
    filename: str
    chars: int


class ActionRequest(BaseModel):
    session_id: str
    action_id: str


class ActionResponse(BaseModel):
    status: str
    error: str | None = None
