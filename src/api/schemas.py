from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "bot"]
    content: str


class RewriteRequest(BaseModel):
    conversation: list[Turn] = Field(..., min_length=1)


class RewriteResponse(BaseModel):
    rewritten_query: str


class CompareResponse(BaseModel):
    base_model_output: str
    fine_tuned_output: str
