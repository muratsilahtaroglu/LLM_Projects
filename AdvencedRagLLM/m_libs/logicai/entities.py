from typing import List, Literal, Optional
from pydantic import BaseModel, Field

__all__ = [
    "TeamsPromptInput",
    "TeamsPromptOutput",
    "ChatResponseFunction"
]

class TeamsPromptInput(BaseModel):
    content: str
    input_files_path: List[str] = Field(default_factory=list)

class TeamsPromptOutput(BaseModel):
    content: Optional[str]
    output_files_path: List[str] = Field(default_factory=list)

class ChatResponseFunction(BaseModel):
    name: str
    arguments: dict
