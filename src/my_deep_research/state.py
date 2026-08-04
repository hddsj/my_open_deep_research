"""State definitions and data structures for the Deep Research agent."""

from typing import Annotated, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ── Structured output models (LLM outputs formatted as these) ──


class ClarifyWithUser(BaseModel):
    """Output when the agent needs to ask the user a clarifying question."""
    
    clarify: bool = Field(
        description="Whether the agent needs to ask the user a clarifying question"
    )
    question: str = Field(
        description="The clarifying question to ask the user"
    )


class ResearchQuestion(BaseModel):
    """A single research question with its rationale."""
    
    question: str = Field(
        description="The research question"
    )
    rationale: str = Field(
        description="Why this question is important for the research"
    )


class Summary(BaseModel):
    """Structured summary of a webpage's content."""
    
    summary: str = Field(
        description="Concise summary preserving key information from the webpage"
    )
    key_excerpts: str = Field(
        description="Important quotes or data points from the webpage"
    )


# ── Graph state definitions ──


def override_reducer(existing: list, new: list) -> list:
    """Custom reducer that replaces the entire list instead of appending."""
    return new


class AgentInputState(MessagesState):
    """Input state for the agent — what the user provides."""
    pass


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    research_brief: Optional[str] = None
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str = ""
