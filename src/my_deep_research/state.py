"""State definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Structured output models (LLM outputs formatted as these) ──


class ClarifyWithUser(BaseModel):
    """Output when the agent needs to ask the user a clarifying question."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope"
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information."
    )


class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research."
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


def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state.
    
    Supports two modes:
    - Normal: appends new_value to current_value (operator.add)
    - Override: replaces current_value entirely when new_value is {"type": "override", "value": [...]}
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """Input state for the agent — what the user provides."""
    pass


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str] = None
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str = ""


class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
