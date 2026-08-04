"""Configuration management for the Deep Research system."""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    TAVILY = "tavily"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum number of retries for structured output calls from models"
    )
    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow the researcher to ask the user clarifying questions before starting research"
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of research units to run concurrently"
    )

    # Search Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="The search API to use for research"
    )
    max_search_results: int = Field(
        default=5,
        description="Maximum number of search results to return per query"
    )
    max_content_length: int = Field(
        default=50000,
        description="Maximum character length of webpage content to include for summarization"
    )

    # Research Configuration
    max_researcher_iterations: int = Field(
        default=5,
        description="Maximum number of iterations for each researcher"
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calls a researcher can make before being forced to stop"
    )
    # Research Configuration
    max_supervisor_iterations: int = Field(
        default=5,
        description="Maximum number of iterations for each supervisor"
    )
    # Model Configuration
    research_model: str = Field(
        default="deepseek-chat",
        description="Model to use for research tasks"
    )
    research_model_max_tokens: int = Field(
        default=8192,
        description="Max tokens for the research model"
    )

    # Supervisor Configuration
    supervisor_model: str = Field(
        default="deepseek-chat",
        description="Model to use for supervisor tasks"
    )
    supervisor_model_max_tokens: int = Field(
        default=8192,
        description="Max tokens for the supervisor model"
    )
    
    summarization_model: str = Field(
        default="deepseek-chat",
        description="Model to use for summarization tasks"
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        description="Max tokens for the summarization model"
    )
    compression_model: str = Field(
        default="deepseek-chat",
        description="Model to use for compression tasks"
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        description="Max tokens for the compression model"
    )
    final_report_model: str = Field(
        default="deepseek-chat",
        description="Model to use for final report generation"
    )
    final_report_model_max_tokens: int = Field(
        default=16384,
        description="Max tokens for the final report model"
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration from a RunnableConfig.
        
        Priority: environment variable > configurable value > field default.
        """
        configurable = (
            config.get("configurable", {}) if config else {}
        )

        values: dict[str, Any] = {}
        for field_name, field_info in cls.model_fields.items():
            env_var = field_name.upper()
            env_value = os.environ.get(env_var)
            if env_value is not None:
                values[field_name] = env_value
            elif field_name in configurable:
                values[field_name] = configurable[field_name]

        return cls(**values)
