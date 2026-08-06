"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from my_deep_research.configuration import Configuration
from my_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    followup_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
    followup_answer_prompt,
)
from my_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    FollowUpDecision,
    SupervisorOutputState,
    SupervisorState,
)
from my_deep_research.utils import (
    execute_tool_safely,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    clarification_model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    response = await clarification_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to end (will be changed to research_supervisor in Phase 5)
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    research_model = (
        configurable_model.with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 3: Store research brief and end (Phase 5 will route to supervisor)
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": [HumanMessage(content=response.research_brief)],
        },
    )


# ── Researcher Subgraph ──


async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.

    Uses available tools (search, think_tool) to gather comprehensive information
    in a tool-calling loop.

    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability

    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and get available tools
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    researcher_prompt = research_system_prompt.format(date=get_today_str())

    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher.

    Handles tool calls, checks iteration limits, and decides whether to
    continue the research loop or proceed to compression.

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits

    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")

    # Step 2: Execute all tool calls in parallel
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "unknown"): tool
        for tool in tools
    }

    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config)
        for tc in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for observation, tc in zip(observations, tool_calls)
    ]

    # Step 3: Check exit conditions
    exceeded_iterations = (
        state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    )

    if exceeded_iterations:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs},
        )

    # Continue research loop
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs},
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise summary.

    Takes all research findings and distills them into a clean, comprehensive
    summary while preserving all important information.

    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings

    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(
        {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.compression_model, config),
            "tags": ["langsmith:nostream"],
        }
    )

    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(
        HumanMessage(content=compress_research_simple_human_message)
    )

    # Step 3: Attempt compression
    try:
        compression_prompt = compress_research_system_prompt.format(
            date=get_today_str()
        )
        messages = [SystemMessage(content=compression_prompt)] + researcher_messages
        response = await synthesizer_model.ainvoke(messages)

        raw_notes_content = "\n".join(
            [
                str(message.content)
                for message in filter_messages(
                    researcher_messages, include_types=["tool", "ai"]
                )
            ]
        )

        return {
            "compressed_research": str(response.content),
            "raw_notes": [raw_notes_content],
        }

    except Exception as e:
        raw_notes_content = "\n".join(
            [
                str(message.content)
                for message in filter_messages(
                    researcher_messages, include_types=["tool", "ai"]
                )
            ]
        )
        return {
            "compressed_research": f"Error synthesizing research: {e}",
            "raw_notes": [raw_notes_content],
        }


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            prompt_content = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )
            response = await configurable_model.with_config(writer_model_config).ainvoke(
                [HumanMessage(content=prompt_content)]
            )
            return {
                "final_report": response.content,
                "messages": [response],
            }

        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error: Token limit exceeded but could not determine model limit. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                }

    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
    }

# Build researcher subgraph
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration,
)

researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)

researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)

researcher_subgraph = researcher_builder.compile()

async def handle_followup(followup_question: str, notes: list, final_report: str, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    api_key = get_api_key_for_model(configurable.research_model, config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
    }
    if api_key:
        model_config["api_key"] = api_key
    model = configurable_model.with_config(configurable=model_config)
    prompt = followup_prompt.format(
            final_report=final_report,
            notes="\n".join(notes),
            followup_question=followup_question)

    result = await model.with_structured_output(FollowUpDecision).ainvoke([
        HumanMessage(content=prompt),
    ])

    print(f"[追问判断] needs_research={result.needs_research}, research_topic={result.research_topic}")
    if result.needs_research:
        yield {"type": "progress", "message": "正在补充搜索..."}
        decision = await researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=result.research_topic)
                ],
                "research_topic": result.research_topic,
            }, config)
        new_content = decision["compressed_research"]
        notes = "\n".join(notes)
        notes += "\n" + new_content
        prompt = followup_answer_prompt.format(
                    final_report=final_report,
                    notes=notes,
                    followup_question=followup_question,
                    new_research=new_content)
        yield {"type": "progress", "message": "搜索完成，正在生成回答..."}
        async for chunk in model.astream([HumanMessage(content=prompt)]):
            yield {"type": "token", "content": chunk.content}
        yield {"type": "done", "searched": True}
    else:
        yield {"type": "token", "content": result.answer}
        yield {"type": "done", "searched": False}


async def supervisor(
    state: SupervisorState, config: RunnableConfig
)->Command[Literal["supervisor_tools"]]:
    # Step 1: Load configuration and get available tools
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])

    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.supervisor_model,
        "max_tokens": configurable.supervisor_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.supervisor_model, config),
        "tags": ["langsmith:nostream"],
    }

    supervisor_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_researcher_iterations=configurable.max_researcher_iterations,
        max_concurrent_research_units=configurable.max_concurrent_research_units,
    )

    supervisor_model = (
        configurable_model.bind_tools([ConductResearch,ResearchComplete,think_tool])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=supervisor_prompt)] + supervisor_messages
    response = await supervisor_model.ainvoke(messages)

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    # Step 1: Extract current state and check early exit
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    exceeded = research_iterations > configurable.max_supervisor_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tc["name"] == "ResearchComplete"
        for tc in most_recent_message.tool_calls
    )
    if exceeded or no_tool_calls or research_complete:
        return Command(
            goto=END,
            update={
                "notes": state.get("notes", []),
                "research_brief": state.get("research_brief", ""),
            }
        )
    all_tool_messages = []

    think_tool_calls = [
        tc for tc in most_recent_message.tool_calls
        if tc["name"] == "think_tool"
    ]
    for tc in think_tool_calls:
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {tc['args']['reflection']}",
            name="think_tool",
            tool_call_id=tc["id"],
        ))
    conduct_research_calls = [
        tc for tc in most_recent_message.tool_calls
        if tc["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        allowed = conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow = conduct_research_calls[configurable.max_concurrent_research_units:]
        research_tasks = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tc["args"]["research_topic"])
                ],
                "research_topic": tc["args"]["research_topic"],
            }, config)
            for tc in allowed
        ]
        results = await asyncio.gather(*research_tasks)
        new_notes = []
        for observation, tc in zip(results, allowed):
            content = observation.get("compressed_research",
                "Error synthesizing research report")
            all_tool_messages.append(ToolMessage(
                content=content,
                name=tc["name"],
                tool_call_id=tc["id"],
            ))
            new_notes.append(content)

        for tc in overflow:
            all_tool_messages.append(ToolMessage(
                content=f"Error: exceeded max concurrent "
                    f"research units ({configurable.max_concurrent_research_units})",
                name="ConductResearch",
                tool_call_id=tc["id"],
            ))
    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": all_tool_messages,
            "notes": new_notes if conduct_research_calls else [],
        },
    )



# Build supervisor subgraph
supervisor_builder = StateGraph(
    SupervisorState,
    output=SupervisorOutputState,
    config_schema=Configuration,
)

supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")

supervisor_subgraph = supervisor_builder.compile()


# ── Build the Main Graph ──

deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration,
)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the graph
deep_researcher = deep_researcher_builder.compile()
