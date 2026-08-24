"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)

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
    suggest_followup_prompt,
    generate_outline_prompt,
    evaluate_report_prompt,
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from my_deep_research.memory import save_memory, retrieve_memory


logger = logging.getLogger(__name__)
# 创建的是一个可配置的模型模板，还没有指定具体用哪个模型
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
        return Command(goto="generate_outline")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    # 根据模型模版创建实例
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
            goto="generate_outline",
            update={"messages": [AIMessage(content=response.verification)]},
        )

async def generate_outline(
    state: AgentState, config: RunnableConfig
):
    """Generate a research outline based on the user's request."""
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    research_model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    prompt_content = generate_outline_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
    )
    
    response = await research_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )

    outline = response.content  # LLM 返回的大纲文本
    user_response = interrupt(outline)  # 暂停！把大纲发给用户，等用户确认  
    
    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=user_response)]},
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
            "supervisor_messages": {"type": "override", "value": [HumanMessage(content=response.research_brief)]},
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

    # 首轮分类，后续轮复用
    if state.get("tool_call_iterations", 0) == 0:
        complexity, routing = await _classify_query(state["research_topic"], config)
    else:
        complexity = state.get("query_complexity", "medium")
        routing = state.get("source_routing", "both")

    # 根据 routing 过滤工具
    local_tools = {"local_knowledge_search"}
    web_tools = {"tavily_search", "duckduckgo_search_tool"}
    if routing == "local":
        tools = [t for t in tools if t.name not in web_tools]
    elif routing == "web":
        tools = [t for t in tools if t.name not in local_tools]
    try:
        # 检索历史研究记忆，如果存在相关记忆则注入 prompt 供 LLM 参考
        memory_context = retrieve_memory(state["research_topic"], top_k=3)
    except Exception as e:
        logger.error(f"[researcher] 检索历史研究记忆失败: {e}")
        memory_context = ""

    if memory_context:
        logger.info(f"[researcher] 注入历史研究记忆:\n{memory_context}")
        researcher_prompt += f"\n\n<Past Research>\n{memory_context}\n</Past Research>"
    else:
        logger.info("[researcher] 无相关历史研究记忆")
        
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
            "query_complexity": complexity,
            "source_routing": routing,
        },
    )

async def _compress_observation(observation: str, research_topic: str, config: RunnableConfig) -> str:
    """对搜索工具返回的原始结果进行上下文压缩，只保留与研究主题相关的核心内容。
    
    短文本（<200字）直接跳过；LLM调用失败时降级返回原文。
    
    Args:
        observation: 搜索工具返回的原始文本。
        research_topic: 当前研究主题，用于指导LLM提取相关内容。
        config: 运行时配置，包含模型参数和API key。
    
    Returns:
        压缩后的文本，或在短文本/失败时返回原文。
    """
    # 模型初始化
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })  
    
    # 执行压缩逻辑：
    #     1.短文本跳过：如果 observation 很短（比如 < 200 字），没必要压缩，直接返回
    #     2.构造 prompt：告诉 LLM "从以下搜索结果中，只提取跟 {research_topic} 相关的核心内容"
    #     3.调 LLM 返回压缩结果，加 try-except 降级（失败就返回原文
    
    # 短文本跳过
    if len(observation) < 200:
        logger.info(f"[compress_observation] 短文本({len(observation)}字)跳过压缩")
        return observation
    
    # 构造 prompt
    prompt = (
        f"请从以下搜索结果中，只提取与'{research_topic}'直接相关的核心内容，"
        "去掉广告、导航、无关段落，保持简洁。\n\n"
        f"【搜索结果】：\n{observation}"
    )

    # 调 LLM 返回压缩结果
    try:
        response = await model.ainvoke(prompt)
        logger.info(f"[compress_observation] 压缩完成: {len(observation)}字 → {len(response.content)}字")
    except Exception as e:
        logger.warning(f"[compress_observation] LLM 调用失败: {e}, 返回原文")
        return observation
    
    return response.content

async def _evaluate_observations(observations: list, tool_calls: list, research_topic: str, config: RunnableConfig) -> bool:
    # 模型初始化
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })

    # 将多个搜索结果拼成一个完整的文本
    search_tools = ("tavily_search", "duckduckgo_search_tool", "local_knowledge_search")
    numbered = []
    for i, obs in enumerate(observations):
        if tool_calls[i]["name"] in search_tools:
            numbered.append(f"{i+1}. {obs}")
    results_text = "\n".join(numbered)

    # 拼接prompt
    prompt = (
        f"请评估以下搜索结果与研究主题'{research_topic}'的相关性。\n"
        "对每条结果回答：相关 或 不相关。\n\n"
        f"{results_text}"
    )

    # 执行LLM调用
    response = await model.ainvoke(prompt)

    # 统计不相关的数量
    content = response.content
    irrelevant_count = content.count("不相关")
    total = len(numbered)
    if total == 0:
        return False
    
    # 计算不相关的比例
    irrelevant_ratio = irrelevant_count / total
    logger.info(f"[evaluate_observations] 不相关占比: {irrelevant_count}/{total} = {irrelevant_ratio:.1%}")
    return irrelevant_ratio > 0.5

async def _rewrite_query(research_topic: str, config: RunnableConfig) -> str:
    # 模型初始化
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })
    # 定义prompt
    prompt = (
        f"原始研究主题是'{research_topic}'，但搜索结果大部分不相关。\n"
        "请改写一个更精准的搜索查询，要求：\n"
        "1. 使用更具体的关键词\n"
        "2. 避免过于宽泛的表述\n"
        "3. 只输出改写后的查询，不要解释\n"
    )
    try:
        response = await model.ainvoke(prompt)
        rewritten = response.content.strip().strip('"').strip("'")
        logger.info(f"[rewrite_query] 改写查询: '{research_topic}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"[rewrite_query] 改写失败: {e}, 使用原始查询")
        return research_topic

async def _classify_query(research_topic: str, config: RunnableConfig) -> str:
    # 模型初始化
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })
    # 定义prompt
    prompt = prompt = (
        f"请判断研究主题'{research_topic}'的两个属性，用 | 分隔输出：\n\n"
        "1. 复杂度（三选一）：\n"
        "- simple：有明确答案的事实性问题（如'Docker默认网段是什么'）\n"
        "- medium：需要理解原理或概念（如'Docker网络原理'）\n"
        "- complex：涉及对比、多维分析、跨领域（如'对比K8s三种CNI方案的性能差异'）\n\n"
        "2. 数据源（三选一）：\n"
        f"- local：主题属于本地知识库覆盖范围（{configurable.knowledge_base_description}）\n"
        "- web：需要最新信息、或主题不在本地知识库范围内\n"
        "- both：既需要本地基础知识，又需要网络补充最新实践\n\n"
        "输出格式：complex | both\n"
        "只输出一行，不要解释。"
    )
    try:
        response = await model.ainvoke(prompt)
        # 将输出结果（"complex | both"）拆分为tuple
        parts = response.content.strip().split("|")
        complexity = parts[0].strip()
        routing = parts[1].strip() if len(parts) > 1 else "both"
        logger.info(f"[classify_query] 分类查询: '{research_topic}' → '{complexity} | {routing}'")
        return complexity, routing
    except Exception as e:
        logger.warning(f"[classify_query] 分类失败: {e}, 默认返回'medium | both'")
        return "medium", "both"

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

    complex_classify = state["query_complexity"]
    princlple = {"simple": 2, "medium": 3, "complex": 4}
    logger.info(f"[researcher_tools] 查询复杂度: {complex_classify}, 最大搜索轮次: {princlple.get(complex_classify, 3)}")
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
    # tool_calls 长这样：
    # [
    #   {"name": "tavily_search", "args": {"query": "Docker网络原理"}, "id": "call_123"},
    #   {"name": "think_tool", "args": {"thought": "..."}, "id": "call_456"},
    # ]
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config)
        for tc in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    # 假设 tool_calls 有 3 个调用：搜索、think、搜索
    # observations = (
    #     "Docker bridge 网络使用 veth pair... 首页|关于我们|广告...",  # [0] tavily_search 返回的网页内容
    #     "我需要从网络和存储两个维度分析...",                          # [1] think_tool 返回的思考
    #     "VXLAN 隧道协议实现跨主机通信... 推荐阅读...",               # [2] duckduckgo_search 返回的网页内容
    # )

    # 对搜索结果进行上下文压缩
    observations = [
        await _compress_observation(obs, state["research_topic"], config)
        if tc["name"] in ("tavily_search", "duckduckgo_search_tool", "local_knowledge_search")
        else obs
        for obs, tc in zip(observations, tool_calls)
    ]

    is_low_quality = await _evaluate_observations(observations, tool_calls, state["research_topic"], config)
    logger.info(f"[researcher_tools] 检索质量评估: {'不合格，触发改写重搜' if is_low_quality else '合格'}")

    if is_low_quality:
        logger.info(f"[researcher_tools] 检索质量不合格，触发查询改写重搜")
        rewritten_query = await _rewrite_query(state["research_topic"], config)
        search_tools = ("tavily_search", "duckduckgo_search_tool", "local_knowledge_search")
        observations = list(observations)  # tuple 转 list 才能赋值
        for i, tc in enumerate(tool_calls):
            if tc["name"] in search_tools:
                # 把原来的 {"query": "Docker网络原理"} 换成 {"query": "改写后的查询"}
                new_args = {**tc["args"], "query": rewritten_query}
                # 重新执行搜索
                logger.info(f"[researcher_tools] 重新搜索: tool={tc['name']}, query='{rewritten_query}'")
                new_result = await execute_tool_safely(tools_by_name[tc["name"]], new_args, config)
                # 替换旧结果
                observations[i] = new_result
        logger.info(f"[researcher_tools] 改写重搜完成，已替换搜索结果")
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
        state.get("tool_call_iterations", 0) >= princlple.get(complex_classify, 3)
    )

    if exceeded_iterations:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs,
             "total_tool_calls": state.get("total_tool_calls", 0) + len(tool_calls),
             "rewrite_count": state.get("rewrite_count", 0) + (1 if is_low_quality else 0),
             "forced_stop": exceeded_iterations},
        )

    # Continue research loop
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs,
        "total_tool_calls": state.get("total_tool_calls", 0) + len(tool_calls),
        "rewrite_count": state.get("rewrite_count", 0) + (1 if is_low_quality else 0),
        "forced_stop": exceeded_iterations},
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
    logger.info(
        f"[trajectory] 搜索轮次: {state.get('tool_call_iterations', 0)}, "
        f"工具调用: {state.get('total_tool_calls', 0)}, "
        f"改写次数: {state.get('rewrite_count', 0)}, "
        f"强制停止: {state.get('forced_stop', False)}"
    )
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
        # 存储压缩后的研究结果到memory（基于Chromadb实现）
        try:
            save_memory(state["research_topic"], str(response.content))
        except Exception as e:
            logger.error(f"[compress_research] 存储压缩后的研究结果到memory失败: {e}")
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

async def evaluate_report(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    research_loops = state.get("research_loops", 0)

    # 已达最大轮次，直接结束
    if research_loops >= configurable.max_research_loops:
        return Command(
            goto=END,
        )
    
    # 配置模型
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # 调用 LLM 评估
    prompt_content = evaluate_report_prompt.format(
        question=state.get("research_brief", ""),
        report=state.get("final_report", ""),
    )
    response = await model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    # 解析结果
    logger.info(f"[evaluate_report] 轮次: {research_loops}/{configurable.max_research_loops}")
    if "VERDICT: PASS" in response.content:
        logger.info("[evaluate_report] ✅ VERDICT: PASS — 报告通过")
        return Command(goto=END)

    # VERDICT: NEEDS_MORE，继续研究
    # 只提取 GAPS 部分
    gaps = ""
    logger.info(f"[evaluate_report] ❌ VERDICT: NEEDS_MORE — 需要补充研究")
    if "GAPS:" in response.content:
        logger.info(f"[evaluate_report] GAPS:\n{response.content.split('GAPS:')[1].strip()}")
        gaps = response.content.split("GAPS:")[1].strip()
    else:
        gaps = response.content
    # 把 gaps 作为消息加入，让 write_research_brief 重新生成研究计划
    return Command(
        goto="write_research_brief",
        update={
            "messages": [HumanMessage(content=f"Please conduct additional research on the following gaps:\n{gaps}")],
            "research_loops": 1,
            "supervisor_messages": {"type": "override", "value": []},
        },
    )   

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

    logger.info(f"[追问判断] needs_research={result.needs_research}, research_topic={result.research_topic}")
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

async def suggest_followups(final_report: str, config: RunnableConfig) -> list:
    configurable = Configuration.from_runnable_config(config)
    api_key = get_api_key_for_model(configurable.research_model, config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
    }
    if api_key:
        model_config["api_key"] = api_key
    model = configurable_model.with_config(configurable=model_config)
    prompt = suggest_followup_prompt.format(final_report=final_report)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    return [line.split('. ', 1)[1] for line in result.content.strip().split('\n') if '. ' in line]

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
deep_researcher_builder.add_node("generate_outline", generate_outline)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("evaluate_report", evaluate_report)  

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("clarify_with_user", "generate_outline")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "evaluate_report") 

# Compile the graph
deep_researcher = deep_researcher_builder.compile(checkpointer=MemorySaver())
