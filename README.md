# My Deep Research Agent

深度研究 Agent，基于 LangGraph 构建。输入一个研究问题，自动执行多轮检索、分析、压缩，最终输出结构化研究报告。

## 出处与分工

本项目以 LangChain 的 [open_deep_research](https://github.com/langchain-ai/open_deep_research)（MIT License）为起点。

**沿用上游的部分**：Multi-Agent 编排骨架 —— `supervisor` / `researcher` / `compress_research` / `final_report_generation` 等节点，`ConductResearch` / `ResearchComplete` / `think_tool` 工具契约，以及对应的 prompt。

**本项目自研的部分**：

| 模块 | 内容 |
|------|------|
| **本地知识库混合检索** | PDF 解析、语义分块、Parent-Child 分块、ChromaDB + BM25 双路召回、RRF 融合、Cross-Encoder 重排、增量索引 |
| **LinUCB 自适应路由** | 手写 Contextual Bandit（NumPy），按查询特征在 local / web / both 之间在线学习路由策略 |
| **MCP 双模式** | 知识库检索可作为标准 MCP 服务暴露；Agent 支持 `direct` / `mcp` 切换，且拒绝静默降级 |
| **研究记忆** | 跨研究的记忆存储、LLM 合并、分层摘要、时间衰减排序（探索性实现） |

## 核心特性

### 研究流程

- **Multi-Agent 协作** — Supervisor 并行调度多个 Researcher sub-agent，各自独立搜索和分析
- **Adaptive RAG** — 根据查询复杂度（simple/medium/complex）动态调整迭代策略
- **Multi-Source Routing** — 自动选择 local/web/both 数据源，匹配最适合的知识来源
- **Corrective RAG** — 检索结果质量低时自动改写查询并重新搜索
- **Human-in-the-Loop** — 研究前可暂停等待用户确认研究大纲

### 知识库

- **本地知识库** — PDF 提取 + ChromaDB 向量检索 + BM25 关键词检索 + Reranker 重排序
- **语义分块** — 跨页合并 + chunk 大小控制
- **Parent-Child Chunking** — child 精确匹配，返回 parent 完整上下文
- **增量索引** — 多书并行，支持增量更新
- **MCP 服务化** — 知识库检索可作为标准 MCP 服务暴露，供任意 MCP 客户端调用；Agent 自身支持 `direct` / `mcp` 两种接入模式切换

### 记忆系统

- **研究记忆** — ChromaDB 向量存储与检索；相似记忆经 LLM 合并；按 topic 字段分组后由 LLM 生成主题摘要；检索时按「相关性 × 时间衰减」重排（探索性实现，未做量化验证）

### 自适应路由（LinUCB）

- **Contextual Bandit** — LinUCB 算法根据查询特征学习最优路由策略
- **冷启动保护** — 前 10 次研究使用 LLM 路由，积累数据后切换到 LinUCB
- **在线学习** — 每次研究结束后根据报告质量自动更新模型
- **持久化** — 模型参数保存到 JSON，重启不丢失

### 质量保障

- **上下文压缩** — 搜索结果经摘要压缩后再送 LLM，减少噪声
- **Context Window 管理** — 防止 token 溢出
- **检索质量评估** — MRR / Precision@K / NDCG 指标
- **报告质量评估** — LLM-as-Judge 自动评分，驱动迭代改进

## 项目结构

```
src/my_deep_research/
├── deep_researcher.py   # LangGraph 主流程（Agent 编排、节点定义）
├── state.py             # 状态定义（AgentState、ResearcherState 等）
├── configuration.py     # 配置管理（模型、搜索 API、参数）
├── prompts.py           # 所有 Prompt 模板
├── utils.py             # 工具函数（搜索、网页抓取、token 计算）
├── knowledge_base.py    # 本地知识库（PDF 解析、ChromaDB、BM25）
├── mcp_server.py        # MCP 服务端（将知识库检索暴露为 MCP 工具）
├── memory.py            # 记忆系统（ChromaDB存储、检索、反思）
├── evaluation.py        # 评估函数（检索质量、报告质量）
└── bandit.py            # LinUCB 自适应路由

web/
├── server.py            # Web 服务端
└── index.html           # Web 前端界面
```

## 快速开始

### 方式一：Docker 部署（推荐，换电脑也能一键启动）

只需安装 [Docker](https://docs.docker.com/get-docker/)，不需要装 Python 和任何依赖。

#### 1. 克隆项目

```bash
git clone <repo-url>
cd my_deep_research_2
```

#### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key        # 可选，默认用 DuckDuckGo
LANGSMITH_API_KEY=your_langsmith_api_key  # 可选，用于追踪调试
```

#### 3. 放入知识库文件（可选）

将 PDF 文件放入 `knowledge_base/` 目录。

#### 4. 构建并启动

```bash
# 构建镜像并启动容器（第一次约 5-10 分钟）
docker-compose up --build

# 或后台运行
docker-compose up --build -d
```

#### 5. 使用

浏览器打开 http://localhost:8000 即可使用。

#### 常用命令

```bash
# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重新构建（代码改动后）
docker-compose up --build
```

> **说明：** 知识库数据（ChromaDB、BM25 缓存）通过数据卷映射到宿主机，容器重启不会丢失。

---

### 方式二：本地开发

#### 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

#### 安装

```bash
git clone <repo-url>
cd my_deep_research_2
uv sync
```

#### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key（同上）。

#### 运行

```bash
# Web 界面
uvicorn web.server:app --host 0.0.0.0 --port 8000

# 或用 LangGraph Studio 直接打开项目（已含 langgraph.json 配置）
```

### 本地知识库

将 PDF 文件放入 `knowledge_base/` 目录，首次运行时自动建立索引。

当前支持的知识库主题：Docker 容器化技术、Python 编程、C++ 编程、C# 编程。

### MCP 知识库服务

知识库检索可以作为标准 MCP 服务对外暴露，供任意 MCP 客户端调用。

启动服务（Streamable HTTP，默认监听 `http://127.0.0.1:8000/mcp`）：

```bash
python -m my_deep_research.mcp_server
```

用 [MCP Inspector](https://github.com/modelcontextprotocol/inspector) 验证：

```bash
npx -y @modelcontextprotocol/inspector
```

在 Inspector 中添加服务器，Transport 选 **Streamable HTTP**，URL 填 `http://127.0.0.1:8000/mcp`，
连接后即可在 Tools 面板看到并调用 `local_knowledge_search`。

让 Agent 自身改走 MCP 模式，把配置项 `kb_mode` 设为 `mcp` 即可。
两种模式共享同一份检索实现，工具名、描述与参数 schema 完全一致
（`tests/test_kb_mode_parity.py` 对此有断言；该检查需手动运行且依赖 MCP 服务在线，尚未纳入 CI）。

> MCP 服务未启动时，`kb_mode=mcp` 会直接报错并提示启动方式，
> 不会静默退回 `direct` —— 避免"以为在用 MCP，实际不是"。

## 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `research_model` | `deepseek-chat` | 研究任务模型 |
| `search_api` | `duckduckgo` | 搜索 API（tavily/duckduckgo） |
| `max_research_loops` | `2` | 最大研究迭代轮次 |
| `max_concurrent_research_units` | `3` | 最大并行研究单元数 |
| `max_search_results` | `5` | 每次搜索返回结果数 |
| `max_researcher_iterations` | `5` | 每个 Researcher 最大迭代次数 |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 向量嵌入模型 |
| `kb_mode` | `direct` | 知识库接入模式（direct 进程内 / mcp 走 MCP 服务） |
| `mcp_kb_url` | `http://127.0.0.1:8000/mcp` | MCP 知识库服务地址 |

所有配置支持通过环境变量覆盖（大写形式，如 `RESEARCH_MODEL`）。

## 技术栈

- **Agent 框架**: LangGraph
- **LLM**: DeepSeek（默认），支持切换 OpenAI 等
- **向量数据库**: ChromaDB
- **关键词检索**: BM25
- **嵌入模型**: BGE-small-zh-v1.5
- **自适应路由**: LinUCB Contextual Bandit (NumPy)
- **Web 搜索**: DuckDuckGo / Tavily
- **工具协议**: MCP (Model Context Protocol) — 服务端 FastMCP，客户端 langchain-mcp-adapters

## License

MIT
