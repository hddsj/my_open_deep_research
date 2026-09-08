# ROADMAP

已知问题、成因分析与排期。按「修复价值 / 修复成本」排序，不按发现顺序。

标注说明：
- **成本** —— 粗略估计，`S` 数行改动、`M` 半天内、`L` 一天以上
- **症状可见性** —— 这个问题会不会自己暴露出来。不可见的优先级更高，因为它不会有人来报 bug

---

## 已完成

### `max_react_tool_calls` 从死配置改为真正的工具调用总数上限

原本定义了但零引用（`configuration.py:65`）。没有直接删掉，而是接上了一道此前完全不存在的防线：`researcher_tools` 原来只按查询复杂度检查"轮次"（`princlple` 字典），管不住"某一轮 LLM 并行发起异常多工具调用"的情况——这个维度过去从未被强制过。

逻辑抽成纯函数 `_check_research_limits`（`deep_researcher.py`），两道防线（轮次 / 工具调用总数）是 `or` 关系，任一触发即停，互相独立、不冲突。默认值 `18`：按 `complex` 档正常预估（4 轮 × 每轮 3 个工具调用 ≈ 12）留 1.5 倍余量，保证它只在异常情况下才会先于轮次上限触发，不会变成事实上的主限制。

`tests/test_research_limits.py` 四个断言覆盖：两道防线各自单独触发、都不触发时继续、`>=` 边界（累计恰好等于上限也算超限，不能等到超过才停）。每条都验证过重新引入对应缺陷会红：`or`→`and` 时两条依赖"任一触发"的断言红；`>=`→`>` 时只有边界那条红。

### BM25 路补上了 `type=child` 过滤

原来 `scores.argsort()[-top_k:][::-1]` 直接从全库（parent+child 混着）切最高的 `top_k` 个，parent 不是均匀分布、会连续聚集（实测某条查询排名 11~14 连续 4 个 parent），导致真正参与融合的 child 候选数量少于 `top_k`。

改用扫描收集：从完整排名（不再提前切片）从高到低遍历，只收集 `type=child`，凑够 `top_k` 就停；候选池不够时自然返回较少数量，不报错——这是检索系统的标准行为（Elasticsearch/Dify 等社区讨论一致）。逻辑抽成纯函数 `_collect_top_child_indices`（`knowledge_base.py`），不碰真实索引，可以直接单测。融合阶段原有的 `if meta.get("type") == "parent": continue` 已删除——收集阶段已经保证只有 child，这行成了死代码。

**实测确认修复生效**：用同一份索引对比改前改后的最终 top5——parent 污染轻的查询（1 个 parent）无变化，符合预期（缺口没能挤进最终排名）；污染重的查询（3 个 parent）top5 逐位置比较只有 3/5 相同，证明修复真的改变了融合结果，不是空转。

`tests/test_bm25_child_filter.py` 四个断言：全是 child 的基准情况、parent 连续聚集时被正确跳过、候选不够时不崩溃、扫描计数精确匹配实测数字（15）。每条都验证过重新引入对应缺陷会红。

副产品：验证过程中发现了 P2-7「去重键」问题的具体实例——两个不同来源的 child 反查到同一个 parent 后，最终结果里出现两条内容相同的记录，因为去重发生在反查之前。已补进对应条目，不在本次修复范围内。

---

## P0 — 功能实际未生效

### 1. LinUCB 奖励回填打在错误的动作上

**成本** `S` · **症状可见性** 无

`evaluate_report` 用下面两行决定奖励更新哪个动作：

```python
state.get("query_complexity", "medium")
state.get("source_routing", "both")
```

但 `AgentState`（`state.py:93`）里没有这两个字段 —— 它们只定义在 researcher 子图的 `ResearcherState`（`state.py:125`、`state.py:143`），而 `ResearcherOutputState` 虽然声明了 `source_routing`，值也没有被 `compress_research` 的返回值填上。

于是 `state.get()` 每次都落到默认值，`bandit_update_reward` 永远更新 `ACTIONS[2]`（`both`）。`local` 和 `web` 的 `A` 矩阵保持初始单位矩阵、`b` 保持零向量。

**后果不只是"学得慢"**：`select_action` 的探索加成 `alpha * sqrt(xᵀA⁻¹x)` 对没有数据的动作最大，所以 LinUCB 会持续偏向那两个从未被更新的动作。**学习方向是错的，不是慢。**

外部完全看不出来：`bandit_model.json` 在更新、`total_updates` 在递增、路由决策照常产出。

**修复方向**：让 researcher 子图把这两个字段透传回主图。`compress_research` 的返回值里补上，`ResearcherOutputState` 补 `query_complexity`，主图 `AgentState` 加对应字段。注意多个 researcher 并发时会有多个值回传，需要决定用哪一个（或者改成按 researcher 粒度回填奖励 —— 这更符合 bandit 的语义，因为路由决策本来就是每个 researcher 各自做的）。

> 顺带记录：这是本项目第四个「静默失效」类 bug，前三个见 README。共同成因是**状态或数据在跨边界时丢失，而流程仍然跑通**。

---

## P1 — 影响可信度

### 2. 检索评测集尚未审阅，缺少混合检索 vs 单路的对比数字

**成本** `L` · **症状可见性** 高（一问就露）

`eval/make_queryset.py` 已能生成草稿，但 `queryset_draft.json` 尚未产出，`reviewed` 字段全为 `false`。目前所有关于检索设计的论述都基于原理，没有实验支撑。

**为什么成本高**：LLM 反推的问题偏字面匹配，会系统性高估 BM25，必须人工逐条改写或剔除。这一步没法自动化，否则偏差就带进评测集了。

**修复方向**：
1. 跑 `make_queryset.py` 生成草稿
2. 人工审 20～30 条（改写措辞、剔除明显抄原文的），标 `reviewed: true`
3. 在这个集合上分别跑「纯向量」「纯 BM25」「RRF 混合」「混合 + 重排」四档，算 Recall@k / MRR
4. 顺手用同一个集合验证 `probes/probe_fusion_diff.py` 的结论

### 3. Corrective RAG 的质量判定依赖关键词计数

**成本** `S` · **症状可见性** 低

`_evaluate_observations` 用 `content.count("不相关")` 统计不相关条数（`deep_researcher.py:370`）。LLM 只要在回答里写一句总结（「以下结果均与主题不相关」），计数就偏了；理论上 `irrelevant_count` 可以超过 `total`，让比例大于 1。

**修复方向**：改用 `with_structured_output`，让模型返回 `list[bool]` 或每条结果的 id + 判定，长度对不上直接报错。

### 4. Docker 未挂载 BM25 缓存

**成本** `S` · **症状可见性** 高（换机器就撞上）

`docker-compose.yml` 的 `x-common` 只挂了 `knowledge_base` 和 `chroma_db`，`bm25_cache.pkl`（约 19 MB）既未挂载也未 `COPY` 进镜像。

容器重建 → BM25 缓存丢失、ChromaDB 仍在 → 命中 `build_index` 的一致性校验 `RuntimeError`。用户此时必须手动删 `chroma_db/` 才能继续，且要等 30 分钟重建。

**修复方向**：给 `x-common` 的 `volumes` 加一条 `./bm25_cache.pkl:/app/bm25_cache.pkl`。注意宿主机上文件不存在时 Docker 会创建成目录，需要先 `touch` 或改成挂载一个 `cache/` 目录并调整 `cache_path`。后者更干净。

### 5. `max_researcher_iterations` 的 UI 控件名不副实

**成本** `S` · **症状可见性** 中（演示时会被发现）

`web/index.html:859` 的 `cfgIterations` 把这个值做成了可调控件，经 `web/server.py:109` 传进 `configurable`。但代码里它只被 `lead_researcher_prompt` 引用（`prompts.py:283`），是给模型的软约束；单个 researcher 的真实轮次上限由复杂度分类硬编码在 `researcher_tools`（`simple` 2 / `medium` 3 / `complex` 4）。

**修复方向**：二选一 —— 接线到真实上限（把 `princlple` 字典改成可配置的基线 + 复杂度系数），或者把控件改名为「Supervisor 任务数建议」并明确它只影响规划提示。

`max_react_tool_calls` 已接线（见下方「已完成」），不再是死配置。

---

## P2 — 正确性没问题，但工程上不干净

### 6. `search()` 是同步阻塞函数，挂在 async 工具下

**成本** `M` · **症状可见性** 低（只表现为慢）

`local_knowledge_search`（`utils.py:355`）是 async 工具，但它调用的 `knowledge_base.search()`（`knowledge_base.py:343`）是纯同步的：Chroma 查询、jieba 分词、Cross-Encoder 推理全在 event loop 里跑。

Supervisor 用 `asyncio.gather` 并发派 3 个 researcher，但它们在本地检索这一段实际是串行的 —— 并发只在等 LLM 和等网络时才真的并发。

**修复方向**：`asyncio.to_thread` 包一层，或者引入线程池。注意 `_bm25_index` / `_reranker` / `_semantic_chunker` 都是模块级全局单例，多线程访问需要确认 `BM25Okapi.get_scores` 和 `CrossEncoder.predict` 的线程安全性（后者底层是 PyTorch，推理本身安全，但要避免同时加载）。

### 7. 融合与重排粒度不一致，且有 N+1 查询

**成本** `M` · **症状可见性** 无

三个独立问题挤在 `search()` 的后半段：

- **粒度**：RRF 用 child 文本的排名融合，之后把 `item["document"]` 换成 parent 全文，Cross-Encoder 对 **parent** 打分。两级排序的输入不是同一个东西。
- **N+1**：parent 反查是逐条 `collection.get(ids=[parent_id])` 循环（`knowledge_base.py:398-404`），应该批量 `get(ids=[...])` 一次取回。
- **去重键**：`key = doc[:100]`，键是在反查 parent **之前**算的。两个不同的 child（原文不同，所以没被当成重复）反查后可能指向**同一个 parent**，最终结果里就会出现两条内容完全相同的记录——修「BM25 候选混进 parent」那个 bug（见上方「已完成」）时实测过一次：查询"容器之间怎么互相通信"，top5 里出现了两条 page=140 的相同内容。反过来，前 100 字恰好相同的不同 chunk（技术书章节开头的模板化段落很容易撞）也会被误判成同一条，在反查前就被错误合并。两个方向都是同一个根因：**去重键选错了阶段**，该在反查之后按最终展示内容（或 parent_id）去重，不是在反查之前按 child 原文去重。

**修复方向**：去重键换成反查后的 parent_id（或最终 document 内容）；parent 反查批量化；粒度问题需要先做决定 —— 是在 child 上重排后再取 parent（重排更准，但 LLM 拿到的上下文没被评估过），还是保持现状。这个决定应该等 P1-2 的评测集就绪后用数据回答。

### 8. 研究记忆的分层摘要会无限累积

**成本** `M` · **症状可见性** 低（渐进劣化）

三个互相放大的问题：

- `_reflect_on_memories` 每次触发都往同一个 collection 新增 `topic_summary` 和 `reflection` 文档（`memory.py:95`、`memory.py:113`），**从不清理旧的**。同一个 topic 会积累多份摘要。
- 触发条件是 `_memory_collection.count() % 5 == 0`（`memory.py:146`），而 `count()` 把这些摘要文档也算进去了，所以触发时机会随摘要增多而漂移。
- `retrieve_memory` 不按 `type` 过滤，会把 `reflection`（跨主题元认知总结）当普通研究记忆检索出来注入 researcher 的 prompt。

另外 `_get_memory_collection` 仍然缓存 collection 对象（`memory.py:22`）—— 知识库侧已在 `ef9b7e6` 修掉这个悬挂引用问题，记忆侧没跟上。

**修复方向**：`type != "research"` 的文档不计入触发计数；生成新摘要前先删同 topic 的旧摘要；`retrieve_memory` 加 `where={"type": "research"}`；去掉 collection 缓存。

### 9. Web 层缺少并发保护

**成本** `M` · **症状可见性** 低（单用户时不出现）

- `research_sessions` 是进程内全局 dict，`save_sessions()` 每次全量重写 `sessions.json`，无锁 —— 两个并发请求同时结束会互相覆盖
- `MemorySaver` 只在单进程内有效，多 worker 部署时 `interrupt` 恢复会找不到 checkpoint
- `/session/{id}` 和 `/followup` 对不存在的 session 返回 `{"error": ...}` 而不是 404

**修复方向**：单机可以用 `asyncio.Lock` + 增量写；要支持多 worker 则需要换 SQLite/Redis checkpointer 和外部 session 存储。这取决于是否真的要多 worker 部署，暂不动。

---

## 尚未接入 CI

`tests/` 下五个测试都能跑，但没有 workflow。

`test_rewrite_args.py`、`test_graph_wiring.py`、`test_research_limits.py`、`test_bm25_child_filter.py` 无外部依赖（不需要 API key，不联网），可以直接接上——后三者首次导入都会拉起 torch 或整套 knowledge_base 依赖，约十几秒到一分钟不等。

`test_kb_mode_parity.py` 依赖 MCP 服务在线（未启动时 skip 并打印原因），接 CI 需要在 job 里先把服务拉起来。

---

## 仓库根目录的遗留脚本

根目录还有约 15 个 `test_*.py`（`test_ranking.py`、`test_crag.py`、`test_retrieval_v2.py` 等），是开发期的手动验证脚本，不是 pytest 能收集的测试，已被 `.gitignore` 排除。

要么删掉，要么挑几个有价值的迁进 `tests/` 改写成真断言 —— 尤其是 `test_ranking.py` 和 `test_retrieval_v2.py`，它们覆盖的正是 P2-7 涉及的检索链路。
