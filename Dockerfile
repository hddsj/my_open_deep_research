# ---- 第 1 层：基础镜像 ----
# 选择 Python 3.11 精简版，体积小
FROM python:3.11-slim

# ---- 第 2 层：系统依赖 ----
# 安装编译工具（某些 Python 包需要 C 编译器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- 第 3 层：工作目录 ----
# 在容器里创建 /app 目录，后续所有操作都在这里
WORKDIR /app

# ---- 第 4 层：先复制依赖文件 ----
# 为什么不直接 COPY . . ？
# 因为 Docker 有"层缓存"机制：如果这一层没变化，就不会重新执行。
# 依赖文件很少变，代码经常变。分开复制可以避免每次改代码都重装依赖。
COPY pyproject.toml .

# ---- 第 5 层：安装 Python 依赖 ----
RUN pip install --no-cache-dir -e "."

# ---- 第 6 层：复制项目源代码 ----
COPY src/ src/
COPY web/ web/
COPY langgraph.json .

# ---- 第 7 层：创建数据目录 ----
# 这些目录在运行时会用到，先创建好
RUN mkdir -p knowledge_base chroma_db

# ---- 第 8 层：暴露端口 ----
# 告诉 Docker 这个容器会用到 8000 端口（仅声明，实际映射在 docker run 时指定）
EXPOSE 8000

# ---- 第 9 层：启动命令 ----
# 容器启动时执行的命令：用 uvicorn 启动 FastAPI 服务
# host=0.0.0.0 表示容器内外都能访问（不写的话只有容器内部能访问）
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
