# ============================================
# Dockerfile (CPU 版本)
# 适用于：无 GPU 的服务器、笔记本、CI/CD
# 镜像体积小，embedding 模型跑在 CPU 上
# ============================================

# ---- 基础镜像 ----
FROM python:3.11-slim

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 复制构建所需文件 ----
COPY pyproject.toml README.md ./
COPY src/ src/

# ---- 安装依赖（强制 CPU 版 PyTorch，减小 ~2GB 镜像体积）----
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "."

# ---- 复制其余文件 ----
COPY web/ web/
COPY langgraph.json .

# ---- 创建数据目录 ----
RUN mkdir -p knowledge_base chroma_db

EXPOSE 8000

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
