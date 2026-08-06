import asyncio
import json
import uuid
from fastapi import FastAPI,Request
from pathlib import Path
from fastapi.responses import StreamingResponse,HTMLResponse
from dotenv import load_dotenv
from my_deep_research.deep_researcher import deep_researcher, handle_followup
load_dotenv()

app = FastAPI()

# 存储每次研究的上下文，追问时用
research_sessions = {}

node_messages = {
    "write_research_brief": "正在生成研究计划...",
    "supervisor": "Supervisor 正在规划研究任务...",
    "researcher": "研究员正在搜索...",
    "compress_research": "正在压缩研究结果...",
    "final_report_generation": "正在生成最终报告...",
}

async def event_generator(query: str, session_id: str):
    inputs = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"allow_clarification": False}}
    
    final_report = ""
    notes = []
    async for event in deep_researcher.astream_events(inputs, config=config, version="v2"):
        if event["event"] == "on_chain_start" and event["name"] in ["write_research_brief", "supervisor", 
        "researcher", "compress_research", "final_report_generation"]:
            yield f"data: {json.dumps({'type': 'progress', 'message': node_messages[event['name']]})}\n\n"
        if event["event"] == "on_chain_end" and event["name"] == "final_report_generation":
            final_report = event["data"]["output"]["final_report"]
        if event["event"] == "on_chain_end" and event["name"] == "research_supervisor":
            notes = event["data"]["output"].get("notes", [])
    yield f"data: {json.dumps({'type': 'report', 'content': final_report})}\n\n"
    research_sessions[session_id] = {"notes": notes, "final_report": final_report}

@app.post("/research")
async def research(request: Request):
    data = await request.json()
    query = data.get("query", "")
    session_id = data.get("session_id", str(uuid.uuid4()))
    return StreamingResponse(event_generator(query, session_id), media_type="text/event-stream")

@app.post("/followup")
async def followup(request: Request):
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id", "")
    if session_id not in research_sessions:
        return {"error": "Session not found"}
    session = research_sessions[session_id]
    config = {"configurable": {"allow_clarification": False}}
    answer = await handle_followup(question, session["notes"], session["final_report"], config)
    return {"answer": answer["answer"], "searched": answer["searched"]}

@app.get("/")
async def index():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
