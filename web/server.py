import asyncio
import json
import uuid
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse,HTMLResponse
from dotenv import load_dotenv
from my_deep_research.deep_researcher import deep_researcher, handle_followup, suggest_followups
load_dotenv()
from pathlib import Path
from datetime import datetime

app = FastAPI()

SESSIONS_FILE = Path(__file__).parent / "sessions.json"

def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(research_sessions, f, ensure_ascii=False, indent=2)

def load_sessions():
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}    


# 存储每次研究的上下文，追问时用
research_sessions = load_sessions()



node_messages = {
    "write_research_brief": "正在生成研究计划...",
    "supervisor": "Supervisor 正在规划研究任务...",
    "researcher": "研究员正在搜索...",
    "compress_research": "正在压缩研究结果...",
    "final_report_generation": "正在生成最终报告...",
}

async def event_generator(query: str, session_id: str, config: dict):
    inputs = {"messages": [{"role": "user", "content": query}]}
    
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
        if event["event"] == "on_tool_start" and event["name"] == "tavily_search":
            queries = event["data"].get("input", {}).get("queries", [])
            for q in queries:
                yield f"data: {json.dumps({'type': 'progress', 'message': f'正在搜索: {q}'})}\n\n"
    yield f"data: {json.dumps({'type': 'report', 'content': final_report})}\n\n"
    suggestions = await suggest_followups(final_report, config)
    yield f"data: {json.dumps({'type': 'suggestions', 'questions': suggestions})}\n\n"
    research_sessions[session_id] = {
        "query": query,
        "created_at": datetime.now().isoformat(),
        "notes": notes,
        "final_report": final_report
    }
    save_sessions()

@app.post("/research")
async def research(request: Request):
    data = await request.json()
    query = data.get("query", "")
    session_id = data.get("session_id", str(uuid.uuid4()))
    config = {"configurable": {
        "allow_clarification": False,
        "research_model": data.get("research_model", "deepseek-chat"),
        "max_search_results": data.get("max_search_results", 5),
        "max_researcher_iterations": data.get("max_researcher_iterations", 5),
        "max_concurrent_research_units": data.get("max_concurrent_research_units", 5),
    }}
    return StreamingResponse(event_generator(query, session_id, config), media_type="text/event-stream")

@app.post("/followup")
async def followup(request: Request):
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id", "")
    if session_id not in research_sessions:
        return {"error": "Session not found"}
    session = research_sessions[session_id]
    config = {"configurable": {"allow_clarification": False}}
    async def followup_stream():
        async for event in handle_followup(question, session["notes"], session["final_report"], config):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(followup_stream(), media_type="text/event-stream")

@app.get("/")
async def index():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)

@app.get("/sessions")
async def list_sessions():
    return [{"id": k, "query": v.get("query",""), "created_at": v.get("created_at","")} 
            for k, v in research_sessions.items()]

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in research_sessions:
        return {"error": "Session not found"}
    s = research_sessions[session_id]
    return {"query": s["query"], "final_report": s["final_report"]}



