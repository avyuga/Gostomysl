from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from workflow import ResearchWorkflow

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = ResearchWorkflow()

@app.get("/")
async def root():
    return {"message": "ArXiv Research System API"}

@app.websocket("/ws/research")
async def research_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            query_data = json.loads(data)
            user_query = query_data['query']
            
            # Create custom workflow with progress updates
            state = {
                'user_query': user_query,
                'status': 'Started'
            }
            
            # Process query
            await websocket.send_json({
                "stage": "query_processing",
                "status": "Processing query..."
            })
            state = await workflow.process_query_node(state)
            await websocket.send_json({
                "stage": "query_processing",
                "status": "Complete",
                "data": state.get('enhanced_queries')
            })
            
            # Search papers
            await websocket.send_json({
                "stage": "searching",
                "status": "Searching ArXiv..."
            })
            state = await workflow.search_papers_node(state)
            await websocket.send_json({
                "stage": "searching",
                "status": "Complete",
                "data": {
                    "count": len(state.get('raw_papers', [])),
                    "papers": state.get('raw_papers', [])[:5]  # Send first 5 for preview
                }
            })
            
            # Rank papers
            await websocket.send_json({
                "stage": "ranking",
                "status": "Ranking papers..."
            })
            state = await workflow.rank_papers_node(state)
            await websocket.send_json({
                "stage": "ranking",
                "status": "Complete",
                "data": {
                    "top_papers": state.get('ranked_papers', [])[:5]
                }
            })
            
            # Summarize papers
            await websocket.send_json({
                "stage": "summarizing",
                "status": "Creating summaries..."
            })
            state = await workflow.summarize_papers_node(state)
            await websocket.send_json({
                "stage": "summarizing",
                "status": "Complete",
                "data": {
                    "summaries": [
                        {"title": p['title'], "summary": p.get('ru_summary', '')[:200]}
                        for p in state.get('summarized_papers', [])[:3]
                    ]
                }
            })
            
            # Filter papers
            await websocket.send_json({
                "stage": "filtering",
                "status": "Filtering relevant papers..."
            })
            state = await workflow.filter_papers_node(state)
            await websocket.send_json({
                "stage": "filtering",
                "status": "Complete",
                "data": {
                    "relevant_count": len(state.get('filtered_papers', []))
                }
            })
            
            # Create analysis
            await websocket.send_json({
                "stage": "analysis",
                "status": "Creating domain analysis..."
            })
            state = await workflow.create_analysis_node(state)
            await websocket.send_json({
                "stage": "analysis",
                "status": "Complete",
                "data": {
                    "plan": state.get('analysis_plan')
                }
            })
            
            # Format document
            await websocket.send_json({
                "stage": "formatting",
                "status": "Formatting document..."
            })
            state = await workflow.format_document_node(state)
            await websocket.send_json({
                "stage": "formatting",
                "status": "Complete",
                "data": {
                    "document": state.get('final_document')
                }
            })
            
            # Send final result
            await websocket.send_json({
                "stage": "complete",
                "status": "Research complete",
                "data": {
                    "document": state.get('final_document'),
                    "papers": state.get('filtered_papers')
                }
            })
            
    except Exception as e:
        await websocket.send_json({
            "stage": "error",
            "status": str(e)
        })
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
