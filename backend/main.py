import asyncio
import datetime
import json
import logging
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
            data = await asyncio.wait_for(websocket.receive_text(), timeout=120)
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

            def datetime_converter(o):
                if isinstance(o, datetime.datetime):
                    return o.isoformat()
                raise TypeError("Type not serializable")

            await websocket.send_json(json.loads(json.dumps({
                "stage": "searching",
                "status": "Complete",
                "data": {
                    "count": len(state.get('raw_papers', [])),
                    "papers": state.get('raw_papers', [])[:5]  # Send first 5 for preview
                }
            }, default=datetime_converter)))
            

            # Rank papers
            await websocket.send_json({
                "stage": "ranking",
                "status": "Ranking papers..."
            })
            state = await workflow.rank_papers_node(state)
            await websocket.send_json(json.loads(json.dumps({
                "stage": "ranking",
                "status": "Complete",
                "data": {
                    "top_papers": state.get('ranked_papers', [])[:5]
                }
            }, default=datetime_converter)))
            

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
            await websocket.send_json(json.loads(json.dumps({
                "stage": "complete",
                "status": "Research complete",
                "data": {
                    "document": state.get('final_document'),
                    "papers": state.get('filtered_papers')
                }
            }, default=datetime_converter)))

    except WebSocketDisconnect as e:
        # Нормальное закрытие вебсокета
        logging.info(f"WebSocket connection closed normally with code {e.code}.")

    except Exception as e:
        tb_str = traceback.format_exc()
        print(tb_str, flush=True)
        await websocket.send_json({
            "stage": "error",
            "status": str(e)
        })
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logging.info(f"Could not close Websocket with exception {e}.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
