from typing import Dict

from agents.gost_formatter import GOSTFormatter
from agents.query_agent import QueryAgent
from agents.ranking_agent import RankingAgent
from agents.search_agent import SearchAgent
from agents.summary_agent import SummaryAgent

from langgraph.graph import END, Graph


class ResearchWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self):
        self.query_agent = QueryAgent()
        self.search_agent = SearchAgent()
        self.ranking_agent = RankingAgent()
        self.summary_agent = SummaryAgent()
        self.formatter = GOSTFormatter()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Graph:
        """Build the workflow graph"""
        workflow = Graph()
        
        # Define nodes
        workflow.add_node("process_query", self.process_query_node)
        workflow.add_node("search_papers", self.search_papers_node)
        workflow.add_node("rank_papers", self.rank_papers_node)
        workflow.add_node("summarize_papers", self.summarize_papers_node)
        workflow.add_node("format_document", self.format_document_node)
        
        # Define edges
        workflow.add_edge("process_query", "search_papers")
        workflow.add_edge("search_papers", "rank_papers")
        workflow.add_edge("rank_papers", "summarize_papers")
        workflow.add_edge("summarize_papers", "format_document")
        workflow.add_edge("format_document", END)
        
        # Set entry point
        workflow.set_entry_point("process_query")
        
        return workflow.compile()
    
    async def process_query_node(self, state: Dict) -> Dict:
        """Process user query"""
        enhanced = self.query_agent.process_query(state['user_query'])
        state['enhanced_queries'] = enhanced
        state['status'] = "Query processed"
        return state
    
    async def search_papers_node(self, state: Dict) -> Dict:
        """Search for papers"""
        queries = state['enhanced_queries']['arxiv_queries']
        papers = await self.search_agent.search_multiple_queries(queries)
        state['raw_papers'] = papers
        state['status'] = f"Found {len(papers)} papers"
        return state
    
    async def rank_papers_node(self, state: Dict) -> Dict:
        """Rank papers"""
        papers = state['raw_papers']
        query = state['user_query']
        ranked = await self.ranking_agent.multi_stage_ranking(papers, query)
        state['ranked_papers'] = ranked
        state['status'] = f"Ranked top {len(ranked)} papers"
        return state
    
    async def summarize_papers_node(self, state: Dict) -> Dict:
        """Summarize papers"""
        papers = state['ranked_papers']
        summarized = await self.summary_agent.summarize_papers(papers)
        state['summarized_papers'] = summarized
        state['status'] = "Papers summarized"
        return state
    
    async def format_document_node(self, state: Dict) -> Dict:
        """Format final document"""
        papers = state['summarized_papers']
        
        document = self.formatter.format_full_document(papers)
        state['final_document'] = document
        state['status'] = "Document formatted"
        return state
    
    async def run(self, user_query: str) -> Dict:
        """Run the complete workflow"""
        initial_state = {
            'user_query': user_query,
            'status': 'Started'
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result
