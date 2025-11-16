import arxiv
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

class SearchAgent:
    """Agent for searching papers on ArXiv"""
    
    def __init__(self, max_results: int = 100):
        self.max_results = max_results
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def search_arxiv(self, query: str, max_results: int = None) -> List[Dict]:
        """Search ArXiv for papers"""
        if max_results is None:
            max_results = self.max_results
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for paper in search.results():
            papers.append({
                "id": paper.entry_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published,
                "updated": paper.updated,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref
            })
        
        return papers
    
    async def search_multiple_queries(self, queries: List[str]) -> List[Dict]:
        """Search multiple queries in parallel"""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(self.executor, self.search_arxiv, query, 30)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Merge and deduplicate results
        all_papers = {}
        for papers in results:
            for paper in papers:
                if paper["id"] not in all_papers:
                    all_papers[paper["id"]] = paper
        
        return list(all_papers.values())
