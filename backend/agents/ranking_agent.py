import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from models.yandex_llm import YandexGPT
from config import Config

class RankingAgent:
    """Agent for ranking search results"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = YandexGPT(
            api_key=Config.YANDEX_API_KEY,
            folder_id=Config.YANDEX_FOLDER_ID,
            model_uri=Config.YANDEX_GPT_MODEL_URI
        )
    
    def rank_bm25(self, papers: List[Dict], query: str, top_k: int = 50) -> List[Dict]:
        """Rank papers using BM25"""
        if not papers:
            return []
        
        # Prepare documents for BM25
        documents = [
            f"{p['title']} {p['summary']}"
            for p in papers
        ]
        
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort by scores
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        return [papers[i] for i in ranked_indices]
    
    def rank_embeddings(self, papers: List[Dict], query: str, top_k: int = 25) -> List[Dict]:
        """Rank papers using embeddings"""
        if not papers:
            return []
        
        # Create embeddings
        documents = [
            f"{p['title']} {p['summary'][:500]}"
            for p in papers
        ]
        
        doc_embeddings = self.embedding_model.encode(documents)
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [papers[i] for i in ranked_indices]
    
    async def rank_with_llm(self, papers: List[Dict], query: str, top_k: int = 10) -> List[Dict]:
        """Rank papers using LLM for relevance assessment"""
        if not papers or len(papers) <= top_k:
            return papers
        
        relevance_prompt = """
        Оцени релевантность статьи запросу от 0 до 10.
        
        Запрос: {query}
        
        Название: {title}
        Аннотация: {summary}
        
        Ответь только числом от 0 до 10.
        """
        
        scored_papers = []
        for paper in papers[:25]:  # Limit to avoid too many API calls
            prompt = relevance_prompt.format(
                query=query,
                title=paper['title'],
                summary=paper['summary'][:500]
            )
            
            try:
                score_text = self.llm(prompt)
                score = float(score_text.strip())
            except:
                score = 5.0  # Default score if parsing fails
            
            paper_with_score = paper.copy()
            paper_with_score['relevance_score'] = score
            scored_papers.append(paper_with_score)
        
        # Sort by relevance score
        scored_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_papers[:top_k]
    
    async def multi_stage_ranking(self, papers: List[Dict], query: str) -> List[Dict]:
        """Perform multi-stage ranking"""
        # Stage 1: BM25
        ranked_bm25 = self.rank_bm25(papers, query, Config.TOP_K_BM25)
        
        # Stage 2: Embeddings
        ranked_embeddings = self.rank_embeddings(ranked_bm25, query, Config.TOP_K_EMBEDDING)
        
        # Stage 3: LLM
        final_ranking = await self.rank_with_llm(ranked_embeddings, query, Config.TOP_K_FINAL)
        
        return final_ranking
