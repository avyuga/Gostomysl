from typing import List, Dict
from datetime import datetime

class GOSTFormatter:
    """Format citations according to GOST standard"""
    
    @staticmethod
    def format_article(paper: Dict) -> str:
        """Format article citation in GOST style"""
        
        # Authors
        authors = paper['authors'][:3]
        if len(authors) > 3:
            authors_str = f"{authors[0]} и др."
        else:
            authors_str = ", ".join(authors)
        
        # Title
        title = paper['title'].replace('\n', ' ')
        
        # Year
        year = paper['published'].year if paper['published'] else datetime.now().year
        
        # URL
        url = f"URL: {paper['pdf_url']}" if paper.get('pdf_url') else ""
        
        # Journal ref
        journal = paper.get('journal_ref', 'ArXiv preprint')
        
        # DOI
        doi = f"DOI: {paper['doi']}" if paper.get('doi') else ""
        
        # Format according to GOST
        citation = f"{authors_str} {title} // {journal}. — {year}."
        
        if doi:
            citation += f" — {doi}."
        if url:
            citation += f" — {url}"
        
        return citation
    
    @staticmethod
    def format_bibliography(papers: List[Dict]) -> str:
        """Format full bibliography in GOST style"""
        
        bibliography = "## Список литературы\n\n"
        
        for i, paper in enumerate(papers, 1):
            citation = GOSTFormatter.format_article(paper)
            bibliography += f"{i}. {citation}\n\n"
        
        return bibliography
    
    @staticmethod
    def format_full_document(analysis: str, papers: List[Dict]) -> str:
        """Format complete document with analysis and bibliography"""
        
        document = analysis
        document += "\n\n"
        document += GOSTFormatter.format_bibliography(papers)
        
        # Add metadata
        metadata = f"""
---
Дата создания: {datetime.now().strftime('%d.%m.%Y')}
Количество источников: {len(papers)}
---
"""
        
        return metadata + "\n\n" + document
