import arxiv
import json

class ResearchEngine:
    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(self, query, max_results=5):
        """Searches Arxiv for papers matching the query."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in self.client.results(search):
            results.append({
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.isoformat(),
                "authors": [a.name for a in result.authors]
            })
        return results

    def suggest_related_work(self, context_hits):
        """
        Suggests search queries or topics based on historical context hits 
        from MemoryManager.
        """
        if not context_hits:
            return []
            
        # Extract themes from past hits (simplified logic)
        themes = [hit.metadata.get("topic") for hit in context_hits if "topic" in hit.metadata]
        unique_themes = list(set(themes))
        
        return unique_themes

if __name__ == "__main__":
    # Quick test
    engine = ResearchEngine()
    print("Searching for 'Large Language Model Memory'...")
    papers = engine.search_papers("Large Language Model Memory", max_results=3)
    for p in papers:
        print(f"- {p['title']} ({p['published']})")
