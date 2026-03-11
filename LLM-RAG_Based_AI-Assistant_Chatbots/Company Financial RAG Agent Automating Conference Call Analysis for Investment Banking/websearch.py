from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
search_tool = TavilySearchResults(max_results=3)

def get_industry_growth(company_name: str):
    """Perform contextual web search for industry outlook."""
    query = f"{company_name} industry outlook 2025 site:moneycontrol.com OR site:reuters.com OR site:business-standard.com"
    results = search_tool.invoke(query)

    if not results:
        return "No relevant industry updates found."

    formatted = []
    for res in results:
        formatted.append(f"- **{res['title']}** ({res['url']}): {res['content'][:200]}...")
    return "\n".join(formatted)