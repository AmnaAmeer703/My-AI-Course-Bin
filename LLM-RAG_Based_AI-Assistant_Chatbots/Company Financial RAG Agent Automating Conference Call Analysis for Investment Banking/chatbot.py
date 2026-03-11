from prompts import PROMPT_TEMPLATE
from llm import get_llm
from retriever import get_relevant_docs
from websearch import get_industry_growth
from logger import log_interaction


def process_query(query: str):
    llm = get_llm()
    docs = get_relevant_docs(query)

    if not docs or len(docs) == 0:
        industry_info = get_industry_growth(query)
        log_interaction(query, industry_info, source="web")
        return f"⚠️ No relevant discussion found in the conference call.\n\n**Industry Insights:**\n{industry_info}"

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"{PROMPT_TEMPLATE}\n\n**Question:** {query}\n\n**Documents:**\n{context}"

    response = llm.invoke(prompt).content
    log_interaction(query, response, source="document")
    return response