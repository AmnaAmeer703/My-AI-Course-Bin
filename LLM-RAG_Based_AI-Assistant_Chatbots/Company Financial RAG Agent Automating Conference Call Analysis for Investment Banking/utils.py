# modules/utils.py
import re

# Simple keyword mapping to an industry (expand as needed)
INDUSTRY_KEYWORDS = {
    "consumer": ["consumer", "fmcg", "fast moving", "foods", "snacks", "tea", "coffee"],
    "retail": ["retail", "furniture", "stores"],
    "insurance": ["insurance", "underwriting", "float", "premium"],
    "it": ["software", "services", "IT", "information technology"],
    "pharma": ["pharma", "pharmaceutical", "drug"],
    "auto": ["automobile", "auto", "car", "vehicle"],
    "agri": ["agri", "agriculture", "food", "wheat", "flour", "aashirvaad"],
}

STABLE_INDUSTRIES = {"consumer", "retail", "utilities", "pharma", "agri"}

def infer_industry_from_text(text, filename=None):
    txt = (text or "").lower()
    # Quick filename hint
    if filename:
        t = filename.lower()
        for key in INDUSTRY_KEYWORDS:
            if key in t:
                return key
    # scan for keywords
    for industry, kws in INDUSTRY_KEYWORDS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", txt):
                return industry
    return None

def classify_company_type(industry):
    """
    Return 'stable' or 'cyclical' or 'growth' depending on industry heuristics.
    """
    if not industry:
        return "unknown"
    if industry in STABLE_INDUSTRIES:
        return "stable"
    if industry in {"auto", "metal", "construction"}:
        return "cyclical"
    if industry in {"it"}:
        return "growth"
    return "stable"

def choose_growth_method(company_type):
    """
    Heuristic: for stable companies prefer published industry estimates or 
    historical CAGRs; for cyclical, present ranges and emphasize cyclicality.
    """
    if company_type == "stable":
        return "published_or_historical"
    if company_type == "growth":
        return "analyst_consensus"
    if company_type == "cyclical":
        return "range_or_historical"
    return "published_or_historical"
