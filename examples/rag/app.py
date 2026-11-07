"""Tiny RAG app (token-overlap retriever + extractive answerer). Swap for your real pipeline: any function
that returns {"answer": ..., "contexts": [{"id":..., "text":...}]} can be evaluated the same way."""
import re
DOCS = {
 "d1": "Refunds are available within 30 days of purchase. Contact support with your order number to request a refund.",
 "d2": "Standard shipping takes 5 to 7 business days. Express shipping takes 2 business days and costs 15 dollars.",
 "d3": "To reset your password, click Forgot Password on the login page and follow the emailed link. The link expires in 24 hours.",
 "d4": "All laptops include a 2 year limited warranty covering manufacturing defects. Accidental damage is not covered.",
 "d5": "We never sell personal data. Account data is deleted within 30 days after you close your account.",
 "d6": "Customer support is available Monday to Friday from 9 AM to 6 PM IST via chat and email.",
 "d7": "We accept credit cards, debit cards, UPI and net banking. Cash on delivery is not available.",
}
STOP = set("the a an is are do you i my to of for how what can in and or does your on it be".split())
def _tok(s): return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}

def answer(question, min_overlap=2):
    q = _tok(question)
    ranked = sorted(((len(q & _tok(t)), i) for i, t in DOCS.items()), reverse=True)
    ctx = [{"id": i, "text": DOCS[i]} for s, i in ranked[:3] if s > 0]
    if not ctx or ranked[0][0] < min_overlap:
        return {"answer": "I don't know based on the available documents.", "contexts": ctx}
    sents = re.split(r"(?<=\.)\s+", ctx[0]["text"])
    best = max(sents, key=lambda s: len(q & _tok(s)))
    return {"answer": best, "contexts": ctx}
