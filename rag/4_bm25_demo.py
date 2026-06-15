import math
import re
from collections import defaultdict


DOCS = {
   "D0": "create_invoice generate invoice for order calculate invoice total",
   "D1": "create_payment process payment validate payment method charge card",
   "D2": "create_statement generate statement for customer calculate amount due",
   "D3": "send_invoice email invoice to customer attach invoice pdf",
   "D4": "process_refund reverse payment refund amount to customer card",
}


DOC_IDS   = list(DOCS.keys())
DOC_TEXTS = list(DOCS.values())


STOP_WORDS = {"a", "an", "the", "and", "or", "to", "in", "on", "for",
             "of", "with", "is", "are", "was", "be", "at", "by"}


def tokenize(text: str) -> list:
   return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP_WORDS]




# ---------------------------------------------------------------------------
# STEP 1 — Inverted Index
# ---------------------------------------------------------------------------


print("--- STEP 1: Inverted Index ---")
print()
print("  Forward index  : document → words it contains   (what you read normally)")
print("  Inverted index : word     → documents containing it  (what search needs)")
print()
print("  Given a query word, the inverted index tells you instantly which documents")
print("  to look at — without scanning every document in the corpus.")
print()


tokenized      = [tokenize(text) for text in DOC_TEXTS]
inverted_index = defaultdict(dict)
doc_lengths    = {}


for doc_id, tokens in zip(DOC_IDS, tokenized):
   doc_lengths[doc_id] = len(tokens)
   for token in tokens:
       inverted_index[token][doc_id] = inverted_index[token].get(doc_id, 0) + 1


# Grid: only terms that appear in more than one doc
shared_terms = sorted(
   t for t, postings in inverted_index.items() if len(postings) > 1
)


col_w = 6
print(f"  {'term':<18}", end="")
for doc_id in DOC_IDS:
   print(f"{doc_id:>{col_w}}", end="")
print()
print(f"  {'-'*18}", end="")
for _ in DOC_IDS:
   print(f"{'-'*col_w}", end="")
print()


for term in shared_terms:
   print(f"  {term:<18}", end="")
   for doc_id in DOC_IDS:
       tf   = inverted_index[term].get(doc_id, 0)
       cell = str(tf) if tf else "."
       print(f"{cell:>{col_w}}", end="")
   print()


print()
print("  Number = how many times the word appears in that document  (term frequency)")
print("  Dot    = word is absent → BM25 skips this document for that term entirely")




# ---------------------------------------------------------------------------
# STEP 2 — The three factors BM25 uses to score
# ---------------------------------------------------------------------------


print()
print("--- STEP 2: How BM25 scores a document ---")
print()
print("  BM25 score  =  Term Frequency  +  Word Rarity  -  Length Penalty")
print()
print("  Factor 1 — Term Frequency (TF)")
print("    More occurrences of the query word in a document → higher score")
print("    But each extra occurrence adds less than the last (diminishing returns)")
print()
print("  Factor 2 — Word Rarity  (IDF — Inverse Document Frequency)")
print("    Word appears in every document → low IDF, nearly useless for ranking")
print("    Word appears in one document   → high IDF, strongly discriminating")
print()
print("  Factor 3 — Document Length Penalty")
print("    A long document contains more words just by size, not relevance")
print("    BM25 slightly penalises documents longer than average")




# ---------------------------------------------------------------------------
# STEP 3 — IDF table: see word rarity in action
# ---------------------------------------------------------------------------


N     = len(DOC_IDS)
avgdl = sum(doc_lengths.values()) / N
k1    = 1.5
b     = 0.75


def idf(term: str) -> float:
   df = len(inverted_index[term])
   return math.log((N - df + 0.5) / (df + 0.5) + 1)


print()
print("--- STEP 3: IDF — word rarity scores ---")
print()
print(f"  {'term':<18}  {'in how many docs':>18}  {'IDF score':>10}  note")
print(f"  {'-'*18}  {'-'*18}  {'-'*10}  {'-'*30}")
for term in shared_terms:
   df       = len(inverted_index[term])
   score    = idf(term)
   note     = "← common, less useful" if df >= 3 else ("← rare, more useful" if df == 1 else "")
   print(f"  {term:<18}  {df:>18}  {score:>10.2f}  {note}")




# ---------------------------------------------------------------------------
# STEP 4 — Ranking: query that BM25 handles well
# ---------------------------------------------------------------------------


def bm25_score(query_tokens: list, doc_id: str) -> float:
   dl    = doc_lengths[doc_id]
   score = 0.0
   for term in query_tokens:
       tf = inverted_index[term].get(doc_id, 0)
       if tf == 0:
           continue
       tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
       score  += idf(term) * tf_norm
   return score


def show_ranking(query: str) -> None:
   tokens = tokenize(query)
   scores = sorted(
       [(doc_id, bm25_score(tokens, doc_id)) for doc_id in DOC_IDS],
       key=lambda x: x[1], reverse=True,
   )
   print(f"  Query : \"{query}\"")
   print(f"  Tokens: {tokens}")
   print()
   for rank, (doc_id, score) in enumerate(scores, 1):
       bar  = "█" * int(score * 8)
       mark = " ✓" if score > 0 else ""
       print(f"  rank {rank}  {doc_id}  score={score:.2f}  {bar:<20}{mark}")
       print(f"          \"{DOCS[doc_id]}\"")




print()
print("--- STEP 4: BM25 ranking — query it handles well ---")
print()
show_ranking("create invoice")


print()
print("  BM25 works perfectly here.")
print("  D0 and D3 rank highest because they both contain 'invoice' multiple times.")
print("  D1/D2/D4 score low or zero — they don't share the exact words.")




# ---------------------------------------------------------------------------
# STEP 5 — Where BM25 fails: the lexical gap
# ---------------------------------------------------------------------------


print()
print("--- STEP 5: Where BM25 fails — the lexical gap ---")
print()
show_ranking("generate bill")


print()
print("  D0 sneaks in at rank 1 — but only because it literally contains 'generate'.")
print("  BM25 did NOT understand that 'bill' = 'invoice'.")
print()
print("  D3 (send_invoice) scores zero — it's clearly about invoices but has no")
print("  exact match for 'generate' or 'bill', so BM25 ignores it entirely.")
print()
print("  Why?")
print("    Query says  'generate'  →  documents say  'create'")
print("    Query says  'bill'      →  documents say  'invoice'")
print()
print("  BM25 only matches exact words. It has no concept of meaning.")
print("  'generate' and 'create' mean the same thing — BM25 doesn't know that.")
print()
print("  This is the lexical gap, and it's exactly why semantic/vector search exists.")
print("  Semantic search embeds both query and document into vector space,")
print("  so 'generate bill' lands close to 'create invoice' even without shared words.")
