import spacy
nlp = spacy.load('en_core_web_md')

# def content_words(doc): # this needs to be critically reflected: what do I want to analyze
   # return [
    #    token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}
    #    and not token.is_stop
 #   ]

def meaningful_chunks(doc):
    chunks = []
    for chunk in doc.noun_chunks:
        chunks.append(chunk.text)
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            chunks.append(token.text)
    return chunks

# debugging
def overlapping_words(sent_a, sent_b, top_n=5):
    doc_a = nlp(sent_a)
    doc_b = nlp(sent_b)

    words_a = content_words(doc_a)
    words_b = content_words(doc_b)

    overlaps = []

    for wa in words_a:
        for wb in words_b:
            if wa.has_vector and wb.has_vector:
                sim = wa.similarity(wb)
                overlaps.append((sim, wa.text, wb.text))

    overlaps.sort(reverse=True)
    return overlaps[:top_n] # de

results = overlapping_words(
    "The fruit fell far from the tree.",
    "The orange tasted sweet."
)

def vector_similarity(sentence_a, sentence_b, sensitivity=0.5):
    doc_a = nlp(sentence_a)
    doc_b = nlp(sentence_b)

    traces = []

    for traces_a in doc_a:
        for traces_b in doc_b:
            if traces_a.has_vector and traces_b.has_vector:
                similarity = traces_a.similarity(traces_b)
                traces.append((traces_a.text, similarity))
                traces.append((traces_b.text, similarity))

        return traces

def leftovers_max(sentence_a, sentence_b):
    doc_a = nlp(sentence_a)
    doc_b = nlp(sentence_b)

    max_sims = {}

    for ta in doc_a:
        if not ta.has_vector:
            continue
        # Track the highest similarity for ta
        best_sim = 0
        for tb in doc_b:
            if tb.has_vector:
                sim = ta.similarity(tb)
                if sim > best_sim:
                    best_sim = sim
        max_sims[ta.text] = best_sim

    for tb in doc_b:
        if not tb.has_vector:
            continue
        best_sim = 0
        for ta in doc_a:
            if ta.has_vector:
                sim = tb.similarity(ta)
                if sim > best_sim:
                    best_sim = sim
        max_sims[tb.text] = best_sim

    return list(max_sims.items())




