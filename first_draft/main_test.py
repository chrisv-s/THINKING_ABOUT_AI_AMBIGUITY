import spacy
import numpy as np

# --------------------------
# Load SpaCy model (medium for word vectors)
# --------------------------
nlp = spacy.load("en_core_web_md")  # might use lg later for better vectors


# Load 1000 most common words source: https://github.com/powerlanguage/word-lists/blob/master/1000-most-common-words.txt
#
with open("../google-10000-english-usa-no-swears.txt", "r") as f:
    common_words = [w.strip().lower() for w in f if w.strip()]

# POS-tag each word and store as SpaCy token
common_lexemes = []
for word in common_words:
    doc = nlp(word)
    if doc and doc[0].has_vector:
        common_lexemes.append(doc[0])


# STEP 0: Fixed Sentence A

sentence_a_text = "The man ate an apple."
doc_a = nlp(sentence_a_text)

def extract_svo_pos(doc):
    subj = verb = obj = None
    for token in doc:
        if subj is None and token.pos_ in {"NOUN", "PRON"}:
            subj = token
        elif verb is None and token.pos_ == "VERB":
            verb = token
        elif obj is None and token.pos_ in {"NOUN", "PRON"} and token != subj:
            obj = token
    return subj, verb, obj

svo_a = extract_svo_pos(doc_a)
print("Fixed Sentence A SVO:", tuple(t.text if t else None for t in svo_a))

#
# STEP 1: Ask user for Sentence B

print("\n--- Enter Sentence B ---")
print("Type any simple sentence like - The wizard drank the potion - The system will extract Subject, Verb, Object automatically.")
user_input = input("Sentence B: ").strip()
user_input = user_input.rstrip(".")
doc_b = nlp(user_input)

svo_b = extract_svo_pos(doc_b)
print("User Sentence B SVO:", tuple(t.text if t else None for t in svo_b))

#
# STEP 2: Compute two-layer midpoint with extended spans
#
def midpoint_vector(tok_a, tok_b):

    vecs = []

    for tok in [tok_a, tok_b]:
        if tok is not None:
            # word-level
            vecs.append(tok.vector)

            # chunk-level: noun chunks
            for chunk in tok.doc.noun_chunks:
                if tok in chunk:
                    vecs.append(chunk.vector)

            # chunk-level: subject-verb
            if tok.dep_ in {"nsubj", "nsubjpass"} and tok.head.pos_ == "VERB":
                sv_span = tok.doc[tok.i : tok.head.i+1]
                vecs.append(sv_span.vector)

            # chunk-level: verb-object
            if tok.dep_ in {"dobj", "attr", "pobj"} and tok.head.pos_ == "VERB":
                vo_span = tok.doc[tok.head.i : tok.i+1]
                vecs.append(vo_span.vector)

    if vecs:
        # Could I weight chunk vectors differently than word vectors?
        return np.mean(vecs, axis=0)
    return None

subj_mid = midpoint_vector(svo_a[0], svo_b[0])
verb_mid = midpoint_vector(svo_a[1], svo_b[1])
obj_mid  = midpoint_vector(svo_a[2], svo_b[2])

#
# STEP 3: Pick candidate that is semantically in the middle
#

def pick_candidate(mid_vector, allowed_pos, banned_words=set(), top_k=10):
    """
    Picks a word from the common words list that is closest to the semantic midpoint (I partially asked ChatGPT to helo
    me with certain parts of the function here).

    Arguments:
    - mid_vector: the vector representing the 'middle meaning' between Sentence A and B
    - allowed_pos: which types of words are allowed (e.g., nouns, verbs)
    - banned_words: words to avoid repeating
    - top_k: number of top candidates to consider before picking

    Returns:
    - chosen_word: the best word
    - chosen_similarity: cosine similarity to the midpoint
    """

    if mid_vector is None:
        return None, None

    # Store all candidate words and their similarity to the midpoint
    candidate_list = []

    for word_token in common_lexemes:  # each word in the common words list
        # Only consider allowed POS and avoid repeated words
        if word_token.pos_ in allowed_pos and word_token.text.lower() not in banned_words:
            # Compute cosine similarity to the midpoint
            similarity_to_midpoint = np.dot(mid_vector, word_token.vector) / (
                np.linalg.norm(mid_vector) * np.linalg.norm(word_token.vector)
            )
            candidate_list.append((word_token.text, similarity_to_midpoint))

    if not candidate_list:
        return None, None

    # Sort candidates from highest similarity to lowest
    candidate_list.sort(key=lambda x: x[1], reverse=True)

    # Keeps only the top_k closest words and picks the most similar word (currently deterministic)
    top_candidates = candidate_list[:top_k]

    chosen_word = top_candidates[0][0]
    chosen_similarity = top_candidates[0][1]

    return chosen_word, chosen_similarity

subj_mid = midpoint_vector(svo_a[0], svo_b[0])
verb_mid = midpoint_vector(svo_a[1], svo_b[1])
obj_mid  = midpoint_vector(svo_a[2], svo_b[2])


used_words = set([tok.text.lower() for tok in svo_a] + [tok.text.lower() for tok in svo_b if tok])
subj_word, subj_sim = pick_candidate(subj_mid, {"NOUN","PRON"}, banned_words=used_words)
verb_word, verb_sim = pick_candidate(verb_mid, {"VERB"}, banned_words=used_words)
obj_word, obj_sim     = pick_candidate(obj_mid, {"NOUN","PRON"}, banned_words=used_words)

generated_svo = (subj_word, verb_word, obj_word)
similarities = (subj_sim, verb_sim, obj_sim)

#
# STEP 4: Assemble third sentence
#
def assemble_svo(svo):
    words = [w for w in svo if w]
    return " ".join(words).capitalize() + "." if words else ""

third_sentence = assemble_svo(generated_svo)

# --------------------------
# STEP 5: Display results with explanations
# --------------------------
print("\n--- Generated Third Sentence ---")
print(third_sentence)

print("\n--- Word-by-Word Explanation ---")
for word, sim, role in zip(generated_svo, similarities, ["subject","verb","object"]):
    print(f"{role.capitalize()}: '{word}'")

# --------------------------
# Questions / Possible Improvements
# --------------------------
# - Should chunk-level vectors be weighted more than word-level vectors?
# - Can we include verb-object and subject-verb phrases more systematically?
# - Is the 1000 most common words list enough for good variety, or should we expand it?
# - Would using a larger spaCy model (en_core_web_lg) improve vector midpoints?
