import spacy
import numpy as np

# Load spaCy model with word vectors (install with: python -m spacy download en_core_web_md)
nlp = spacy.load("en_core_web_md")

print("=" * 60)
print("VECTOR SPACE SENTENCE BLENDER")
print("=" * 60)
print("This program finds the 'semantic midpoint' between two sentences.")
print("Enter two simple sentences → Get a third sentence 'in between' them in vector space!")

# Load 10K most common English words and convert to spaCy tokens
print("Please wait for a few seconds... ", end="")
with open("google-10000-english-usa-no-swears.txt", "r") as f:
    common_words = [w.strip().lower() for w in f if w.strip()]

common_lexemes = []
for word in common_words:
    doc = nlp(word)
    if doc and doc[0].has_vector:
        common_lexemes.append(doc[0])
print(f"✓ {len(common_lexemes)} words loaded")


# FUNCTION 1: Extract Subject-Verb-Object from sentence

def extract_svo_pos(doc):
    """
    Identifies Subject (nsubj), Verb (VERB/AUX phrases), Object (dobj) using
    spaCy's dependency parsing. Handles verbs like "is running" as single verb phrase.
    """
    subj = verb = obj = None
    for token in doc:
        # SUBJECT: First noun/pronoun with nsubj relation (I asked ChatGPT to write a code snippet here
        # to skip possessives)
        if subj is None and token.dep_ in {"nsubj", "nsubjpass"}:
            if token.pos_ == "PRON" and token.morph.get("Poss", False):
                continue
            subj = token

        # VERB: First VERB/AUX. Merge AUX+VERB into span like "is swimming"
        elif verb is None and token.pos_ in {"VERB", "AUX"}:
            if token.pos_ == "AUX" and token.head.pos_ == "VERB":
                aux_span = doc[token.i: token.head.i + 1]
                verb = aux_span
            else:
                verb = token

        # OBJECT
        elif obj is None and token.dep_ in {"dobj", "obj", "attr", "pobj"}:
            obj = token
    return subj, verb, obj


# FUNCTION 2: Compute average vector

def midpoint_vector(tok_a, tok_b):
    """
    Creates "super vector" by averaging multiple semantic perspectives on each word.
    For example, instead of just the word-level "cat", we get the average of: "cat" + "the cat" + "cat chased"
    I am actually not sure whether this approach is thaat much better. But I wanted to try it anyways.
    """
    vecs = []

    for tok in [tok_a, tok_b]:
        if tok is not None:
            # LEVEL 1: BASIC WORD EMBEDDING (ALWAYS INCLUDED)
            vecs.append(tok.vector)

            if hasattr(tok, 'dep_'):
                # LEVEL 2: NOUN PHRASE CONTEXT
                for chunk in tok.doc.noun_chunks:
                    if tok in chunk:
                        vecs.append(chunk.vector)

                # LEVEL 3: SUBJECT-VERB PHRASE
                # If this token is SUBJECT, also include "SUBJECT + VERB" phrase
                if tok.dep_ in {"nsubj", "nsubjpass"} and tok.head.pos_ in {"VERB", "AUX"}:
                    # Slice from subject to its verb: tok.i=1, tok.head.i=2 → doc[1:3]
                    sv_span = tok.doc[tok.i: tok.head.i + 1]
                    vecs.append(sv_span.vector)

                # LEVEL 4: VERB-OBJECT PHRASE (ONLY FOR OBJECTS)
                # If this token is OBJECT, include "VERB + OBJECT" phrase
                if tok.dep_ in {"dobj", "obj", "attr", "pobj"} and tok.head.pos_ in {"VERB", "AUX"}:
                    # Slice from verb to object: head.i=2, tok.i=3 → doc[2:4]
                    vo_span = tok.doc[tok.head.i: tok.i + 1]
                    vecs.append(vo_span.vector)

    # Averages all these perspectives together
    return np.mean(vecs, axis=0) if vecs else None


# FUNCTION 3: Searches for the midpoint word

def pick_candidate(mid_vector, allowed_pos, banned_words=set(),
                   start_threshold=0.85, step=0.05, min_threshold=0.5):
    """
    Finds word from common 10K vocab most similar to midpoint vector.
    Starts strict (0.85 similarity), relaxes until good match found.
    Filters by POS (NOUN/VERB/AUX) and excludes used words.
    """
    if mid_vector is None:
        return None, None

    mid_norm = np.linalg.norm(mid_vector)
    if mid_norm == 0:
        return None, None

    current_threshold = start_threshold
    while current_threshold >= min_threshold:
        candidates = []
        for word_token in common_lexemes:
            # POS filter
            if word_token.pos_ not in allowed_pos:
                continue
            # Banned words filter
            if word_token.text.lower() in banned_words:
                continue
            # Zero vector filter
            if word_token.vector_norm == 0:
                continue
            # Possessive FILTER: No "his/her/its" as subjects
            if word_token.pos_ == "PRON" and "Poss" in word_token.morph:
                continue

            # Cosine similarity
            similarity = np.dot(mid_vector, word_token.vector) / (
                    mid_norm * word_token.vector_norm
            )

            if similarity >= current_threshold:
                candidates.append((word_token.text, similarity))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]

        current_threshold -= step

    return None, None


# FUNCTION 4: Build the third sentence

def assemble_svo(svo):
    words = [w for w in svo if w and str(w).strip()]
    if len(words) >= 2:  # Need subject + verb minimum
        # Add articles for natural English
        sentence = f"The {words[0]} {words[1]}"
        if len(words) > 2:
            sentence += f" the {words[2]}"
        return sentence.capitalize() + "."
    return "No valid sentence generated."


# MAIN PROGRAM FLOW

# Sentence A input
print("\n" + "=" * 60)
print("📝 ENTER SENTENCE A")
print("Use simple active sentences like:")
print("- 'The cat chased the mouse'")
print("- 'Birds fly in the sky'")
print("- 'He is swimming now' (works with 'is/has/will')")
user_input_a = input("\nSentence A: ").strip().rstrip(".")
doc_a = nlp(user_input_a)
svo_a = extract_svo_pos(doc_a)
print(f"\nDetected: Subject='{svo_a[0]}' Verb='{svo_a[1]}' Object='{svo_a[2]}'")

# Sentence B input
print("\n" + "=" * 60)
print("📝 ENTER SENTENCE B")
user_input_b = input("\nSentence B: ").strip().rstrip(".")
doc_b = nlp(user_input_b)
svo_b = extract_svo_pos(doc_b)
print(f"\nDetected: Subject='{svo_b[0]}' Verb='{svo_b[1]}' Object='{svo_b[2]}'")

# Compute semantic midpoints
print("\n" + "=" * 60)
print("🧮 COMPUTING SEMANTIC MIDPOINTS...")
subj_mid = midpoint_vector(svo_a[0], svo_b[0])
verb_mid = midpoint_vector(svo_a[1], svo_b[1])
obj_mid = midpoint_vector(svo_a[2], svo_b[2])

# Find best words at each position
used_words = set([tok.text.lower() for tok in svo_a if tok] + [tok.text.lower() for tok in svo_b if tok])

print("🔍 Searching vocabulary for midpoint words...")
subj_word, subj_sim = pick_candidate(subj_mid, {"NOUN"}, banned_words=used_words)
verb_word, verb_sim = pick_candidate(verb_mid, {"VERB", "AUX"}, banned_words=used_words)
obj_word, obj_sim = pick_candidate(obj_mid, {"NOUN"}, banned_words=used_words)

# Generate final sentence
generated_svo = (subj_word, verb_word, obj_word)
third_sentence = assemble_svo(generated_svo)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nGENERATED SENTENCE:")
print(f"\"{third_sentence}\"")


print("\n" + "=" * 60)

