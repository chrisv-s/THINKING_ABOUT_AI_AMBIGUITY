Phase 1: Conceptualization — Why Ambiguity Matters

When I started this project, I wanted to explore a question that’s been on my mind: why does AI-generated text often feel vague or ambiguous? In human language, ambiguity usually has a reason — a sentence can be interpreted in multiple coherent ways. AI, however, often produces text that looks ambiguous but is really just statistical uncertainty. My goal was to see what happens if a system tries to generate a sentence that sits between two distinct inputs — something that relates to both without explicitly choosing a meaning. The idea was simple: take two sentences, analyze them semantically, and create a third that lies somewhere in the “semantic middle” using vectors, embeddings, and probabilities, without relying on grammar rules or templates.

Human ambiguity usually has a reason: a sentence can be interpreted in multiple coherent ways. AI-generated text, however, often appears ambiguous but is really probabilistic fuzziness, a result of statistical averaging rather than intentional multiple meanings.

I wanted to explore what happens if a system generates a sentence between two distinct inputs. By “between,” I mean a mathematical averaging in vector space, not a semantic compromise in the human sense. The system doesn’t understand the inputs; it only knows which word embeddings are closest to an averaged vector.

Cosine similarity is used to quantify this closeness:

```
similarity_to_midpoint = np.dot(mid_vector, word_token.vector) / (
    np.linalg.norm(mid_vector) * np.linalg.norm(word_token.vector)
)
```

Here, mid_vector is the average of the vectors of two tokens (from Sentence A and B), and word_token.vector is the candidate word’s vector. The cosine similarity measures how “aligned” a candidate word is with the semantic midpoint.

Example
Sentence A: The man ate an apple.
Sentence B: The father drank water.
Generated: village eat ocean

The output does not meaningfully describe both inputs. It simply lies between them in vector space. This illustrates computational semantic muddle: AI can produce text that feels like it might relate to multiple meanings without actually reconciling them.

Phase 2: Building the Subject-Verb-Object Extraction 

The first practical step was figuring out the input sentences. I used subject-verb-object (SVO) structures as the building blocks since they’re very simple and interpretable. SpaCy helped me identify nouns, pronouns, and verbs, and extract the core triplets. This worked well for short sentences, though I noticed it can fail with longer, more complex sentences — sometimes missing the subject or object. 
I experimented with two-layer semantic representation:
Word-level vectors: single tokens
Chunk/phrase-level vectors: noun phrases, subject-verb pairs, verb-object pairs

```
def midpoint_vector(tok_a, tok_b):
    vecs = []
    for tok in [tok_a, tok_b]:
        if tok is not None:
            vecs.append(tok.vector)  # word
            for chunk in tok.doc.noun_chunks:
                if tok in chunk:
                    vecs.append(chunk.vector)  # noun phrase
            if tok.dep_ in {"nsubj", "nsubjpass"} and tok.head.pos_ == "VERB":
                sv_span = tok.doc[tok.i:tok.head.i+1]
                vecs.append(sv_span.vector)
            if tok.dep_ in {"dobj", "attr", "pobj"} and tok.head.pos_ == "VERB":
                vo_span = tok.doc[tok.head.i:tok.i+1]
                vecs.append(vo_span.vector)
    return np.mean(vecs, axis=0) if vecs else None

```

This allowed me to explore how SVO components relate to each other and how combining token and phrase vectors influences the semantic midpoint. But I decided to scratch this part as it made the project quite complex and I wanted to keep the main analysis quite simple.


Phase 3: Candidate Selection 

Next, I needed to pick actual words from a vocabulary. I initially tried SpaCy’s full lexicon but ran into errors because Lexeme objects don’t have POS tags directly. My solution/idea was to include a words list: SpaCy could tag them, and I could pick candidates close to the midpoint of the analysed vectors and suitable for the SVO-structure. Even so, the system sometimes repeated one of the original words or produced odd sentences like “him eat water.” 

I also experimented with:

- Combining word-level and phrase-level vectors to get a more “context-aware” midpoint 
- Using Markovify to build candidate phrases, but it was difficult to integrate while keeping SVO structure interpretable 
- Adding soft randomness to allow more varied outputs

These steps highlighted fuzzy semantic regions: areas in vector space where words are similar but do not fully resolve the original meanings.

Phase 4: Third Sentence Generation 

Once words were chosen, generating the sentence was simple: just assemble the SVO words in order, optionally adding articles for readability. The resulting sentences felt mostly super random and funny — like “village eat ocean” from inputs such as “The man ate an apple” and “The father drank water.” These outputs confirmed my hypothesis: the language model doesn’t need to understand to produce text that seems meaningful. The challenge is that vector analysis/midpoint often landed in a semantic place that missed a lot of context, far from words that make perfect sense together. This explains why outputs are abstract or slightly nonsensical, even when SVO extraction and candidate selection are carefully done. It showed the limitations of semantic midpoints in embedding space. It is not a fully realistic representation of how language models work, but it serves as an experiment in computational/mathematical semantic averaging.

Phase 5: Reflection and Visualization 

After realizing that my approach to look at midpoints of vetors did not really tackle ambiguity produced by language models or look at the statistical averagenes produced by the models, I wanted to a basic simulation of statistical ambiguity in language by using commonly occurring vague expressions. This builds on the idea that generated text often seems ambiguous not because it “understands” multiple meanings, but because it averages over frequently seen patterns in its training data. 

To explore this, I created a small dataset of ambiguous VO pairs and subjects. I prompted ChatGPT as follows:

“Please generate 1000 short, grammatically correct, commonly occurring English phrases or VO pairs that are vague and flexible enough to fit multiple contexts, suitable for testing computational ambiguity.”

This gave me a curated set of phrases that, while artificial, allowed me to experiment with statistical averageness. I then classified these phrases using SpaCy, extracted their SVO-like structure, and scored them against the SVO tokens from both input sentences:


The system selects the best subject and best VO pair by average similarity to both sentences. Finally, SpaCy’s lemmatization and inflection tools ensure grammatical agreement, so the generated sentence remains syntactically correct, e.g.:

```
def phrase_similarity(phrase, ref_tokens):
    doc = nlp(phrase)
    sims = []
    for tok in doc:
        for ref in ref_tokens:
            sims.append(tok.similarity(ref))
    return np.mean(sims) if sims else -1
```

The system selects the best subject and best VO pair by average similarity to both sentences. Finally, SpaCy’s lemmatization and inflection tools ensure grammatical agreement, so the generated sentence remains syntactically correct, e.g.:

```
Input A: The man ate an apple.
Input B: The wizard hit the tree.
Generated (prototype): someone take part something
```

This prototype explicitly shows how much influence the dataset exerts: the “ambiguity” here is not naturally emergent from the AI model but constructed from the pre-selected vague expressions. Nonetheless, it allows me to reflect on how statistical patterns can create the illusion of ambiguity and to test ideas inspired by my professor’s feedback.

### Limitations and Reflections

This project helped me see the limits of trying to find a “middle meaning” in generated text. The vector midpoint is just a mathematical average — it doesn’t produce a sentence that actually captures both inputs in a human sense. What the system generates is often abstract or nonsensical, showing how language models handles meaning probabilistically rather than semantically.

Using the curated ambiguous phrases for the second part of the project highlighted that “ambiguity” can also be constructed. The output is strongly influenced by the dataset, so it’s not natural AI ambiguity but a way to explore statistical averaging and vagueness.

I also learned that looking at SVO elements separately and considering how subjects, verbs, and objects interact is tricky — combining word- and phrase-level vectors gives more interesting results, but it still doesn’t guarantee coherence.

Overall, the project shows that the generated text feels ambiguous for a reason: its outputs are shaped by probabilities over patterns it has seen, and any “semantic middle” we try to force is really just a mathematical experiment. This helped me reflect on my research question: why AI text seems ambiguous, what that ambiguity means, and how much is real versus constructed. It also gave me a hands-on way to test these ideas, even if the results are rough or artificial.