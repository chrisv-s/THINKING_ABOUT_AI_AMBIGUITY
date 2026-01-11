# AI Text Ambiguity Project

## What This Project Is About

This project started because I was curious why AI-generated text often feels “ambiguous” and what that actually means. In human language, ambiguity usually has a reason — a sentence can genuinely be interpreted in multiple ways. With AI, though, what looks like ambiguity is often just statistical uncertainty: the AI generates plausible but vague outputs without true understanding.

My main goal was to explore what happens when meaning is averaged in a computational sense. I built a system that takes two sentences and generates a third sentence that lies somewhere in the **semantic middle**. This doesn’t mean the new sentence truly captures the meaning of both inputs — rather, it is a **mathematical midpoint in vector space**. The system uses word embeddings to find words that are, in a sense, “between” the two inputs. The results are usually abstract, neutral, or funny, highlighting how AI language models smooth over differences probabilistically.

## How It Works — Main Pipeline (`main.py`)

1. **Analyze the sentences**: I use **spaCy** to extract subject-verb-object (SVO) structures from two input sentences.
2. **Compute semantic vectors**: For each SVO element, the code considers word-level vectors and optionally phrase/chunk-level vectors (e.g., noun phrases, subject-verb or verb-object spans).
3. **Pick candidate words**: Using a curated list of common English words, the system selects words closest to the semantic midpoint while respecting part-of-speech constraints.
4. **Assemble the sentence**: The chosen words are combined into a new SVO sentence.

The goal is not grammatical perfection — it is to experiment with what a **mathematical “middle meaning”** looks like in AI-generated text.

### Examples

Sentence A: "The cat chased the mouse"  
Sentence B: "My mother cooks the best food"  
GENERATED: "The dog shot the meat"

My explanation guess: cat+mother→dog (common agent), chased+cooks→shot (action verbs), mouse+food→meat (consumables).

Sentence A: "Thailand is a country"  
Sentence B: "The man wanted pictures"  
GENERATED: "The islands do the video"

My explanation guess: Thailand+man→islands (geographic), is+wanted→do (generic verb), country+pictures→video (visual media).

Sentence A: "The library has many books"  
Sentence B: "My team lost the football tournament"  
GENERATED: "The excellence lead the sports"

What do YOU think about this case?

This output is abstract, funny, and illustrates how semantic averaging can produce plausible but nonsensical sentences.

## Part 2 — Ambiguity Simulation (`ambiguity_simulation.py`)

Based on my professor’s feedback, I also wanted to explore **statistical averageness**: could we use a curated set of **commonly occurring, vague phrases** to simulate ambiguity?  

- I created a text file (`common_ambiguous_phrases.txt`) containing **high-frequency, grammatically correct VO pairs** like “do something,” “take part,” “deal with.”  
- Separately, I defined a list of ambiguous subjects like “someone,” “people,” or “they.”  
- The code compares each SVO element of the two input sentences to the vectors of these phrases.  
- Instead of calculating a midpoint, the system selects the **phrase with the highest average similarity to both sentences**.  
- The chosen subject and VO pair are combined to generate a final “ambiguity sentence.”

This second part is more of a **prototype**, showing what is possible when we experiment with curated ambiguity. It does not reflect natural AI ambiguity — I am heavily influencing the outputs by pre-defining the phrases in the list — but it provides a **controlled way to test how vague, statistically common expressions behave** in combination with the semantic vectors.

Example Output: 

--- Enter Sentence A ---
Sentence A: The wizard killed the goblin. 

--- Enter Sentence B ---
Sentence B: My mother ate an apple.
Sentence A SVO: ('wizard', 'killed', 'goblin')
Sentence B SVO: ('My', 'ate', 'mother')

--- Generated Third Sentence ---
Someone breaks something.


## Observations / Reflections

- **Computational vs. human ambiguity**: What the system produces is not human-like ambiguity. The midpoint is mathematical, and phrases chosen from the ambiguity simulation are curated.  
- **Abstract outputs**: The sentences are often funny or nonsensical, e.g., “Someone take action,” showing how AI smooths over differences probabilistically.
- **Limitations**: SpaCy’s SVO extraction is not perfect for complex sentences, vector averaging does not always lead to a result as there are semantic "voids", and the curated phrases strongly shape the simulation.  
- **Reflection**: The project helped me understand the “fuzziness” of AI-generated language and the difference between true human ambiguity and statistical or semantic averaging. It also demonstrates how small interventions — like a curated phrase list — can produce controllable ambiguity for experiments.

## How to Try It

1. Run `main.py` for the semantic middle experiment.  
2. Run `ambiguity_simulation.py` for the curated ambiguity test.  
3. Enter two simple sentences when prompted and observe the generated outputs.

## Dependencies

- Python 3  
- [spaCy](https://spacy.io/) with the `en_core_web_md` model  
- NumPy  
- [lemminflect](https://github.com/bjascob/lemminflect) (for verb inflection)
