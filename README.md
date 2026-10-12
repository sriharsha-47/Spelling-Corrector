# Efficient Spelling Corrector (NLP Group Assignment, Question 3)

A spelling-correction system that detects and fixes both **non-word errors** (misspellings not in the vocabulary) and **real-word errors** (valid words used incorrectly in context) within an **edit distance of 1**. The vocabulary, word frequencies, and bigram context are all derived from the **NLTK Brown Corpus**.

Two candidate-generation strategies are implemented side by side so their quality and speed can be compared on identical inputs:

| Method | Approach | Notes |
|---|---|---|
| **A** | Standard Edit Distance 1 (brute force) | Generates every deletion, transposition, replacement, and insertion, then filters by vocabulary |
| **B** | Symmetric Delete | Precomputed index of one-character deletions; no 26-letter alphabet at query time |

Method B returns the same corrections as Method A while being about **7.4x faster** in the benchmark.

---

## Project Structure

| File | Lines | Purpose |
|---|---|---|
| `Spelling_Corrector.py` | 1,518 | Core library: normalization, Brown loading, unigram/bigram models, Method A, Method B, non-word and real-word correction, sentence reconstruction, and the `SpellingCorrector` class |
| `benchmark.py` | 138 | "Speed Demon" benchmark: times Method A vs. Method B on the same 1,000 non-word errors |
| `evaluate.py` | 259 | Accuracy evaluation on a 10% sample of Brown sentences with injected errors |
| `cli.py` | 82 | Interactive terminal app with highlighted corrections and latency reporting |
| `Test_spell_corrector.py` | 123 | Manual smoke test (non-word, real-word, candidate generation, clean sentences) |

`benchmark.py`, `evaluate.py`, and `cli.py` all import the same `SpellingCorrector` class, so both methods are always compared under identical conditions.

---

## Requirements

- Python 3.8+
- [NLTK](https://www.nltk.org/)

```bash
pip install nltk
```

The Brown Corpus is downloaded automatically on first run. To fetch it manually:

```bash
python -m nltk.downloader brown
```

---

## Usage

### Interactive CLI

```bash
python cli.py
```

Type a sentence and the corrector prints the fixed version, with changed words highlighted in yellow (ANSI) and the correction latency in milliseconds. Exit with `exit`, `Ctrl+C`, or EOF.

```
> I hav a good feeling about this.
I had a good feeling about this.   (2.53 ms)
```

### Speed benchmark

```bash
python benchmark.py --seed 42 --threshold 3.0
```

Builds exactly 1,000 one-edit non-word errors, runs a warmup call, then times `correct_word()` for each method on the identical batch (model construction excluded).

### Accuracy evaluation

```bash
python evaluate.py --seed 42 --threshold 3.0
```

Samples 10% of Brown sentences, injects one non-word and one real-word error per sentence, and scores both methods on the same generated cases.

### Smoke test

```bash
python Test_spell_corrector.py
```

### Using the library

```python
from Spelling_Corrector import get_corrector

corrector = get_corrector()   # cached; the Brown model is built once
print(corrector.correct("This is a test sentnce."))
# This is a test sentence.
```

`get_corrector()` caches the model, so repeated calls do not rebuild it. `SpellingCorrector.model_info()` returns the model statistics shown below.

---

## How It Works

### 1. Model preparation

- Loads `brown.sents()` and normalizes words (lower-cased, keeping only letters and apostrophes).
- Builds a unigram vocabulary and a bigram model with **add-k smoothing** (`k = 0.1`).

| Metric | Value |
|---|---|
| Brown sentences | 57,340 |
| Vocabulary size | 47,140 |
| Total word tokens | 1,014,325 |
| Bigram contexts | 47,141 |
| Real-word threshold | 3.0 (log-probability units) |

### 2. Candidate generation

- **Method A** (`edits1`, `method_a_candidates`): builds all edit-distance-1 strings and intersects them with the vocabulary.
- **Method B** (`build_delete_dictionary`, `generate_deletes`, `method_b_candidates`): looks up the word and its own deletions in a precomputed delete index. A final `is_edit_distance_one` check guarantees Method B never returns a candidate Method A would reject.

### 3. Correction logic

**Non-word errors** (`correct_nonword`): generate candidates and return the one with the highest unigram count. No context is used.

**Real-word errors** (`correct_real_word`): every word is scored against its candidates plus itself:

```
score(c) = log P(c | prev) + log P(next | c) + 0.25 * log P_mle(c)
```

A candidate replaces the original word only if all three guards pass:

1. Its score beats the original's by at least the threshold (default 3.0).
2. It does not worsen either neighbouring bigram by more than 1.0 log-unit.
3. It improves at least one side by 0.75 or more.

---

## Results

All figures were produced by running the submitted code end-to-end against a fresh NLTK Brown download with `--seed 42 --threshold 3.0`.

### Accuracy (5,653 generated test cases)

| Method | Non-word accuracy | Real-word accuracy |
|---|---|---|
| A: Standard Edit Distance 1 | 88.84% (5,022 / 5,653) | 83.11% (4,698 / 5,653) |
| B: Symmetric Delete | 88.84% (5,022 / 5,653) | 83.11% (4,698 / 5,653) |

### Speed (1,000 words)

| Method | Total time | Avg. per word |
|---|---|---|
| A: Standard Edit Distance 1 | 0.134183 s | 0.134183 ms |
| B: Symmetric Delete | 0.018157 s | 0.018157 ms |

**Method B is 7.39x faster** at identical accuracy.

### CLI latency (Method B, post warm-up)

| Input | Output | Latency |
|---|---|---|
| I hav a good feeling about this. | I had a good feeling about this. | 2.53 ms |
| This is a test sentnce. | This is a test sentence. | 2.17 ms |
| I would like to sea the world. | I would like to see the world. | 3.07 ms |
| Please meat me at the station. | Please beat me at the station. | 2.54 ms |

---

## Known Limitations

- **Context-blind non-word correction.** Ambiguous repairs are decided by frequency alone (`hav` becomes `had` rather than `have`).
- **Sparse bigrams can cause overcorrection.** For example, `meat` becomes `beat` (expected `meet`), and `has a grate idea` becomes `was a rate idea` because `great` is more than one edit from `grate` and so is never a candidate.
- **Unfiltered vocabulary.** Brown's proper nouns and informal spellings (`lenin`, `goin`) sit in the vocabulary with no frequency floor, which occasionally yields implausible corrections.
- **Edit distance 1 only.** Words two or more edits away (`lernin` to `learning`) are out of reach.
- **Highlighting assumes one-to-one word replacement.** `highlight_changes()` aligns original and corrected text by whitespace split. This is safe today, since corrections are always one word for one word, but would misalign if that changed.
- **Per-sentence cost.** `correct_real_word()` re-scores every word, which matters if the component is reused in a tighter per-token latency budget (e.g., Question 4).

### Ideas for improvement

- Trigram-aware or frequency-floored scoring for real-word correction.
- A minimum-frequency filter or proper-noun filtering on the vocabulary.
- Flagging only suspicious words before running the full real-word scorer.

---

## Reproducibility

Defaults are `--seed 42` and `--threshold 3.0`. Results will vary slightly if either is changed or if a different Brown Corpus version is used.
