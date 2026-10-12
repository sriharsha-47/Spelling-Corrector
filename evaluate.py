"""
Part 4 - Evaluation for the Brown Corpus spelling corrector.

Requirements implemented from the assignment:
1. Use 10% of Brown sentences.
2. For each selected sentence, randomly choose one word.
3. Create a one-edit non-word error and a one-edit real-word error.
4. Evaluate both Method A and Method B on both test sets.
5. Print reproducible accuracy and useful error statistics.

Run:
    python evaluate.py

Optional:
    python evaluate.py --seed 42 --threshold 3.0
"""

import argparse
import random
from collections import Counter

try:
    from Spelling_Corrector import (
        SpellingCorrector,
        edits1,
        normalize_word,
        tokenize_text,
    )
except ModuleNotFoundError:
    from Spelling_Corrector import (
        SpellingCorrector,
        edits1,
        normalize_word,
        tokenize_text,
    )


def is_word_token(token):
    return bool(normalize_word(token)) and token[0].isalpha()


def make_nonword_error(word, vocabulary, rng):
    """Return a random edit-distance-1 string that is not in vocabulary."""
    candidates = [
        candidate
        for candidate in edits1(word)
        if candidate and candidate not in vocabulary
        and candidate.isalpha()
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def make_realword_error(word, vocabulary, rng):
    """Return a different Brown-vocabulary word at edit distance exactly 1."""
    candidates = [
        candidate
        for candidate in edits1(word)
        if candidate != word and candidate in vocabulary
        and candidate.isalpha()
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def replace_word_in_sentence(sentence_tokens, word_index, replacement):
    """Replace one word token while preserving the sentence punctuation."""
    tokens = list(sentence_tokens)
    seen = 0
    for i, token in enumerate(tokens):
        if is_word_token(token):
            if seen == word_index:
                tokens[i] = replacement
                return tokens
            seen += 1
    raise IndexError("Word index was not found in sentence tokens")


def detokenize(tokens):
    """Small reconstruction helper matching the project's punctuation style."""
    text = ""
    no_space_before = {".", "!", "?", ",", ";", ":"}
    for token in tokens:
        if not text:
            text = token
        elif token in no_space_before:
            text += token
        else:
            text += " " + token
    return text


def extract_word_tokens(sentence):
    return [normalize_word(t) for t in tokenize_text(sentence) if normalize_word(t)]


def build_test_cases(corrector, fraction, seed):
    """Generate paired non-word/real-word cases from exactly 10% of Brown sentences."""
    rng = random.Random(seed)
    sentences = corrector.sentences
    sample_size = int(len(sentences) * fraction)
    selected = rng.sample(list(sentences), sample_size)

    cases = []
    skipped = 0

    for sentence in selected:
        tokens = tokenize_text(" ".join(sentence))
        words = [normalize_word(t) for t in tokens if is_word_token(t)]
        eligible = list(range(len(words)))
        rng.shuffle(eligible)

        found = False
        for word_index in eligible:
            original = words[word_index]
            if len(original) < 2 or not original.isalpha():
                continue

            nonword = make_nonword_error(original, corrector.vocabulary, rng)
            realword = make_realword_error(original, corrector.vocabulary, rng)

            if nonword is None or realword is None:
                continue

            corrupted_nonword_tokens = replace_word_in_sentence(
                tokens, word_index, nonword
            )
            corrupted_realword_tokens = replace_word_in_sentence(
                tokens, word_index, realword
            )

            cases.append({
                "clean_sentence": detokenize(tokens),
                "nonword_sentence": detokenize(corrupted_nonword_tokens),
                "realword_sentence": detokenize(corrupted_realword_tokens),
                "original_word": original,
                "nonword": nonword,
                "realword": realword,
                "word_index": word_index,
            })
            found = True
            break

        if not found:
            skipped += 1

    return cases, sample_size, skipped


def evaluate_nonword(corrector, cases):
    correct = 0
    attempted = 0
    unchanged = 0

    for case in cases:
        result = corrector.correct_word(case["nonword"])
        attempted += 1
        if result == case["original_word"]:
            correct += 1
        if result == case["nonword"]:
            unchanged += 1

    return {
        "correct": correct,
        "total": attempted,
        "accuracy": correct / attempted if attempted else 0.0,
        "unchanged": unchanged,
    }


def evaluate_realword(corrector, cases):
    correct = 0
    attempted = 0
    unchanged = 0

    for case in cases:
        corrected_sentence, _ = corrector.correct(case["realword_sentence"])
        corrected_words = extract_word_tokens(corrected_sentence)
        idx = case["word_index"]

        predicted = corrected_words[idx] if idx < len(corrected_words) else None
        attempted += 1
        if predicted == case["original_word"]:
            correct += 1
        if predicted == case["realword"]:
            unchanged += 1

    return {
        "correct": correct,
        "total": attempted,
        "accuracy": correct / attempted if attempted else 0.0,
        "unchanged": unchanged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    print("=" * 72)
    print("PART 4 - SPELLING CORRECTOR EVALUATION")
    print("=" * 72)
    print(f"Random seed : {args.seed}")
    print(f"Threshold   : {args.threshold}")

    print("\nBuilding Brown models once for test generation...")
    generator = SpellingCorrector(method="B", threshold=args.threshold).build()

    cases, requested_size, skipped = build_test_cases(
        generator, fraction=0.10, seed=args.seed
    )

    print(f"Brown sentences             : {len(generator.sentences):,}")
    print(f"10% requested               : {requested_size:,}")
    print(f"Usable paired test cases    : {len(cases):,}")
    print(f"Sentences skipped           : {skipped:,}")
    print("(A case is used only when both one-edit error types can be generated.)")

    if not cases:
        raise RuntimeError("No evaluation cases could be generated.")

    # Evaluate each method using the exact same generated cases.
    for method in ("A", "B"):
        print("\n" + "-" * 72)
        print(f"METHOD {method}")
        print("-" * 72)

        corrector = SpellingCorrector(
            method=method,
            threshold=args.threshold,
        ).build()

        nonword = evaluate_nonword(corrector, cases)
        realword = evaluate_realword(corrector, cases)

        print(
            f"Non-word accuracy : {nonword['accuracy']:.4%} "
            f"({nonword['correct']}/{nonword['total']})"
        )
        print(f"Non-word unchanged: {nonword['unchanged']:,}")
        print(
            f"Real-word accuracy: {realword['accuracy']:.4%} "
            f"({realword['correct']}/{realword['total']})"
        )
        print(f"Real-word unchanged: {realword['unchanged']:,}")

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print("\nImportant: Method A and Method B were evaluated on the same generated cases.")
    print("Use benchmark.py separately for the required exact 1,000-word timing test.")


if __name__ == "__main__":
    main()
