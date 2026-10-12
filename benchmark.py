"""
Part 4 - Speed Demon benchmark.

The assignment requires exactly 1,000 misspelled words, with the same batch
passed through Method A and Method B. Only the non-word correction logic is
benchmarked; model construction is excluded from the timed section.

Run:
    python benchmark.py

Optional:
    python benchmark.py --seed 42 --threshold 3.0
"""

import argparse
import random
import time

try:
    from Spelling_Corrector import SpellingCorrector, edits1
except ModuleNotFoundError:
    from Spelling_Corrector import SpellingCorrector, edits1


BATCH_SIZE = 1000


def make_nonword_batch(corrector, size, seed):
    """Create exactly `size` one-edit non-word errors from Brown vocabulary."""
    rng = random.Random(seed)
    vocabulary = list(corrector.vocabulary)
    batch = []

    # Shuffle once so repeated high-frequency words do not dominate the batch.
    rng.shuffle(vocabulary)

    for original in vocabulary:
        if len(original) < 2 or not original.isalpha():
            continue

        candidates = [
            candidate
            for candidate in edits1(original)
            if candidate.isalpha() and candidate not in corrector.vocabulary
        ]
        if not candidates:
            continue

        batch.append(rng.choice(candidates))
        if len(batch) == size:
            break

    if len(batch) != size:
        raise RuntimeError(
            f"Could only generate {len(batch)} non-word errors; "
            f"needed exactly {size}."
        )

    return batch


def time_method(corrector, batch):
    """Time only the non-word correction loop, not model construction."""
    start = time.perf_counter()

    outputs = [
        corrector.correct_word(word)
        for word in batch
    ]

    elapsed = time.perf_counter() - start
    return elapsed, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    print("=" * 72)
    print("PART 4 - SPEED DEMON BENCHMARK")
    print("=" * 72)
    print(f"Batch size  : {BATCH_SIZE:,} words")
    print(f"Random seed : {args.seed}")
    print(f"Threshold   : {args.threshold}")

    print("\nBuilding models (NOT included in timing)...")
    method_a = SpellingCorrector(method="A", threshold=args.threshold).build()
    method_b = SpellingCorrector(method="B", threshold=args.threshold).build()

    # Generate the batch once from the same source vocabulary.
    batch = make_nonword_batch(method_b, BATCH_SIZE, args.seed)

    # Warm-up calls prevent first-call effects from contaminating the measured loop.
    method_a.correct_word(batch[0])
    method_b.correct_word(batch[0])

    # EXACT SAME batch for both methods.
    time_a, output_a = time_method(method_a, batch)
    time_b, output_b = time_method(method_b, batch)

    avg_a = (time_a / BATCH_SIZE) * 1000
    avg_b = (time_b / BATCH_SIZE) * 1000

    speedup = time_a / time_b if time_b > 0 else float("inf")

    print("\n" + "-" * 72)
    print("RESULTS")
    print("-" * 72)
    print(f"Method A total time : {time_a:.6f} seconds")
    print(f"Method A per word   : {avg_a:.6f} ms")
    print(f"Method B total time : {time_b:.6f} seconds")
    print(f"Method B per word   : {avg_b:.6f} ms")
    print(f"A/B runtime ratio   : {speedup:.2f}x")
    print(f"Exact same batch    : {len(batch) == BATCH_SIZE}")
    print(f"Outputs produced    : {len(output_a):,} / {len(output_b):,}")

    print("\n" + "-" * 72)
    print("WHY METHOD B SHOULD BE FASTER")
    print("-" * 72)
    print(
        "Method A explicitly constructs the full edit-distance-1 search space "
        "(deletions, transpositions, replacements, and insertions) and then "
        "checks candidates against the vocabulary. Method B uses a precomputed "
        "deletion index, so candidate retrieval is primarily dictionary lookup "
        "plus a small number of one-character deletions. The measured runtime "
        "therefore reflects the lower per-word candidate-generation cost of the "
        "Symmetric Delete approach."
    )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
