import time

from Spelling_Corrector import get_corrector


def highlight_changes(original, corrected):
    """
    Highlight words that changed using ANSI yellow.
    """

    original_words = original.split()
    corrected_words = corrected.split()

    YELLOW = "\033[93m"
    RESET = "\033[0m"

    result = []

    for original_word, corrected_word in zip(
        original_words, corrected_words
    ):
        if original_word != corrected_word:
            result.append(f"{YELLOW}{corrected_word}{RESET}")
        else:
            result.append(corrected_word)

    # Handle any difference in word count safely
    if len(corrected_words) > len(original_words):
        result.extend(corrected_words[len(original_words):])

    return " ".join(result)


def main():
    print("=" * 72)
    print("SPELLING CORRECTOR - TERMINAL")
    print("=" * 72)
    print("Type a sentence to correct it.")
    print("Type 'exit' to quit.")
    print()

    # Build/load the corrector once.
    corrector = get_corrector()

    while True:
        try:
            sentence = input("Enter sentence: ")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if sentence.strip().lower() == "exit":
            print("Goodbye!")
            break

        if not sentence.strip():
            print("Please enter a sentence.")
            print()
            continue

        start_time = time.perf_counter()

        corrected_sentence, _ = corrector.correct(sentence)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        highlighted = highlight_changes(
            sentence,
            corrected_sentence
        )

        print()
        print("Corrected:")
        print(highlighted)
        print(f"Latency: {latency_ms:.3f} ms")
        print()


if __name__ == "__main__":
    main()