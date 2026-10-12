from Spelling_Corrector import SpellingCorrector


def main():
    print("=" * 60)
    print("SPELLING CORRECTOR TEST")
    print("=" * 60)

    print("\n[1] Building model...")
    corrector = SpellingCorrector()
    corrector.build()

    print("Model built successfully.")
    print("Model information:")
    print(corrector.model_info())

    # TEST 1: Non-word errors
    print("\n" + "=" * 60)
    print("TEST 1: NON-WORD ERRORS")
    print("=" * 60)

    nonword_tests = [
        "I hav a good feeling about this.",
        "This is a test sentnce.",
        "I woud like to learn Python.",
        "Please corect this sentnce.",
        "Machine lernin is interesting.",
        "I am goin to college.",
    ]

    for sentence in nonword_tests:
        corrected, changes = corrector.correct(sentence)

        print("\nOriginal : ", sentence)
        print("Corrected: ", corrected)

        if changes:
            print("Changes  :")
            for change in changes:
                print(
                    f"  {change['original']} -> "
                    f"{change['corrected']} "
                    f"({change['type']})"
                )
        else:
            print("Changes  : None")

    # TEST 2: Real-word errors
    print("\n" + "=" * 60)
    print("TEST 2: REAL-WORD ERRORS")
    print("=" * 60)

    realword_tests = [
        "I would like to sea the world.",
        "Please meat me at the station.",
        "I want to bye a new book.",
        "He went too the market.",
        "She has a grate idea.",
    ]

    for sentence in realword_tests:
        corrected, changes = corrector.correct(sentence)

        print("\nOriginal : ", sentence)
        print("Corrected: ", corrected)

        if changes:
            print("Changes  :")
            for change in changes:
                print(
                    f"  {change['original']} -> "
                    f"{change['corrected']} "
                    f"({change['type']})"
                )
        else:
            print("Changes  : None")

    # TEST 3: Candidate generation

    print("\n" + "=" * 60)
    print("TEST 3: CANDIDATE GENERATION")
    print("=" * 60)

    test_words = [
        "hav",
        "sentnce",
        "woud",
        "lern",
        "meat"
    ]

    for word in test_words:
        candidates = corrector.candidates(word)

        print(f"\nWord: {word}")
        print("Candidates:", sorted(candidates)[:20])

    # TEST 4: Clean sentences
 
    print("\n" + "=" * 60)
    print("TEST 4: CORRECT SENTENCES")
    print("=" * 60)

    clean_tests = [
        "I am a student.",
        "This is a simple sentence.",
        "Machine learning is interesting.",
        "I like Python.",
    ]

    for sentence in clean_tests:
        corrected, changes = corrector.correct(sentence)

        print("\nOriginal :", sentence)
        print("Corrected:", corrected)
        print("Changed  :", len(changes) > 0)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()