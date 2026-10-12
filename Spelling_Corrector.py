"""
Efficient Spelling Corrector

Features:
    1. Brown Corpus vocabulary
    2. Unigram frequency model
    3. Bigram language model
    4. Method A - Standard Edit Distance 1
    5. Method B - Symmetric Delete
    6. Non-word error correction
    7. Real-word error correction using context
    8. Sentence-level correction

Designed as a reusable backend module for:
    - Streamlit deployment
    - Evaluation
    - Benchmarking
    - CLI applications
"""

import math
import re

from collections import Counter, defaultdict
from functools import lru_cache

import nltk
from nltk.corpus import brown


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_K = 0.1
DEFAULT_THRESHOLD = 3.0

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_word(word):
    """
    Convert a word to lowercase and remove unwanted characters.

    Examples:
        "Hello!"  -> "hello"
        "WORLD"   -> "world"
        "can't"   -> "can't"
    """

    word = word.lower()

    return re.sub(r"[^a-z']", "", word)


def tokenize_text(text):
    """
    Tokenize a sentence while preserving basic punctuation.

    Example:
        "I hav a good sentence."

    Returns:
        ["I", "hav", "a", "good", "sentence", "."]
    """

    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,;:]",
        text
    )


# ============================================================
# BROWN CORPUS
# ============================================================

def load_brown():
    """
    Load Brown Corpus sentences.

    If Brown is not installed, download it automatically.
    """

    try:
        return brown.sents()

    except LookupError:

        print("Brown Corpus not found.")
        print("Downloading Brown Corpus...")

        nltk.download("brown", quiet=True)

        return brown.sents()


# ============================================================
# VOCABULARY + UNIGRAM MODEL
# ============================================================

def build_vocabulary(sentences):
    """
    Build vocabulary and unigram frequency model.

    Returns:
        vocabulary:
            set of unique words

        unigram_counts:
            Counter containing word frequencies
    """

    unigram_counts = Counter()

    for sentence in sentences:

        for word in sentence:

            word = normalize_word(word)

            if word:
                unigram_counts[word] += 1

    vocabulary = set(unigram_counts.keys())

    return vocabulary, unigram_counts


# ============================================================
# BIGRAM LANGUAGE MODEL
# ============================================================

def build_bigram_model(sentences):
    """
    Build bigram frequency model.

    bigram_counts[word1][word2]
        = number of times word2 follows word1

    context_counts[word1]
        = number of times word1 occurs as a context
    """

    bigram_counts = defaultdict(Counter)
    context_counts = Counter()

    for sentence in sentences:

        words = []

        for word in sentence:

            word = normalize_word(word)

            if word:
                words.append(word)

        if not words:
            continue

        # Sentence boundary markers
        words = ["<s>"] + words + ["</s>"]

        for i in range(len(words) - 1):

            word1 = words[i]
            word2 = words[i + 1]

            bigram_counts[word1][word2] += 1
            context_counts[word1] += 1

    return bigram_counts, context_counts


# ============================================================
# METHOD A
# STANDARD EDIT DISTANCE 1
# ============================================================

def edits1(word):
    """
    Generate all possible strings at edit distance 1.

    Operations:

        1. Deletion
        2. Transposition
        3. Replacement
        4. Insertion
    """

    word = normalize_word(word)

    candidates = set()

    # --------------------------------------------------------
    # DELETIONS
    # --------------------------------------------------------

    for i in range(len(word)):

        candidate = (
            word[:i] +
            word[i + 1:]
        )

        candidates.add(candidate)

    # --------------------------------------------------------
    # TRANSPOSITIONS
    # --------------------------------------------------------

    for i in range(len(word) - 1):

        candidate = (
            word[:i]
            + word[i + 1]
            + word[i]
            + word[i + 2:]
        )

        candidates.add(candidate)

    # --------------------------------------------------------
    # REPLACEMENTS
    # --------------------------------------------------------

    for i in range(len(word)):

        for letter in ALPHABET:

            candidate = (
                word[:i]
                + letter
                + word[i + 1:]
            )

            candidates.add(candidate)

    # --------------------------------------------------------
    # INSERTIONS
    # --------------------------------------------------------

    for i in range(len(word) + 1):

        for letter in ALPHABET:

            candidate = (
                word[:i]
                + letter
                + word[i:]
            )

            candidates.add(candidate)

    return candidates


def method_a_candidates(word, vocabulary):
    """
    Standard Edit Distance 1 candidate generation.

    Only candidates existing in the vocabulary
    are returned.
    """

    candidates = edits1(word)

    return {
        candidate
        for candidate in candidates
        if candidate in vocabulary
    }


# ============================================================
# METHOD B
# SYMMETRIC DELETE
# ============================================================

def generate_deletes(word):
    """
    Generate all possible one-character deletions.

    Example:

        hello

        ello
        hllo
        helo
        hell
    """

    word = normalize_word(word)

    return {
        word[:i] + word[i + 1:]
        for i in range(len(word))
    }


def build_delete_dictionary(vocabulary):
    """
    Build the Symmetric Delete index.

    Every possible one-character deletion maps
    back to the vocabulary words that produced it.

    Example:

        "ello" -> {"hello"}
        "hllo" -> {"hello"}
        "helo" -> {"hello"}
        "hell" -> {"hello"}
    """

    delete_dictionary = defaultdict(set)

    for word in vocabulary:

        for deleted in generate_deletes(word):

            delete_dictionary[deleted].add(word)

    return dict(delete_dictionary)


def method_b_candidates(
    word,
    vocabulary,
    delete_dictionary
):
    """
    Generate edit-distance-1 candidates using the
    precomputed Symmetric Delete index.

    The delete dictionary stores:

        deleted_form -> vocabulary words

    This allows candidate generation without creating
    the full replacement/insertion alphabet space used
    by Method A.

    Handles:

        1. Deletion
        2. Insertion
        3. Replacement
        4. Adjacent transposition

    Examples:

        hav -> have
        sentnce -> sentence
        teh -> the
    """

    word = normalize_word(word)

    if not word:
        return set()

    candidates = set()

    # --------------------------------------------------------
    # CASE 1
    #
    # Candidate is one character LONGER than observed word.
    #
    # Example:
    #
    #     hav -> have
    #
    #     have
    #       |
    #       delete "e"
    #       v
    #     hav
    #
    # The delete dictionary directly stores:
    #
    #     "hav" -> {"have", ...}
    # --------------------------------------------------------

    candidates.update(
        delete_dictionary.get(word, set())
    )

    # --------------------------------------------------------
    # CASE 2
    #
    # Candidate has the SAME length.
    #
    # Replacement and adjacent transposition errors share
    # at least one deletion with the correct word.
    #
    # Example:
    #
    #     teh -> the
    #
    # deletions of "teh":
    #
    #     eh
    #     th
    #     te
    #
    # "the" also produces:
    #
    #     he
    #     te
    #     th
    #
    # Shared deletion forms allow us to recover "the".
    # --------------------------------------------------------

    word_deletes = generate_deletes(word)

    for deleted in word_deletes:

        candidates.update(
            delete_dictionary.get(deleted, set())
        )

    # --------------------------------------------------------
    # CASE 3
    #
    # Candidate is one character SHORTER than observed word.
    #
    # Example:
    #
    #     hause -> hose
    #
    # A correct shorter candidate may itself be one of the
    # observed word's deletions.
    #
    # The delete dictionary indexes longer vocabulary words,
    # so we explicitly check these shorter forms against
    # the vocabulary.
    # --------------------------------------------------------

    for deleted in word_deletes:

        if deleted in vocabulary:
            candidates.add(deleted)

    # --------------------------------------------------------
    # SAFETY FILTER
    #
    # Symmetric Delete is a candidate-generation strategy.
    # The final candidates must still be genuine one-edit
    # corrections according to the assignment's allowed
    # operations.
    # --------------------------------------------------------

    candidates.intersection_update(vocabulary)

    candidates = {
        candidate
        for candidate in candidates
        if is_edit_distance_one(word, candidate)
    }

    return candidates


# ============================================================
# UNIGRAM SCORING
# ============================================================

def best_unigram_candidate(
    candidates,
    unigram_counts
):
    """
    Select candidate with highest unigram frequency.
    """

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda word: (
            unigram_counts.get(word, 0),
            word
        )
    )


# ============================================================
# BIGRAM PROBABILITY
# ============================================================

def bigram_probability(
    word1,
    word2,
    bigram_counts,
    context_counts,
    vocabulary_size,
    k=DEFAULT_K
):
    """
    Calculate add-k smoothed bigram probability.

    P(word2 | word1)

        count(word1, word2) + k
        -----------------------
        count(word1) + kV
    """

    numerator = (
        bigram_counts[word1][word2]
        + k
    )

    denominator = (
        context_counts[word1]
        + k * vocabulary_size
    )

    return numerator / denominator


# ============================================================
# BIGRAM PHRASE SCORE
# ============================================================

def phrase_bigram_log_probability(
    words,
    bigram_counts,
    context_counts,
    vocabulary_size,
    k=DEFAULT_K
):
    """
    Calculate log probability of a sequence of words.
    """

    if not words:
        return float("-inf")

    normalized = [
        normalize_word(word)
        for word in words
    ]

    normalized = [
        word
        for word in normalized
        if word
    ]

    sequence = (
        ["<s>"]
        + normalized
        + ["</s>"]
    )

    score = 0.0

    for i in range(len(sequence) - 1):

        probability = bigram_probability(
            sequence[i],
            sequence[i + 1],
            bigram_counts,
            context_counts,
            vocabulary_size,
            k
        )

        score += math.log(probability)

    return score


# ============================================================
# NON-WORD ERROR CORRECTION
# ============================================================

def correct_nonword(
    word,
    vocabulary,
    unigram_counts,
    delete_dictionary,
    method="B"
):
    """
    Correct a word that is NOT present in vocabulary.

    Method A:
        Standard Edit Distance 1

    Method B:
        Symmetric Delete
    """

    word = normalize_word(word)

    # Already correct
    if word in vocabulary:
        return word

    method = method.upper()

    if method == "A":

        candidates = method_a_candidates(
            word,
            vocabulary
        )

    elif method == "B":

        candidates = method_b_candidates(
            word,
            vocabulary,
            delete_dictionary
        )

    else:

        raise ValueError(
            "method must be 'A' or 'B'"
        )

    best = best_unigram_candidate(
        candidates,
        unigram_counts
    )

    if best is None:
        return word

    return best


# ============================================================
# EDIT DISTANCE SAFETY CHECK
# ============================================================

def is_edit_distance_one(source, target):
    """
    Check whether two normalized words differ by exactly one
    allowed spelling edit:

        deletion
        insertion
        replacement
        adjacent transposition
    """

    source = normalize_word(source)
    target = normalize_word(target)

    if source == target:
        return True

    if abs(len(source) - len(target)) > 1:
        return False

    # --------------------------------------------------------
    # SOURCE IS ONE CHARACTER SHORTER
    # --------------------------------------------------------

    if len(source) + 1 == len(target):

        i = 0
        j = 0
        differences = 0

        while i < len(source) and j < len(target):

            if source[i] != target[j]:

                differences += 1
                j += 1

                if differences > 1:
                    return False

            else:

                i += 1
                j += 1

        return True

    # --------------------------------------------------------
    # SOURCE IS ONE CHARACTER LONGER
    # --------------------------------------------------------

    if len(target) + 1 == len(source):

        return is_edit_distance_one(
            target,
            source
        )

    # --------------------------------------------------------
    # SAME LENGTH
    #
    # Replacement or adjacent transposition.
    # --------------------------------------------------------

    if len(source) == len(target):

        diff = [
            i
            for i, (a, b)
            in enumerate(zip(source, target))
            if a != b
        ]

        # One-character replacement
        if len(diff) == 1:
            return True

        # Adjacent transposition
        if len(diff) == 2:

            i, j = diff

            return (
                j == i + 1
                and source[i] == target[j]
                and source[j] == target[i]
            )

    return False


# ============================================================
# REAL-WORD CANDIDATES
# ============================================================

def get_real_word_candidates(
    word,
    vocabulary,
    delete_dictionary,
    method="B"
):
    """
    Generate candidates for a word that already exists
    in the vocabulary.

    The original word is always included.
    """

    word = normalize_word(word)

    if method.upper() == "A":

        candidates = method_a_candidates(
            word,
            vocabulary
        )

    elif method.upper() == "B":

        candidates = method_b_candidates(
            word,
            vocabulary,
            delete_dictionary
        )

    else:

        raise ValueError(
            "method must be 'A' or 'B'"
        )

    candidates.add(word)

    return candidates


# ============================================================
# REAL-WORD ERROR CORRECTION
# ============================================================

def correct_real_word(
    previous_word,
    word,
    next_word,
    vocabulary,
    bigram_counts,
    context_counts,
    delete_dictionary,
    unigram_counts,
    method="B",
    threshold=DEFAULT_THRESHOLD,
    k=DEFAULT_K
):
    """
    Correct a real-word error using the Brown bigram language model.

    The original word is compared against edit-distance-1
    candidates.

    A correction is made only when:

        1. The total local context score improves enough.

        2. The candidate does not badly damage either
           neighboring bigram.

    This conservative rule prevents valid words from being
    unnecessarily replaced by another valid vocabulary word.
    """

    word = normalize_word(word)

    previous_word = (
        normalize_word(previous_word)
        if previous_word
        else None
    )

    next_word = (
        normalize_word(next_word)
        if next_word
        else None
    )

    vocabulary_size = len(vocabulary)

    candidates = get_real_word_candidates(
        word,
        vocabulary,
        delete_dictionary,
        method
    )

    # Safety filter
    candidates = {
        candidate
        for candidate in candidates
        if (
            candidate == word
            or is_edit_distance_one(
                word,
                candidate
            )
        )
    }

    if not candidates:
        return word

    total_unigrams = sum(
        unigram_counts.values()
    )

    # --------------------------------------------------------
    # LOCAL BIGRAM SCORE
    # --------------------------------------------------------

    def side_score(left, right):

        if not left or not right:
            return 0.0

        return math.log(
            bigram_probability(
                left,
                right,
                bigram_counts,
                context_counts,
                vocabulary_size,
                k
            )
        )

    # --------------------------------------------------------
    # CANDIDATE SCORE
    # --------------------------------------------------------

    def candidate_details(candidate):

        left = side_score(
            previous_word,
            candidate
        )

        right = side_score(
            candidate,
            next_word
        )

        prior = 0.25 * math.log(
            (
                unigram_counts.get(
                    candidate,
                    0
                )
                + k
            )
            /
            (
                total_unigrams
                + k * vocabulary_size
            )
        )

        return (
            left + right + prior,
            left,
            right
        )

    # --------------------------------------------------------
    # ORIGINAL WORD
    # --------------------------------------------------------

    (
        original_score,
        original_left,
        original_right
    ) = candidate_details(word)

    best_word = word
    best_score = original_score
    best_left = original_left
    best_right = original_right

    # --------------------------------------------------------
    # FIND BEST CANDIDATE
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate == word:
            continue

        (
            score,
            left,
            right
        ) = candidate_details(candidate)

        if score > best_score:

            best_word = candidate
            best_score = score
            best_left = left
            best_right = right

    # No candidate improved the score
    if best_word == word:
        return word

    # --------------------------------------------------------
    # SCORE DELTAS
    # --------------------------------------------------------

    left_delta = (
        best_left - original_left
        if previous_word
        else 0.0
    )

    right_delta = (
        best_right - original_right
        if next_word
        else 0.0
    )

    total_delta = (
        best_score - original_score
    )

    side_deltas = []

    if previous_word:
        side_deltas.append(left_delta)

    if next_word:
        side_deltas.append(right_delta)

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if total_delta < threshold:
        return word

    # Prevent severe degradation of one side
    if side_deltas and min(side_deltas) < -1.0:
        return word

    # Require meaningful improvement on at least one side
    if side_deltas and max(side_deltas) < 0.75:
        return word

    return best_word


# ============================================================
# SENTENCE CORRECTION
# ============================================================

def correct_sentence(
    sentence,
    vocabulary,
    unigram_counts,
    bigram_counts,
    context_counts,
    delete_dictionary,
    method="B",
    threshold=DEFAULT_THRESHOLD,
    k=DEFAULT_K
):
    """
    Correct a complete sentence.

    Pipeline:

        1. Tokenize
        2. Correct non-word errors
        3. Correct real-word errors
        4. Reconstruct sentence

    Returns:

        corrected_sentence
        corrections
    """

    tokens = tokenize_text(sentence)

    # --------------------------------------------------------
    # Separate words and punctuation
    # --------------------------------------------------------

    words = []

    for token in tokens:

        clean = normalize_word(token)

        if clean:
            words.append(clean)

    # --------------------------------------------------------
    # STEP 1
    # NON-WORD CORRECTION
    # --------------------------------------------------------

    corrected_words = []

    corrections = []

    for word in words:

        if word not in vocabulary:

            corrected = correct_nonword(
                word,
                vocabulary,
                unigram_counts,
                delete_dictionary,
                method
            )

            if corrected != word:

                corrections.append({
                    "original": word,
                    "corrected": corrected,
                    "type": "non-word"
                })

            corrected_words.append(
                corrected
            )

        else:

            corrected_words.append(
                word
            )

    # --------------------------------------------------------
    # STEP 2
    # REAL-WORD CORRECTION
    # --------------------------------------------------------

    final_words = corrected_words.copy()

    for i, word in enumerate(
        corrected_words
    ):

        previous_word = (
            corrected_words[i - 1]
            if i > 0
            else None
        )

        next_word = (
            corrected_words[i + 1]
            if i < len(corrected_words) - 1
            else None
        )

        corrected = correct_real_word(
            previous_word,
            word,
            next_word,
            vocabulary,
            bigram_counts,
            context_counts,
            delete_dictionary,
            unigram_counts,
            method,
            threshold,
            k
        )

        if corrected != word:

            corrections.append({
                "original": word,
                "corrected": corrected,
                "type": "real-word"
            })

            final_words[i] = corrected

    # --------------------------------------------------------
    # RECONSTRUCT SENTENCE
    # --------------------------------------------------------

    corrected_sentence = reconstruct_sentence(
        tokens,
        final_words
    )

    return (
        corrected_sentence,
        corrections
    )


# ============================================================
# SENTENCE RECONSTRUCTION
# ============================================================

def restore_case(original, corrected):
    """
    Restore the capitalization pattern of the original token.
    """

    if not corrected:
        return corrected

    if original.isupper():
        return corrected.upper()

    if original[:1].isupper():
        return corrected.capitalize()

    return corrected


def reconstruct_sentence(
    original_tokens,
    corrected_words
):
    """
    Reconstruct sentence while preserving basic punctuation.
    """

    result = []

    word_index = 0

    for token in original_tokens:

        clean = normalize_word(token)

        if clean:

            result.append(
                restore_case(
                    token,
                    corrected_words[
                        word_index
                    ]
                )
            )

            word_index += 1

        else:

            result.append(token)

    text = ""

    for token in result:

        if token in ".,!?;:":

            text = text.rstrip() + token

        else:

            if text:
                text += " "

            text += token

    return text


# ============================================================
# SPELLING CORRECTOR CLASS
# ============================================================

class SpellingCorrector:
    """
    Main spelling-corrector class.

    This class is the main interface that should be used
    by app.py, evaluate.py, and benchmark.py.
    """

    def __init__(
        self,
        method="B",
        k=DEFAULT_K,
        threshold=DEFAULT_THRESHOLD
    ):
        """
        Initialize the spelling corrector.

        method:
            "A" = Standard Edit Distance
            "B" = Symmetric Delete

        k:
            Add-k smoothing parameter.

        threshold:
            Minimum contextual improvement required
            for real-word correction.
        """

        self.method = method.upper()

        self.k = k
        self.threshold = threshold

        self.sentences = None

        self.vocabulary = set()

        self.unigram_counts = Counter()

        self.bigram_counts = defaultdict(
            Counter
        )

        self.context_counts = Counter()

        self.delete_dictionary = {}

        self.is_loaded = False

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    def build(self):
        """
        Build all required NLP models.
        """

        self.sentences = load_brown()

        (
            self.vocabulary,
            self.unigram_counts
        ) = build_vocabulary(
            self.sentences
        )

        (
            self.bigram_counts,
            self.context_counts
        ) = build_bigram_model(
            self.sentences
        )

        self.delete_dictionary = (
            build_delete_dictionary(
                self.vocabulary
            )
        )

        self.is_loaded = True

        return self

    # --------------------------------------------------------
    # CORRECT WORD
    # --------------------------------------------------------

    def correct_word(self, word):
        """
        Correct a single word.
        """

        if not self.is_loaded:
            self.build()

        clean_word = normalize_word(word)

        if clean_word in self.vocabulary:
            return clean_word

        return correct_nonword(
            clean_word,
            self.vocabulary,
            self.unigram_counts,
            self.delete_dictionary,
            self.method
        )

    # --------------------------------------------------------
    # CORRECT SENTENCE
    # --------------------------------------------------------

    def correct(self, sentence):
        """
        Correct a complete sentence.
        """

        if not self.is_loaded:
            self.build()

        return correct_sentence(
            sentence,
            self.vocabulary,
            self.unigram_counts,
            self.bigram_counts,
            self.context_counts,
            self.delete_dictionary,
            self.method,
            self.threshold,
            self.k
        )

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    def candidates(self, word):
        """
        Return candidate corrections for a word.
        """

        if not self.is_loaded:
            self.build()

        word = normalize_word(word)

        if self.method == "A":

            return method_a_candidates(
                word,
                self.vocabulary
            )

        return method_b_candidates(
            word,
            self.vocabulary,
            self.delete_dictionary
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    def model_info(self):
        """
        Return basic model statistics.
        """

        if not self.is_loaded:
            self.build()

        return {
            "vocabulary_size": len(
                self.vocabulary
            ),

            "total_words": sum(
                self.unigram_counts.values()
            ),

            "bigram_contexts": len(
                self.context_counts
            ),

            "method": self.method,

            "smoothing_k": self.k,

            "threshold": self.threshold
        }


# ============================================================
# READY-TO-USE MODEL FUNCTION
# ============================================================

@lru_cache(maxsize=2)
def get_corrector(
    method="B",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD
):
    """
    Create and cache a spelling-corrector model.

    The cache is particularly useful for deployment because
    the Brown Corpus and model do not need to be rebuilt
    every time a user submits text.
    """

    corrector = SpellingCorrector(
        method=method,
        k=k,
        threshold=threshold
    )

    corrector.build()

    return corrector


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SPELLING CORRECTOR")
    print("=" * 60)

    print("\nLoading model...")

    corrector = get_corrector(
        method="B"
    )

    print("\nModel information:")

    info = corrector.model_info()

    for key, value in info.items():

        print(
            f"{key}: {value:,}"
            if isinstance(value, int)
            else f"{key}: {value}"
        )

    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    test_sentences = [
        "I hav a good feeling about this.",
        "This is a test sentnce.",
        "I would like to sea the world.",
        "Please meat me at the station."
    ]

    for sentence in test_sentences:

        corrected, changes = (
            corrector.correct(sentence)
        )

        print("\nOriginal:")
        print(sentence)

        print("\nCorrected:")
        print(corrected)

        print("\nChanges:")

        if changes:

            for change in changes:

                print(
                    f"  {change['original']} "
                    f"-> {change['corrected']} "
                    f"({change['type']})"
                )

        else:

            print("  No changes")

        print("-" * 60)