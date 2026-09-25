"""Generative boxed-answer prompt and target formatting for Greek ARC."""

# Greek choice labels. These are Greek code points, not Latin A/B/C/D/E.
# ARC questions have 3-5 choices, hence the fifth label Ε.
LABELS = ["Α", "Β", "Γ", "Δ", "Ε"]

PROMPT = (
    "Αυτό είναι μια ερώτηση Επιστημών. Επίλεξε τη σωστή απάντηση.\n\n"
    "Ερώτηση: {question}\n"
    "{choices}\n\n"
    "Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. "
    "Χρησιμοποίησε ακριβώς μία από τις εξής μορφές: {boxed}. "
    "Μην προσθέσεις άλλο κείμενο.\n\n"
    "Απάντηση:"
)


def doc_to_text(doc):
    """
    Format the question with choices for Greek ARC.

    Args:
        doc: Dictionary with 'question' and 'choices' fields, where
            'choices' is a dict with 'text' and 'label' lists.

    Returns:
        Formatted question string
    """
    question = doc["question"]
    texts = doc["choices"]["text"]
    labels = doc["choices"]["label"]

    if not 1 < len(texts) <= len(LABELS):
        raise ValueError(
            f"Greek ARC expects 2-{len(LABELS)} choices, got {len(texts)}"
        )
    if len(texts) != len(labels):
        raise ValueError(
            f"Choices text/label length mismatch: {len(texts)} vs {len(labels)}"
        )

    # Format choices with Greek labels.
    choices_text = "\n".join(
        f"{LABELS[i]}. {choice}" for i, choice in enumerate(texts)
    )
    boxed_choices = " ή ".join(
        f"\\boxed{{{label}}}" for label in LABELS[: len(texts)]
    )

    return PROMPT.format(question=question, choices=choices_text, boxed=boxed_choices)


def doc_to_target(doc):
    """Return the normalized Greek answer label used for exact-match scoring."""
    answer_key = doc["answerKey"]
    labels = doc["choices"]["label"]
    if answer_key not in labels:
        raise ValueError(f"answerKey {answer_key!r} not in choice labels {labels}")
    return LABELS[labels.index(answer_key)]


def doc_to_boxed_target(doc):
    """Format few-shot answers exactly as requested from the model."""
    return f"\\boxed{{{doc_to_target(doc)}}}"
