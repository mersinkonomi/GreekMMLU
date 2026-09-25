"""Generative boxed-answer prompt and target formatting for the Greek
(ell_Grek) Belebele task. Only belebele_ell_Grek.yaml uses this module; the
other language configs keep the shared _default_template_yaml."""

# Greek choice labels. These are Greek code points, not Latin A/B/C/D.
LABELS = ["Α", "Β", "Γ", "Δ"]

PROMPT = (
    "Διάβασε το ακόλουθο απόσπασμα και επίλεξε τη σωστή απάντηση.\n\n"
    "Απόσπασμα: {passage}\n\n"
    "Ερώτηση: {question}\n"
    "{choices}\n\n"
    "Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. "
    "Χρησιμοποίησε ακριβώς μία από τις εξής μορφές: {boxed}. "
    "Μην προσθέσεις άλλο κείμενο.\n\n"
    "Απάντηση:"
)


def doc_to_text(doc):
    """
    Format the passage and question with choices for Greek Belebele.

    Args:
        doc: Dictionary with 'flores_passage', 'question' and
            'mc_answer1'..'mc_answer4' fields

    Returns:
        Formatted question string
    """
    passage = doc["flores_passage"]
    question = doc["question"]
    choices = [
        doc["mc_answer1"],
        doc["mc_answer2"],
        doc["mc_answer3"],
        doc["mc_answer4"],
    ]

    if len(choices) != len(LABELS):
        raise ValueError(
            f"Greek Belebele expects {len(LABELS)} answers, got {len(choices)}"
        )

    # Format choices with Greek labels.
    choices_text = "\n".join(
        f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices)
    )
    boxed_choices = " ή ".join(f"\\boxed{{{label}}}" for label in LABELS)

    return PROMPT.format(
        passage=passage, question=question, choices=choices_text, boxed=boxed_choices
    )


def doc_to_target(doc):
    """Return the normalized Greek answer label used for exact-match scoring."""
    answer_num = int(doc["correct_answer_num"])
    if not 1 <= answer_num <= len(LABELS):
        raise ValueError(f"Invalid correct_answer_num {answer_num}")
    return LABELS[answer_num - 1]


def doc_to_boxed_target(doc):
    """Format few-shot answers exactly as requested from the model."""
    return f"\\boxed{{{doc_to_target(doc)}}}"
