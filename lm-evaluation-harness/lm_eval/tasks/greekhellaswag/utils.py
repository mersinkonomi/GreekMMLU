"""Generative boxed-answer prompt and target formatting for Greek HellaSwag."""

import re

import datasets


# Greek choice labels. These are Greek code points, not Latin A/B/C/D.
LABELS = ["Α", "Β", "Γ", "Δ"]

PROMPT = (
    "Διάβασε το παρακάτω απόσπασμα και επίλεξε την πιο κατάλληλη συνέχεια.\n\n"
    "{query}\n"
    "{choices}\n\n"
    "Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. "
    "Χρησιμοποίησε ακριβώς μία από τις εξής μορφές: {boxed}. "
    "Μην προσθέσεις άλλο κείμενο.\n\n"
    "Απάντηση:"
)


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }

    # Dataset.from_list instead of dataset.map: the shared HF datasets cache at
    # /shared/models/huggingface is not writable, so map() fails with EPERM.
    return datasets.Dataset.from_list([_process_doc(doc) for doc in dataset])


def doc_to_text(doc):
    """
    Format the context with the candidate endings for Greek HellaSwag.

    Args:
        doc: Dictionary with 'query' and 'choices' fields

    Returns:
        Formatted question string
    """
    query = doc["query"]
    choices = doc["choices"]

    if len(choices) != len(LABELS):
        raise ValueError(
            f"Greek HellaSwag expects {len(LABELS)} endings, got {len(choices)}"
        )

    # Format choices with Greek labels.
    choices_text = "\n".join(
        f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices)
    )
    boxed_choices = " ή ".join(f"\\boxed{{{label}}}" for label in LABELS)

    return PROMPT.format(query=query, choices=choices_text, boxed=boxed_choices)


def doc_to_target(doc):
    """Return the normalized Greek answer label used for exact-match scoring."""
    gold = doc["gold"]
    if not 0 <= gold < len(LABELS):
        raise ValueError(f"Invalid gold index {gold}")
    return LABELS[gold]


def doc_to_boxed_target(doc):
    """Format few-shot answers exactly as requested from the model."""
    return f"\\boxed{{{doc_to_target(doc)}}}"
