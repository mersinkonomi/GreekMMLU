"""GreekMMLU prompt and target formatting helpers."""


PROMPT = (
    "Αυτό είναι μια ερώτηση {}. Επίλεξε τη σωστή απάντηση.\n\n"
    "Ερώτηση: {}\n"
    "{}\n\n"
    "Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. "
    "Χρησιμοποίησε ακριβώς μία από τις εξής μορφές: {}. "
    "Μην προσθέσεις άλλο κείμενο.\n\n"
    "Απάντηση:"
)


# subjects_gr
subjects_gr = {
    "Economics": "Οικονομικών",
    "Education": "Παιδαγωγικής",
    "Medicine": "Ιατρικής",
    "Electrical Engineering": "Ηλεκτρολόγων Μηχανικών",
    "Greek Mythology": "Ελληνικής Μυθολογίας",
    "Computer Networks & Security": "Δικτύων Υπολογιστών και Ασφάλειας",
    "Law": "Νομικής",
    "Physics": "Φυσικής",
    "Government and Politics": "Διακυβέρνησης και Πολιτικής",
    "Art": "Τέχνης",
    "Greek Literature": "Νεοελληνικής Λογοτεχνίας",
    "World History": "Παγκόσμιας Ιστορίας",
    "General Knowledge": "Γενικών Γνώσεων",
    "World Religions": "Παγκόσμιων Θρησκειών",
    "Mathematics": "Μαθηματικών",
    "Clinical Knowledge": "Κλινικών Γνώσεων",
    "Driving Rules": "Κανόνων Οδικής Κυκλοφορίας",
    "Biology": "Βιολογίας",
    "Civil Engineering": "Πολιτικών Μηχανικών",
    "Computer Science": "Επιστήμης Υπολογιστών",
    "Geography": "Γεωγραφίας",
    "Chemistry": "Χημείας",
    "Prehistory": "Προϊστορίας",
    "Agriculture": "Γεωργίας",
    "Modern Greek Language": "Νεοελληνικής Γλώσσας",
    "Accounting": "Λογιστικής",
    "Greek History": "Ελληνικής Ιστορίας",
    "Management": "Διοίκησης Επιχειρήσεων",
    "Greek Traditions": "Ελληνικών Παραδόσεων",
    "Maritime Safety and Rescue Operations": "Ναυαγοσωστικων Λειτουργιών και Ασφάλειας στη Θάλασσα",
}



# Greek choice labels. These are Greek code points, not Latin A/B/C/D.
LABELS = ["Α", "Β", "Γ", "Δ"]


def doc_to_text(doc):
    """
    Format the question with choices for Greek MMLU.
    
    Args:
        doc: Dictionary with 'question' and 'choices' fields
        
    Returns:
        Formatted question string
    """
    question = doc["question"]
    choices = doc["choices"]
    subject = doc["subject"]
    
    # Convert English subject to Greek
    subject_gr = subjects_gr.get(subject, subject)
    
    if not 1 < len(choices) <= len(LABELS):
        raise ValueError(
            f"GreekMMLU expects 2-{len(LABELS)} choices, got {len(choices)}"
        )

    # Format choices with Greek labels.
    formatted_choices = []
    for i, choice in enumerate(choices):
        formatted_choices.append(f"{LABELS[i]}. {choice}")
    
    choices_text = "\n".join(formatted_choices)
    boxed_choices = " ή ".join(f"\\boxed{{{label}}}" for label in LABELS[: len(choices)])
    
    return PROMPT.format(subject_gr, question, choices_text, boxed_choices)


def doc_to_choice(doc):
    """
    Extract choice labels based on number of choices.
    
    Args:
        doc: Dictionary with 'choices' field
        
    Returns:
        List of choice labels (e.g., ['Α', 'Β', 'Γ', 'Δ'])
    """
    num_choices = len(doc["choices"])
    return LABELS[:num_choices]


def doc_to_target(doc):
    """Return the normalized Greek answer label used for exact-match scoring."""
    answer_index = int(doc["answer"])
    num_choices = len(doc["choices"])
    if not 0 <= answer_index < num_choices <= len(LABELS):
        raise ValueError(
            f"Invalid answer index {answer_index} for {num_choices} choices"
        )
    return LABELS[answer_index]


def doc_to_boxed_target(doc):
    """Format few-shot answers exactly as requested from the model."""
    return f"\\boxed{{{doc_to_target(doc)}}}"
