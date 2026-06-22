import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # Your implementation here
    counts = Counter(labels)
    total = len(labels)

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy
    pass


def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Your implementation here
    parent_labels = [
        ex[target_attr]
        for ex in examples
    ]

    parent_entropy = calculate_entropy(parent_labels)

    total_examples = len(examples)

    weighted_entropy = 0.0

    values = set(
        ex[attr]
        for ex in examples
    )

    for value in values:

        subset = [
            ex
            for ex in examples
            if ex[attr] == value
        ]

        subset_labels = [
            ex[target_attr]
            for ex in subset
        ]

        weight = len(subset) / total_examples

        weighted_entropy += (
            weight *
            calculate_entropy(subset_labels)
        )

    return parent_entropy - weighted_entropy
    pass


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # Your implementation here
    labels = [
        ex[target_attr]
        for ex in examples
    ]

    counts = Counter(labels)

    max_count = max(counts.values())

    candidates = [
        label
        for label, count in counts.items()
        if count == max_count
    ]

    return sorted(candidates)[0]
    pass


def learn_decision_tree(
    examples,
    attributes,
    target_attr
):

    labels = [
        ex[target_attr]
        for ex in examples
    ]

    # all labels same
    if len(set(labels)) == 1:
        return labels[0]

    # no attributes left
    if not attributes:
        return majority_class(
            examples,
            target_attr
        )

    best_attr = max(
        attributes,
        key=lambda a:
            calculate_information_gain(
                examples,
                a,
                target_attr
            )
    )

    tree = {
        best_attr: {}
    }

    values = set(
        ex[best_attr]
        for ex in examples
    )

    remaining_attrs = [
        a
        for a in attributes
        if a != best_attr
    ]

    for value in values:

        subset = [
            ex
            for ex in examples
            if ex[best_attr] == value
        ]

        if not subset:

            tree[best_attr][value] = (
                majority_class(
                    examples,
                    target_attr
                )
            )

        else:

            subtree = learn_decision_tree(
                subset,
                remaining_attrs,
                target_attr
            )

            tree[best_attr][value] = subtree

    return tree
