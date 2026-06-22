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
