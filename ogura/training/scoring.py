"""Pairwise allowances for scoring only, never for rendering or CTC labels."""


def allow_punctuation_space(reference, prediction):
    """Ignore inserted spaces after periods/closing parentheses or before opening ones.

    Real sample line-02 has a period with a large right side bearing: its
    visible gap is indistinguishable from an explicit space. Give full credit
    for reference '.' versus prediction '. ', including after width aliases.
    In line-04, the gap in ") (" is likewise visually consistent with the
    parentheses side bearings. Accept space after )/） or before (/（, but
    not after an opening parenthesis or before a closing one. Line-04 spaces
    inside words and numbers remain errors and should be improved by training.
    Preserve actual reference spaces: their deletion is still an error. Other
    punctuation and spaces elsewhere are unaffected. This is an evaluation
    convention, not a classifier merge or an alternative CTC training target.
    """
    if ' ' not in prediction or not any(c in reference for c in '.()（）'):
        return prediction

    def free_space(i, j):
        if prediction[j] != ' ':
            return False
        before = reference[i-1] if i else ''
        after = reference[i] if i < len(reference) else ''
        return ((before in ('.', ')', '）') and j > 0 and prediction[j-1] == before) or
                (after in ('(', '（') and prediction[j+1:j+2] == after))
    n, m = len(reference), len(prediction)
    initial = [0]; initial_paths = [None]
    for j in range(m):
        free = free_space(0, j)
        initial.append(initial[-1] + (0 if free else 1))
        initial_paths.append('free' if free else 'insert')
    costs = [initial]
    paths = [initial_paths]
    for i, a in enumerate(reference, 1):
        row = [i]; path = ['delete']
        for j, b in enumerate(prediction, 1):
            free = free_space(i, j-1)
            choices = [(costs[i-1][j-1] + (a != b), 'match'),
                       (costs[i-1][j] + 1, 'delete'),
                       (row[-1] + (0 if free else 1), 'free' if free else 'insert')]
            value, op = min(choices, key=lambda x: x[0])
            row.append(value); path.append(op)
        costs.append(row); paths.append(path)
    i, j = n, m; retained = []
    while i or j:
        op = paths[i][j]
        if op == 'match':
            retained.append(prediction[j-1]); i -= 1; j -= 1
        elif op == 'delete': i -= 1
        else:
            if op != 'free': retained.append(prediction[j-1])
            j -= 1
    return ''.join(reversed(retained))
