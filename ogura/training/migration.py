"""Warm-start a coarser label space without discarding learned image features."""
import hashlib
from pathlib import Path

import torch

from .render import Vocabulary


def migrate_classifier(model, state, vocabulary, channels, model_type, vocabulary_path=None):
    settings = state if 'characters' in state else state.get('identity', {}).get('settings', {})
    if settings.get('channels') != channels or settings.get('model_type', 'small') != model_type:
        raise ValueError('Migration requires the same architecture and channels')
    old_aliases = state.get('identity', {}).get('character_aliases')
    if 'source_characters' in state:
        source = state['source_characters']
    elif 'characters' in state and not old_aliases:
        source = state['characters']
    else:
        if vocabulary_path is None:
            raise ValueError('Migration needs a verified source vocabulary')
        digest = hashlib.sha256(Path(vocabulary_path).read_bytes()).hexdigest()
        if digest != state.get('identity', {}).get('vocabulary_sha256'):
            raise ValueError('Source vocabulary hash differs from checkpoint')
        source = Vocabulary.read(vocabulary_path).source_characters
    old = Vocabulary(source, old_aliases)
    if 'characters' in state and list(old.characters) != state['characters']:
        raise ValueError('Checkpoint class ordering is inconsistent')
    if (old.aliases.compose_katakana and not vocabulary.aliases.compose_katakana) or (old.aliases.collapse_spaces and not vocabulary.aliases.collapse_spaces):
        raise ValueError('Migration cannot undo sequence or whitespace normalization')
    if set(source) != set(vocabulary.source_characters):
        raise ValueError('Migration requires the same source characters; additions/removals are unsupported')
    # An already merged class cannot be separated back into its original characters.
    for c in source:
        if vocabulary.aliases.normalize(c) != vocabulary.aliases.normalize(old.aliases.normalize(c)):
            raise ValueError('Migration cannot split an existing alias class')
    contributors = [[] for _ in range(len(vocabulary))]
    contributors[0] = [0]
    for c, index in old.ids.items():
        target = vocabulary.aliases.normalize(c)
        if len(target) != 1 or target not in vocabulary.ids:
            raise ValueError(f'No one-character destination for old class {c!r}')
        contributors[vocabulary.ids[target]].append(index)
    if any(not indices for indices in contributors):
        raise ValueError('New class has no source weights')
    weights = state['model']
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise ValueError('Checkpoint model keys differ')
    for key, value in weights.items():
        shape = expected[key].shape
        if key in ('classifier.weight', 'classifier.bias'):
            shape = (len(old), *shape[1:])
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'Checkpoint shape mismatch: {key}')
    migrated = dict(weights)
    for key in ('classifier.weight', 'classifier.bias'):
        # Copy singleton rows exactly; averaging is only a starting point for retraining.
        migrated[key] = torch.stack([
            weights[key][indices[0]].clone() if len(indices) == 1 else weights[key][indices].mean(dim=0)
            for indices in contributors
        ])
    model.load_state_dict(migrated, strict=True)
    return dict(method='mean-classifier-rows-v1',
                old_classes_including_blank=len(old), new_classes_including_blank=len(vocabulary),
                source_aliases=old.aliases.config, target_aliases=vocabulary.aliases.config,
                merged_groups=[dict(representative=vocabulary.characters[i-1],
                                    source_classes=[old.characters[j-1] for j in indices])
                               for i, indices in enumerate(contributors) if i and len(indices) > 1])
