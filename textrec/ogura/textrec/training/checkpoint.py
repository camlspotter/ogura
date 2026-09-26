"""Atomic checkpoints, previous-generation recovery, and RNG snapshots."""
import os
from pathlib import Path
import pickle
import random
import tempfile
import warnings

import numpy as np
import torch


class IdentityMismatch(ValueError):
    pass


def rng_state():
    name, keys, pos, gaussian, cached = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (name, keys.tolist(), pos, gaussian, cached),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Checkpoints:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest = self.directory / 'latest.pt'
        self.previous = self.directory / 'previous.pt'
        self.latest_valid = False

    def load(self, identity, compatible_code_hashes=()):
        errors = []
        for path in (self.latest, self.previous):
            if not path.exists():
                continue
            try:
                state = torch.load(path, map_location='cpu', weights_only=True)
                if not isinstance(state, dict) or not {'identity', 'model', 'optimizer', 'scheduler', 'scaler', 'position', 'rng'} <= state.keys():
                    raise ValueError('Incomplete checkpoint')
            except (OSError, RuntimeError, EOFError, ValueError, IndexError, pickle.UnpicklingError) as exc:
                errors.append(f'{path.name}: {exc}')
                continue
            saved_identity = dict(state['identity'])
            if saved_identity.get('training_code_sha256') in compatible_code_hashes:
                saved_identity['training_code_sha256'] = identity['training_code_sha256']
                if 'settings' in saved_identity:
                    saved_identity['settings'] = dict(saved_identity['settings'])
                    saved_identity['settings'].setdefault('model_type', 'small')
            if saved_identity != identity:
                raise IdentityMismatch('Dataset, vocabulary, font, code, runtime, or training configuration changed')
            self.latest_valid = path == self.latest
            if path == self.previous:
                warnings.warn('Recovering previous.pt because latest.pt is absent or unreadable: ' + '; '.join(errors))
            return state
        raise FileNotFoundError('No readable checkpoint: ' + '; '.join(errors))

    def save(self, state):
        fd, name = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if self.latest_valid and self.latest.exists():
                os.replace(self.latest, self.previous)
            os.replace(temporary, self.latest)
            sync_directory(self.directory)
            self.latest_valid = True
        finally:
            temporary.unlink(missing_ok=True)


def write_best(directory, best):
    """Materialize the best snapshot committed inside the restart checkpoint."""
    path = Path(directory) / 'best.pt'
    if best is None:
        path.unlink(missing_ok=True)
        return
    fd, name = tempfile.mkstemp(prefix='.best-', suffix='.tmp', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(best,stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
