"""Fixed rendering, no gradients, and preservation of all training RNG states."""
import math
import time

import torch

from .checkpoint import restore_rng, rng_state
from .metrics import add_totals, batch_totals, decode, empty_totals, summary, evaluation_texts, weighted_edit_distance
from .model import ctc_loss
from .render import BatchRenderer


def evaluate(model, dataset, vocabulary, batch_size, device, evaluation_aliases=None):
    previous_mode = model.training
    random_state = rng_state()
    totals = empty_totals()
    weighted_errors = 0.0
    renderer = BatchRenderer(vocabulary)
    if not len(dataset):raise ValueError('Validation dataset is empty')
    if device.type == 'cuda':torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                batch = renderer([dataset[i] for i in range(start,min(start+batch_size,len(dataset)))])
                logits = model(batch.images.to(device))
                loss = float(ctc_loss(logits,batch))
                if not math.isfinite(loss):raise FloatingPointError('Non-finite validation loss')
                predictions = decode(logits,model.output_lengths(batch.image_widths),vocabulary)
                refs, preds = evaluation_texts(batch.texts, predictions, evaluation_aliases)
                weighted_errors += sum(weighted_edit_distance(r, p) for r, p in zip(refs, preds))
                add_totals(totals,batch_totals(predictions,batch.texts,loss,0,evaluation_aliases))
        if device.type == 'cuda':torch.cuda.synchronize(device)
        totals['seconds'] = time.perf_counter()-started
        return dict(summary(totals), weighted_character_errors=weighted_errors,
                    weighted_cer=weighted_errors/totals['reference_characters'])
    finally:
        model.train(previous_mode)
        restore_rng(random_state)
