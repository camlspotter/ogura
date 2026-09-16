"""Fixed rendering, no gradients, and preservation of all training RNG states."""
import math
import time

import torch

from .checkpoint import restore_rng, rng_state
from .metrics import add_totals, batch_totals, decode, empty_totals, summary
from .model import ctc_loss
from .render import BatchRenderer


def evaluate(model, dataset, vocabulary, batch_size, device):
    previous_mode = model.training
    random_state = rng_state()
    totals = empty_totals()
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
                add_totals(totals,batch_totals(predictions,batch.texts,loss,0))
        if device.type == 'cuda':torch.cuda.synchronize(device)
        totals['seconds'] = time.perf_counter()-started
        return summary(totals)
    finally:
        model.train(previous_mode)
        restore_rng(random_state)
