"""Sparse training errors with resolved rendering inputs, without image storage."""
from dataclasses import asdict
from .metrics import edit_distance, evaluation_texts

EVERY_BATCHES = 100
MAX_SAMPLES = 3


def error_rows(samples, references, predictions, widths, epoch, batch, step, evaluation_aliases=None):
    refs, preds = evaluation_texts(references, predictions, evaluation_aliases)
    ranked=[]
    for i,(ref,pred) in enumerate(zip(refs,preds)):
        errors=edit_distance(ref,pred)
        if errors:ranked.append((i,errors,errors/len(ref)))
    ranked.sort(key=lambda r:(-r[2],r[0]))
    return [dict(epoch=epoch,batch=batch,step=step,rank=rank+1,batch_sample_index=i,
                 sample_id=samples[i].sample_id,input_text=samples[i].text,
                 reference=references[i],prediction=predictions[i],
                 scoring_reference=refs[i],scoring_prediction=preds[i],errors=errors,cer=cer,
                 image_width=int(widths[i]),render_params=asdict(samples[i].render_params),
                 selection='top-errors-in-every-100th-batch',prediction_timing='before-optimizer-update')
            for rank,(i,errors,cer) in enumerate(ranked[:MAX_SAMPLES])]
