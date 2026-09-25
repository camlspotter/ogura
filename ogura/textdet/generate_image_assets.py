"""Generate reusable, unreviewed image assets with Z-Image-Turbo; no remote execution."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODEL = 'Tongyi-MAI/Z-Image-Turbo'
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'


def read_prompts(path, limit=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', row.get('id', '')) or row['id'] in ids:
            raise ValueError('Prompt IDs must be unique safe filenames')
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Each entry needs a nonempty prompt')
        ids.add(row['id'])
    if not rows:
        raise ValueError('Empty prompts')
    return rows[:limit] if limit else rows


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_sheet(output, rows):
    sheet = Image.new('RGB', (5*256, ((len(rows)+4)//5)*280), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with Image.open(output/'images'/f"{row['id']}.png") as src:
            thumb = src.convert('RGB')
            thumb.thumbnail((248,248))
            x,y = i%5*256, i//5*280
            sheet.paste(thumb, (x,y))
            draw.text((x+4,y+252), row['id'], fill='black')
    sheet.save(output/'contact-sheet.jpg')


def download(cache):
    from huggingface_hub import snapshot_download
    print(f'Downloading {MODEL}@{REVISION} to {cache}', flush=True)
    return snapshot_download(MODEL, revision=REVISION, cache_dir=str(cache),
        allow_patterns=['model_index.json','scheduler/*','tokenizer/*','text_encoder/*',
                        'transformer/*','vae/*','LICENSE*','README.md'])


def generate(args, rows):
    import torch
    from diffusers import ZImagePipeline
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for generation; use check for a lightweight local test')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script requires a GPU with bfloat16 support')
    config = dict(model=MODEL, revision=REVISION, width=args.size, height=args.size,
                  steps=9, guidance_scale=0.0, seed=args.seed, prompts=rows)
    manifest_path = args.output/'manifest.json'
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config:
            raise ValueError('Resume configuration or prompts differ')
        for item in manifest['images']:
            path = args.output/'images'/f"{item['id']}.png"
            if digest(path) != item['sha256']:
                raise ValueError(f'Image changed: {path}')
    else:
        (args.output/'images').mkdir(parents=True)
        manifest = dict(status='running', review_status='unreviewed_do_not_use_as_negative_yet',
                        config=config, images=[])
    done = {item['id'] for item in manifest['images']}
    save_json(manifest_path, manifest)
    try:
        if len(done) < len(rows):
            # Establish a real CUDA context before CPU weights and file cache
            # consume shared DRAM on GB10. Availability checks alone are not enough.
            print('Initializing CUDA before loading model weights...', flush=True)
            torch.cuda.init()
            probe = torch.empty(1, device='cuda')
            torch.cuda.synchronize()
            del probe
            free, total = torch.cuda.mem_get_info()
            print(f'CUDA memory before loading: {free / 2**30:.1f} GiB free / '
                  f'{total / 2**30:.1f} GiB total', flush=True)
            snapshot = download(args.cache)
            pipe = ZImagePipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
                                                  local_files_only=True)
            # Default SDPA: no Flash Attention dependency or upstream modification.
            if args.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to('cuda')
            for i, row in enumerate(rows):
                if row['id'] in done:
                    continue
                print(f"{i+1}/{len(rows)}: {row['id']}", flush=True)
                with torch.inference_mode():
                    result = pipe(prompt=row['prompt'], height=args.size, width=args.size,
                                  num_inference_steps=9, guidance_scale=0.0,
                                  generator=torch.Generator('cuda').manual_seed(args.seed+i))
                path = args.output/'images'/f"{row['id']}.png"
                temporary = path.with_suffix('.tmp.png')
                result.images[0].save(temporary)
                temporary.replace(path)
                manifest['images'].append(dict(id=row['id'], seed=args.seed+i, sha256=digest(path)))
                save_json(manifest_path, manifest)
        contact_sheet(args.output, rows)
        manifest['status'] = 'complete'
        manifest.pop('error', None)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        save_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check','download','generate'])
    parser.add_argument('--prompts', type=Path, default=ROOT/'prompts/image_assets_sample.jsonl')
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/z-image-turbo')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/image-assets-sample-v1')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu-offload', action='store_true')
    args = parser.parse_args()
    if args.size < 64 or args.size % 16 or (args.limit is not None and args.limit < 1):
        parser.error('size must be a positive multiple of 16 (at least 64); limit must be positive')
    if any(not p.resolve().is_relative_to(ROOT) for p in (args.cache,args.output)):
        parser.error('Cache and output must be under ogura/textdet')
    rows = read_prompts(args.prompts, args.limit)
    if args.action == 'check':
        print(json.dumps(dict(model=MODEL, revision=REVISION, prompts=len(rows), size=args.size,
                              ids=[r['id'] for r in rows]), indent=2))
    elif args.action == 'download':
        print(download(args.cache))
    else:
        generate(args, rows)


if __name__ == '__main__':
    main()
