"""Pythia: the two-digit addition ability and its LoRA transfer.

The R2b protocol, per (model, revision, stage1_n, seed):

    stage 1   LoRA (r16, lr 3e-4, batch 16, 1000 steps) on stage1_n train
              pairs of a+b; the train accuracy and the held-out a+b accuracy
              are recorded
    stage 2   the same adapter continues for 300 steps on 200 examples of
              a+2b; the held-out a+2b accuracy is recorded
    baselines once per revision and seed, not per stage1_n: a fresh LoRA
              straight on the same 200 a+2b examples, and the stage-1 adapter
              (the one of the largest stage1_n) fine-tuned on copy-a and on
              max(a,b)

The prompt format is the source format, "<a>+<b>=" with the answer after the
equals sign. The fine-tune loss is taken over the answer tokens only.

    python nseed_backbone\\pythia_transfer.py --feasibility
    python nseed_backbone\\pythia_transfer.py --model EleutherAI/pythia-1b \
        --revisions step143000 --stage1-n 200 1000 4000 \
        --out runs_nseed_r2b/pythia_transfer_r2b.csv
    python nseed_backbone\\pythia_transfer.py --pack --out results_queue5.zip

Produces: Figure 3 (the LoRA transfers).
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import sys
import time
import zipfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODEL_1B = 'EleutherAI/pythia-1b'
MODEL_14B = 'EleutherAI/pythia-1.4b'
MODEL_28B = 'EleutherAI/pythia-2.8b'
REVISIONS = {
    MODEL_1B: ['step0', 'step1000', 'step2000', 'step4000', 'step8000', 'step16000',
               'step32000', 'step64000', 'step100000', 'step143000'],
    MODEL_14B: ['step0', 'step1000', 'step2000', 'step4000', 'step8000', 'step16000',
                'step32000', 'step64000', 'step100000', 'step143000'],
    MODEL_28B: ['step0', 'step4000', 'step16000', 'step64000', 'step143000'],
}
N_SHOTS = 3
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 3e-4
TRAIN_STEPS = 1000
STAGE2_STEPS = 600
BATCH_SIZE = 16
EVAL_BATCH = 32
STAGE1_N = (1000, 4000)
STAGE1_TASKS = ('add', 'shuffled')
TARGET_MODULES = ['query_key_value', 'dense']
FIELDS = ['model', 'revision', 'step', 'stage1_n', 'stage1_task', 'seed',
          'stage1_train_acc', 'stage1_heldout', 'stage2_a2b', 'direct_short',
          'direct_long', 'gain_over_best_direct']
R2B_FOLDER = 'runs_nseed_r2d'


def revision_step(revision):
    digits = ''.join(character for character in revision if character.isdigit())
    return int(digits) if digits else -1


def load_model(model_name, revision, device):
    """The checkpoint at `revision`, in half precision on the device."""
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 dtype=torch.float16)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        for candidate in (tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token):
            if candidate is not None:
                tokenizer.pad_token = candidate
                break
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    tokenizer.padding_side = 'left'
    return model, tokenizer


def problems(count, seed, low=10, high=99):
    """`count` random pairs of two-digit numbers with a fixed seed."""
    generator = random.Random(seed)
    return [(generator.randint(low, high), generator.randint(low, high))
            for _ in range(count)]


def shot_prompt(shots, query):
    """The few-shot prompt: "<a>+<b>=<sum>" lines and the query at the end."""
    lines = [f'{a}+{b}={a + b}' for a, b in shots]
    lines.append(f'{query[0]}+{query[1]}=')
    return '\n'.join(lines)


def answer_of(a, b, mode):
    if mode == 'sum':
        return a + b
    if mode == 'a2b':
        return a + 2 * b
    if mode == 'copy':
        return a
    if mode == 'max':
        return max(a, b)
    raise ValueError(f'unknown mode {mode!r}')


def build_examples(pairs, mode):
    """The training texts and answers for one mode."""
    return [(f'{a}+{b}=', str(answer_of(a, b, mode))) for a, b in pairs]


def build_shuffled_examples(pairs, seed):
    """The a+b questions with their answers permuted among the questions.

    The permutation is fixed for the seed, so the model can memorise the
    mapping while the held-out a+b accuracy stays at chance.
    """
    answers = [str(a + b) for a, b in pairs]
    random.Random(seed).shuffle(answers)
    return [(f'{a}+{b}=', answer) for (a, b), answer in zip(pairs, answers)]


def stage1_examples(pairs, task, seed):
    """The stage-1 training examples of a task."""
    if task == 'shuffled':
        return build_shuffled_examples(pairs, seed)
    return build_examples(pairs, 'sum')


@torch.no_grad()
def batched_correct(model, tokenizer, prompts, answers, device, batch_size=EVAL_BATCH):
    """How many prompts continue with the expected answer."""
    correct = 0
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        chunk_answers = answers[start:start + batch_size]
        encoded = tokenizer(chunk_prompts, return_tensors='pt', padding=True).to(device)
        generated = model.generate(**encoded, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        length = encoded['input_ids'].shape[1]
        for row, answer in enumerate(chunk_answers):
            text = tokenizer.decode(generated[row, length:], skip_special_tokens=True)
            if text.strip().startswith(str(answer)):
                correct += 1
    return correct


def accuracy(model, tokenizer, pairs, mode, device, shots=None):
    """The accuracy on `pairs`, with the few-shot prompt when shots are given."""
    if shots:
        prompts = [shot_prompt(shots, (a, b)) for a, b in pairs]
    else:
        prompts = [f'{a}+{b}=' for a, b in pairs]
    answers = [answer_of(a, b, mode) for a, b in pairs]
    if not pairs:
        return float('nan')
    return batched_correct(model, tokenizer, prompts, answers, device) / len(pairs)


def example_accuracy(model, tokenizer, examples, device):
    """The accuracy on prepared (prompt, answer) examples."""
    if not examples:
        return float('nan')
    prompts = [text for text, _answer in examples]
    answers = [answer for _text, answer in examples]
    return batched_correct(model, tokenizer, prompts, answers, device) / len(examples)


def fine_tune(model, tokenizer, examples, device, steps=TRAIN_STEPS,
              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, lora_seed=0):
    """LoRA fine-tune on the answer tokens; returns the model and the step time.

    A model that already carries an adapter is continued, so that stage 2 and
    the copy-a and max baselines start from the stage-1 adapter.
    """
    if not isinstance(model, PeftModel):
        torch.manual_seed(int(lora_seed))
        config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                             bias='none', task_type='CAUSAL_LM',
                             target_modules=TARGET_MODULES)
        model = get_peft_model(model, config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate)
    generator = random.Random(0)
    step_time = 0.0
    for _step in range(steps):
        batch = [examples[generator.randrange(len(examples))] for _ in range(batch_size)]
        texts = [text for text, _answer in batch]
        answers = [answer for _text, answer in batch]
        full = [text + answer for text, answer in batch]
        encoded = tokenizer(full, return_tensors='pt', padding=True,
                            truncation=True, max_length=64).to(device)
        prompt_lengths = [len(tokenizer(text)['input_ids']) for text in texts]
        labels = encoded['input_ids'].clone()
        labels[encoded['attention_mask'] == 0] = -100
        for row, length in enumerate(prompt_lengths):
            labels[row, :length] = -100
        tick = time.time()
        loss = model(**encoded, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_time = time.time() - tick
    model.eval()
    return model, step_time


def run_feasibility(args):
    device = args.device
    model, tokenizer = load_model(args.model, args.revision, device)
    pairs = problems(100, 11)
    shots = [(10, 11), (12, 34), (25, 25)]
    started = time.time()
    value = accuracy(model, tokenizer, pairs, 'sum', device, shots=shots)
    print(f'3-shot addition accuracy on 100 problems: {value:.4f} '
          f'({time.time() - started:.1f} s)')
    examples = build_examples(problems(20, 12), 'a2b')
    _tuned, step_time = fine_tune(model, tokenizer, examples, device, steps=2,
                                  batch_size=args.batch_size, lora_seed=1)
    print(f'one LoRA training step at batch size {args.batch_size}: {step_time:.3f} s')
    return value, step_time


def run_revision_seed(args, device, model_name, revision, seed, sizes, tasks):
    """The stage-1/stage-2 rows and the shared baselines of one revision and seed."""
    train_pool = problems(max(sizes), 2000 + seed)
    heldout = {'sum': problems(args.test_examples, 3000 + seed),
               'a2b': problems(args.test_examples, 4000 + seed),
               'copy': problems(args.test_examples, 5000 + seed),
               'max': problems(args.test_examples, 6000 + seed)}
    train_a2b = problems(args.stage2_examples, 7000 + seed)

    rows = []
    for size in sizes:
        for task in tasks:
            pairs = train_pool[:size]
            examples = stage1_examples(pairs, task, seed)
            model, tokenizer = load_model(model_name, revision, device)
            stage1, _t = fine_tune(model, tokenizer, examples,
                                   device, steps=args.steps, batch_size=args.batch_size,
                                   lora_seed=1000 + seed)
            train_acc = example_accuracy(stage1, tokenizer, examples, device)
            heldout_acc = accuracy(stage1, tokenizer, heldout['sum'], 'sum', device)
            stage2, _t = fine_tune(stage1, tokenizer, build_examples(train_a2b, 'a2b'),
                                   device, steps=args.stage2_steps,
                                   batch_size=args.batch_size)
            stage2_a2b = accuracy(stage2, tokenizer, heldout['a2b'], 'a2b', device)
            rows.append(dict(model=model_name, revision=revision,
                             step=revision_step(revision), stage1_n=size,
                             stage1_task=task, seed=seed,
                             stage1_train_acc=train_acc,
                             stage1_heldout=heldout_acc, stage2_a2b=stage2_a2b,
                             direct_short='', direct_long='',
                             gain_over_best_direct=''))
            print(f'{model_name} {revision} stage1_n={size} task={task} seed {seed}: '
                  f'train {train_acc:.4f} heldout {heldout_acc:.4f} '
                  f'stage2 {stage2_a2b:.4f}', flush=True)
            del stage1, stage2, model
            torch.cuda.empty_cache()

    base_model, tokenizer = load_model(model_name, revision, device)
    direct_short, _t = fine_tune(base_model, tokenizer,
                                 build_examples(train_a2b, 'a2b'), device,
                                 steps=args.stage2_steps, batch_size=args.batch_size,
                                 lora_seed=2000 + seed)
    short_acc = accuracy(direct_short, tokenizer, heldout['a2b'], 'a2b', device)
    direct_long, _t = fine_tune(base_model, tokenizer,
                                build_examples(train_a2b, 'a2b'), device,
                                steps=args.steps + args.stage2_steps,
                                batch_size=args.batch_size, lora_seed=3000 + seed)
    long_acc = accuracy(direct_long, tokenizer, heldout['a2b'], 'a2b', device)
    for row in rows:
        row['direct_short'] = short_acc
        row['direct_long'] = long_acc
        row['gain_over_best_direct'] = row['stage2_a2b'] - max(short_acc, long_acc)
    del direct_short, direct_long, base_model
    torch.cuda.empty_cache()
    print(f'{model_name} {revision} seed {seed} baselines: direct_short {short_acc:.4f} '
          f'direct_long {long_acc:.4f}', flush=True)
    return rows


def write_rows(path, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(out.getvalue(), encoding='utf-8')
    return path


def read_rows(path):
    """The rows of a previous run of the same output file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def run_job(args):
    device = args.device
    table = {(row['model'], row['revision'], str(row['stage1_n']), row['stage1_task'],
              str(row['seed'])): row for row in read_rows(args.out)}
    revisions = args.revisions or REVISIONS.get(args.model, [])
    for revision in revisions:
        for seed in args.seeds:
            pending = [(size, task) for size in sorted(args.stage1_n)
                       for task in args.stage1_task
                       if (args.model, revision, str(size), task, str(seed)) not in table]
            if not pending:
                print(f'{args.model} {revision} seed {seed}: already done', flush=True)
                continue
            try:
                for row in run_revision_seed(args, device, args.model, revision, seed,
                                             sorted({size for size, _task in pending}),
                                             sorted({task for _size, task in pending})):
                    key = (row['model'], row['revision'], str(row['stage1_n']),
                           row['stage1_task'], str(row['seed']))
                    table[key] = row
            except Exception as exc:  # noqa: BLE001 - one revision must not kill the job
                print(f'{args.model} {revision} seed {seed} FAILED: '
                      f'{type(exc).__name__}: {exc}', flush=True)
            write_rows(args.out, [table[key] for key in sorted(table)])
        clear_model_cache(args.model)
    return [table[key] for key in sorted(table)]


def clear_model_cache(model_name):
    """Drop the downloaded weights of one model from the Hugging Face cache.

    The revisions of a model are one cache entry each, and the queue only
    visits every revision once, so the weights are removed after the last seed
    of a revision to keep the cache inside the disk.
    """
    import shutil
    from huggingface_hub import constants as hf_constants
    hub = Path(hf_constants.HF_HUB_CACHE)
    try:
        for folder in (hub / ('models--' + model_name.replace('/', '--')),
                        hub / 'blobs', hub / '.locks'):
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
                print(f'cleared {folder}', flush=True)
    except Exception as exc:  # noqa: BLE001 - the cache is only a disk saving
        print(f'could not clear the cache of {model_name}: {exc}', flush=True)


# ---------------------------------------------------------------------------
# packing and the report
# ---------------------------------------------------------------------------
def write_report(path, rows):
    """One table per (model, revision, stage1_n, stage1_task), the means over seeds."""
    lines = ['# REPORT_job7', '',
             '| model revision | stage1_n | stage1_task | n_seeds | stage1_train '
             '| stage1_heldout | stage2_a2b | direct_short | direct_long '
             '| gain_over_best_direct | gain_sd |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    keys = sorted({(row['model'], row['revision'], str(row['stage1_n']),
                    row['stage1_task']) for row in rows})
    for model_name, revision, size, task in keys:
        subset = [row for row in rows
                  if (row['model'], row['revision'], str(row['stage1_n']),
                      row['stage1_task']) == (model_name, revision, size, task)]
        cells = [f'{model_name} {revision}', size, task, str(len(subset))]
        for field in ('stage1_train_acc', 'stage1_heldout', 'stage2_a2b',
                      'direct_short', 'direct_long', 'gain_over_best_direct'):
            values = [float(row[field]) for row in subset
                      if row.get(field) not in (None, '')]
            cells.append(f'{statistics.fmean(values):.4f}' if values else 'nan')
        gains = [float(row['gain_over_best_direct']) for row in subset
                 if row.get('gain_over_best_direct') not in (None, '')]
        cells.append(f'{statistics.stdev(gains):.4f}' if len(gains) > 1 else 'nan')
        lines.append('| ' + ' | '.join(cells) + ' |')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def step_pack(args):
    """Pack only the new folder and write the report."""
    root = Path(args.root) if args.root else Path.cwd()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = root / R2B_FOLDER
    members = []
    rows = []
    if base.is_dir():
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.csv', '.json'):
                members.append(path)
        rows = read_rows(base / 'pythia_transfer_r2d.csv')
    report = root / 'nseed_backbone' / 'REPORT_job7.md'
    write_report(report, rows)
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    print(f'packed {len(members)} files into {out_path}')
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feasibility', action='store_true')
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--model', default=MODEL_1B)
    parser.add_argument('--revision', default='step143000')
    parser.add_argument('--revisions', nargs='+', default=None)
    parser.add_argument('--stage1-n', nargs='+', type=int, default=list(STAGE1_N))
    parser.add_argument('--stage1-task', nargs='+', default=list(STAGE1_TASKS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(9)))
    parser.add_argument('--out', default='runs_nseed_r2d/pythia_transfer_r2d.csv')
    parser.add_argument('--root', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=TRAIN_STEPS)
    parser.add_argument('--stage2-steps', type=int, default=STAGE2_STEPS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--stage2-examples', type=int, default=500)
    parser.add_argument('--test-examples', type=int, default=300)
    parser.add_argument('--ability-problems', type=int, default=200)
    args = parser.parse_args(argv)

    if args.feasibility:
        run_feasibility(args)
        return 0
    if args.pack:
        step_pack(args)
        return 0
    run_job(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
