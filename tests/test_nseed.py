"""Acceptance tests for the N-seed backbone harness (spec section 8).

Run from the project root:

    python -m unittest -v nseed_backbone.test_nseed

or, from inside `nseed_backbone`:

    python -m unittest -v test_nseed
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402  (common binds the MSVC runtime first)
from common import COUNT, N_CONTEXTS, P  # noqa: E402


def exploding_worker(seed, config):
    """A worker whose job raises, used to test the failure reporting."""
    raise RuntimeError(f'the worker for seed {seed} exploded')


def silent_worker(seed, config):
    """A worker that finishes without ever reaching a terminal job state."""
    common.log_line(f'[test] silent worker for seed {seed}')


class MaskAndSplitTest(unittest.TestCase):
    """Test 1: the coverage mask and the symmetry-clean split."""

    def test_mask_is_identical_across_contexts_and_has_1392_pairs(self):
        for seed in (3, 5, 7):
            mask = common.mask_for_seed(seed)
            self.assertEqual(tuple(mask.shape), (N_CONTEXTS, P * P))
            self.assertEqual(int(mask[0].sum()), COUNT)
            self.assertEqual(COUNT, 1392)
            for context in range(1, N_CONTEXTS):
                self.assertTrue(bool((mask[context] == mask[0]).all()),
                                f'seed {seed}: context {context} mask differs')

    def test_mask_follows_the_specified_generator(self):
        seed = 5
        generator = torch.Generator().manual_seed(100000 + seed)
        selected = torch.randperm(P * P, generator=generator)[:COUNT]
        mask = common.mask_for_seed(seed)
        self.assertTrue(bool(mask[0][selected].all()))
        self.assertEqual(int(mask[0].sum()), COUNT)

    def test_dev_split_is_nonempty_and_disjoint_from_train_in_every_context(self):
        for seed in (3, 5, 7):
            mask = common.mask_for_seed(seed)
            strata, exposure = common.split_for_mask(mask)
            self.assertGreater(len(strata['dev']), 0, f'seed {seed}: empty dev split')
            self.assertGreater(len(strata['audit']), 0, f'seed {seed}: empty audit split')
            self.assertEqual(len(strata['train']), COUNT)
            train_ids = set(int(i) for i in strata['train'])
            dev_ids = [int(i) for i in strata['dev']]
            transpose = exposure['transpose_ids']
            for dev_id in dev_ids:
                self.assertNotIn(dev_id, train_ids,
                                 f'seed {seed}: dev pair {dev_id} is in train')
                transpose_id = int(transpose[dev_id])
                self.assertNotIn(transpose_id, train_ids,
                                 f'seed {seed}: transpose {transpose_id} of dev pair '
                                 f'{dev_id} is in train')
            for context in range(N_CONTEXTS):
                self.assertTrue(bool(mask[context][dev_ids].logical_not().all()))


class LabelTest(unittest.TestCase):
    """Test 2: the task labels are the ones the spec prescribes."""

    def test_labels_match_an_independent_recomputation(self):
        pairs = torch.cartesian_prod(torch.arange(P), torch.arange(P))
        a, b = pairs[:, 0], pairs[:, 1]
        permutation = common.input_permutation()
        for context in range(N_CONTEXTS):
            multiplier = context + 1
            expected = {
                'source': (a + b) % P,
                'affine': (a + multiplier * b) % P,
                'scaled': (multiplier * (a + b)) % P,
                'input_perm': (permutation[a] + permutation[b]) % P,
            }
            for task, want in expected.items():
                data = common.build_task(task)
                self.assertEqual(tuple(data['x'].shape), (N_CONTEXTS * P * P, 4))
                rows = common.context_rows(torch.arange(P * P), context)
                got = data['y'][rows]
                self.assertTrue(bool((got == want).all()),
                                f'task {task} context {context}: labels differ')

    def test_affine_contexts_are_not_functions_of_a_plus_b(self):
        data = common.build_task('affine')
        source = common.build_task('source')
        for context in range(1, N_CONTEXTS):
            rows = common.context_rows(torch.arange(P * P), context)
            differing = int((data['y'][rows] != source['y'][rows]).sum())
            self.assertGreater(differing, 0,
                               f'affine context {context} coincides with a+b')

    def test_input_permutation_is_the_specified_one(self):
        generator = torch.Generator().manual_seed(555)
        expected = torch.randperm(P, generator=generator)
        self.assertTrue(bool((common.input_permutation() == expected).all()))


class EvaluationTest(unittest.TestCase):
    """Test 3: the run's evaluation agrees with a plain-loop reimplementation."""

    def test_dev_accuracy_matches_a_plain_loop(self):
        torch.manual_seed(11)
        model = common.make_model(common.CONFIG)
        model.eval()
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        data = common.build_task('source')
        evaluation = common.evaluate(model, data, strata)

        for context in range(N_CONTEXTS):
            full_rows = common.context_rows(strata['dev'], context)
            self.assertAlmostEqual(evaluation[f'dev_acc_c{context}'],
                                   common.accuracy_on(model, data, full_rows), places=12)
            rows = full_rows[:50]
            correct = 0
            for row in rows.tolist():
                logits = model(data['x'][row].unsqueeze(0))
                correct += int(int(logits.argmax(-1).item()) == int(data['y'][row].item()))
            plain = correct / len(rows)
            self.assertAlmostEqual(common.accuracy_on(model, data, rows), plain, places=12)
        self.assertAlmostEqual(evaluation['dev_mean'],
                               sum(evaluation[f'dev_acc_c{c}'] for c in range(N_CONTEXTS))
                               / N_CONTEXTS, places=12)
        self.assertAlmostEqual(evaluation['dev_worst'],
                               min(evaluation[f'dev_acc_c{c}'] for c in range(N_CONTEXTS)),
                               places=12)

    def test_train_accuracy_matches_a_plain_loop(self):
        torch.manual_seed(12)
        model = common.make_model(common.CONFIG)
        model.eval()
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        data = common.build_task('source')
        evaluation = common.evaluate(model, data, strata)
        full_rows = common.index_for(strata['train'])
        self.assertAlmostEqual(evaluation['train_acc'],
                               common.accuracy_on(model, data, full_rows), places=12)
        rows = full_rows[:50]
        correct = 0
        for row in rows.tolist():
            logits = model(data['x'][row].unsqueeze(0))
            correct += int(int(logits.argmax(-1).item()) == int(data['y'][row].item()))
        self.assertAlmostEqual(common.accuracy_on(model, data, rows),
                               correct / len(rows), places=12)


class ResumeTest(unittest.TestCase):
    """Test 4: an interrupted run resumes to the identical state."""

    def _train(self, model, optimizer, data, rows, steps):
        for _ in range(steps):
            common.train_step(model, optimizer, data, rows)

    def _resume_equality(self, deterministic, threads, device='cpu', tolerance=0.0):
        previous = os.environ.get(common.DETERMINISTIC_ENV)
        os.environ[common.DETERMINISTIC_ENV] = '1' if deterministic else ''
        try:
            runtime = common.setup(threads, device)
            self.assertEqual(runtime['deterministic'], deterministic and device == 'cpu')
            self.assertEqual(runtime['device'], device)
            mask = common.mask_for_seed(5)
            strata, _ = common.split_for_mask(mask)
            data = common.build_task('source', device)
            rows = common.index_for(strata['train'], device)

            straight = common.init_model(5, device)
            straight_optimizer = common.make_optimizer(straight)
            self._train(straight, straight_optimizer, data, rows, 60)

            temp = Path(tempfile.mkdtemp(prefix='nseed_resume_'))
            try:
                resumed = common.init_model(5, device)
                resumed_optimizer = common.make_optimizer(resumed)
                self._train(resumed, resumed_optimizer, data, rows, 30)
                common.save_resumable(temp / 'latest.pt', resumed, resumed_optimizer, 30, [])
                fresh = common.init_model(5, device)
                fresh_optimizer = common.make_optimizer(fresh)
                common.restore_resumable(temp / 'latest.pt', fresh, fresh_optimizer)
                self._train(fresh, fresh_optimizer, data, rows, 30)

                worst = 0.0
                for name, parameter in straight.named_parameters():
                    other = dict(fresh.named_parameters())[name]
                    if tolerance == 0.0:
                        self.assertTrue(bool(torch.equal(parameter.detach(), other.detach())),
                                        f'resumed parameter {name} differs')
                    else:
                        difference = float((parameter.detach() - other.detach()).abs().max())
                        worst = max(worst, difference)
                self.assertLessEqual(worst, tolerance,
                                     f'resumed weights differ by {worst} (tolerance {tolerance})')
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        finally:
            if previous is None:
                os.environ.pop(common.DETERMINISTIC_ENV, None)
            else:
                os.environ[common.DETERMINISTIC_ENV] = previous

    def test_resume_matches_uninterrupted_determinism_off(self):
        self._resume_equality(deterministic=False, threads=1, device='cpu')

    def test_resume_matches_uninterrupted_determinism_on(self):
        self._resume_equality(deterministic=True, threads=1, device='cpu')

    def _gpu_resume_numbers(self, steps=30):
        """(noise between two identical GPU runs, difference of a resumed run)."""
        device = 'cuda'
        runtime = common.setup(1, device)
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        data = common.build_task('source', device)
        rows = common.index_for(strata['train'], device)

        def fresh():
            model = common.init_model(5, device)
            return model, common.make_optimizer(model)

        def train(model, optimizer, count):
            for _ in range(count):
                optimizer.zero_grad(set_to_none=True)
                logits = model(data['x'][rows])
                loss = torch.nn.functional.cross_entropy(logits, data['y'][rows])
                loss.backward()
                optimizer.step()

        def max_difference(left, right):
            return max(float((p.detach() - q.detach()).abs().max())
                       for p, q in zip(left.parameters(), right.parameters()))

        first, first_optimizer = fresh()
        train(first, first_optimizer, steps)
        second, second_optimizer = fresh()
        train(second, second_optimizer, steps)
        noise = max_difference(first, second)

        temp = Path(tempfile.mkdtemp(prefix='nseed_gpu_resume_'))
        try:
            half = steps // 2
            resumed, resumed_optimizer = fresh()
            train(resumed, resumed_optimizer, half)
            common.save_resumable(temp / 'latest.pt', resumed, resumed_optimizer, half, [])
            restored, restored_optimizer = fresh()
            common.restore_resumable(temp / 'latest.pt', restored, restored_optimizer)
            train(restored, restored_optimizer, steps - half)
            difference = max_difference(first, restored)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        return runtime, noise, difference

    def test_gpu_resume_within_tolerance(self):
        if not common.cuda_available():
            self.skipTest('no CUDA device available')
        runtime, noise, difference = self._gpu_resume_numbers()
        print(f'\n  GPU resume: run-to-run noise {noise:.3e}, '
              f'resume difference {difference:.3e}, '
              f'deterministic={runtime["deterministic"]}')
        if runtime['deterministic']:
            self.assertLessEqual(difference, 1e-5,
                                 f'resumed weights differ by {difference} (tolerance 1e-5)')
        else:
            bound = max(1e-5, 10.0 * noise)
            self.assertLessEqual(difference, bound,
                                 f'resumed weights differ by {difference}; the GPU noise floor '
                                 f'is {noise}, so the bound is {bound}')


class ParityTest(unittest.TestCase):
    """CPU and GPU must agree early on and are allowed to drift apart later."""

    def _loss_trace(self, device, steps, seed=5):
        common.setup(1, device)
        mask = common.mask_for_seed(seed)
        strata, _ = common.split_for_mask(mask)
        data = common.build_task('source', device)
        rows = common.index_for(strata['train'], device)
        model = common.init_model(seed, device)
        optimizer = common.make_optimizer(model)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(data['x'][rows])
            loss = torch.nn.functional.cross_entropy(logits, data['y'][rows])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def test_cpu_and_gpu_agree_early(self):
        if not common.cuda_available():
            self.skipTest('no CUDA device available')
        cpu_losses = self._loss_trace('cpu', 200)
        gpu_losses = self._loss_trace('cuda', 200)
        differences = [abs(a - b) for a, b in zip(cpu_losses, gpu_losses)]
        early = max(differences[:20])
        self.assertLess(early, 1e-4,
                        f'CPU and GPU losses differ by {early} in the first 20 steps')
        later = max(differences[20:])
        print(f'\n  CPU/GPU parity: max |dloss| over the first 20 steps = {early:.3e}, '
              f'over steps 21-200 = {later:.3e}; '
              f'final CPU loss {cpu_losses[-1]:.6f} vs GPU {gpu_losses[-1]:.6f}')


class NormControlTest(unittest.TestCase):
    """Test 5: the norm control rescales the model to the target norm."""

    def test_rescale_matches_the_target_norm(self):
        common.setup(4, 'cpu')
        source = common.init_model(5)
        target = common.init_model(6)
        target_norm = common.global_norm(target)
        scale = common.rescale_to_norm(source, target_norm)
        self.assertAlmostEqual(common.global_norm(source) / target_norm, 1.0, places=6)
        self.assertGreater(scale, 0.0)

    def test_rescale_leaves_the_direction_unchanged(self):
        common.setup(4, 'cpu')
        source = common.init_model(5)
        before = {name: parameter.detach().clone()
                  for name, parameter in source.named_parameters()}
        target_norm = common.global_norm(source) * 3.0
        common.rescale_to_norm(source, target_norm)
        for name, parameter in source.named_parameters():
            original = before[name]
            keep = original.abs() > 1e-12
            if int(keep.sum()) == 0:
                self.assertEqual(int((parameter.detach() != 0).sum()), 0,
                                 f'parameter {name} was zero and must stay zero')
                continue
            ratio = parameter.detach()[keep] / original[keep]
            self.assertTrue(bool(torch.allclose(ratio, ratio[0].expand_as(ratio),
                                                rtol=1e-4, atol=1e-6)),
                            f'parameter {name} was not scaled by a single scalar')


class FourierTest(unittest.TestCase):
    """Test 6: the embedding Fourier metrics."""

    def test_single_frequency_embedding(self):
        embedding = torch.zeros(P, 3)
        for a in range(P):
            embedding[a, 0] = math.cos(2 * math.pi * 7 * a / P)
        metrics = common.fourier_metrics(embedding)
        self.assertAlmostEqual(metrics['top5_share'], 1.0, places=6)
        self.assertEqual(metrics['peak_k'], 7)
        self.assertAlmostEqual(metrics['nonconst_share'], 1.0, places=6)

    def test_constant_embedding_has_no_nonconstant_energy(self):
        embedding = torch.ones(P, 2)
        energy, power = common.embedding_spectrum(embedding)
        self.assertAlmostEqual(float(energy.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(power[0]), float(power.sum()), places=4)


class StageSelectionTest(unittest.TestCase):
    """Stage selection reads the saved evaluation rows only (spec section 3b)."""

    def _run(self, temp):
        from run_seed import SeedRun
        return SeedRun(7, temp / 'seed_7', threads=2, resume=False, tiers=(),
                       log=lambda *a: None)

    def _row(self, step, train_acc, dev_mean, dev_worst):
        row = common.make_row(7, 'source', 'source', 'scratch', step,
                              {'train_acc': train_acc,
                               'dev_acc_c0': dev_mean, 'dev_acc_c1': dev_mean,
                               'dev_acc_c2': dev_mean, 'dev_acc_c3': dev_mean,
                               'dev_mean': dev_mean, 'dev_worst': dev_worst})
        return row

    def test_memorized_transition_and_grokked_are_picked(self):
        temp = Path(tempfile.mkdtemp(prefix='nseed_stages_'))
        try:
            run = self._run(temp)
            (run.ckpts).mkdir(parents=True, exist_ok=True)
            for step in (500, 1000, 1500, 2000, 2500, 3000):
                common.save_model_checkpoint(run.ckpts / f'step_{step:07d}.pt',
                                             common.init_model(7), step)
            common.save_model_checkpoint(run.ckpts / 'grokked.pt', common.init_model(7),
                                         2500)
            rows = [
                self._row(500, 0.99, 0.0, 0.0),
                self._row(1000, 1.0, 0.0, 0.0),
                self._row(1500, 1.0, 0.0, 0.0),
                self._row(2000, 1.0, 0.5, 0.5),
                self._row(2500, 1.0, 1.0, 1.0),
            ]
            stages = run.select_stages(rows, grok_step=2500, status='grokked')
            self.assertIn('memorized', stages)
            self.assertIn('transition', stages)
            self.assertIn('grokked', stages)
            self.assertEqual(stages['memorized']['step'], 1000)
            self.assertEqual(stages['transition']['step'], 2000)
            self.assertEqual(stages['grokked']['step'], 2500)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_transition_is_marked_missing_when_no_checkpoint_fits(self):
        temp = Path(tempfile.mkdtemp(prefix='nseed_stages_'))
        try:
            run = self._run(temp)
            (run.ckpts).mkdir(parents=True, exist_ok=True)
            common.save_model_checkpoint(run.ckpts / 'step_0000500.pt',
                                         common.init_model(7), 500)
            rows = [self._row(500, 1.0, 1.0, 1.0)]
            stages = run.select_stages(rows, grok_step=2500, status='grokked')
            self.assertNotIn('transition', stages)
            self.assertNotIn('memorized', stages)
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class JobScheduleTest(unittest.TestCase):
    """Test 5a: the job schedule respects the source -> stage -> transfer order."""

    def test_transfer_jobs_start_after_their_source_and_stage(self):
        from run_batch import JOB_LATE, JOB_SOURCE, JOB_STAGE, JOB_TRANSFER, JobScheduler

        temp = Path(tempfile.mkdtemp(prefix='nseed_jobs_'))
        try:
            events = temp / 'events'
            events.mkdir()
            config = dict(runs_dir=str(temp), threads=1, tiers=['A'], resume=False,
                          source_cap=1, eval_every=1, ckpt_every=1, resumable_every=1,
                          transfer_steps=1, transfer_eval_every=1, late_offset=1,
                          max_train_pairs=4, stop=False)
            scheduler = JobScheduler(2, config, executor=fake_job_executor)

            submitted = []
            done = []

            def submit(job):
                submitted.append(job['id'])
                scheduler.submit(job)

            def on_result(result):
                done.append(result['id'])
                if result.get('kind') == JOB_SOURCE:
                    submit({'id': f'stage:{result["id"]}', 'kind': JOB_STAGE,
                            'seed': result['id']})
                elif result.get('kind') == JOB_STAGE:
                    submit({'id': f'transfer:{result["id"]}', 'kind': JOB_TRANSFER,
                            'seed': result['id'], 'task': 'affine', 'init': 'grokked'})

            for seed in (11, 12):
                submit({'id': f'source:{seed}', 'kind': JOB_SOURCE, 'seed': seed})
            scheduler.run(on_result=on_result)

            self.assertEqual(len(done), 6, f'expected 6 jobs, got {done}')
            for seed in (11, 12):
                source = read_event(events, f'source:{seed}')
                stage = read_event(events, f'stage:source:{seed}')
                transfer = read_event(events, f'transfer:stage:source:{seed}')
                self.assertGreaterEqual(stage['start'], source['end'],
                                        f'stage for seed {seed} started before its source ended')
                self.assertGreaterEqual(transfer['start'], stage['end'],
                                        f'transfer for seed {seed} started before its stage ended')
        finally:
            shutil.rmtree(temp, ignore_errors=True)


def fake_job_executor(job, config, log):
    """Fake job used by the dependency test; records start and end timestamps."""
    path = Path(config['runs_dir']) / 'events' / f'{safe_name(job["id"])}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(start=time.time())
    path.write_text(json.dumps(record), encoding='utf-8')
    time.sleep(0.15)
    record['end'] = time.time()
    path.write_text(json.dumps(record), encoding='utf-8')
    return dict(kind=job['kind'])


def safe_name(job_id):
    """A file-name-safe version of a job id (colons are invalid on Windows)."""
    return job_id.replace(':', '_')


def read_event(events, job_id):
    path = Path(events) / f'{safe_name(job_id)}.json'
    return json.loads(path.read_text(encoding='utf-8'))


class TransferResumeTest(unittest.TestCase):
    """Test 5b: an interrupted transfer job resumes from its own state file."""

    def _fixture(self, out):
        """A seed folder with a grokked checkpoint and the matching evaluation rows."""
        (out / 'ckpts').mkdir(parents=True, exist_ok=True)
        model = common.init_model(77)
        rows = []
        for step, train_acc, dev in ((250, 0.99, 0.0), (500, 1.0, 1.0)):
            evaluation = {'train_acc': train_acc,
                          'dev_acc_c0': dev, 'dev_acc_c1': dev,
                          'dev_acc_c2': dev, 'dev_acc_c3': dev,
                          'dev_mean': dev, 'dev_worst': dev}
            row = common.make_row(77, 'source', 'source', 'scratch', step, evaluation,
                                  stage_clean_acc=dev, norm=common.global_norm(model))
            rows.append(row)
            common.save_model_checkpoint(out / 'ckpts' / f'step_{step:07d}.pt', model, step,
                                         row, extra={'seed': 77, 'stage': 'source'})
        common.save_model_checkpoint(out / 'ckpts' / 'grokked.pt', model, 500, rows[-1],
                                     extra={'seed': 77, 'stage': 'grokked',
                                            'global_norm': common.global_norm(model)})
        common.write_metrics(out / 'metrics.csv', rows)
        common.write_summary(out / 'summary.json',
                             dict(seed=77, mask_seed=77, status='grokked', grok_step=500))

    def test_transfer_resumes_from_the_saved_step(self):
        from run_seed import SeedRun

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_xfer_'))
        try:
            out = temp / 'seed_77'
            self._fixture(out)
            run = SeedRun(77, out, threads=1, resume=False, tiers=('A',),
                          source_cap=1, eval_every=1, ckpt_every=1, resumable_every=1,
                          transfer_steps=6, transfer_eval_every=2,
                          max_train_pairs=8, log=lambda *a: None)
            stages = run.stage_job()
            self.assertIn('grokked', stages['stages'])

            state_path = out / 'ckpts' / 'transfer_affine_grokked_state.pt'
            first = SeedRun(77, out, threads=1, resume=False, tiers=('A',),
                            source_cap=1, eval_every=1, ckpt_every=1, resumable_every=1,
                            transfer_steps=4, transfer_eval_every=2,
                            max_train_pairs=8, log=lambda *a: None)
            first.transfer_job('affine', 'grokked')
            self.assertTrue(state_path.exists())
            payload = common.load_checkpoint(state_path)
            self.assertEqual(int(payload['step']), 4)

            rows_after_first = common.read_metrics(out / 'metrics.csv')
            self.assertTrue(any(r['stage'] == 'transfer' and int(r['step']) == 4
                                for r in rows_after_first))

            summary = common.read_summary(out / 'summary.json')
            summary['transfers'].pop('affine|grokked', None)
            common.write_summary(out / 'summary.json', summary)
            resumed = SeedRun(77, out, threads=1, resume=True, tiers=('A',),
                              source_cap=1, eval_every=1, ckpt_every=1, resumable_every=1,
                              transfer_steps=6, transfer_eval_every=2,
                              max_train_pairs=8, log=lambda *a: None)
            resumed.transfer_job('affine', 'grokked')

            payload = common.load_checkpoint(state_path)
            self.assertEqual(int(payload['step']), 6)
            rows = common.read_metrics(out / 'metrics.csv')
            steps = sorted(int(r['step']) for r in rows
                           if r['stage'] == 'transfer' and r['init'] == 'grokked')
            self.assertEqual(steps, [2, 4, 6], f'unexpected transfer steps {steps}')
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class LocalizeTest(unittest.TestCase):
    """The localization machinery: transplants, knockouts, freeze and step-0 eval."""

    def test_transplant_swaps_only_the_numeric_rows(self):
        import fourier_metrics as fourier

        recipient = common.make_model(common.CONFIG).state_dict()
        donor = common.make_model(common.CONFIG).state_dict()
        merged = fourier.transplant(recipient, donor)
        self.assertTrue(bool(torch.equal(merged['token_embed.weight'][:common.P],
                                         donor['token_embed.weight'][:common.P])))
        self.assertTrue(bool(torch.equal(merged['token_embed.weight'][common.P:],
                                         recipient['token_embed.weight'][common.P:])))
        for name, value in recipient.items():
            if name == 'token_embed.weight':
                continue
            self.assertTrue(bool(torch.equal(merged[name], value)),
                            f'{name} changed in the transplant')

    def _synthetic_embedding(self):
        p = common.P
        a = torch.arange(p, dtype=torch.float64)
        embedding = torch.zeros(p, 4, dtype=torch.float64)
        embedding[:, 0] = 3.0 * torch.cos(2 * math.pi * 3 * a / p)
        embedding[:, 1] = 1.5 * torch.sin(2 * math.pi * 7 * a / p)
        embedding[:, 2] = 0.5 * torch.cos(2 * math.pi * 11 * a / p)
        return embedding

    def test_knockout_on_known_frequencies(self):
        import fourier_metrics as fourier

        embedding = self._synthetic_embedding()
        energies = fourier.frequency_energies(embedding)
        top_k, share, total = fourier.top_frequencies(embedding, 0.90)
        self.assertEqual(top_k[0], 3)
        self.assertGreater(float(energies[2]), float(energies[6]))
        self.assertGreater(float(energies[6]), float(energies[10]))
        self.assertEqual(top_k, [3, 7], 'the 90% rule picked the wrong frequencies')
        self.assertGreaterEqual(share, 0.90)
        self.assertLess(sum(float(energies[k - 1]) for k in top_k[:-1]) / total, 0.90)

        key, key_info = fourier.knockout(embedding, 'key', top_k=[3])
        after = fourier.frequency_energies(key)
        self.assertAlmostEqual(float(after[2]), 0.0, places=6)
        self.assertAlmostEqual(float(after[6]), float(energies[6]), places=4)
        self.assertAlmostEqual(float(after[10]), float(energies[10]), places=4)

        nonkey, _ = fourier.knockout(embedding, 'nonkey', top_k=[3])
        after = fourier.frequency_energies(nonkey)
        self.assertAlmostEqual(float(after[2]), float(energies[2]), places=4)
        self.assertAlmostEqual(float(after[6]), 0.0, places=6)
        self.assertAlmostEqual(float(after[10]), 0.0, places=6)

        scale, _ = fourier.knockout(embedding, 'scale', top_k=[3])
        key_energy = fourier.non_dc_energy(key)
        scale_energy = fourier.non_dc_energy(scale)
        self.assertAlmostEqual(scale_energy / key_energy, 1.0, places=6)

    def test_random_dimension_knockout(self):
        import fourier_metrics as fourier

        embedding = self._synthetic_embedding()
        top_k, _, _ = fourier.top_frequencies(embedding, 0.90)
        dims = 2 * len(top_k)

        results = [fourier.knockout(embedding, 'rand_dim', top_k=top_k, seed=seed)[0]
                   for seed in (0, 1, 2)]
        self.assertFalse(bool(torch.equal(results[0], results[1])))
        self.assertFalse(bool(torch.equal(results[1], results[2])))

        for result in results:
            self.assertTrue(bool(torch.allclose(result.mean(dim=0), embedding.mean(dim=0),
                                                atol=1e-10)),
                            'the constant direction was not left untouched')

        # A signal that lies entirely inside a known random 2K-dimensional
        # subspace must be removed completely by the matching draw.
        basis = fourier.random_orthogonal_basis(common.P, dims, 0)
        inside = basis @ torch.randn(dims, 3, generator=torch.Generator().manual_seed(5),
                                     dtype=torch.float64)
        signal = torch.zeros(common.P, 3, dtype=torch.float64)
        signal += inside
        knocked, info = fourier.knockout(signal, 'rand_dim', top_k=top_k, seed=0)
        self.assertLess(fourier.non_dc_energy(knocked) / fourier.non_dc_energy(signal),
                        1e-12, 'the random subspace was not removed exactly')
        self.assertEqual(info['K'], len(top_k))

    def test_step0_source_evaluation_matches_the_stored_accuracy(self):
        import run_localize

        common.setup(1, 'cpu')
        device = 'cpu'
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        source_data = common.build_task('source', device)

        temp = Path(tempfile.mkdtemp(prefix='nseed_step0_'))
        try:
            model = common.init_model(5, device)
            stored = common.evaluate(model, source_data, strata)
            row = common.make_row(5, 'source', 'source', 'scratch', 10, stored,
                                  stage_clean_acc=stored['dev_mean'])
            common.save_model_checkpoint(temp / 'grokked.pt', model, 10, row)
            payload = common.load_checkpoint(temp / 'grokked.pt')
            loaded = common.make_model(common.CONFIG)
            loaded.load_state_dict(payload['model'])
            loaded = loaded.to(device)
            measured = run_localize.source_evaluation(loaded, source_data, strata)
            self.assertAlmostEqual(measured['source_dev_mean'], stored['dev_mean'], places=12)
            self.assertAlmostEqual(measured['source_train_acc'], stored['train_acc'], places=12)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        stored_path = common.PARENT / 'runs_nseed' / 'seed_5' / 'summary.json'
        if stored_path.exists():
            summary = common.read_summary(stored_path)
            stage = (summary.get('stages') or {}).get('grokked')
            ckpt = common.PARENT / 'runs_nseed' / 'seed_5' / 'ckpts' / 'grokked.pt'
            if stage and ckpt.exists():
                device = 'cuda' if common.cuda_available() else 'cpu'
                common.setup(1, device)
                source_data = common.build_task('source', device)
                payload = common.load_checkpoint(ckpt)
                model = common.make_model(common.CONFIG)
                model.load_state_dict(payload['model'])
                model = model.to(device)
                measured = run_localize.source_evaluation(model, source_data, strata)
                self.assertAlmostEqual(measured['source_dev_mean'],
                                       common.as_float(stage['dev_mean']), places=9)

    def test_freeze_keeps_frozen_entries_bit_identical(self):
        from run_freeze import freeze_plan, frozen_restorer

        common.setup(1, 'cpu')
        device = 'cpu'
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        data = common.build_task('source', device)
        rows = common.index_for(strata['train'], device)

        for variant in ('train_all', 'freeze_numeric_embed', 'embed_only',
                        'mlp_unembed_only'):
            model = common.init_model(5, device)
            source = {name: parameter.detach().clone()
                      for name, parameter in model.named_parameters()}
            optimizer_params, frozen = freeze_plan(model, variant)
            restore = frozen_restorer(model, frozen)
            optimizer = common.make_optimizer(model, optimizer_params)
            for _ in range(5):
                common.train_step(model, optimizer, data, rows)
                restore()

            def unchanged(name):
                return bool(torch.equal(model.state_dict()[name].detach(), source[name]))

            if variant in ('freeze_numeric_embed', 'mlp_unembed_only'):
                self.assertTrue(bool(torch.equal(
                    model.state_dict()['token_embed.weight'].detach()[:common.P],
                    source['token_embed.weight'][:common.P])),
                    f'{variant}: the frozen numeric embedding rows changed')
                self.assertFalse(unchanged('token_embed.weight'),
                                 f'{variant}: the trainable context rows did not change')
            if variant == 'embed_only':
                self.assertFalse(unchanged('token_embed.weight'),
                                 'embed_only: the embedding did not change')
                for name in source:
                    if name != 'token_embed.weight':
                        self.assertTrue(unchanged(name),
                                        f'embed_only: frozen tensor {name} changed')
            if variant == 'mlp_unembed_only':
                for name in source:
                    if not name.startswith(('mlp_in.', 'mlp_out.', 'unembed.',
                                            'token_embed.')):
                        self.assertTrue(unchanged(name),
                                        f'mlp_unembed_only: frozen tensor {name} changed')
            changed = [name for name, parameter in model.named_parameters()
                       if not bool(torch.equal(parameter.detach(), source[name]))]
            self.assertTrue(changed, f'{variant}: no trainable tensor changed')


def fake_race_executor(job, config, log):
    """Fake job that appends a row to the seed's metrics CSV."""
    path = Path(config['runs_dir']) / 'metrics.csv'
    rows = common.read_metrics(path)
    time.sleep(0.2)
    evaluation = {'train_acc': 0.5, 'dev_acc_c0': 0.1, 'dev_acc_c1': 0.1,
                  'dev_acc_c2': 0.1, 'dev_acc_c3': 0.1, 'dev_mean': 0.1, 'dev_worst': 0.1}
    row = common.make_row(job['seed'], 'transfer', 'affine', job['init'], 100, evaluation)
    rows = common.merge_rows(rows, [row])
    common.write_metrics(path, rows)
    return dict(ok=True)


class SameSeedRaceTest(unittest.TestCase):
    """Regression test for the Batch 1 bug: two jobs of one seed must not lose rows."""

    def test_two_jobs_of_the_same_seed_lose_no_rows(self):
        from run_batch import JobScheduler

        temp = Path(tempfile.mkdtemp(prefix='nseed_race_'))
        try:
            config = dict(runs_dir=str(temp), threads=1, tiers=['A'], resume=False,
                          source_cap=1, eval_every=1, ckpt_every=1, resumable_every=1,
                          transfer_steps=1, transfer_eval_every=1, late_offset=1,
                          max_train_pairs=4, stop=False, device='cpu')
            scheduler = JobScheduler(2, config, executor=fake_race_executor)
            scheduler.submit({'id': 'transfer:5:affine:grokked', 'kind': 'transfer',
                              'seed': 5, 'task': 'affine', 'init': 'grokked'})
            scheduler.submit({'id': 'transfer:5:affine:memorized', 'kind': 'transfer',
                              'seed': 5, 'task': 'affine', 'init': 'memorized'})
            scheduler.run()
            rows = common.read_metrics(temp / 'metrics.csv')
            inits = sorted(row['init'] for row in rows)
            self.assertEqual(inits, ['grokked', 'memorized'],
                             f'rows were lost: {inits}')
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class ComponentTest(unittest.TestCase):
    """The component study: hybrids, step-0 evaluation, new labels, status.py."""

    def _states(self):
        torch.manual_seed(21)
        recipient = common.make_model(common.CONFIG).state_dict()
        torch.manual_seed(22)
        donor = common.make_model(common.CONFIG).state_dict()
        return {key: value.detach().clone() for key, value in recipient.items()}, \
               {key: value.detach().clone() for key, value in donor.items()}

    def test_hybrid_copies_only_the_named_components(self):
        recipient, donor = self._states()
        for name in common.COMPONENTS:
            merged = common.build_hybrid(recipient, donor, [name])
            entries = common.component_entries(recipient, name)
            touched = set()
            for key, rows in entries:
                touched.add(key)
                if rows is None:
                    self.assertTrue(bool(torch.equal(merged[key], donor[key])),
                                    f'{name}: {key} is not the donor\'s')
                else:
                    self.assertTrue(bool(torch.equal(merged[key][rows], donor[key][rows])),
                                    f'{name}: {key}[{rows}] is not the donor\'s')
            for key, value in recipient.items():
                if key in touched:
                    continue
                self.assertTrue(bool(torch.equal(merged[key], value)),
                                f'{name}: {key} should have stayed with the recipient')

    def test_numeric_and_other_embed_share_a_tensor(self):
        recipient, donor = self._states()
        numeric = common.build_hybrid(recipient, donor, ['numeric_embed'])
        other = common.build_hybrid(recipient, donor, ['other_embed'])
        embed = recipient['token_embed.weight']
        self.assertTrue(bool(torch.equal(numeric['token_embed.weight'][:common.P],
                                         donor['token_embed.weight'][:common.P])))
        self.assertTrue(bool(torch.equal(numeric['token_embed.weight'][common.P:],
                                         embed[common.P:])))
        self.assertTrue(bool(torch.equal(other['token_embed.weight'][common.P:],
                                         donor['token_embed.weight'][common.P:])))
        self.assertTrue(bool(torch.equal(other['token_embed.weight'][:common.P],
                                         embed[:common.P])))
        self.assertTrue(bool(torch.equal(other['pos_embed'], donor['pos_embed'])))
        self.assertTrue(bool(torch.equal(numeric['pos_embed'], recipient['pos_embed'])))

    def test_hybrid_copying_nothing_reproduces_the_stored_accuracy(self):
        import run_localize

        common.setup(1, 'cpu')
        device = 'cpu'
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        source_data = common.build_task('source', device)
        temp = Path(tempfile.mkdtemp(prefix='nseed_hybrid_'))
        try:
            model = common.init_model(5, device)
            stored = common.evaluate(model, source_data, strata)
            common.save_model_checkpoint(temp / 'grokked.pt', model, 10)
            payload = common.load_checkpoint(temp / 'grokked.pt')
            merged = common.build_hybrid(payload['model'], payload['model'], [])
            hybrid = common.make_model(common.CONFIG)
            hybrid.load_state_dict(merged)
            hybrid = hybrid.to(device)
            measured = run_localize.source_evaluation(hybrid, source_data, strata)
            self.assertAlmostEqual(measured['source_dev_mean'], stored['dev_mean'], places=12)
            self.assertAlmostEqual(measured['source_train_acc'], stored['train_acc'], places=12)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        stored_path = common.PARENT / 'runs_nseed' / 'seed_5' / 'summary.json'
        if stored_path.exists():
            summary = common.read_summary(stored_path)
            stage = (summary.get('stages') or {}).get('grokked')
            ckpt = common.PARENT / 'runs_nseed' / 'seed_5' / 'ckpts' / 'grokked.pt'
            if stage and ckpt.exists():
                device = 'cuda' if common.cuda_available() else 'cpu'
                common.setup(1, device)
                source_data = common.build_task('source', device)
                payload = common.load_checkpoint(ckpt)
                merged = common.build_hybrid(payload['model'], payload['model'], [])
                hybrid = common.make_model(common.CONFIG)
                hybrid.load_state_dict(merged)
                hybrid = hybrid.to(device)
                measured = run_localize.source_evaluation(hybrid, source_data, strata)
                self.assertAlmostEqual(measured['source_dev_mean'],
                                       common.as_float(stage['dev_mean']), places=9)

    def test_copy_a_and_selector_labels(self):
        p = common.P
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        for context in range(common.N_CONTEXTS):
            expected_copy = a % p
            expected_selector = torch.where(
                torch.as_tensor([context in (0, 2)] * (p * p)), a % p, b % p)
            copy_data = common.build_task('copy_a')
            selector_data = common.build_task('selector')
            rows = common.context_rows(torch.arange(p * p), context)
            self.assertTrue(bool((copy_data['y'][rows] == expected_copy).all()),
                            f'copy_a context {context} labels differ')
            self.assertTrue(bool((selector_data['y'][rows] == expected_selector).all()),
                            f'selector context {context} labels differ')
        selector_data = common.build_task('selector')
        rows = common.context_rows(torch.arange(p * p), 1)
        self.assertTrue(bool((selector_data['y'][rows] == (b % p)).all()))

    def test_neuron_selectivity_on_synthetic_activations(self):
        import body_metrics

        p = common.P
        repeats = 32
        generator = torch.Generator().manual_seed(0)
        key_sum = torch.arange(p).repeat(repeats)
        a_values = key_sum.clone()
        for value in range(p):
            index = (key_sum == value).nonzero().flatten()
            a_values[index] = torch.randperm(p, generator=generator)[:repeats]
        activations = torch.zeros(p * repeats, 3, dtype=torch.float64)
        activations[:, 0] = torch.cos(2 * math.pi * 5 * key_sum / p)
        activations[:, 1] = torch.randn(p * repeats, generator=generator, dtype=torch.float64)
        activations[:, 2] = torch.randn(p * repeats, generator=generator, dtype=torch.float64)

        r2, means, total = body_metrics.selectivity_r2(activations, key_sum, p)
        self.assertAlmostEqual(float(r2[0]), 1.0, places=9)
        frequency, share = body_metrics.dominant_frequency(means)
        self.assertEqual(int(frequency[0]), 5)
        # a real cosine splits its energy with its conjugate, so the strongest
        # frequency holds about half of the total
        self.assertGreater(float(share[0]), 0.4)
        self.assertLess(float(share[0]), 0.6)
        r2_a, _, _ = body_metrics.selectivity_r2(activations, a_values, p)
        self.assertLess(float(r2_a[0]), 0.1)
        self.assertLess(float(r2[1]), 0.2)

    def test_status_script_writes_nothing(self):
        import subprocess

        runs_dir = common.PARENT / 'runs_nseed'
        before = {path: (path.stat().st_mtime, path.stat().st_size)
                  for path in runs_dir.rglob('*') if path.is_file()}
        result = subprocess.run([sys.executable, str(HERE / 'status.py'),
                                 '--log', str(runs_dir / 'batch.log')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PROMPT STEP', result.stdout)
        after = {path: (path.stat().st_mtime, path.stat().st_size)
                 for path in runs_dir.rglob('*') if path.is_file()}
        self.assertEqual(before, after, 'status.py modified a file in the run folder')


class FactorialTest(unittest.TestCase):
    """The factorial hybrids and the two embedding controls."""

    def _states(self):
        torch.manual_seed(31)
        memorized = common.make_model(common.CONFIG).state_dict()
        torch.manual_seed(32)
        grokked = common.make_model(common.CONFIG).state_dict()
        torch.manual_seed(33)
        donor = common.make_model(common.CONFIG).state_dict()
        return ({k: v.detach().clone() for k, v in memorized.items()},
                {k: v.detach().clone() for k, v in grokked.items()},
                {k: v.detach().clone() for k, v in donor.items()})

    def test_factorial_hybrids_copy_only_the_subset(self):
        memorized, grokked, _ = self._states()
        subsets = [(), ('numeric_embed',), ('attn', 'mlp'),
                   ('numeric_embed', 'attn', 'mlp', 'unembed')]
        for subset in subsets:
            merged = common.build_hybrid(memorized, grokked, list(subset))
            copied = set()
            for name in subset:
                for key, rows in common.component_entries(memorized, name):
                    copied.add(key)
                    if rows is None:
                        self.assertTrue(bool(torch.equal(merged[key], grokked[key])),
                                        f'{subset}: {key} is not the grokked tensor')
                    else:
                        self.assertTrue(bool(torch.equal(merged[key][rows], grokked[key][rows])),
                                        f'{subset}: {key}[{rows}] is not the grokked slice')
            for key, value in memorized.items():
                if key in copied:
                    continue
                self.assertTrue(bool(torch.equal(merged[key], value)),
                                f'{subset}: {key} should have stayed with the memorized model')

    def test_embed_random_matched_has_the_grokked_norm(self):
        import run_factorial

        memorized, grokked, donor = self._states()
        model = run_factorial.build_job_model({}, 'embed_random_matched', (),
                                               memorized, grokked, donor, 'cpu')
        target = float(grokked['token_embed.weight'][:common.P].double().norm())
        actual = float(model.token_embed.weight[:common.P].double().norm())
        self.assertAlmostEqual(actual / target, 1.0, places=6)
        self.assertTrue(bool(torch.equal(
            model.token_embed.weight[common.P:], grokked['token_embed.weight'][common.P:])))

    def test_embed_rowperm_keeps_the_rows(self):
        import run_factorial

        memorized, grokked, donor = self._states()
        model = run_factorial.build_job_model({}, 'embed_rowperm', (),
                                               memorized, grokked, donor, 'cpu')
        original = grokked['token_embed.weight'][:common.P]
        permuted = model.token_embed.weight[:common.P]
        self.assertFalse(bool(torch.equal(original, permuted)))
        original_rows = sorted(tuple(row.tolist()) for row in original)
        permuted_rows = sorted(tuple(row.tolist()) for row in permuted)
        self.assertEqual(original_rows, permuted_rows,
                         'the permuted embedding does not hold the same rows')

    def test_neuron_freq_on_synthetic_neurons(self):
        import neuron_freq

        p = common.P
        a = torch.arange(p).float().view(-1, 1)
        b = torch.arange(p).float().view(1, -1)
        for b_frequency, expected_class in ((6, 2), (p - 6, 2)):
            grid = torch.relu(torch.cos(2 * math.pi * 3 * a / p)
                              + torch.cos(2 * math.pi * b_frequency * b / p))
            activations = grid.reshape(p * p, 1)
            data = neuron_freq.neuron_spectrum(activations)
            self.assertEqual(int(data['fa'][0]), 3)
            self.assertEqual(int(data['fb'][0]), 6)
            self.assertEqual(int(data['ratio_class'][0]), expected_class)
            self.assertTrue(bool(data['clean'][0]))
            self.assertGreaterEqual(float(data['conc_a'][0]), 0.5)
            self.assertGreaterEqual(float(data['conc_b'][0]), 0.5)


class Batch4bTest(unittest.TestCase):
    """The Batch 4b job types: clean scratch, age-matched sources, attention stats."""

    def test_scratch_clean_worker_writes_its_output(self):
        """The scratch-clean worker must survive the pool and write its files."""
        import run_scratch_clean

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_clean_worker_'))
        previous_log = common.BATCH_LOG
        try:
            code = run_scratch_clean.main([
                '--seeds', '9101', '--tasks', 'affine', '--procs', '1',
                '--steps', '20', '--eval-every', '10', '--save-every', '10',
                '--out', str(temp / 'scratch_clean'), '--device', 'cpu',
                '--batch-log', str(temp / 'batch.log')])
            self.assertEqual(code, 0)
            out = temp / 'scratch_clean' / 'scratch_9101' / 'affine'
            self.assertTrue((out / 'summary.json').exists())
            self.assertTrue((out / 'ckpts' / 'affine_final.pt').exists())
            summary = common.read_summary(out / 'summary.json')
            self.assertTrue(summary['transfers']['affine|scratch']['done'])
        finally:
            common.set_batch_log(previous_log)
            shutil.rmtree(temp, ignore_errors=True)

    def test_scratch_clean_runner_exits_nonzero_when_the_worker_fails(self):
        """A worker that cannot write its output must make the runner fail."""
        import run_scratch_clean

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_clean_fail_'))
        previous_log = common.BATCH_LOG
        try:
            blocker = temp / 'blocker'
            blocker.write_text('not a directory', encoding='utf-8')
            code = run_scratch_clean.main([
                '--seeds', '9102', '--tasks', 'affine', '--procs', '1',
                '--steps', '20', '--eval-every', '10', '--save-every', '10',
                '--out', str(blocker / 'scratch_clean'), '--device', 'cpu',
                '--batch-log', str(temp / 'batch.log')])
            self.assertEqual(code, 1)
        finally:
            common.set_batch_log(previous_log)
            shutil.rmtree(temp, ignore_errors=True)

    def test_parallel_reports_a_failed_worker(self):
        """A worker whose job raises must make the pool report a failure."""
        from parallel import run_parallel

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_parallel_'))
        previous_log = common.BATCH_LOG
        try:
            common.set_batch_log(None)
            config = dict(runs_dir=str(temp), device='cpu', threads=1, batch_log=None)
            common.planned_line(9103, ['job_a'])
            report = run_parallel(1, [9103], exploding_worker, config)
            self.assertFalse(report.ok)
            self.assertTrue(report.failed)
            self.assertIn((9103, 'job_a'), report.missing)
        finally:
            common.set_batch_log(previous_log)
            shutil.rmtree(temp, ignore_errors=True)

    def test_parallel_reports_a_job_that_never_finished(self):
        """A planned job that never reached a terminal state must be reported."""
        from parallel import run_parallel

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_parallel_'))
        previous_log = common.BATCH_LOG
        try:
            common.set_batch_log(None)
            config = dict(runs_dir=str(temp), device='cpu', threads=1, batch_log=None)
            common.planned_line(9104, ['job_a'])
            report = run_parallel(1, [9104], silent_worker, config)
            self.assertFalse(report.ok)
            self.assertFalse(report.failed)
            self.assertIn((9104, 'job_a'), report.missing)
        finally:
            common.set_batch_log(previous_log)
            shutil.rmtree(temp, ignore_errors=True)

    def test_attention_reconstruction_matches_the_module_weights(self):
        """The rebuilt query-key row must agree with the module's own weights."""
        import attention_stats

        common.setup(1, 'cpu')
        model = common.make_model(common.CONFIG)
        model.eval()
        difference = attention_stats.qk_self_check(
            model, attention_stats.context_tokens(0, 'cpu'))
        self.assertLess(difference, 1e-5,
                        f'the rebuilt attention row differs by {difference}')

    def test_entropy_is_the_mean_of_the_rows(self):
        import attention_stats

        rows = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
        expected = (math.log(2) + math.log(4)) / 2
        self.assertAlmostEqual(attention_stats.entropy(rows), expected, places=6)

    def test_attention_rows_are_finite_and_complete(self):
        import attention_stats

        common.setup(1, 'cpu')
        model = common.make_model(common.CONFIG)
        model.eval()
        data = common.build_task('source', 'cpu')
        mask = common.mask_for_seed(5)
        strata, _ = common.split_for_mask(mask)
        rows = attention_stats.attention_rows(model, data, strata, 'cpu')
        self.assertEqual(len(rows), N_CONTEXTS * model.attn.num_heads)
        for row in rows:
            self.assertTrue(math.isfinite(row['entropy']))
            self.assertGreaterEqual(row['entropy'], 0.0)
            self.assertLessEqual(row['entropy'], math.log(4) + 1e-6)
            self.assertLessEqual(row['weight_to_a'] + row['weight_to_b'], 1.0 + 1e-6)

    def test_skipped_job_line_is_accepted(self):
        """A job that is skipped on resume must still produce a job line."""
        previous = common.BATCH_LOG
        try:
            common.set_batch_log(None)
            common.skipped_job_line(9106, 'fact_empty', 2000, {'final_dev_mean': None})
            common.skipped_job_line(9106, 'fact_attn', 2000, {'final_dev_mean': 0.5})
            self.assertIn((9106, 'fact_empty'), common.DONE_JOBS)
            self.assertIn((9106, 'fact_attn'), common.DONE_JOBS)
        finally:
            common.set_batch_log(previous)

    def test_factorial_cell_filter_reaches_the_worker(self):
        import run_factorial

        config = run_factorial.worker_config(
            {'device': 'cpu'}, '/tmp/out', 10, 10, False, ['factorial'], ['empty', 'attn'])
        self.assertEqual(config['cells'], {'empty', 'attn'})
        names = [name for name, _, _ in
                 run_factorial.job_plan(tuple(config['families']), config['cells'])]
        self.assertEqual(sorted(names), ['fact_attn', 'fact_empty'])

    def test_analysis_reads_the_clean_scratch_folders(self):
        """The analysis must find the summary of every task of every seed."""
        import analyze

        temp = Path(tempfile.mkdtemp(prefix='nseed_analyze_'))
        try:
            for seed in (9101, 9102):
                for task in ('affine', 'input_perm'):
                    folder = temp / 'scratch_clean' / f'scratch_{seed}' / task
                    folder.mkdir(parents=True, exist_ok=True)
                    common.write_summary(folder / 'summary.json', dict(
                        seed=seed, init_seed=seed, task=task, mask_seed=3,
                        transfers={f'{task}|scratch': dict(
                            task=task, done=True, steps=50000,
                            final_dev_mean=0.5, final_dev_worst=0.4,
                            final_train_acc=1.0, best_dev_mean=0.6,
                            steps_to_30=1234, global_norm_init=42.0)}))
            rows = analyze.clean_scratch_rows(temp)
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['task'] for r in rows}, {'affine', 'input_perm'})
            for row in rows:
                self.assertEqual(row['run_kind'], 'scratch_clean')
                self.assertEqual(row['mask_seed'], 3)
                self.assertEqual(row['final_dev_mean'], 0.5)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_scratch_clean_resume_gives_identical_tensors(self):
        import run_scratch_clean

        previous = os.environ.get(common.DETERMINISTIC_ENV)
        os.environ[common.DETERMINISTIC_ENV] = '1'
        temp = Path(tempfile.mkdtemp(prefix='nseed_clean_'))
        try:
            common.setup(1, 'cpu')
            straight = temp / 'straight'
            run_scratch_clean.run_scratch_clean(
                9001, 'affine', straight, steps=40, eval_every=20, save_every=20,
                device='cpu', log=lambda *a: None)
            resumed = temp / 'resumed'
            run_scratch_clean.run_scratch_clean(
                9001, 'affine', resumed, steps=20, eval_every=20, save_every=20,
                device='cpu', log=lambda *a: None)
            run_scratch_clean.run_scratch_clean(
                9001, 'affine', resumed, steps=40, eval_every=20, save_every=20,
                device='cpu', resume=True, log=lambda *a: None)

            left = common.load_checkpoint(straight / 'ckpts' / 'affine_final.pt')['model']
            right = common.load_checkpoint(resumed / 'ckpts' / 'affine_final.pt')['model']
            self.assertEqual(sorted(left), sorted(right))
            for name, value in left.items():
                self.assertTrue(bool(torch.equal(value, right[name])),
                                f'the resumed tensor {name} differs')
        finally:
            shutil.rmtree(temp, ignore_errors=True)
            if previous is None:
                os.environ.pop(common.DETERMINISTIC_ENV, None)
            else:
                os.environ[common.DETERMINISTIC_ENV] = previous

    def test_source_snapshot_is_written_at_the_requested_step(self):
        from run_seed import SeedRun

        common.setup(1, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_snapshot_'))
        try:
            mask = common.mask_for_seed(5)
            strata, _ = common.split_for_mask(mask)
            data = common.build_task('source', 'cpu')
            run = SeedRun(5, temp / 'seed_5', threads=1, resume=False, tiers=(),
                          device='cpu', source_cap=10, eval_every=5, ckpt_every=5,
                          resumable_every=5, log=lambda *a: None)
            run._ensure_dirs()
            run.train_source(strata, data, run._summary(),
                             snapshot_steps={7: 'age_matched.pt'})
            snapshot = temp / 'seed_5' / 'ckpts' / 'age_matched.pt'
            self.assertTrue(snapshot.exists(), 'the snapshot checkpoint was not written')
            payload = common.load_checkpoint(snapshot)
            self.assertEqual(int(payload['step']), 7)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_attention_entropy_and_spectral_concentration(self):
        import attention_stats

        uniform = torch.full((4,), 0.25)
        self.assertAlmostEqual(attention_stats.entropy(uniform), math.log(4), places=6)
        peaked = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(attention_stats.entropy(peaked), 0.0, places=6)

        p = common.P
        a = torch.arange(p, dtype=torch.float64).view(-1, 1)
        b = torch.arange(p, dtype=torch.float64).view(1, -1)
        grid = torch.cos(2 * math.pi * 3 * a / p) * torch.cos(2 * math.pi * 4 * b / p)
        share, conc_a, conc_b = attention_stats.spectral_concentration(grid)
        self.assertAlmostEqual(share, 1.0, places=9)
        self.assertAlmostEqual(conc_a, 1.0, places=9)
        self.assertAlmostEqual(conc_b, 1.0, places=9)

    def test_missing_reference_is_reported_as_na(self):
        import analyze

        self.assertEqual(analyze.fmt(None), 'n/a')
        self.assertEqual(analyze.fmt(0.5), '0.5000')
        rows = [
            dict(run_kind='factorial', seed=5, init='fact_attn', components='attn',
                 ratio_to_grokked='0.5', final_dev_mean='0.5'),
            # an obstacle job whose reference cell (the same subset, grokked) is
            # not in the table at all
            dict(run_kind='obstacle', seed=5, init='mem_reset_mlp', components='mlp',
                 ratio_to_grokked='0.9', final_dev_mean='0.9'),
        ]
        text = analyze.factorial_lines(rows)
        self.assertIn('n/a', text)
        self.assertIn('does not matter', text)


class SmokeTest(unittest.TestCase):
    """Test 7: a short run writes valid CSVs."""

    def test_short_source_run_writes_valid_csvs(self):
        from run_seed import SeedRun

        common.setup(4, 'cpu')
        temp = Path(tempfile.mkdtemp(prefix='nseed_smoke_'))
        try:
            run = SeedRun(4242, temp / 'seed_4242', threads=4, resume=False,
                          tiers=('A',), source_cap=300, eval_every=100, ckpt_every=150,
                          transfer_steps=20, transfer_eval_every=10,
                          max_train_pairs=139, log=lambda *a: None)
            summary = run.run()
            self.assertIn(summary['status'], ('grokked', 'nongrok'))
            metrics = common.read_metrics(temp / 'seed_4242' / 'metrics.csv')
            self.assertGreater(len(metrics), 0)
            for row in metrics:
                self.assertIn(row['stage'], ('source', 'transfer'))
                self.assertIn(row['task'], common.TASKS)
                self.assertTrue(0.0 <= float(row['train_acc']) <= 1.0)
                self.assertTrue(0.0 <= float(row['dev_mean']) <= 1.0)
            self.assertTrue((temp / 'seed_4242' / 'summary.json').exists())
            self.assertTrue((temp / 'seed_4242' / 'ckpts').is_dir())
        finally:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
