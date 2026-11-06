"""Check every available trained checkpoint on synthetic inputs, before MILK outcomes."""
import argparse
import datetime
import gc
import pathlib
import time
import traceback
import uuid
import numpy as np
import milk_common as C
import run_milk as E


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--available', action='store_true')
    g.add_argument('--require-complete', action='store_true')
    p.add_argument('--campaign', choices=('PAA60', 'PAA30'))
    args = p.parse_args()
    if args.require_complete and args.campaign:
        p.error('--campaign is only for --available; formal preflight always requires all 135')
    records, missing = C.inventory()
    records = [r for r in records if pathlib.Path(r['path']).is_file()]
    available_total = len(records)
    if args.campaign:
        records = [r for r in records if r['campaign'] == args.campaign]
        selected_arms = C.ARMS[3:7] if args.campaign == 'PAA60' else C.ARMS[7:]
        expected = {(f, a, seed) for f, a, seed in C.matrix() if a in selected_arms}
        actual = {(r['fold'], r['arm'], r['seed']) for r in records}
        if actual != expected or len(records) != len(expected):
            raise RuntimeError('The selected campaign must contain every frozen fold and seed')
    started = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report_path = C.ROOT / 'reports' / ('trained_preflight_' + stamp + '_' + uuid.uuid4().hex[:8] + '.json')
    report = dict(status='CHECKING', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  mode='require_complete' if args.require_complete else 'available_only',
                  available=available_total, selected=len(records), campaign=args.campaign, planned=135, missing=missing, checks=[],
                  clinical_images_used=False, clinical_labels_used=False,
                  synthetic_images_per_model=3, cpu_cuda_equivalence_tested=False,
                  report_path=str(report_path))

    def save():
        report['seconds'] = time.monotonic() - started
        C.save(report_path, report)
        C.save(C.ROOT / 'reports/trained_checkpoint_preflight.json', report)

    if args.require_complete and (missing or len(records) != 135):
        report['status'] = 'WAITING_FOR_COMPLETE_135_FROZEN_WEIGHTS'
        save()
        print('STOP: all 135 frozen checkpoints are required; no MILK images accessed.', flush=True)
        return 2
    if not records:
        raise RuntimeError('No available checkpoint to check')
    try:
        env = E.runtime()
        import torch
        from information_conditions import restrict_metadata
        qualification = C.read(C.ROOT / 'reports/cpu_pipeline_qualification.json')
        if (qualification['status'] != 'CPU_SYNTHETIC_RUNTIME_QUALIFIED'
                or qualification['environment'] != env
                or qualification['source_binding'] != E.source_binding()):
            raise RuntimeError('Exact runtime/source qualification is required')
        S = C.setup()
        S.verify_manifest(materials=True)
        encoding = C.read(C.ROOT / 'reports/input_encoding_audit.json')['folds']
        report['environment'] = env
        report['source_binding'] = E.source_binding()
        report['preflight_source_sha256'] = C.sha(__file__)
        report['encoding_audit_sha256'] = C.sha(C.ROOT / 'reports/input_encoding_audit.json')
        rng = np.random.default_rng(20261104)
        rgb = rng.integers(0, 256, (3, 224, 224, 3), dtype=np.uint8)
        x = E.normalize(rgb)
        full = np.array([[.25, 0, 1, 0, 0], [-.5, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=np.float32)
        save()
        for i, r in enumerate(records, 1):
            task = C.task_id(r['fold'], r['arm'], r['seed'])
            report['current_task'] = task
            if C.sha(r['path']) != r['checkpoint_sha256']:
                raise RuntimeError('Frozen weight hash mismatch: ' + task)
            # Own training artifacts only; the frozen campaign digest is checked above.
            ck = torch.load(r['path'], map_location='cpu', weights_only=False)
            if C.json_hash(ck['config']) != C.json_hash(r['config']):
                raise RuntimeError('Checkpoint/config disagreement: ' + task)
            if ck['config']['metadata_age_train_stats'] != encoding[r['fold']]['training_age_stats']:
                raise RuntimeError('Training age statistics disagreement: ' + task)
            state = ck['model_state_dict']
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError('Non-tensor state: ' + key)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite model state: ' + key)
            model = E.create_model(r['arm'])
            model.load_state_dict(state, strict=True)
            del ck, state
            meta = torch.from_numpy(restrict_metadata(full, C.CONDITIONS[r['arm']]))
            with torch.inference_mode():
                logits = model(x, meta)
                if logits.shape != (3, 6) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite or incorrect-shape synthetic logits: ' + task)
            report['checks'].append(dict(task=task, checkpoint_sha256=r['checkpoint_sha256'],
                                        config_sha256=C.json_hash(r['config']), strict_load=True,
                                        training_age_stats_match=True, finite_state=True,
                                        synthetic_forward_shape=[3, 6], finite_forward=True))
            save()
            print(f'PREFLIGHT PASS {i}/{len(records)} {task}', flush=True)
            del model, logits, meta
            gc.collect()
        report.pop('current_task', None)
        report['status'] = ('SELECTED_CAMPAIGN_TRAINED_CHECKPOINTS_PASS' if args.campaign
                            else 'ALL_135_TRAINED_CHECKPOINTS_PASS' if not missing and len(records) == 135
                            else 'AVAILABLE_TRAINED_CHECKPOINTS_PASS_WAITING_FOR_REMAINDER')
        save()
        print(report['status'], 'checked=', len(report['checks']), flush=True)
        return 0
    except BaseException:
        report['status'] = 'FAILED_BEFORE_MILK_OUTCOMES'
        report['traceback'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
