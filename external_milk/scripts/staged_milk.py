"""Staged MILK inference: freeze 105 now; release metrics only after all 135.
The numerical engine in run_milk.py is unchanged. A scoped output-path adapter
writes the first batch to a separate cache. Completion validates and copies the
same arrays into the full-run binding without recalculating those predictions.
"""
import argparse
import contextlib
import datetime
import gc
import json
import os
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import milk_common as C
import run_milk as E


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stage_dir():
    return C.ROOT / 'evaluation/staged_105'


def amendment_path():
    return C.ROOT / 'prepared/staged_execution_amendment_20261104.json'


def stage_paths(record):
    p = stage_dir() / 'predictions' / (C.task_id(record['fold'], record['arm'], record['seed']) + '.npz')
    return p, p.with_suffix('.receipt.json')


@contextlib.contextmanager
def staging_paths():
    original = E.prediction_paths
    E.prediction_paths = stage_paths
    try:
        yield
    finally:
        E.prediction_paths = original


def stub(record):
    return {k: record[k] for k in ['fold', 'arm', 'seed', 'checkpoint_sha256', 'config', 'campaign']}


def key(record):
    return record['fold'], record['arm'], record['seed']


def grid(records, initial=False):
    expected = {x for x in C.matrix() if not initial or x[1] in C.ARMS[:7]}
    if len(records) != len(expected) or {key(r) for r in records} != expected:
        raise ValueError('Complete fixed 105/135 fold-seed grid required; no arbitrary subset')


def initial_records():
    records, _ = C.inventory()
    selected = [r for r in records if r['arm'] in C.ARMS[:7]]
    grid(selected, initial=True)
    coverage = C.read(C.ROOT / 'reports/available_105_weight_verification.json')
    if coverage['status'] != 'AVAILABLE_105_FROZEN_MODELS_VERIFIED':
        raise ValueError('Complete 105-model preflight required')
    checked = {x['task']: x for x in coverage['checks']}
    for r in selected:
        proof = checked[C.task_id(*key(r))]
        if proof['checkpoint_sha256'] != r['checkpoint_sha256'] or proof['config_sha256'] != C.json_hash(r['config']):
            raise ValueError('Preflight no longer matches frozen input')
    return selected


def check_amendment():
    a = C.read(amendment_path())
    expected = {
        'original_cohort_lock_sha256': C.sha(C.ROOT / 'prepared/cohort_lock.json'),
        'original_analysis_plan_sha256': C.sha(C.ROOT / 'prepared/pre_outcome_plan.json'),
        'pending_PAA30_plan_sha256': C.sha(C.PAA30 / 'analysis_plan.json'),
        'pending_PAA30_source_manifest_sha256': C.sha(C.PAA30 / 'source_manifest.json'),
        'adapter_sha256': C.sha(__file__),
    }
    if any(a[k] != value for k, value in expected.items()) or a['engine_source_binding'] != E.source_binding():
        raise ValueError('Amendment input/code identity changed; preserve existing results')
    if not a['analysis_release_requires_all_135_frozen'] or not a['no_performance_review_before_all_135_frozen']:
        raise ValueError('Required analysis embargo missing')
    return a


def qualified_runtime():
    env = E.runtime()
    q = C.read(C.ROOT / 'reports/cpu_pipeline_qualification.json')
    if q['status'] != 'CPU_SYNTHETIC_RUNTIME_QUALIFIED' or q['environment'] != env or q['source_binding'] != E.source_binding():
        raise ValueError('Original qualified numerical engine/runtime must be unchanged')
    C.setup().verify_manifest(materials=True)
    return env


def prepare():
    if amendment_path().exists():
        check_amendment()
        print('Existing pre-outcome staging amendment verified.', flush=True)
        return
    if (C.ROOT / 'evaluation/run_lock.json').exists() or any((C.ROOT / 'evaluation').glob('**/*.npz')):
        raise ValueError('Cannot create a pre-outcome amendment after prediction files exist')
    records = initial_records()
    test = C.read(C.ROOT / 'reports/staged_pipeline_tests.json')
    if test['status'] != 'PASS' or test['adapter_sha256'] != C.sha(__file__):
        raise ValueError('Exact staged adapter must pass its synthetic tests first')
    a = dict(status='PRE_OUTCOME_EXECUTION_TIMING_AMENDMENT', created_utc=now(),
        reason='User requested overlapping local inference on 105 already frozen models with the separately running, prespecified PAA30 training campaign.',
        initial_models=105, pending_models=30, planned_total=135,
        original_requirement='All 135 models frozen before any MILK inference.',
        revised_requirement='Compute and cache only the fixed first 105 models now; no model-performance inspection, metrics or bootstrap until PAA30 is fully frozen. Then compute the remaining 30 and analyze all 135 together.',
        no_MILK_predictions_computed_or_viewed_at_amendment=True,
        analysis_release_requires_all_135_frozen=True,
        no_performance_review_before_all_135_frozen=True,
        no_outcomes_sent_to_training_worker=True,
        no_changes_to_training_cohort_preprocessing_model_selection_or_statistics=True,
        not_public_preregistration=True,
        reporting='Disclose the staged chronology. Do not claim that all 135 training runs ended before the first MILK prediction.',
        original_cohort_lock_sha256=C.sha(C.ROOT/'prepared/cohort_lock.json'),
        original_analysis_plan_sha256=C.sha(C.ROOT/'prepared/pre_outcome_plan.json'),
        pending_PAA30_plan_sha256=C.sha(C.PAA30/'analysis_plan.json'),
        pending_PAA30_source_manifest_sha256=C.sha(C.PAA30/'source_manifest.json'),
        adapter_sha256=C.sha(__file__), engine_source_binding=E.source_binding(),
        synthetic_adapter_test_sha256=C.sha(C.ROOT/'reports/staged_pipeline_tests.json'),
        prior_105_preflight_sha256=C.sha(C.ROOT/'reports/available_105_weight_verification.json'),
        initial_records=[stub(r) for r in records])
    C.save(amendment_path(), a)
    print('Timing amendment saved before any clinical prediction. Original locks preserved.', flush=True)


@contextlib.contextmanager
def process_guard(mode):
    folder = C.ROOT / 'evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / 'RUNNING.json'
    with p.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), mode=mode, started_utc=now()), f)
    try:
        yield
    except BaseException:
        C.save(folder / ('failure_' + uuid.uuid4().hex + '.json'), dict(mode=mode, utc=now(), traceback=traceback.format_exc(), outputs_retained=True))
        raise
    finally:
        p.unlink()


def case_arrays(cases):
    return (np.array([c['isic_id'] for c in cases]), np.array([int(c['label']) for c in cases]), np.array([c['lesion_id'] for c in cases]))


def placeholder(digest):
    return dict(status='STAGED_PREDICTIONS_ONLY_ANALYSIS_LOCKED', staged_run_sha256=digest,
                all135_frozen_before_evaluation=False, analysis_release_requires_all_135_frozen=True)


def run_initial():
    a = check_amendment()
    records = initial_records()
    if a['initial_records'] != [stub(r) for r in records]:
        raise ValueError('The initial frozen 105-model registry changed')
    env = qualified_runtime()
    cases, indices, _ = C.check_cohort()
    ids, labels, lesions = case_arrays(cases)
    run = dict(cohort_lock_sha256=C.sha(C.ROOT/'prepared/cohort_lock.json'), environment=env,
        source_binding=E.source_binding(), adapter_sha256=C.sha(__file__), amendment_sha256=C.sha(amendment_path()),
        records=[stub(r) for r in records], first_batch=105, planned_total=135,
        all135_frozen_before_evaluation=False, performance_analysis_permitted=False)
    digest = C.json_hash(run)
    with process_guard('STAGED_105_INFERENCE_NO_METRICS'):
        run_path = stage_dir() / 'run_lock.json'
        if run_path.exists() and C.read(run_path) != run:
            raise ValueError('Staged run identity changed')
        C.save(run_path, run)
        main_lock = C.ROOT/'evaluation/run_lock.json'
        sentinel = placeholder(digest)
        if main_lock.exists() and C.read(main_lock) != sentinel:
            raise ValueError('Another full-run identity already exists')
        C.save(main_lock, sentinel)  # Prevent the old full-run path from ignoring this chronology.
        outputs = []
        with np.load(C.ROOT/'prepared/candidate_metadata_inputs.npz', allow_pickle=False) as metadata, staging_paths():
            if not np.array_equal(metadata['isic_id'][indices], ids):
                raise ValueError('Metadata/image order mismatch')
            for r in records:
                C.save(stage_dir()/'progress.json', dict(status='RUNNING_PREDICTIONS_NO_METRICS', pid=os.getpid(), updated_utc=now(), completed=len(outputs), total=105, current_task=C.task_id(*key(r))))
                receipt = E.predict(r, cases, indices, digest, metadata)
                outputs.append(receipt)
                print('STAGED MODEL SAVED', len(outputs), '/ 105', C.task_id(*key(r)), flush=True)
        C.save(stage_dir()/'stage_complete.json', dict(status='ALL_105_STAGED_PREDICTIONS_VERIFIED_NO_METRICS', completed_utc=now(), run_sha256=digest, files=outputs, metrics_computed=False))
        C.save(stage_dir()/'progress.json', dict(status='105_COMPLETE_WAITING_FOR_PAA30_FREEZE_NO_METRICS', completed=105, total=105, updated_utc=now()))
        print('All 105 predictions saved. No performance metrics computed. Wait for all 30 remaining models to freeze.', flush=True)


def adopt_prediction(r, ids, labels, lesions, old_digest, new_digest):
    old_binding = C.json_hash(dict(run=old_digest, checkpoint=r['checkpoint_sha256'], config=C.json_hash(r['config'])))
    new_binding = C.json_hash(dict(run=new_digest, checkpoint=r['checkpoint_sha256'], config=C.json_hash(r['config'])))
    with staging_paths():
        old_receipt = E.existing_prediction(r, ids, labels, lesions, old_binding)
    if old_receipt is None:
        raise ValueError('Expected staged output missing')
    cached = E.existing_prediction(r, ids, labels, lesions, new_binding)
    if cached is not None:
        if cached.get('reused_staged_prediction_sha256') != old_receipt['prediction_sha256']:
            raise ValueError('Full-run output has different provenance')
        return cached
    source, _ = stage_paths(r)
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    # Only the provenance binding changes; logits/probabilities are copied exactly.
    arrays['binding_sha256'] = np.array(new_binding)
    E.validate_arrays(arrays, ids, labels, lesions, new_binding)
    target, receipt_path = E.prediction_paths(r)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.partial_' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        np.savez_compressed(f, **arrays)
    if target.exists():
        raise FileExistsError('Never overwrite previous predictions')
    temp.rename(target)
    result = dict(old_receipt, path=str(target.relative_to(C.ROOT)), binding_sha256=new_binding,
        prediction_sha256=C.sha(target), reused_staged_prediction_sha256=old_receipt['prediction_sha256'],
        reused_staged_path=str(source.relative_to(C.ROOT)), numerical_arrays_unchanged=True, adopted_utc=now())
    C.save(receipt_path, result)
    return result


def complete():
    records, missing = C.inventory()
    if missing:
        raise ValueError('All 135 frozen models required before completion or analysis')
    grid(records)
    a = check_amendment()
    env = qualified_runtime()
    preflight = C.read(C.ROOT/'reports/trained_checkpoint_preflight.json')
    if preflight['status'] != 'ALL_135_TRAINED_CHECKPOINTS_PASS' or preflight['environment'] != env or preflight['source_binding'] != E.source_binding():
        raise ValueError('Complete 135-model trained preflight required')
    checked = {x['task']: x for x in preflight['checks']}
    if len(checked) != 135:
        raise ValueError('Incomplete preflight')
    for r in records:
        p = checked[C.task_id(*key(r))]
        if p['checkpoint_sha256'] != r['checkpoint_sha256'] or p['config_sha256'] != C.json_hash(r['config']):
            raise ValueError('Preflight/model mismatch')
    old_run = C.read(stage_dir()/'run_lock.json')
    old_digest = C.json_hash(old_run)
    staged = C.read(stage_dir()/'stage_complete.json')
    if staged['status'] != 'ALL_105_STAGED_PREDICTIONS_VERIFIED_NO_METRICS' or staged['run_sha256'] != old_digest or len(staged['files']) != 105:
        raise ValueError('All initial 105 cached outputs must finish first')
    initial = [r for r in records if r['arm'] in C.ARMS[:7]]
    if old_run['records'] != [stub(r) for r in initial] or old_run['records'] != a['initial_records']:
        raise ValueError('Frozen initial models changed')
    if old_run['environment'] != env or old_run['source_binding'] != E.source_binding() or old_run['amendment_sha256'] != C.sha(amendment_path()):
        raise ValueError('Staged outputs have a different protocol/runtime')
    cases, indices, _ = C.check_cohort()
    if old_run['cohort_lock_sha256'] != C.sha(C.ROOT/'prepared/cohort_lock.json'):
        raise ValueError('Cohort changed')
    ids, labels, lesions = case_arrays(cases)
    run = dict(cohort_lock_sha256=old_run['cohort_lock_sha256'], environment=env, source_binding=E.source_binding(),
        records=[stub(r) for r in records], all135_frozen_before_evaluation=False,
        all135_frozen_before_analysis=True, clinical_outcome_used_for_selection=False,
        staged_execution=dict(amendment_sha256=C.sha(amendment_path()), adapter_sha256=C.sha(__file__), initial_run_sha256=old_digest, initial_models=105, later_models=30))
    digest = C.json_hash(run)
    with process_guard('COMPLETE_135_FROM_STAGED_105'):
        run_path = C.ROOT/'evaluation/run_lock.json'
        previous = C.read(run_path)
        if previous != placeholder(old_digest) and previous != run:
            raise ValueError('Unrelated existing full run')
        C.save(stage_dir()/'run_lock_before_completion.json', placeholder(old_digest))
        C.save(run_path, run)
        outputs = [adopt_prediction(r, ids, labels, lesions, old_digest, digest) for r in initial]
        with np.load(C.ROOT/'prepared/candidate_metadata_inputs.npz', allow_pickle=False) as metadata:
            if not np.array_equal(metadata['isic_id'][indices], ids):
                raise ValueError('Metadata order mismatch')
            for r in records:
                if r['arm'] not in C.ARMS[:7]:
                    outputs.append(E.predict(r, cases, indices, digest, metadata))
        C.save(C.ROOT/'evaluation/evaluation_complete.json', dict(status='ALL_135_MILK_EVALUATIONS_VERIFIED', run_sha256=digest, files=outputs,
            staged_models_reused=105, all135_frozen_before_first_prediction=False, all135_frozen_before_analysis=True))
        print('All 135 predictions verified; initial 105 reused exactly. Full prespecified analysis can now run.', flush=True)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--run-existing', action='store_true')
    g.add_argument('--complete', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.run_existing:
        run_initial()
    else:
        complete()


if __name__ == '__main__':
    main()
