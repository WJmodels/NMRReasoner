"""Run README commands against local assets; record each command and result."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
output = Path(__file__).resolve().parent
mode = 'training_fixed'
records = []
env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

def run(name, args):
    name = 'fixed_' + name
    command = [sys.executable, *map(str, args)]
    start = time.time()
    with (output / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=command, returncode=result.returncode, seconds=time.time()-start))
    (output / (mode + '_commands.json')).write_text(json.dumps(records, indent=2))
    print(name, result.returncode, round(time.time()-start, 2), flush=True)
    if result.returncode:
        print((output/(name+'.log')).read_text()[-6000:], flush=True)
        raise SystemExit(result.returncode)

models = {}
for task, folder, prefix in [('se', 'Structure elucidation', 'SE'), ('ncsp', 'NMR chemical shift prediction', 'NCSP')]:
    model = root / 'weight' / folder / f'{prefix}-CoT-1_MultiNMR_real200k_weight'
    models[task] = model
    if mode == 'inference':
        data = root / 'data' / folder / 'test/MultiNMR-200k' / f'{prefix}_MultiNMR_200k_test_arrow'
        preset = 'experiments/local.json' if task == 'se' else 'experiments/shift_prediction.json'
        options = ['--config', preset, '--data', data, '--model', model]
        run(task+'_inspect', ['test.py', 'inspect', *options])
        dest = output / task / 'inference'
        run(task+'_infer', ['test.py', *options, '--output', dest])
        run(task+'_evaluate', ['test.py', 'evaluate', '--data', dest/'ground_truth', '--predictions', dest/'predictions.txt', '--output', dest/'rescore.json'])
        assert json.loads((dest/'metrics.json').read_text()) == json.loads((dest/'rescore.json').read_text())
    elif mode == 'training_fixed':
        dest = output/task/'training_fixed'
        run(task+'_train', ['train.py', '--data', output/task/'arrow/train8', '--model', model, '--epochs', '1', '--batch-size', '1', '--output', dest])
        assert (dest/'final_model/config.json').is_file(), 'Training did not produce a full model'
        assert list((dest/'final_model').glob('*.safetensors')), 'No saved weights'
        run(task+'_trained_inspect', ['test.py', 'inspect', '--data', output/task/'arrow/test', '--model', dest/'final_model'])
        run(task+'_trained_infer', ['test.py', '--data', output/task/'arrow/test', '--model', dest/'final_model', '--limit', '1', '--output', output/task/'trained_inference_fixed'])

if mode == 'inference':
    base = output/'se/inference'
    # A real second decoding run produces independent candidates for voting.
    beam = output/'se/beam'
    run('se_beam', ['test.py', '--data', base/'ground_truth', '--model', models['se'], '--beam-width', '3', '--output', beam])
    run('se_vote', ['test.py', 'vote', '--data', base/'ground_truth', '--predictions', base/'predictions.jsonl', beam/'predictions.jsonl', '--topk', '10', '--workers', '1', '--output', output/'se/vote'])
    run('regression', ['-m', 'unittest', 'discover', '-s', 'checks', '-v'])
elif mode == 'vllm':
    for task in ['se', 'ncsp']:
        run(task+'_vllm', ['test.py', '--data', output/task/'arrow/test', '--model', models[task], '--backend', 'vllm', '--limit', '1', '--output', output/task/'vllm'])
