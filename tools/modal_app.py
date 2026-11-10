"""Modal runs.

    modal run tools/modal_app.py::main                 # reality check: 32 real torch.distributed processes x 4 strategies
    modal run tools/modal_app.py::main --world 8 --steps 6   # a cheap dry run of the same
    modal run tools/modal_app.py::notebook             # execute the full notebook (32 virtual GPUs) on a 16-core CPU box

The notebook needs ~1 GB at the peak of a 32-GPU step (32 ranks' activations alive at once) on top of everything it
keeps; my 8 GB laptop ran out of memory in the timing grid, so the reported run executes here and the executed notebook,
figures and results are copied back.

Writes results/real_dist.json. A CPU container is enough: the point is the bookkeeping (what each rank holds,
what it sends) and whether PyTorch's own ZeRO implementations train the same model my simulator does.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.15.0', 'numpy', index_url='https://download.pytorch.org/whl/cpu')
         .add_local_dir(ROOT / 'data', '/root/data')
         .add_local_python_source('s12'))

app = modal.App('era-v5-s12-zero', image=image)

nb_image = (modal.Image.debian_slim(python_version='3.12')
            .pip_install('torch==2.15.0', index_url='https://download.pytorch.org/whl/cpu')
            .pip_install('numpy', 'pandas', 'matplotlib', 'nbformat', 'nbconvert', 'ipykernel', 'jinja2')
            .add_local_dir(ROOT / 'data', '/repo/data')
            .add_local_dir(ROOT / 's12', '/repo/s12', ignore=['__pycache__'])
            .add_local_dir(ROOT / 'tools', '/repo/tools', ignore=['__pycache__'])
            .add_local_file(ROOT / 'results' / 'real_dist.json', '/repo/results/real_dist.json'))


@app.function(image=nb_image, cpu=16, memory=16384, timeout=5400)
def run_notebook_remote() -> dict:
    import os
    import subprocess
    os.chdir('/repo')
    env = dict(os.environ, PYTHONPATH='/repo', S12_PROFILE='full', MPLBACKEND='module://matplotlib_inline.backend_inline')
    subprocess.run(['python', 'tools/build_notebook.py'], check=True, env=env)
    p = subprocess.run(['python', 'tools/run_notebook.py', '--profile', 'full', '--timeout', '7200',
                        '--out', 'zero_on_32_virtual_gpus.ipynb'], env=env, capture_output=True, text=True)
    files = {}
    for rel in ['zero_on_32_virtual_gpus.ipynb', 'results/summary.full.json', 'results/site.full.json']:
        if os.path.exists(rel):
            files[rel] = open(rel, 'rb').read()
    for f in sorted(os.listdir('figures')) if os.path.isdir('figures') else []:
        files['figures/' + f] = open('figures/' + f, 'rb').read()
    return {'files': files, 'stdout': p.stdout[-4000:], 'stderr': p.stderr[-6000:], 'code': p.returncode}


@app.local_entrypoint()
def notebook():
    r = run_notebook_remote.remote()
    for rel, b in r['files'].items():
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
        print('copied', rel, len(b))
    print(r['stdout'])
    if r['code']:
        print('NOTEBOOK FAILED', r['stderr'])


@app.function(cpu=16, memory=32768, timeout=3600, max_containers=4)
def run_strategy(strategy: str, world: int, steps: int) -> dict:
    import os
    os.chdir('/root')
    from s12.real_dist import launch
    return launch(strategy, world, steps, f'/tmp/real_{strategy}.json')


@app.local_entrypoint()
def main(world: int = 32, steps: int = 30):
    from itertools import repeat
    strategies = ['ddp', 'zero1', 'fsdp2_zero2', 'fsdp2_zero3']
    out = {}
    for r in run_strategy.map(strategies, repeat(world), repeat(steps)):
        r0 = r['ranks'][0]
        out[r['strategy']] = r
        print(f"{r['strategy']:<12} world {r['world']}  torch {r['torch']}  loss {r['losses'][-1]:.4f}  "
              f"B/param rank0: params {r0['params'] / r['psi']:.3f} grads {r0['grads'] / r['psi']:.3f} "
              f"opt {r0['opt'] / r['psi']:.3f}  comm {r0['comm_bytes']}  wall {r0['wall_s']:.0f}s")
    path = ROOT / 'results' / 'real_dist.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    print('wrote', path)
