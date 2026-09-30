"""Fetch pinned author source and run fresh extraction in isolated historical Python.

This explicit compute entry point may download git source, Python and dependencies.
It never installs into the modern compute environment or reuses published vectors.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from .refusal_direction import COMMIT

AUTHOR_URL='https://github.com/andyrdt/refusal_direction'


def ensure_author_repo(repo):
    repo=Path(repo).resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',AUTHOR_URL,str(repo)],check=True)
        # Check out only source and raw split inputs, not published run artifacts.
        subprocess.run(['git','-C',str(repo),'sparse-checkout','set','--no-cone',
            '/pipeline/*.py','/pipeline/model_utils/','/pipeline/submodules/','/pipeline/utils/',
            '/dataset/*.py','/dataset/splits/','/LICENSE','/requirements.txt'],check=True)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',COMMIT],check=True)
    if not (repo/'.git').exists():
        raise ValueError(f'{repo} exists but is not an author git checkout')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:
        raise ValueError(f'{repo}: expected author commit {COMMIT}, got {head}; refusing to alter an existing checkout')
    dirty=subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--','pipeline','dataset'],text=True)
    if dirty:
        raise ValueError('Author pipeline/dataset checkout has modifications; exact extraction is blocked')
    return repo


def command(repo, output, uv='uv'):
    requirements=Path(__file__).with_name('refusal-extraction-requirements.txt').resolve()
    return [uv,'run','--isolated','--no-project','--python','3.10',
            '--with-requirements',str(requirements),'python','-m',
            'manifold_attractors.maps.refusal_direction','--author-repo',str(repo),
            '--output',str(output)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-repo',type=Path,default=Path('external/refusal_direction'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    uv=shutil.which('uv')
    if uv is None:raise RuntimeError('uv is required for isolated historical Python/dependency preparation')
    repo=ensure_author_repo(a.author_repo)
    env=os.environ.copy()
    # Historical Python cannot install this Python3.13 project, but the minimal
    # extraction module itself is Python3.10-compatible and needs no package build.
    source=Path(__file__).resolve().parents[2]
    env['PYTHONPATH']=str(source)
    subprocess.run(command(repo,a.output.resolve(),uv),env=env,check=True)

if __name__=='__main__':main()
