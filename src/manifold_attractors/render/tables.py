"""Render the three included manuscript tables from freshly generated results.

Numerical selection and LaTeX formatting preserve paper ed8b89b's table scripts.
No old table values or experiment results are bundled as inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ..pipeline import ROOT

MODELS = [('llama3', 'Llama 3.1 8B'), ('olmo3', 'OLMo 3 7B'), ('qwen3', 'Qwen 3 8B')]
METHODS = {'raw_centroid': 'Untuned', 'additive': 'Tuned', 'adaptive_straight': 'Adaptive', 'oracle': 'Oracle'}


def markdown(columns, rows):
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(escape, columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(map(escape, row)) + ' |' for row in rows]
    return '\n'.join(lines)+'\n'


def norm_classifier(data, output, layers_per_panel=15, layers=None):
    import numpy as np
    import pandas as pd
    controls = ['Gaussian', 'Shuffled']
    if layers is None:
        layers = np.rint(np.linspace(1, 36, 15)).astype(int).tolist()
    frames = {}
    for key, label in MODELS:
        with np.load(data/f'{key}.npz') as archive:
            layer_ids, scores = archive['layers'], archive['scores']
            assert archive['classes'].tolist() == ['natural', 'gaussian', 'shuffled']
            assert scores.shape[:2] == (len(layer_ids), 3) and np.isfinite(scores).all()
            differences = scores[:, 0, None, :, None] - scores[:, 1:, None, :]
            values = ((differences > 0) + 0.5*(differences == 0)).mean(axis=(-2, -1))
            frames[label] = pd.DataFrame(values.T, index=controls, columns=layer_ids)
    order = pd.MultiIndex.from_product([controls, frames], names=['Class', 'Model'])
    table = (pd.concat(frames, names=['Model', 'Class']).swaplevel().reindex(order)
             .sort_index(axis='columns').rename_axis(columns='Layer')).loc[:, layers]
    panels = []
    for start in range(0, len(table.columns), layers_per_panel):
        panel = table.iloc[:, start:start+layers_per_panel]
        panels.append(panel.style.format(precision=3, na_rep=r'{\textemdash}').to_latex(
            siunitx=True, hrules=True, multirow_align='naive',
            column_format='ll'+'S[table-format=1.3, round-mode=places, round-precision=3]'*len(panel.columns)))
    output.write_text('\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n'
                      +'\\par\\smallskip\n'.join(panels)+'\\endgroup\n')
    rows = [[*index, *['—' if pd.isna(v) else f'{v:.3f}' for v in values]]
            for index, values in table.iterrows()]
    output.with_suffix('.md').write_text(markdown(['Class','Model',*[str(x) for x in layers]], rows))
    return {'layers': layers, 'rows': [dict(class_name=index[0], model=index[1],
              values=[None if pd.isna(v) else float(v) for v in values]) for index,values in table.iterrows()]}


def pair_barrier(data, output):
    summary = json.loads(data.read_text())
    assert summary['cutoff'] is None
    rows, values = [], []
    for index, (key, label) in enumerate([('gaussian','Gaussian'),('shuffled','Shuffled'),('norm-matched','Norm-matched')], 1):
        if key not in summary['models'][MODELS[0][0]]['comparisons']:
            continue
        cells = []
        for model, _ in MODELS:
            result = summary['models'][model]
            positive, negative = result['scores'][0], result['scores'][index]
            wins = sum(x>y for x in positive for y in negative)
            ties = sum(x==y for x in positive for y in negative)
            value = (wins+0.5*ties)/(len(positive)*len(negative))
            assert value == result['comparisons'][key]['auc']
            cells.append(value)
        formatted = [f'{x:.3f}' for x in cells]
        rows.append(label+' & '+' & '.join(formatted)+r' \\')
        values.append({'class_name': label, 'values': cells})
    numeric = 'S[table-format=1.3, round-mode=places, round-precision=3]'
    lines = [r'\begin{tabular}{l'+numeric*3+'}',r'\toprule',
             r'Synthetic class & {Llama 3.1 8B} & {OLMo 3 7B} & {Qwen 3 8B} \\',
             r'\midrule',*rows,r'\bottomrule',r'\end{tabular}']
    output.write_text('\n'.join(lines)+'\n')
    output.with_suffix('.md').write_text(markdown(['Synthetic class',*[label for _,label in MODELS]],
        [[row['class_name'],*[f'{v:.3f}' for v in row['values']]] for row in values]))
    return {'rows': values, 'cutoff': None}


def adaptive_steering(data, output):
    import pandas as pd
    results = json.loads((data/'summary.json').read_text())
    oracle = json.loads((data/'linear_oracle/summary.json').read_text())
    def normalize(summary, group):
        return pd.concat({dataset: pd.json_normalize(list(result[group].values())).set_axis(result[group])
            for dataset,result in summary['datasets'].items()}, names=['dataset','method_key'])
    baseline = normalize(results,'methods')
    oracle = normalize(oracle,'oracles').rename(index=lambda track:f'{track}/oracle',level='method_key')
    scores = pd.concat([baseline,oracle]).reset_index()
    scores[['track','method']] = scores.pop('method_key').str.split('/',expand=True)
    mass = scores.filter(like='mean_masses.').rename(columns=lambda c:c.removeprefix('mean_masses.'))
    greedy = scores.filter(like='argmax_counts.').rename(columns=lambda c:c.removeprefix('argmax_counts.'))
    fractions = mass.where(scores['track'].eq('mass'),greedy/results['n_per_test'],axis=0)
    code = results.get('target','zh')
    name = {'zh':'Chinese','el':'Greek'}[code]; key = name.lower()
    scores = scores.assign(**{f'non_{key}':100*(fractions['Latin']+fractions['Other alphabet']),
                              key:100*fractions[name]})
    tracks = {'mass':r'Mass (\%)','greedy':r'Greedy (\%)'}
    datasets = {'wiki_test':r'\textsc{id}','fineweb_test':r'\textsc{ood}'}
    outcomes = {f'non_{key}':rf'non-\texttt{{{code}}} $\downarrow$',key:rf'\texttt{{{code}}} $\uparrow$'}
    numeric = (scores.pivot(index='method',columns=['track','dataset'],values=list(outcomes))
        .reindex(index=METHODS,columns=pd.MultiIndex.from_product([outcomes,tracks,datasets])))
    table = (numeric.rename(index=METHODS).rename(columns=outcomes,level=0)
        .rename(columns=tracks,level=1).rename(columns=datasets,level=2).rename_axis(columns=[None,None,None]))
    if table.isna().any().any():
        raise ValueError('Missing adaptive steering values after pivot/reindex')
    table.insert(0,('','','Method'),table.index)
    styled = table.style.hide(axis='index').format(precision=2)
    styled.map(lambda value:'bfseries:;',subset=(['Adaptive'],table.columns[1:]))
    latex = styled.to_latex(siunitx=True,hrules=True,multicol_align='c',
        column_format='l'+'S[table-format=2.2]'*4+'|'+'S[table-format=2.2]'*4)
    output.write_text('\\begingroup\n\\small\n\\sisetup{detect-weight=true}\n'
        '\\setlength{\\tabcolsep}{3pt}\n'+latex+'\\endgroup\n')
    rows = [[METHODS[index],*[f'{v:.2f}' for v in values]] for index,values in numeric.iterrows()]
    labels = {f'non_{key}':f'Non-{name} ↓', key:f'{name} ↑', 'mass':'Mass (%)',
              'greedy':'Greedy (%)', 'wiki_test':'ID', 'fineweb_test':'OOD'}
    output.with_suffix('.md').write_text(markdown(['Method',*[' / '.join(labels[x] for x in column)
        for column in numeric.columns]],rows))
    return {'columns':[list(c) for c in numeric.columns],
        'rows':[{'method':METHODS[index],'values':[float(v) for v in values]} for index,values in numeric.iterrows()],
        'n_per_test':results['n_per_test'], f'non_{key}_excludes_neutral':True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('table',choices=['norm_growth_classifier','pair_barrier_validation','adaptive_steering'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--layers-per-panel',type=int,default=15)
    p.add_argument('--layers',type=int,nargs='+')
    a = p.parse_args()
    if a.layers_per_panel<1:p.error('--layers-per-panel must be positive')
    output = a.output or ROOT/'tab'/f'{a.table}.tex'; output.parent.mkdir(parents=True,exist_ok=True)
    if a.table=='norm_growth_classifier': values=norm_classifier(a.data,output,a.layers_per_panel,a.layers)
    elif a.table=='pair_barrier_validation': values=pair_barrier(a.data,output)
    else: values=adaptive_steering(a.data,output)
    values['table']=a.table
    values['render_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {output}, Markdown and numeric JSON')


if __name__=='__main__':main()
