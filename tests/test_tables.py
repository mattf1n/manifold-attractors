import json
import numpy as np

from manifold_attractors.render.tables import norm_classifier, pair_barrier, adaptive_steering
from manifold_attractors.pipeline import ROOT, load_tasks, order_tasks


def test_all_active_paper_tables_have_fresh_generation_edges():
    tasks = load_tasks()
    references = json.loads((ROOT/'configs/reference_tables.json').read_text())['tables']
    assert {entry['name'] for entry in references} == {'norm_growth_classifier','pair_barrier_validation','adaptive_steering'}
    for entry in references:
        name = 'render-table-'+entry['name']
        assert tasks[name]['environment'] == 'table'
        assert tasks[name]['deps']
        assert len(order_tasks(tasks, [name])) > 1


def test_layer_auc_half_ties_and_missing_model_layers(tmp_path):
    for model, count in [('llama3',32),('olmo3',32),('qwen3',36)]:
        scores = np.tile(np.array([[2.,3.],[0.,2.],[3.,4.]])[None], (count,1,1))
        np.savez(tmp_path/f'{model}.npz', layers=np.arange(1,count+1), scores=scores,
                 classes=['natural','gaussian','shuffled'])
    output = tmp_path/'table.tex'
    result = norm_classifier(tmp_path,output,layers=[1,34,36])
    assert result['rows'][0]['values'] == [.875,None,None]
    assert result['rows'][2]['values'] == [.875,.875,.875]
    assert result['rows'][3]['values'] == [.125,None,None]
    assert '{\\textemdash}' in output.read_text()
    assert '0.875' in output.with_suffix('.md').read_text()


def test_pair_auc_recomputed_from_scores(tmp_path):
    model = {'scores':[[2,3],[0,2],[3,4]], 'comparisons':{'gaussian':{'auc':.875},'shuffled':{'auc':.125}}}
    source = tmp_path/'summary.json'; source.write_text(json.dumps({'cutoff':None,
        'models':{key:model for key in ['llama3','olmo3','qwen3']}}))
    result = pair_barrier(source,tmp_path/'table.tex')
    assert result['rows'][0]['values'] == [.875]*3
    assert result['rows'][1]['values'] == [.125]*3


def test_adaptive_excludes_neutral_not_complement(tmp_path):
    result = {'mean_masses':{'Latin':.2,'Other alphabet':.1,'Chinese':.6,'Neutral':.1},
              'argmax_counts':{'Latin':3,'Other alphabet':1,'Chinese':5,'Neutral':1}}
    methods = {f'{track}/{method}':result for track in ['mass','greedy']
               for method in ['raw_centroid','additive','adaptive_straight']}
    (tmp_path/'summary.json').write_text(json.dumps({'n_per_test':10,'datasets':{
        dataset:{'methods':methods} for dataset in ['wiki_test','fineweb_test']}}))
    (tmp_path/'linear_oracle').mkdir()
    (tmp_path/'linear_oracle/summary.json').write_text(json.dumps({'datasets':{
        dataset:{'oracles':{track:result for track in ['mass','greedy']}}
        for dataset in ['wiki_test','fineweb_test']}}))
    values = adaptive_steering(tmp_path,tmp_path/'table.tex')
    for row in values['rows']:
        assert np.allclose(row['values'],[30,30,40,40,60,60,50,50])
    assert 'Adaptive' in (tmp_path/'table.md').read_text()
    assert '\\bfseries' in (tmp_path/'table.tex').read_text()
