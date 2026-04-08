# Verified Run (Full Transcript)

This document records a full, reproducible harness run using:
- [`data/training.json`](../data/training.json)
- [`data/prompts.json`](../data/prompts.json)
- Defaults in [`eval_harness.py`](../eval_harness.py): `EVAL_SEED=0`, `STATE_DIM=32`, `RELAX_STEPS=2`, `RELAX_ALPHA=0.25`, `TRAIN_EPOCHS=1500`
- CPU execution

The transcript includes forward-state and converged-attractor diagnostics, stability checks, interpolation probes, basin comparisons, decoding outputs, generation metrics, and gate summary. Training logs include `ctr` because contrastive pairs are enabled in `training.json`.

Small numeric drift across machines is expected (PyTorch and hardware differences).

## Command

```bash
python3 main.py
```

## Full Terminal Output (Exit Code `0`)

```
(.venv) boggersthefish@BoggersThePC:~/TinyLLM$ python3 main.py 
Frozen eval harness (seed=0, state_dim=32, epochs=1500)
Corpus pair (first two lines): the cat chases the dog | the dog runs to the barn
  first_divergence_index (full strings)=1
  first_divergence_index (after first token)=0 shared_prefix_until_divergence=[]

Training on: ['the cat chases the dog', 'the dog runs to the barn', 'the bird flies in the sky', 'the fish swims in the lake', 'the mouse eats the cheese']
epoch    0  loss 2.9161  ctr 0.1721
epoch  100  loss 0.5268  ctr 0.1721
epoch  200  loss 0.4275  ctr 0.1721
epoch  300  loss 0.4014  ctr 0.1721
epoch  400  loss 0.3903  ctr 0.1721
epoch  500  loss 0.3844  ctr 0.1721
epoch  600  loss 0.3808  ctr 0.1721
epoch  700  loss 0.3784  ctr 0.1721
epoch  800  loss 0.3768  ctr 0.1721
epoch  900  loss 0.3756  ctr 0.1721
epoch 1000  loss 0.3747  ctr 0.1721
epoch 1100  loss 0.3740  ctr 0.1721
epoch 1200  loss 0.3735  ctr 0.1721
epoch 1300  loss 0.3730  ctr 0.1721
epoch 1400  loss 0.3726  ctr 0.1721
epoch 1499  loss 0.3723  ctr 0.1721

mean_corpus_ce: 0.355128
next-token tests: 5/5
  [ok] 'the cat' -> 'chases' (expected 'chases')
  [ok] 'the dog' -> 'runs' (expected 'runs')
  [ok] 'the bird' -> 'flies' (expected 'flies')
  [ok] 'the fish' -> 'swims' (expected 'swims')
  [ok] 'the mouse' -> 'eats' (expected 'eats')
ambiguous_prefix: 'the'
ambiguous_entropy: 1.622345
ambiguous_top5: [('mouse', 0.22662661969661713), ('cat', 0.22647841274738312), ('dog', 0.18170131742954254), ('fish', 0.1812954992055893), ('bird', 0.18121987581253052)]

Pairwise cosine similarity between animal states:
            the bird   the cat   the dog  the fish the mouse
the bird      1.0000    0.6048    0.6173    0.4171    0.5317
the cat       0.6048    1.0000    0.3340    0.2763    0.4895
the dog       0.6173    0.3340    1.0000    0.2538    0.2725
the fish      0.4171    0.2763    0.2538    1.0000    0.4439
the mouse     0.5317    0.4895    0.2725    0.4439    1.0000

Trajectory norms:
step_name : ||state||  (cos to previous)
start              norm=0.0000
token:the          norm=5.6920  cos(prev)=nan
relax:0            norm=4.4405  cos(prev)=0.9861
relax:1            norm=3.5858  cos(prev)=0.9821
token:cat          norm=7.2470  cos(prev)=0.5332
relax:0            norm=5.6654  cos(prev)=0.9877
relax:1            norm=4.6149  cos(prev)=0.9833
token:chases       norm=6.9905  cos(prev)=0.5551
relax:0            norm=5.3317  cos(prev)=0.9894
relax:1            norm=4.1558  cos(prev)=0.9844
token:the          norm=7.4932  cos(prev)=0.6586
relax:0            norm=5.8631  cos(prev)=0.9927
relax:1            norm=4.6886  cos(prev)=0.9906

Relax-until-convergence (extra dynamics on prefix final states):
the cat
  steps_to_converge=48
  final_norm=3.3118
the dog
  steps_to_converge=50
  final_norm=2.4691
the bird
  steps_to_converge=50
  final_norm=2.0614
the fish
  steps_to_converge=50
  final_norm=2.5417
the mouse
  steps_to_converge=48
  final_norm=3.1481
the cat chases
  steps_to_converge=50
  final_norm=3.0446
the dog runs
  steps_to_converge=50
  final_norm=2.7790
the bird flies
  steps_to_converge=46
  final_norm=2.4316
the fish swims
  steps_to_converge=49
  final_norm=3.1275
the mouse eats
  steps_to_converge=50
  final_norm=2.8095
the dog chases
  steps_to_converge=49
  final_norm=3.0445
the mouse chases
  steps_to_converge=50
  final_norm=3.0448
the cat runs
  steps_to_converge=50
  final_norm=2.7789
the cat chases the dog
  steps_to_converge=50
  final_norm=2.4692
the cat chases the mouse
  steps_to_converge=50
  final_norm=3.1481
the cat chases the bird
  steps_to_converge=50
  final_norm=2.0615
the mouse eats the cheese
  steps_to_converge=50
  final_norm=3.1253
the mouse eats the dog
  steps_to_converge=50
  final_norm=2.4690
the bird flies to the dog
  steps_to_converge=50
  final_norm=2.4691
the bird flies to the mouse
  steps_to_converge=48
  final_norm=3.1481

Attractor norm summary (L2):
  converged states (all relax-until-convergence prefixes):
    mean=2.7741  std=0.3699  min=2.0614  max=3.3118
  forward states (noun prefixes only, after relax substeps per token):
    mean=4.4209  std=0.6575  min=3.2414  max=5.2708

Attractor stability test:
# cos ≥ 0.95 → strong attractor basin
# cos 0.8–0.95 → weak basin
# cos < 0.8 → unstable basin

prefix      mean_cos_recovery    std_cos_recovery    mean_steps    min_cos_recovery
-----------------------------------------------------------------------------------
the cat                1.0000              0.0000         29.50              1.0000
the dog                1.0000              0.0000         32.20              1.0000
the bird               1.0000              0.0000         30.20              1.0000
the fish               1.0000              0.0000         33.10              1.0000
the mouse              1.0000              0.0000         29.70              1.0000

Attractor interpolation test (cat → dog)

t            cos(final, cat)       cos(final, dog)
--------------------------------------------------
0.0                   1.0000               -0.1390
0.1                   0.9947               -0.0501
0.2                   0.9765                0.0533
0.3                   0.9416                0.1709
0.4                   0.8847                0.3031
0.5                   0.7986                0.4493
0.6                   0.6744                0.6057
0.7                   0.5055                0.7606
0.8                   0.2955                0.8923
0.9                   0.0679                0.9752
1.0                  -0.1392                1.0000

Interpretation:
  • if the system converges cleanly to cat or dog → basins are well separated
  • if a boundary region appears → basin boundary detected
  • if a new attractor appears → emergent semantic state

Attractor interpolation test (cat chases → dog runs)

t        cos(final, cat chases)     cos(final, dog runs)
--------------------------------------------------------
0.0                      1.0000                   0.2117
0.1                      0.9971                   0.2692
0.2                      0.9863                   0.3411
0.3                      0.9633                   0.4312
0.4                      0.9208                   0.5415
0.5                      0.8493                   0.6672
0.6                      0.7428                   0.7918
0.7                      0.6086                   0.8929
0.8                      0.4656                   0.9583
0.9                      0.3309                   0.9910
1.0                      0.2118                   1.0000

Interpretation:
  • clean convergence toward one verb-prefix attractor → verb basins well separated
  • boundary region in t → verb basin boundary (or mixed conditioning)
  • distinct intermediate attractor → emergent state under blended token embed

Pairwise cosine similarity between converged attractor states:
                the bird     the cat     the dog    the fish   the mouse
the bird          1.0000      0.3801      0.2262     -0.0090      0.3724
the cat           0.3801      1.0000     -0.1390     -0.2575      0.4158
the dog           0.2262     -0.1390      1.0000     -0.3284     -0.1672
the fish         -0.0090     -0.2575     -0.3284      1.0000     -0.1287
the mouse         0.3724      0.4158     -0.1672     -0.1287      1.0000

Pairwise cosine similarity between verb-conditioned attractor states:
                  the bird flies  the cat chases    the dog runs  the fish swims  the mouse eats
the bird flies            1.0000         -0.4850         -0.2347          0.4784         -0.6034
the cat chases           -0.4850          1.0000          0.2117         -0.4176          0.7323
the dog runs             -0.2347          0.2117          1.0000         -0.0774          0.1500
the fish swims            0.4784         -0.4176         -0.0774          1.0000         -0.3271
the mouse eats           -0.6034          0.7323          0.1500         -0.3271          1.0000

Verb basin comparison:
                      the cat chases    the dog chases  the mouse chases      the dog runs      the cat runs
the cat chases                1.0000            1.0000            1.0000            0.2117            0.2117
the dog chases                1.0000            1.0000            1.0000            0.2117            0.2117
the mouse chases              1.0000            1.0000            1.0000            0.2118            0.2117
the dog runs                  0.2117            0.2117            0.2118            1.0000            1.0000
the cat runs                  0.2117            0.2117            0.2117            1.0000            1.0000

  cos("the cat chases", "the dog chases")=1.0000
  cos("the cat chases", "the mouse chases")=1.0000
  cos("the dog runs", "the cat runs")=1.0000

Object basin comparison:
cosine similarity matrix between these converged states.
                                the cat chases the dog   the cat chases the mouse    the cat chases the bird  the mouse eats the cheese     the mouse eats the dog
the cat chases the dog                          1.0000                    -0.1671                     0.2263                    -0.1043                     1.0000
the cat chases the mouse                       -0.1671                     1.0000                     0.3725                    -0.3303                    -0.1673
the cat chases the bird                         0.2263                     0.3725                     1.0000                    -0.0715                     0.2262
the mouse eats the cheese                      -0.1043                    -0.3303                    -0.0715                     1.0000                    -0.1043
the mouse eats the dog                          1.0000                    -0.1673                     0.2262                    -0.1043                     1.0000

  cos("the cat chases the dog", "the cat chases the mouse")=-0.1671
  cos("the cat chases the dog", "the cat chases the bird")=0.2263
  cos("the mouse eats the cheese", "the mouse eats the dog")=-0.1043

Global object basin comparison:
cosine similarity matrix between these states.
                                    the cat chases the dog       the mouse eats the dog    the bird flies to the dog     the cat chases the mouse  the bird flies to the mouse
the cat chases the dog                              1.0000                       1.0000                       1.0000                      -0.1671                      -0.1672
the mouse eats the dog                              1.0000                       1.0000                       1.0000                      -0.1673                      -0.1673
the bird flies to the dog                           1.0000                       1.0000                       1.0000                      -0.1672                      -0.1672
the cat chases the mouse                           -0.1671                      -0.1673                      -0.1672                       1.0000                       1.0000
the bird flies to the mouse                        -0.1672                      -0.1673                      -0.1672                       1.0000                       1.0000

  cos("the cat chases the dog", "the mouse eats the dog")=1.0000
  cos("the cat chases the dog", "the bird flies to the dog")=1.0000
  cos("the cat chases the mouse", "the bird flies to the mouse")=1.0000

Noun vs verb converged attractor states (same subject; verb extends prefix):
  cos("the cat", "the cat chases")=-0.2169
  cos("the dog", "the dog runs")=-0.0439

cos(state("the cat"), state("the cat chases"))=-0.2169 (converged attractor states)

Forward pass hidden state (after 2 relax substeps per token) vs converged attractor:
the cat
  cos(prefix_state , attractor_state)=0.4207
the dog
  cos(prefix_state , attractor_state)=0.1995
the bird
  cos(prefix_state , attractor_state)=0.5772
the fish
  cos(prefix_state , attractor_state)=0.1058
the mouse
  cos(prefix_state , attractor_state)=0.3497

Greedy generations (prefixes from data/prompts.json):
  prefix='the cat' -> the cat chases the dog runs to the barn dog runs to the barn barn dog runs to the barn barn barn dog runs to the barn barn dog runs
  prefix='the cat chases the' -> the cat chases the bird flies in the sky the sky the barn barn to the barn barn barn dog runs to the barn barn dog runs to the barn barn dog
  prefix='the dog' -> the dog runs to the barn barn barn dog runs to the barn barn dog runs to the barn barn dog runs to the barn barn dog runs to the
  prefix='the bird flies' -> the bird flies in the sky the mouse the cheese lake lake the dog runs to the barn barn dog runs to the barn barn dog runs to the barn barn
  prefix='the mouse eats the' -> the mouse eats the cat chases the dog runs to the barn barn dog runs to the barn barn dog runs to the barn barn dog runs to the barn barn dog
  prefix='the' -> the bird flies in the sky the barn the dog runs to the barn barn barn dog runs to the barn barn dog runs to the barn barn dog

Generation metrics (full-string corpus match + repeated n-grams):
  exact_match_rate: 0.0000  (6 prompts)
  longest_repeated_ngram (max over prompts): 5

All gates passed.
(.venv) boggersthefish@BoggersThePC:~/TinyLLM$ 
```

## Interpretation

This run demonstrates:
- Branch next-token checks and configured gates pass.
- The ambiguous prefix `the` distributes probability mass across multiple subjects.
- Converged attractor geometry differs from forward-pass geometry and reveals clearer basin structure.
- Generation with nonzero temperature still exhibits template mixing and long repeated n-grams.

For strict argmax decoding, set `greedy_temperature` to `0`.
