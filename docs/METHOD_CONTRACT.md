# Fixed method and evaluation rules

CellAudit separates language-model candidate generation from deterministic
testing of the selected model's stated input use.

## Model search

Prediction-score discovery instantiates the existing CellScientist search
policy. CellAudit couples its generated candidates to a fixed execution
harness and deterministic source and fitted-model input-use tests.

- Folds 1--2 fit model parameters; Fold 3 supplies target-bearing discovery
  feedback.
- Each proposal occupies one physical candidate slot, including proposals that
  fail validation or training.
- The local executor owns data loading, preprocessing, fold assignment, target
  construction, evaluation, checkpoint selection, and resource accounting.
- Provider requests contain the candidate interface, fixed
  diagnostics, and compact trajectory memory.
- One campaign contains ten trajectories under one seed list, fixed campaign
  configuration, and completion record.
- For prediction-score discovery, the method-level model is frozen only after all
  ten trajectories complete. Every executable candidate is ranked by maximum
  Fold-3 Global PCC, then minimum Fold-3 MSE, minimum parameter count, and a
  stable trajectory/candidate/hash identifier. Fold 4 and Fold 5 do not enter
  this ranking.

## Code-path checking

The path-constrained compiler validates component choices and emits the
executable PyTorch source, metadata, exported symbols, tensor interfaces,
training hooks, and an explicit compiler-owned perturbation route. The falsification-guided compiler additionally defines a
centered perturbation increment over a fixed control-profile predictor.
The deterministic abstract-syntax-tree checker verifies that the cited function accepts
and uses the stated input and tests a small set of known algebraic failures.
It does not execute or trace the complete model; dynamic code outside these rules
remains unresolved, and whole-model dependence is determined by the held-out
input intervention. The checker reports implemented,
implementation-contradicted, unresolved, or not-claimed.

## Held-out input-use testing

- All within-policy model identities, source hashes, fit seeds, intervention-map
  construction, map seeds, threshold-construction rule, and decision rules
  are fixed before Fold 4. The numerical model-specific threshold is computed
  from the fixed reference scores on the fold being tested. The
  prediction-score audit contains five paired refits of one automatically
  frozen method-level model.
- Fixed matched permutations, constructed without response values or model scores, test perturbation identity, control profile,
  perturbation--context interaction, and identifiable dose contrasts.
- For small-molecule inputs, compound permutations hold dose fixed and
  dose maps hold chemical identity fixed. Two-player Shapley terms resolve
  predictive utility across chemical identity and dose; the both-shuffled
  residual closes the full-minus-control-only increment, and the factorial
  interaction remains a separately reported diagnostic.
- Model-specific map-variation references calibrate passed,
  behaviorally unsupported, inconclusive, and not-identifiable decisions.
- In the five-refit prediction-score panel, each refit receives its own status
  and the aggregate status requires agreement in at least four of five refits.
  The path-constrained and falsification-guided panels report counts across all
  ten selected models without a vote.
- The prediction-score panel first freezes one source within each policy using
  Fold 3. Its disclosed across-policy rule then selects one of those sources
  using mean Fold-4 Global PCC. The other two search settings carry every
  Fold-3 trajectory winner through both folds without Fold-4 reselection.
- Fold-4 execution binds every model evaluation and summary to the authorization
  record that opens Fold 5.
- Fold 5 repeats the same construction and decision rule with new fixed
  permutation seeds on the second held-out fold. For prediction-score discovery,
  the frozen source is refit under the same five seed identities on Folds 1--2,
  its checkpoint is selected on Fold 3, and only then is Fold 5 loaded. This is
  an independently refit replication of the frozen source, not reuse of Fold-4
  weights. Path-constrained and falsification-guided studies replay their
  already frozen trajectory checkpoints.

## Experiment-log contents

Search logs include parent identity, hypothesis or design card, source,
preflight, repairs, checkpoint, Fold-3 diagnostics, selection decision,
provider usage, token accounting, and compute time. Audit logs include the
selected model, fit seed, intervention permutations, continuous effects,
target-loss gain, code-check result, behavioral result, and integrity hashes.

This sequence makes every reported conclusion traceable to the selected model
and held-out evaluation that produced it.
