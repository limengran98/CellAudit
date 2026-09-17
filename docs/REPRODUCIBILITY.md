# Reproducibility guide

CellAudit separates software verification, fresh model discovery, frozen-source
refitting, and checkpoint replay. Each level has a different input contract and
produces its own integrity receipts.

## 1. Verify the released software

Install the package, check the registered task bindings, verify repository
integrity, and run the CPU regression suite:

```bash
python -m pip install -e .
cellaudit validate
python scripts/verify_repository.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The tests cover the task/fold contract, provider lock, candidate materialization,
source checker, coordinate decomposition, method-level source selection,
five-refit panel construction, and Fold-5 source-refit boundary.

## 2. Run fresh discovery

Fresh discovery requires the four registered profile files in the locations
listed in the root README and an OpenAI-compatible provider configured through
environment variables. A formal campaign writes:

- `campaign_freeze.json`, binding task, seeds, budget, and configuration;
- one directory per trajectory, including failed candidates and repairs;
- `campaign_summary.json`, binding all selected trajectory endpoints; and
- `_SUCCESS`, written only after every registered trajectory is complete.

Use a new output directory for each run:

```bash
cellaudit campaign \
  --stage open \
  --task bbbc047 \
  --trajectories 10 \
  --seed-base 2026080701 \
  --device cuda:0 \
  --output-root runs/bbbc047_open_discovery
```

Provider output is an input to discovery, so a fresh campaign is a new search
replicate even when its data, seeds, and candidate budget match an earlier run.
The deterministic executor and evaluator remain fixed by the campaign receipt.

## 3. Refit a frozen source

For prediction-score discovery, `prepare-open-refits` ranks every executable
candidate in the complete campaign using the registered Fold-3 rule, freezes
one source, and performs five paired Folds-1--3 refits. Each result binds the
same source hash and a separate checkpoint hash:

```bash
cellaudit audit prepare-open-refits \
  --task bbbc047 \
  --discovery-root runs/bbbc047_open_discovery \
  --seed-base 2026080901 \
  --device cuda:0 \
  --output-root runs/bbbc047_open_refits
```

Path-constrained and falsification-guided campaigns use their already frozen
trajectory winners. The audit freeze rejects incomplete campaigns, changed
source files, mismatched task identities, or an unexpected number of endpoints.

## 4. Replay a frozen checkpoint

Checkpoint replay performs inference only. It requires:

1. the audit registry and freeze receipt;
2. the exact selected source or structured design;
3. its matching fitted checkpoint;
4. prepared data for the authorized held-out fold; and
5. the registered input-replacement construction and seeds.

Fold 4 is run and summarized before Fold 5 can be authorized:

```bash
cellaudit audit run --output-root runs/bbbc047_open_audit --fold 4 --device cuda:0
cellaudit audit summarize --output-root runs/bbbc047_open_audit --fold 4
cellaudit audit authorize-fold5 --output-root runs/bbbc047_open_audit
cellaudit audit run --output-root runs/bbbc047_open_audit --fold 5 --device cuda:0
cellaudit audit summarize --output-root runs/bbbc047_open_audit --fold 5
```

The replay records checkpoint and source hashes, map seeds, donor-map hashes,
continuous effects, model-specific reference variation, categorical decisions,
elapsed time, and `provider_calls: 0`.

## Statistical units

- A discovery trajectory is one search-level observation.
- A paired refit seed is one frozen-source observation.
- For constrained discovery settings, one trajectory-selected endpoint is one
  endpoint-level observation.
- The 32 matched replacement maps are repeated measurements within a fitted
  model. They calibrate model-specific effects and are not training replicates.

Continuous evidence should be retained alongside categorical status. A
source-level finding, prediction change, target-loss gain, and cross-fold
replication answer different questions and are not interchangeable.

## Adding a new task

A new task should define, before discovery:

- fit, selection, audit, and replication roles;
- biological context, perturbation, and optional attribute coordinates;
- response construction and evaluation units;
- legal matched replacements and minimum donor coverage;
- source claims and deterministic checker mappings; and
- prediction, loss, checkpoint, and endpoint-selection rules.

The executor should keep these fields outside candidate control and bind them
to every campaign and audit receipt.
