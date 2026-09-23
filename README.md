# AdsDrift: A One-Pass Conditional Generator for Adsorption Structures

> **Project status**
>
> AdsDrift is a research prototype developed by the original author. Active
> development has paused, and the project is now released under the Apache-2.0
> License so that new maintainers, reproduction efforts, and follow-up research
> can continue from this point. The repository is provided as is. It is not a
> production-ready software release, and ongoing maintenance is not guaranteed.

AdsDrift is designed for catalyst screening. Given a fixed surface-adsorbate
system, it maps randomly initialized structures distributed across a surface to
a set of candidate structures near low-energy endpoints in a single generator
forward pass. The training target is the distribution of low-energy
configurations for that system. Generated structures can subsequently be
deduplicated, ranked by energy, and subjected to short relaxations.

## What is included

This repository contains the current model architecture, training and
generation code, example configurations, and data-preparation utilities. It
does **not** distribute training data, trained AdsDrift checkpoints, run
outputs, or MACE-MH-1 weights. MACE-MH-1 must be obtained separately from its
[official model page](https://huggingface.co/mace-foundations/mace-mh-1) and
used in accordance with the upstream ASL license. Set
`ADSDRIFT_MACE_CHECKPOINT` to the local checkpoint path.

Prospective contributors should begin with [CONTRIBUTING.md](CONTRIBUTING.md).
Third-party code origins and license notices are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How AdsDrift uses MACE-MH-1

AdsDrift uses **MACE-MH-1 as a frozen, differentiable feature encoder during
training**. Representations for the adsorbate and movable surface atoms are
read from both MACE-MH-1 interaction layers using the `oc20_usemppbe` head.
These representations define the feature space in which the Drifting
distribution-learning objective is constructed. All MACE-MH-1 parameters
remain frozen and the model stays in evaluation mode, but the computational
graph from generated coordinates to MACE features is retained so that feature
loss gradients can propagate back into the AdsDrift generator.

MACE-MH-1 is neither the trainable AdsDrift generator nor an energy labeler for
final structure ranking. Once training is complete, generator-only one-pass
structure generation does not run MACE. MACE-MH-1 weights are not redistributed
in this repository.

This repository describes the historical `test_18` implementation. It inherits
from `test_16` and changes only the within-block aggregation order from parallel
three-branch aggregation to "self-attention residuals, then cross-attention
residuals, then FFN." Cross-attention reads the features already updated by
self-attention in the same layer. Initialization, including the full-graph
geometric embedding and crystal conditioning, is unchanged. Fixed-layer
features are updated while their coordinates remain fixed. The implementation
retains the frozen MACE feature space and the joint, non-isolated Drifting
coordinate gradient restored on 12 September 2026. The parameter count is
unchanged.

To generate PDB files directly from a clean crystal or slab and an adsorbate,
including a vacuum layer and the bundled adsorbate library, use
[`generate/generate_pdb.py`](generate/generate_pdb.py). This generation path
does not require MACE, another machine-learned force field, or a positive-sample
bank. Run the script with `--help` for its command-line options.

## 1. Repository structure and data flow

```text
AdsDrift/
├── README.md                         # Architecture, data, and usage guide
├── config/                           # Fixed-system training configurations
├── data/prepare_dataset.py           # Filtering, endpoint export, and indexing
├── generate/                         # Crystal + adsorbate -> generator -> PDB
├── plot/plot_train.py                # Six-panel training dashboard
├── model/
│   ├── __init__.py                   # Package declarations and exports
│   ├── model.py                      # Model composition and train/sample entry points
│   ├── drifting/
│   │   ├── loss.py                   # Joint kernel, drift fields, losses, and metrics
│   │   ├── mace_features.py          # Frozen MACE configuration and feature extraction
│   │   ├── mace_pt/                  # Local external weights, ignored by Git
│   │   └── distributed_objective.py  # Joint multi-GPU candidate objective
│   ├── generator/
│   │   ├── generator.py              # Three-branch local equivariant generator
│   │   └── core/                     # Consolidated equivariant operators
│   ├── condition/
│   │   ├── schema.py                 # Strict factorized-condition schema
│   │   └── encoder.py                # Periodic primitive-cell condition encoder
│   ├── initialize/
│   │   ├── generate_random_inputs.py # R0 sampling, coverage checks, and plots
│   │   ├── inputs_to_structures.py   # NPZ-to-ASE structure conversion
│   │   └── r0_resampling.py          # Online resampling by system and epoch
│   └── utils/
│       ├── data_loader.py                       # Conditional data loading and batching
│       ├── train.py                             # Single-GPU training
│       ├── multi_gpu_training.py                # Multi-GPU training
│       ├── generate_structures.py               # One-pass checkpoint inference
│       ├── structure_validation.py              # Periodic geometry and deduplication
│       └── screening_evaluation_metrics.py      # Screening and coverage metrics
```

The historical development tree included tests for condition permutation
invariance, factor sensitivity, rotational equivariance, and backpropagation.
Not all historical test assets are included in this public snapshot.

To plot training curves, run the following command from `AdsDrift/plot`:

```bash
python plot_train.py /path/to/test_or_run_directory
```

The script reads `metrics.jsonl` from the specified directory and creates or
updates `training_dashboard.png` in the same directory. Its six panels show the
raw drift loss `mean(V²)`, mode-attraction metrics, normalized optimization
loss, structural displacement, gradients and learning rate, and training
throughput. The default smoothing is a sliding median over 51 records and can
be changed with `--window`. If `training_metrics_summary.json` contains oracle
displacement statistics, the corresponding reference lines are shown.
Otherwise, they are omitted. The plotting script does not load checkpoints or
modify statistics files.

The training data flow is:

```text
condition c + random noise --initialization--> R0 --Gθ--> generated coordinates R_hat
                                                       |
                                                frozen MACE phi
                                                       |
                                            generated features phi(R_hat)
                                                       |
low-energy endpoints --pre-extraction--> positive features
                                                       |
                                           joint distance and drift field
                                                       |
                                 stop-gradient targets and joint feature loss
                                                       |
                   one dL/dR -> translation/rotation/internal weighting -> update θ
```

`AdsorptionDriftingObjective` in `model.py` combines the generator, the frozen
feature encoder, and the Drifting loss.
`build_model(generator_config, drifting_config, coordinate_gradient_balancing_config)`
accepts generator, Drifting, and gradient-balancing settings and loads the
configured MACE encoder.

The shared joint kernel uses four feature branches to compute sample weights
and mode-attraction fields:

$$
\{h_{\rm ads}^{(0)},h_{\rm ads}^{(1)},
  h_{\rm surf}^{(0)},h_{\rm surf}^{(1)}\}
\longrightarrow (L_{\rm ads},L_{\rm surf}).
$$

Only the vector-Jacobian product (VJP) from loss to coordinates is routed by
branch:

$$
\frac{\partial L}{\partial R_i}=
\begin{cases}
\partial L_{\rm ads}/\partial R_i,&i\in\text{adsorbate},\\
\partial L_{\rm surf}/\partial R_i,&i\in\text{movable surface},\\
0,&i\in\text{fixed/padding}.
\end{cases}
$$

The implementation performs one MACE forward pass. The two branches obtain
their coordinate VJPs separately and pass them to the generator through a
straight-through proxy scalar whose numerical value equals the original loss.
Logs additionally store both branch losses and the norms and ratios of the two
masked cross-gradients.

## 2. Conditions, learned variables, and tensor shapes

Following `test_6`, `test_18` expresses each condition as a product of
construction factors rather than an indivisible system identifier:

$$
c=(S_{\rm prim},hkl,s_{\rm term},b,M,n_{\rm layer},v_{\rm vac},\varepsilon,A).
$$

Here, $S_{\rm prim}=(Z_{\rm prim},F_{\rm prim},L_{\rm prim})$ contains the
elements, fractional coordinates, and lattice of the standardized primitive
cell. The reduced Miller index $hkl$ is defined relative to the standardized
conventional cell $L_{\rm conv}$. The variables $s_{\rm term}$ and $b$ specify
the termination shift and the selected surface side. $M$ is a nonsingular
three-dimensional integer matrix mapping the primitive cell to the oriented,
expanded bulk lattice. The remaining factors specify the slab-layer count,
vacuum thickness, strain, and adsorbate identity. `condition_factors.json`
stores both $L_{\rm prim}$ and $L_{\rm conv}$, preventing a conventional cell
from being mislabeled as a primitive cell or a Miller index from being
interpreted in the wrong reciprocal basis. "Factorized" means that these
variables can be controlled separately through the interface; it does not
assume statistical independence in the data.

The complete slab $(Z,t,L,R_0)$ remains the local physical graph input, but it
no longer carries the entire condition representation:

$$
c\xrightarrow{\text{surface construction}}S_{\rm slab},\qquad
(S_{\rm slab},R_0,c)\xrightarrow{G_\theta}\widehat R.
$$

The current model generates positions, orientations, and surface relaxation.
It does not generate elements, primitive cells, Miller indices, or supercell
matrices. These conditions are supplied at inference time and may be scanned
compositionally in future work.

| Input | Shape | Meaning |
| --- | --- | --- |
| `atomic_numbers` | [B,N] | Atomic number of each atom |
| `roles` | [B,N] | padding=0, fixed=1, movable surface=2, adsorbate=3 |
| `atom_mask` | [B,N] | Valid-atom mask |
| `cell` | [B,3,3] | Full cell vectors in Å |
| `r0_positions` | [B,G,N,3] | G random initial structures per condition, in Å |

Each system directory must also contain `condition_factors.json`, which
explicitly records the primitive cell, facet, termination, supercell matrix,
layer count, vacuum, strain, and adsorbate element sequence. See
`data/condition_factors.example.json` for an example. Legacy data without this
file fail immediately. `test_18` does not infer these variables from the final
slab because the inverse mapping is generally non-unique for reconstructions,
alloys, and distinct terminations.

OC20 tags 0, 1, and 2 map to internal roles 1, 2, and 3, respectively. The
current training loader requires B=1. G defaults to 100 and is configurable in
YAML.

The generator merges B and G into B×G independent structures. Attention occurs
between atoms within each structure. The 100 candidates do not attend to one
another inside the generator, but they jointly participate in the downstream
distribution comparison used by the Drifting loss.

## 3. Random initialization R0

Initialization reads frame 0 of a reference trajectory. All reference-surface
atoms remain fixed at their input positions, while the adsorbate is translated
and rotated as a rigid body to produce valid initial structures that cover the
surface.

Each candidate uses a seven-dimensional Gaussian proposal:

$$
z\sim\mathcal N(0,I_7),\qquad u_k=\Phi(z_k),\quad k=1,2,3.
$$

The first three components are mapped through the Gaussian cumulative
distribution function to uniform variables for in-plane fractional coordinates
and adsorption height. By default, equal-area stratification divides G=100
samples across a 10×10 grid, with one accepted site per stratum. Adsorption
height is divided into 100 strata and randomly assigned to the lateral strata.
The default height range is 1.2–3.0 Å, measured along the surface normal from
the highest surface atom to the lowest adsorbate atom.

The remaining four components are normalized to a unit quaternion, producing a
uniform SO(3) rotation. Initialization preserves the adsorbate's internal
relative geometry and randomizes only its global orientation. Rotation has no
effect on a single-atom adsorbate. Molecular torsions are not sampled
separately.

Sampling also checks contacts across periodic boundaries, vacuum clearance,
and lateral deduplication. The default minimum lateral separation is 0.1 Å.
After stratification and rejection sampling, R0 is a geometrically constrained
distribution rather than a collection of independent Gaussian atomic
coordinates.

`r0_resampling.py` derives deterministic seeds from the system identifier,
base random seed, and epoch, then rebuilds R0 from the saved reference structure
and sampling metadata. It validates the reference-file hash, atom order, and
cell. Online mode produces 100 new inputs per epoch, enabling the model to learn
the mapping from previously unseen initializations.

## 4. Generator Gθ

See the [generator README](model/generator/README.md) for the full equations,
local-edge definitions, and output constraints.

### 4.1 SO(3) atomic representations

Each node stores irreducible representations from $l=0$ through
$l=l_{\max}$:

$$
X_i\in\mathbb R^{(l_{\max}+1)^2\times C},\qquad
X_{i,0,:}=E_Z(Z_i)+E_{\mathrm{role}}(t_i)+B_{t_i}(c).
$$

Initialization also adds geometric edge embeddings generated from radial
distance bases and Wigner-D rotations on a unified all-atom periodic graph,
including the scalar components. This behavior is retained from the earlier
implementation, so interface geometry is present before the first layer. The
network does not embed absolute Cartesian coordinates; global translation
therefore does not change the predicted displacements.

### 4.2 Three-branch local equivariant blocks

Each layer contains three parameter-distinct EquiformerV3 attention paths:

```text
full slab --K/V--> full-slab queries            (slab self-attention)
adsorbate --K/V--> adsorbate queries            (adsorbate self-attention)
full slab --K/V--> adsorbate queries --+
adsorbate --K/V--> full-slab queries   --+--     (shared interface cross-attention)
```

Fixed-layer atoms act as queries, keys, and values. Every valid atom participates
in attention and FFN residual updates, while padding does not. Updated
fixed-layer features continue into the next layer, but their coordinates remain
fixed by the output mask. All four layers use smooth radial cutoffs, SO(2) graph
attention, Wigner-D rotations, and SwiGLU-$S^2$ feed-forward layers. The default
settings are $C=128$, $l_{\max}=3$, and $m_{\max}=2$. Computation is restricted
to periodic edges within the relation-specific cutoffs; there is no global
$N^2$ attention or global mode-routing mechanism. Role-specific FiLM
conditioning for fixed, movable-surface, and adsorbate atoms is applied only to
the $l=0$ scalar channels, preserving the rotation rules of higher-order
tensors.

Within each layer, SS and AA are computed in parallel and applied as residual
updates. The updated features are normalized again and passed through the
existing FiLM transformation before SA reads them. The SA residual is applied
before the FFN. Both attention normalization steps share the existing
parameters, and the SS, AA, and SA residual coefficients remain
$1/\sqrt2$. Parameter counts, initialization weights, neighbor graphs, and the
single coordinate readout are unchanged; only the feature-update order differs.

### 4.3 Three geometric output heads

| Atomic subset | Output | Constraint |
| --- | --- | --- |
| Fixed layer | Input coordinates | No movement |
| Movable surface | Per-atom 3D residual | Read from each atom's $l=1$ features; scale 0.5 Å |
| Adsorbate as a whole | One 3D center displacement | Mean adsorbate $l=1$ output; scale 3 Å |
| Adsorbate internal geometry | Per-atom 3D residual | Mean residual removed; scale 1 Å |

For adsorbate atoms, the output is

$$
\hat R_i=\operatorname{wrap}_{ab}(\bar R_0+\Delta c)
 +(R_{0,i}-\bar R_0)+\Delta r_i,
\qquad\sum_{i\in\mathrm{ads}}\Delta r_i=0.
$$

Only the molecular center is wrapped into the in-plane periodic cell. Individual
atoms are not wrapped, preserving molecular continuity across cell boundaries.
Center and internal displacements are separated to avoid duplicating the
global translational degree of freedom. The listed scales are output
multipliers, not hard displacement limits.

The three output heads use shared linear projections only along the channel
dimension and do not mix the three spatial components of $l=1$. Coordinate
residuals therefore rotate with the structure. The output heads are initialized
to zero, so the initial output is the periodic equivalent of R0. The internal
residual of a single-atom adsorbate is always zero. One generator forward pass
produces the final candidate coordinates. MACE is not required for
generator-only structure generation after training.

## 5. Frozen MACE feature space

Training uses MACE-MH-1 with the `oc20_usemppbe` head. The model reads
512-channel features from both interaction layers. MACE parameters remain
frozen and the encoder stays in evaluation mode, while the graph from generated
coordinates to features remains differentiable so that gradients can propagate
back into the generator.

| Feature branch | Per-structure shape | Readout |
| --- | --- | --- |
| `scalar_ads` | [4,512] | Adsorbate scalar mean and standard deviation from two layers |
| `message_l1_ads` | [4,512] | Mean and standard deviation of adsorbate l=1 message magnitudes from two layers |
| `scalar_movable` | [2,M,512] | Movable-surface scalar features from two layers, preserving atom identity |
| `message_l1_movable` | [2,M,512] | Movable-surface l=1 message magnitudes from two layers |

M is the number of movable surface atoms. The $l=1$ readout uses the Euclidean
norm of the three components and currently does not retain the full directional
vector. Statistical pooling gives the adsorbate a fixed-size representation,
whereas movable-layer features preserve the fixed within-system atom order.

The loss uses at most 4+4+2+2=12 feature groups. Standard-deviation groups
degenerate for single-atom adsorbates and are skipped when they fall below the
feature-scale threshold. The earlier single-O system used eight groups in
practice. Fixed-layer atoms have no separate loss readout, but still provide
environmental context in the MACE and generator graphs.

Positive-sample features are precomputed during data preparation. Generated
sample features are computed online at every step. All positive branches must
refer to the same endpoint index, preventing the adsorbate and surface features
of different configurations from being mixed.

## 6. Drifting loss

Each system is evaluated independently. Generated samples form the negative
set, and positive samples are low-energy endpoints from the same system.
Positive and negative samples are not matched across different catalyst
systems.

### 6.1 Joint distance

For feature group $g$, the scale $s_g$ is estimated from the mean
generated-to-positive distance in the current batch. Each group is also divided
by the square root of its channel count so that groups contribute comparably to
the joint distance:

$$
\tilde f_g=f_g/s_g,\qquad
j=\frac{1}{\sqrt Q}\operatorname{concat}_g
       \left(\tilde f_g/\sqrt{d_g}\right).
$$

Q is the number of valid feature groups, and $d_g$ is the flattened dimension
of group $g$. Gradients through the feature scales and joint-kernel weights are
stopped, while gradients through the generated features are retained. The
larger dimension associated with multiple movable atoms therefore does not
automatically increase a group's weight in proportion to its channel count.

### 6.2 Attraction and repulsion

At each temperature $\tau$, logits are constructed from Euclidean distances in
the joint feature space:

$$
\ell_{ij}=-\|j_i-j_j\|_2/\tau,\qquad
A_{ij}=\sqrt{\operatorname{softmax}_{\rm row}(\ell)_{ij}
                  \operatorname{softmax}_{\rm col}(\ell)_{ij}}.
$$

The columns contain both positive and generated samples. The negative
self-weight of each generated sample is set to zero. Cross-normalization of
positive and negative mass yields shared $W^+$ and $W^-$, which are then used
for every feature group:

$$
V_g^{(\tau)}=W^+\tilde F_g^+-W^-\tilde F_g^-.
$$

Positive samples provide attraction, and other generated samples provide
repulsion. Repulsion can help preserve diversity, but finite training does not
guarantee equal generation frequency for every mode.

### 6.3 Optimization objective and logging

The drift field at each temperature is normalized by its own RMS before the
fields are summed to construct a stop-gradient target:

$$
T_g=\operatorname{sg}\!\left(\tilde F_g^-
 +\sum_\tau\frac{V_g^{(\tau)}}{\operatorname{RMS}(V_g^{(\tau)})}\right),
\qquad
L=\sum_g\operatorname{mean}\!\left[(\tilde F_g^--T_g)^2\right].
$$

`raw_drift_loss`, used to monitor convergence, is
$\operatorname{mean}(V^2)$ before drift RMS normalization, averaged over
feature groups and temperatures. Here, V is computed in the feature-scale
normalized space. `loss` is the normalized optimization objective above, so
the two curves need not have the same scale or trend.

The three dimensionless kernel temperatures are set directly in YAML as
`drifting.temperatures: [0.02, 0.05, 0.2]`, following the dataset calibration
recorded in the archived development report. Training no longer reads
`temperature_calibration.json` or supports `temperatures: auto`. The objective
contains no direct paired-coordinate supervision, energy MAE, or force MAE.

### 6.4 Adsorbate coordinate-gradient projection and weighting

The joint feature loss retains all adsorbate-surface interactions but first
computes one MACE coordinate VJP with respect to isolated coordinate leaf
tensors:

$$
g=\partial_R(L_{\rm ads}+L_{\rm surf}).
$$

The adsorbate gradient is decomposed orthogonally into translation,
infinitesimal rigid-body rotation, and internal deformation:

$$
g_{\rm ads}=g_t+g_r+g_i.
$$

The default YAML allocation assigns 20% of gradient energy to translation, 20%
to rotation, and 60% to internal deformation. Each nondegenerate component is
rescaled by its own norm, and the combined field preserves the total energy of
the candidate's original MACE adsorbate gradient. Surface gradients are left
unchanged.

The weighted field is detached and passed to the generator through a linear
proxy loss. Only one first-order MACE coordinate gradient is required; no
Hessian is computed. Startup logs report
`coordinate_gradient_routing=mace_projected_balanced_first_order`, set
`coordinate_routing_enabled=1`, and record component ratios and scale factors
before and after weighting.

## 7. Training data and sampling unit

The complete preprocessed dataset is not included in this repository. The
original development environment used the following layout:

```text
dataset/drift_oc20dense_0p50eV_v1
```

This version selected 905 systems with at least two low-energy modes from 973
audited systems. It contains 15,207 modes, 31,147 endpoints, and 90,500
pregenerated R0 structures. The acceptance window is 0.50 eV above the lowest
observed accepted endpoint within each system. Modes are defined by the audit's
`symmetry_rmsd_0.1A` clustering key.

Each batch contains one system and generates 100 candidates by default. Each
epoch visits every configured system once in random order. A full 905-system
configuration therefore has 905 batches per epoch, whereas a single-system
configuration has one batch per epoch.

By default, one endpoint is sampled from each mode. If a mode has multiple
members, one member is chosen randomly. All four feature branches use the same
sampled endpoint indices. Positive modes therefore have equal sampling
opportunity during training, but generation frequencies are not thermodynamic
occupancies. The objective is to cover the low-energy modes, not to produce
exactly equal output counts.

Online resampling must be enabled explicitly in single-GPU configurations:

```yaml
data:
  r0_per_system: 100
  positives_per_mode: 1
  max_positive_modes: null
  online_r0_resampling:
    enabled: true
    base_seed: 20260907
    max_attempts_per_sample: 100
    max_bank_retries: 4
```

The fixed training configurations enable this option. If disabled, training
uses the R0 structures cached in the dataset. The current multi-GPU runner
retains the sharded fixed-R0 implementation; online resampling is implemented
in the single-GPU runner.

## 8. Data preparation

After constructing the structure and feature banks, each system must be given
trusted construction factors. For example:

```bash
python -m AdsDrift.data.prepare_condition_factors \
  --system-directory /path/to/systems/0_1190_0 \
  --primitive-structure /path/to/bulk-primitive.cif \
  --conventional-structure /path/to/bulk-conventional.cif \
  --miller 1 1 1 --termination-shift 0.25 --top \
  --supercell-matrix 3 0 0 0 3 0 0 0 1 \
  --slab-layers 4 --vacuum 15
```

The primitive cell, termination shift, and supercell matrix must come from the
data-generation record or another trusted source. They cannot be inferred
reliably from the appearance of the final slab. The script validates primitive
cell periodicity, the Miller index, the integer matrix, and adsorbate atom order.

`data/prepare_dataset.py` consolidates three historical data scripts. Its
subcommands can be run independently:

| Subcommand | Purpose | Main inputs |
| --- | --- | --- |
| `index` | Generate system indices, seeds, and counts | audit-root, trajectory-root, tag-mapping |
| `trajectories` | Copy one representative full trajectory and clean surface per mode | audit-root, trajectory-root, mapping-root |
| `positives` | Export one representative final frame per mode for one system | system-dir, audit-json, tag-mapping |

The shared `--window-ev` option defaults to 0.5, and `--cluster-key` defaults to
the symmetry-clustering key above. The legacy spellings `--window` and
`--window-eV` are also accepted. `index` accepts `--expected-systems 905` to
validate the full system count, while `trajectories` accepts
`--expected-samples 15274` to validate the representative-trajectory count.
Omit these checks for subsets.

`index` only constructs the index and leaves `complete: false` in the manifest.
It does not generate R0 or extract MACE features. Representative-endpoint
export also does not expand every mode member. A complete training bank still
requires initialization and feature preparation. This repository contains the
initialization utilities in `model/initialize/`; historical experiment
directories and uncurated feature-export scripts are not part of this public
snapshot.

```bash
python model/AdsDrift/data/prepare_dataset.py index \
  --audit-root analysis/oc20_dense_positive_audit_20260905/full_results_0p50eV \
  --trajectory-root data/oc20_dense_extracted/trajs \
  --tag-mapping data/oc20_dense_mappings_extracted/oc20dense_tags.pkl \
  --output-root dataset/drift_oc20dense_0p50eV_v2 \
  --minimum-modes 2 --random-count 100 --expected-systems 905

python model/AdsDrift/data/prepare_dataset.py positives \
  --system-dir test_1/model/0_1190_0 \
  --audit-json analysis/oc20_dense_positive_audit_20260905/local_results/systems/0_1190_0.json \
  --tag-mapping analysis/oc20_dense_positive_audit_20260905/mappings/oc20dense_tags.pkl \
  --output-dir model/AdsDrift/data/positive_0p50eV
```

Indexing and endpoint export refuse to overwrite existing results. Original
trajectory export can resume when invoked with the same input configuration;
it rejects conflicting configurations and same-size files with different
contents.

## 9. Installation and usage

The top-level Python package is named `AdsDrift`, and internal imports use
`from AdsDrift... import ...`.

Clone the repository, install its dependencies, and add the repository's parent
directory to the Python search path:

```bash
git clone https://github.com/Prome-theus1/AdsDrift.git
cd AdsDrift
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"
```

You can then run `python -m AdsDrift...` from the parent directory. Configuration
arguments must point to the actual files. Relative model and data paths are
resolved from the runtime working directory. For GPU training, install the
PyTorch build matching the target CUDA environment before installing the other
dependencies.

```bash
# Unified entry point
python -m AdsDrift.model.model --help

# Single-GPU training; requires the dataset and frozen MACE weights
python -m AdsDrift.model.model train \
  --config AdsDrift/config/train_single_0_1190_0.yaml

# Multi-GPU training
torchrun --nproc_per_node=2 -m AdsDrift.model.model train-distributed \
  --config AdsDrift/config/train_single_0_1190_0.yaml

# One-pass generation from an existing condition bank
python -m AdsDrift.model.model sample \
  --checkpoint /path/to/epoch_5000.pt \
  --system-directory dataset/drift_oc20dense_0p50eV_v1/systems/0_1190_0 \
  --output-directory model/AdsDrift/runs/inference_0_1190_0

# Random initialization with the default local example
python -m AdsDrift.model.initialize.generate_random_inputs --num-samples 100

# Structure-conversion help
python -m AdsDrift.model.initialize.inputs_to_structures --help
```

`model/model.py` can also be executed directly. The `--help` output of each
subcommand documents the full single-GPU training, distributed training, and
sampling interfaces.

The current `sample` command reuses the complete condition-bank loader, so its
input directory still requires positive-sample metadata and feature files even
though the generator forward pass itself does not consume the positive samples.

Training paths are resolved from the runtime working directory. MACE weights
are not distributed with this repository. The compatibility default is
`model/drifting/mace_pt/macemh1model.pt`; alternatively, set
`ADSDRIFT_MACE_CHECKPOINT=/path/to/mace-mh-1.model`. The head, SHA256 checksum,
microbatch size of 8, activation checkpointing, and numerical thresholds are
defined centrally by `MACEFeatureConfig`. Training checkpoints record the
effective MACE configuration for provenance.

Outputs are written to `training.run_directory`, which defaults to
`model/AdsDrift/runs/production`; `--run-directory` overrides it. The deprecated
`paths.run_directory` setting is no longer used. Default configurations enable
TF32, a MACE microbatch size of 8, and activation checkpointing.

The public snapshot preserves generator parameter names, tensor shapes, and the
saved `generator` state-dict format. Existing checkpoints can therefore load
the generator from its new location, although legacy configuration paths must
be updated.

Learning-rate warmup is controlled by `training.warmup_epochs`, which defaults
to 10 and can be disabled with 0. Internally, the duration is
`warmup_epochs × optimization steps per epoch`. Each system contributes one
update, so the number of optimization steps equals the number of systems. The
same rule is used for single- and multi-GPU training and is not divided by the
GPU count. Cosine decay follows warmup. If the total number of epochs does not
exceed the warmup duration, the entire run remains in warmup. Replace the
legacy `warmup_steps` option with `warmup_epochs`.

## 10. Validation status and current limitations

Some historical experimental evidence and reproduction materials are not part
of this public snapshot. The repository retains the current model
implementation, configurations, and data-preparation entry points. A priority
for future maintainers is to add a small redistributable test dataset and an
end-to-end reproduction workflow.

A generated distribution that approaches the reference endpoints does not
imply that every generated structure is a strict stationary point or a
mathematical global minimum. Validation should also examine movable-surface
errors, collisions, residual forces, energies, and performance on unseen
systems. Independent energy and force validation remains future work. The
current training stage optimizes only the distribution objective in the frozen
feature space described above.

## License

Original AdsDrift code is licensed under the
[Apache License 2.0](LICENSE). Components adapted from EquiformerV3, FAIR
Chemistry, and e3nn retain their MIT licenses and attribution; see
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External
model weights and datasets are not covered by this repository's license.
