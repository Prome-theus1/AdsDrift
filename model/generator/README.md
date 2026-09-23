# AdsDrift EquiformerV3 Local Equivariant Generator

This directory contains the historical `test_18` generator. It inherits from
`test_16` and changes only the within-block aggregation order to
"self-attention residuals, then cross-attention residuals, then FFN."
Initialization and the parameter count are unchanged. Fixed-layer atoms remain
queries, keys, and values and their features are updated, but their coordinates
stay fixed. The model entry point is `generator.py`. The factorized-condition
schema and encoder are in the adjacent `../condition/` directory. Low-level
operators consolidated from EquiformerV3 are organized under `core/`.

## 1. Inputs and outputs

Given a factorized condition

$$
c=(S_{\rm prim},hkl,s_{\rm term},b,M,n_{\rm layer},v_{\rm vac},\varepsilon,A),
$$

and a bank of random initial structures under that condition,

$$
R_0\in\mathbb R^{B\times G\times N\times3},
$$

the generator produces the output in one forward pass:

$$
\widehat R=G_\theta(R_0;Z,t,L,c).
$$

$Z$, $t$, and $L$ describe the fully constructed slab graph. The condition $c$
explicitly records the primitive cell, facet, termination, supercell matrix,
layer count, vacuum, strain, and adsorbate. The complete graph provides the
local physics, while $c$ exposes the underlying construction variables to the
network. There is no system-ID embedding.

The condition encoder builds a minimum-image periodic graph from primitive-cell
fractional coordinates and passes messages using distance, displacement along
the surface normal, and in-plane distance. The pooled primitive-cell
representation is concatenated with invariants including primitive and actual
slab lattice metrics, the reduced Miller index, interplanar spacing,
termination phase, the supercell matrix, and its physical metrics. Permuting
primitive-cell atoms or globally rotating the structure does not change the
condition scalars.

The encoder outputs an initial role bias $B_t(c)$ and a scale and shift
$(\gamma_t^{(l)},\beta_t^{(l)})$ for each role in every layer:

$$
X_{i,0}^{(0)}=E_Z(Z_i)+E_t(t_i)+B_{t_i}(c),
$$

$$
\widetilde X_{i,0}^{(l)}=X_{i,0}^{(l)}
\left[1+s_c\tanh\gamma_{t_i}^{(l)}(c)\right]
+s_c\beta_{t_i}^{(l)}(c).
$$

Only the $l=0$ scalars are modulated before entering slab self-attention,
adsorbate self-attention, and interface cross-attention. The transformation
rules of tensors with $l>0$ remain unchanged, preserving SE(3) equivariance of
the coordinate output.

The model changes only the coordinates of role-2 and role-3 atoms:

$$
\widehat R_i=R_{0,i},\qquad t_i=1.
$$

## 2. Periodic local graphs

For source atom $j$, target atom $i$, and periodic translation
$n\in\mathbb Z^3$, define the directed edge vector

$$
r_{ji}^{(n)}=R_{0,j}+n^\top L-R_{0,i},
\qquad d_{ji}^{(n)}=\|r_{ji}^{(n)}\|_2.
$$

By default, only $n_a,n_b\in\{-1,0,1\}$ and $n_c=0$ are enumerated. At most
the 24 nearest edges per query are retained for each relation. The three edge
types are

$$
\mathcal E_{SS}=\{j\in S_{\rm all}\to i\in S_{\rm all}:d_{ji}<r_S\},
$$

$$
\mathcal E_{AA}=\{j\in A\to i\in A:j\ne i,\ d_{ji}<r_A\},
$$

$$
\mathcal E_{SA}=\{S_{\rm all}\to A\}\cup\{A\to S_{\rm all}\},
\qquad d_{ji}<r_C.
$$

The default cutoffs are $r_S=6.0$ Å, $r_A=4.5$ Å, and $r_C=6.0$ Å. The full
slab contains both fixed and movable layers. Fixed-layer atoms participate as
queries in slab self-attention and adsorbate-to-slab cross-attention and remain
subject to the same distance cutoffs and neighbor limits.

## 3. SO(3) input representation

Each node feature is

$$
X_i\in\mathbb R^{(l_{\max}+1)^2\times C},
\qquad C=128,\quad l_{\max}=3,\quad m_{\max}=2.
$$

The scalar channels are initialized as

$$
X_{i,l=0}^{(0)}=E_Z(Z_i)+E_t(t_i)+B_{t_i}(c).
$$

Local edge distances are expanded over 64 Gaussian radial basis functions:

$$
g_k(d)=\exp[-\gamma(d-\mu_k)^2].
$$

The radial network generates $m=0$ coefficients in an edge-aligned local frame.
Inverse Wigner-D rotation then produces the spherical edge embedding:

$$
X_i^{(0)}\leftarrow X_i^{(0)}+
\frac1{\sqrt{K}}
\sum_{(j\to i)\in\mathcal E_{\rm all}}
D^{(l)}(Q_{ji})^{-1}\rho_l(g(d_{ji}),E_Z^{\rm src}(Z_j),E_Z^{\rm dst}(Z_i)).
$$

$Q_{ji}$ aligns the edge direction with the local axis. Absolute coordinates do
not enter the scalar MLP, so the network is invariant to global translation.
The spherical-harmonic and Wigner-D path ensures that higher-order features
transform according to their irreducible representations under rotation. This
initialization is retained from `test_16`. The geometric edge embedding covers
$l=0$ through $l_{\max}$, so the all-atom graph introduces interface
environment information before the first layer. Initialization is not isolated
in this version.

## 4. EquiformerV3 graph attention

For relation $q\in\{SS,AA,SA\}$, node features first undergo equivariant
normalization and the existing condition modulation $C_k$, which changes only
$l=0$:

$$
\bar X=C_k\!\left(\operatorname{EqNorm}(X)\right).
$$

For every edge, source and target representations are rotated into an
edge-aligned frame and combined with element embeddings and radial weights:

$$
M_{ji}^{q}=\operatorname{SO2Conv}_2\!\left(
\sigma_{S^2}\!\left[
\operatorname{SO2Conv}_1
\left(D(Q_{ji})\,\Psi(\bar X_j,\bar X_i;d_{ji})\right)
\right]\right).
$$

When `use_add_merge=true`, $\Psi$ applies separate radial channel weights to
the source and target and adds the results, avoiding the approximately twofold
SO(2) input cost of concatenation. $\sigma_{S^2}$ is EquiformerV3's
SwiGLU-$S^2$ activation.

Attention weights are rotationally invariant:

$$
\alpha_{ji,h}^{q}=
\operatorname{softmax}_{j\in\mathcal N_q(i)}
\left(a_h^\top u_{ji,h}^{q}\right).
$$

The multi-head values are weighted, rotated back to the global frame, and
aggregated by target node:

$$
A_i^q=\sum_{j\in\mathcal N_q(i)}
D(Q_{ji})^{-1}\left(\alpha_{ji}^q M_{ji}^q\right).
$$

The three branches do not share parameters. The two cross-attention directions
share one interface parameter set. Let $m=\mathbf1_{S_{\rm all}\cup A}$. Each
layer first applies the self-attention residual updates:

$$
U = X + \frac{m}{\sqrt2}\odot
\left[A^{SS}(\bar X)+A^{AA}(\bar X)\right].
$$

$$
\bar U=C_k(N_{\rm attn}(U)),\qquad
V=U+\frac{m}{\sqrt2}\odot A^{SA}(\bar U).
$$

Cross-attention reads the updated $U$, not the old $X$. The FFN is applied
afterward:

$$
X^+=V+m\odot\operatorname{FFN}_{\mathrm{SwiGLU}-S^2}
\!\left(C_k(N_{\rm ffn}(V))\right).
$$

Both $N_{\rm attn}$ applications reuse the same existing normalization
parameters, and $C_k$ reuses the existing FiLM transformation. Every attention
residual retains its original coefficient $1/\sqrt2$. No parameters are added,
and gradients are not truncated. The target sets of the two self-attention
paths do not overlap.

Four layers are stacked by default. Fixed-node features $X_i$ receive attention
and FFN residuals alongside all other valid nodes. Their self-updated
representations are read by cross-attention within the same layer and passed to
the next layer. Padding features are not updated. These updates affect hidden
representations rather than intermediate coordinates. The output heads modify
movable-surface and adsorbate coordinates only once, at the end of the network.

## 5. Smooth cutoffs

Distance-aware attention uses a fifth-order polynomial envelope. Let
$x=d/r_c$:

$$
e(d)=
\begin{cases}
1+ax^p+bx^{p+1}+cx^{p+2},&x<1,\\
0,&x\ge1,
\end{cases}
$$

where $p=5$, $a=-(p+1)(p+2)/2$, $b=p(p+2)$, and $c=-p(p+1)/2$. The envelope
contributes to the exponential weight used by softmax and multiplies the edge
embedding, causing interactions to decay smoothly to zero at the cutoff.

## 6. Reading coordinate displacements from l=1

After the final equivariant normalization, the model reads

$$
V_i=X_{i,l=1}\in\mathbb R^{3\times C}.
$$

Each output head applies a channel-shared linear combination:

$$
v_{i,k}=\sum_{c=1}^{C}w_cV_{i,kc},\qquad k\in\{x,y,z\}.
$$

The same $w_c$ is used for all three spatial components, preserving the
rotation transformation rule of $l=1$. The surface displacement is

$$
\Delta R_i^{S}=s_S v_i^S,\qquad i\in S_{\rm mov}.
$$

The adsorbate output is decomposed into global and internal changes:

$$
\Delta c=s_c\frac1{|A|}\sum_{i\in A}v_i^c,
$$

$$
\Delta r_i=s_r\left(v_i^r-\frac1{|A|}\sum_{j\in A}v_j^r\right),
\qquad \sum_{i\in A}\Delta r_i=0.
$$

The final adsorbate coordinates are

$$
\widehat R_i=
\operatorname{wrap}_{ab}(\bar R_0+\Delta c)
+(R_{0,i}-\bar R_0)+\Delta r_i,
\qquad i\in A.
$$

Only the adsorbate center is wrapped; individual atoms are not. This avoids
splitting a molecule that crosses a periodic boundary. The defaults are
$s_S=0.5$ Å, $s_c=3.0$ Å, and $s_r=1.0$ Å. All three heads are initialized to
zero, so the untrained model begins as an exact identity mapping.

## 7. Current limitations

- This is a one-pass conditional generator, not a force field or a multistep
  relaxation method.
- There is no global mode token or basin-routing mechanism. Multimodality comes
  entirely from distinct $R_0$ samples and their local interactions.
- Adsorbate conformations change through per-atom $l=1$ internal residuals,
  without explicit bond-length or bond-angle constraints. Geometric validity
  remains jointly constrained by the data, Drifting features, and structural
  checks.
- Legacy CatFlow checkpoints are incompatible with this architecture and should
  fail explicitly when loaded.

## 8. Default scale

The fixed-system configurations are stored in `../../config/`. They use the
same model settings: four layers, 128 channels, eight heads,
$l_{\max}=3$, $m_{\max}=2$, an FFN width of 256, and at most 24 neighbors per
relation. Only the data systems and output directories differ.
