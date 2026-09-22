"""Consolidated EquiformerV3-derived operator library for AdsDrift.

The package deliberately exposes only three implementation modules:

``geometry``
    Edge frames, Wigner-D matrices, and SO(2)/SO(3) operators.
``layers``
    Radial, normalization, activation, dropout, and native graph-softmax layers.
``attention``
    Edge embedding, equivariant graph attention, and equivariant feed-forward
    blocks used by the AdsDrift generator.

The upstream energy/force model wrappers and output heads are not part of this
package.  AdsDrift-specific composition lives in the parent ``generator.py``
module.
"""
