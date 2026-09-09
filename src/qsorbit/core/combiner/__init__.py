"""Non-coherent combining across receive branches.

One of the ``core/`` packages the original layout reserved, filled in by
Chunk E. The layering matches :mod:`qsorbit.core.sdr` and
:mod:`qsorbit.core.rotor`: a module holding the decision logic, and a
package that re-exports it, so a second combining strategy arrives as
another module here rather than as a rewrite of this one.

**Non-coherent is the design, not a limitation.** Two dongles have two
crystals and two PLLs, so two branches' recovered audio is
phase-unrelated and summing it would not reinforce anything. Selection
diversity picks one at a time instead, which reduces the whole problem
to *when to switch* -- and that question is answered by a margin
measured at the bench rather than assumed. See
:mod:`qsorbit.core.combiner.selector` for where the number comes from.
"""

from qsorbit.core.combiner.selector import (
    DEFAULT_MARGIN_DB,
    BranchReading,
    BranchSelector,
    SelectorStats,
    pair_simultaneous,
)

__all__ = [
    "DEFAULT_MARGIN_DB",
    "BranchReading",
    "BranchSelector",
    "SelectorStats",
    "pair_simultaneous",
]
