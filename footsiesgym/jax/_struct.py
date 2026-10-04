"""Immutable classes that are also JAX pytrees.

``@struct`` turns a class into a frozen dataclass whose fields are pytree
leaves, so instances can be passed through ``jax.jit``, ``jax.vmap`` and
``jax.lax`` loops, and their methods vmapped as plain functions
(``jax.vmap(Fighter.update_movement)(fighters)``). ``Struct`` adds the few
operations the game code needs on whole instances.
"""

import dataclasses

import jax
import jax.numpy as jnp


def struct(cls):
    """Frozen dataclass registered as a JAX pytree (every field is a leaf)."""
    cls = dataclasses.dataclass(frozen=True)(cls)
    fields = [f.name for f in dataclasses.fields(cls)]
    return jax.tree_util.register_dataclass(cls, data_fields=fields, meta_fields=[])


class Struct:
    def replace(self, **changes):
        """A copy with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def where(self, cond, otherwise):
        """``self`` where ``cond`` holds, else ``otherwise`` (leafwise)."""
        return jax.tree.map(lambda a, b: jnp.where(cond, a, b), self, otherwise)

    def at(self, i):
        """Element ``i`` along the leading axis of every field."""
        return jax.tree.map(lambda x: x[i], self)

    @classmethod
    def stack(cls, *items):
        """Stack instances along a new leading axis."""
        return jax.tree.map(lambda *xs: jnp.stack(xs), *items)
