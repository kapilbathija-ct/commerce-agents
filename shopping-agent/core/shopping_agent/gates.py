# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The cart gates, and the writes they guard.

A cart write accepts only product ids that catalog or order tools returned this session
(or lines already in the cart), holds an add of a product that still has options to choose
and points at its variants, caps the resulting line quantity at the config's limit, and
reports any cap it applied; writes for one session are serialized because a turn's tool
calls run concurrently.

Forked from upstream. Three divergences, all recorded in ``DIVERGENCES.md`` at the repo
root, which is the list an upstream contribution reads from:

1. **The precedence ``or``-chain is gone.** ``gated_add_to_cart`` no longer decides the
   order of provenance, options and the capacity limits inside its own body. It receives a
   :class:`CartGateDecision` — the answer of whatever chain the deployment registered —
   and applies it. The predicates themselves (``check_provenance``, ``check_options``,
   :class:`QuantityAllowance`) are unchanged and are what a chain wraps, so nothing the
   reference gated has been re-implemented or dropped.
2. **``max_cart_lines`` and ``max_quantity_per_item`` return held outcomes, not errors.**
   A held outcome names the gate and the reason; an error does not, and the two are
   relayed to the model differently.
3. **The gate phase and the per-session cart lock are mutually exclusive at runtime.**
   ``cart_critical_section`` refuses to be entered while a gate phase is running, and
   ``gate_phase`` refuses to be entered while the critical section is held, so no gate can
   perform a platform read inside a section that serializes per session.

``REFERENCE_ORDERING`` keeps the upstream order available as one named, replaceable object
for a deployment that registers no chain of its own. It is the reference's behaviour, not
a second authority: exactly one decision source answers any single call, and a deployment
that supplies its own never reaches this one.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from commerce_common.streaming import AgentEvent, ToolOutcome

from .backend import StorefrontBackend
from .config import ShoppingAgentConfig
from .fencing import STOREFRONT_FENCE
from .serialization import cart_payload, cart_summary
from .types import Cart, Order, Product, ShoppingSessionContext, ShoppingSessionState

PROVENANCE_GATE = "provenance"
OPTIONS_GATE = "options"

#: The two capacity limits, named. They were anonymous ``ToolOutcome.error`` strings
#: upstream, which is exactly why they could not be held outcomes: a held outcome names
#: its gate, and these had no names to give.
CART_CAPACITY_GATE = "cart_capacity"
PER_ITEM_QUANTITY_GATE = "per_item_quantity"


def provenance_error(product_id: str) -> str:
    # get_product_details comes first in the hint: text search does not match ids, and
    # an empty search reads to the model as proof the product does not exist.
    return (
        f"product_id {product_id} was not returned by catalog or order tools in this "
        "session. Resolve it first: call get_product_details with this exact id (text "
        "search does not match product ids), or find it via search or order history, "
        "then add it using a product_id from those results."
    )


def check_provenance(state: ShoppingSessionState, product_id: str) -> ToolOutcome | None:
    """The held outcome when ``product_id`` has no session provenance, else None."""
    if product_id in state.seen_products:
        return None
    return ToolOutcome.held(PROVENANCE_GATE, provenance_error(product_id))


def options_error(product: Product) -> str:
    # Option names are catalog text arriving outside the fence: sanitized and kept
    # short. The values themselves are in the fenced record the model already holds.
    names = STOREFRONT_FENCE.sanitize_text(", ".join(product.options), max_chars=60)
    return (
        f"product_id {product.product_id} has options still to choose ({names}), so the "
        "cart takes one of its variants. Settle each option from what the customer said "
        "or the customer's profile, ask once with the values as chips when one is still "
        "open, then add the matching variant's product_id from the variants "
        "get_product_details returns for this id."
    )


def check_options(state: ShoppingSessionState, product_id: str) -> ToolOutcome | None:
    """The held outcome when the record for ``product_id`` is a family with options
    still to choose; its variants are what the cart takes."""
    product = state.seen_products.get(product_id)
    if product is None or not product.has_options:
        return None
    return ToolOutcome.held(OPTIONS_GATE, options_error(product))


def remember_order_items(state: ShoppingSessionState, orders: Sequence[Order]) -> None:
    """Items on the customer's own orders count as provenance, so a reorder needs no search."""
    state.remember_products(
        [
            Product(
                product_id=item.product_id,
                title=item.title,
                price=item.price,
                option_values=item.option_values,
                variant_of=item.variant_of,
            )
            for order in orders
            for item in order.items
        ]
    )


# -- the capacity limits, as arithmetic a chain can register -------------------------
#
# Divergence 2. The numbers, the headroom subtraction, the `allowed <= 0` rejection and
# the clamp wording are the reference's, moved out of `gated_add_to_cart`'s body and into
# a value so that the same arithmetic answers twice: once in the gate phase, above the
# lock, where a refusal costs no write; and once inside the critical section against the
# cart as it actually is, which is what keeps two concurrent adds from jointly exceeding
# the cap. Two applications of one computation, never two computations.


def cart_full_error(max_cart_lines: int) -> str:
    # Upstream: "The cart is full." — unnamed, and an error. The line count is added
    # because a held outcome states its reason and "full" is not a reason on its own.
    return (
        f"The cart is full: it already holds {max_cart_lines} different lines, which is "
        f"this store's limit. Nothing was added. Remove a line before adding another, or "
        f"add more of something the cart already has."
    )


def per_item_limit_error(max_quantity_per_item: int) -> str:
    return (
        f"This item is already at the per-item limit of {max_quantity_per_item}, so "
        f"nothing was added. Tell the customer the limit rather than retrying."
    )


def capped_disclosure(max_quantity_per_item: int) -> str:
    return f" (capped at the per-item limit of {max_quantity_per_item})"


@dataclass(frozen=True)
class QuantityVerdict:
    """What the allowance decided: a quantity to write, or a refusal naming its gate."""

    allowed: int
    requested: int
    max_quantity_per_item: int
    refusal_gate: str | None = None
    refusal_reason: str | None = None

    @property
    def refused(self) -> bool:
        return self.refusal_gate is not None

    def as_held(self) -> ToolOutcome | None:
        """Divergence 2, in one place: the refusal as a **held** outcome, never an error.

        One conversion, called by both applications of the allowance, so the pre-lock
        refusal and the in-lock refusal are the same bytes to the model.
        """
        if self.refusal_gate is None or self.refusal_reason is None:
            return None
        return ToolOutcome.held(self.refusal_gate, self.refusal_reason)

    @property
    def disclosure(self) -> str:
        """The clamp-and-disclose suffix, empty when nothing was clamped."""
        if self.refused or self.allowed >= self.requested:
            return ""
        return capped_disclosure(self.max_quantity_per_item)


@dataclass(frozen=True)
class QuantityAllowance:
    """The config's two cart limits as arithmetic, applicable against any cart."""

    max_quantity_per_item: int
    max_cart_lines: int

    @classmethod
    def from_config(cls, config: ShoppingAgentConfig) -> QuantityAllowance:
        return cls(
            max_quantity_per_item=config.max_quantity_per_item,
            max_cart_lines=config.max_cart_lines,
        )

    def for_add(self, cart: Cart, product_id: str, requested: int) -> QuantityVerdict:
        """An add: a new line needs a free slot, and an existing line needs headroom."""
        existing = next((i for i in cart.items if i.product_id == product_id), None)
        if existing is None and len(cart.items) >= self.max_cart_lines:
            return QuantityVerdict(
                allowed=0,
                requested=requested,
                max_quantity_per_item=self.max_quantity_per_item,
                refusal_gate=CART_CAPACITY_GATE,
                refusal_reason=cart_full_error(self.max_cart_lines),
            )
        headroom = max(0, self.max_quantity_per_item - (existing.quantity if existing else 0))
        allowed = min(requested, headroom)
        if allowed <= 0:
            return QuantityVerdict(
                allowed=0,
                requested=requested,
                max_quantity_per_item=self.max_quantity_per_item,
                refusal_gate=PER_ITEM_QUANTITY_GATE,
                refusal_reason=per_item_limit_error(self.max_quantity_per_item),
            )
        return QuantityVerdict(
            allowed=allowed,
            requested=requested,
            max_quantity_per_item=self.max_quantity_per_item,
        )

    def for_set(self, requested: int) -> QuantityVerdict:
        """An update sets a quantity rather than adding to one, so it clamps and never
        refuses — the reference's behaviour, unchanged."""
        return QuantityVerdict(
            allowed=min(requested, self.max_quantity_per_item),
            requested=requested,
            max_quantity_per_item=self.max_quantity_per_item,
        )


# -- what a write requires before it runs -------------------------------------------


class UngatedCartWrite(RuntimeError):
    """A cart write reached this module with no gate decision behind it.

    Raised rather than defaulted. A write that gated nothing is indistinguishable from a
    write every gate passed, which is the one failure a gate chain exists to make
    impossible — so the absence of a decision is a wiring error and says so.
    """


@runtime_checkable
class CartGateDecision(Protocol):
    """The answer of whatever chain the deployment registered, as a cart write needs it.

    Two members and no inheritance, so a deployment's own registry can satisfy it without
    importing anything from this package.
    """

    def held_outcome(self) -> ToolOutcome | None:
        """The refusal to relay, or ``None`` when every applicable gate passed."""
        ...

    @property
    def allowance(self) -> QuantityAllowance | None:
        """The capacity arithmetic to apply inside the critical section.

        Required for an add, which is the only write that can exceed a limit. ``None`` is
        accepted on update and remove.
        """
        ...


def require_decision(decision: CartGateDecision | None, tool: str) -> CartGateDecision:
    if decision is None:
        raise UngatedCartWrite(
            f"{tool} was called with decision=None. Every cart write takes the decision of "
            f"a registered gate chain; pass `REFERENCE_ORDERING.decide(...)` for the "
            f"upstream order, or your own chain's answer. There is deliberately no default: "
            f"an ungated write and a write every gate passed would otherwise be the same "
            f"call."
        )
    return decision


# -- the lock, and the boundary the gate phase sits above ---------------------------
#
# Divergence 3. The gates read the cart, compute, then write; a second mutation for the
# same session in the same gather must not interleave with that. Locks live only while
# held. What is new is that the two regions are now *named* and mutually exclusive, so
# "no gate performs a platform read inside the critical section" is enforced rather than
# reviewed.

_cart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

_gate_phase_running: ContextVar[bool] = ContextVar("shopping_agent_gate_phase", default=False)
_cart_lock_held: ContextVar[bool] = ContextVar("shopping_agent_cart_lock", default=False)


class GatePhaseInsideCartLock(RuntimeError):
    """A gate phase was entered while the per-session cart lock was held.

    A platform read there sits in a section serialized per session, against a drawer
    latency budget that does not accommodate it. The phase belongs above the lock.
    """


class CartLockInsideGatePhase(RuntimeError):
    """A cart write was attempted from inside a gate phase.

    A gate that writes is a gate that cannot be re-ordered: its effect would land before
    a higher-band gate had spoken, which is the ordering the chain exists to guarantee.
    """


def _cart_lock(session: ShoppingSessionContext) -> asyncio.Lock:
    lock = _cart_locks.get(session.session_id)
    if lock is None:
        lock = _cart_locks[session.session_id] = asyncio.Lock()
    return lock


def gate_phase_is_running() -> bool:
    """True while a gate phase is evaluating in this task."""
    return _gate_phase_running.get()


def cart_lock_is_held() -> bool:
    """True while this task holds a session's cart lock."""
    return _cart_lock_held.get()


@contextmanager
def gate_phase() -> Iterator[None]:
    """Mark a gate phase. Refuses to run inside the cart's critical section."""
    if _cart_lock_held.get():
        raise GatePhaseInsideCartLock(
            "a gate phase was entered while the per-session cart lock was held. Evaluate "
            "the chain above the lock: a gate's platform read in there serializes every "
            "write for that session behind it."
        )
    token = _gate_phase_running.set(True)
    try:
        yield
    finally:
        _gate_phase_running.reset(token)


@asynccontextmanager
async def cart_critical_section(session: ShoppingSessionContext) -> AsyncIterator[None]:
    """The per-session lock, with the cart read and the write inside it."""
    if _gate_phase_running.get():
        raise CartLockInsideGatePhase(
            "a cart write was attempted from inside a gate phase. A gate decides; it does "
            "not write."
        )
    async with _cart_lock(session):
        token = _cart_lock_held.set(True)
        try:
            yield
        finally:
            _cart_lock_held.reset(token)


def _written(text: str, cart: Cart) -> ToolOutcome:
    return ToolOutcome(text, [AgentEvent.cart_update(cart_payload(cart))])


# -- the writes ----------------------------------------------------------------------


async def gated_add_to_cart(
    *,
    backend: StorefrontBackend,
    session: ShoppingSessionContext,
    product_id: str,
    quantity: int,
    decision: CartGateDecision | None = None,
) -> ToolOutcome:
    """Apply ``decision``, then write. This function decides nothing itself.

    ``config`` and ``state`` are gone from the signature (divergence 1): the limits now
    arrive as ``decision.allowance`` and provenance and options are the chain's business.
    A caller that has neither has not run a chain, and ``require_decision`` says so.
    """
    answered = require_decision(decision, "add_to_cart")
    if (held := answered.held_outcome()) is not None:
        return held
    allowance = answered.allowance
    if allowance is None:
        raise UngatedCartWrite(
            "add_to_cart reached the write with no quantity allowance. An add is the only "
            "write that can exceed a cart limit, so the chain must attach the arithmetic "
            "that bounds it; `QuantityAllowance.from_config(config)` is the reference's."
        )
    requested = max(1, quantity)
    async with cart_critical_section(session):
        # The second application of the one allowance, against the cart as it is rather
        # than as the phase read it. It can only tighten: two concurrent adds serialize
        # here, so they cannot jointly exceed the per-item cap.
        current = await backend.get_cart(session)
        verdict = allowance.for_add(current, product_id, requested)
        if (held := verdict.as_held()) is not None:
            return held
        cart = await backend.add_to_cart(session, product_id, verdict.allowed)
    # The confirmation names the id only: titles are catalog text and stay inside fences.
    return _written(
        f"Added {product_id} x{verdict.allowed}{verdict.disclosure}. "
        f"Cart now has {cart_summary(cart)}.",
        cart,
    )


async def gated_update_cart_item(
    *,
    backend: StorefrontBackend,
    session: ShoppingSessionContext,
    product_id: str,
    quantity: int,
    decision: CartGateDecision | None = None,
) -> ToolOutcome:
    answered = require_decision(decision, "update_cart_item")
    if (held := answered.held_outcome()) is not None:
        return held
    allowance = answered.allowance
    if allowance is None:
        raise UngatedCartWrite(
            "update_cart_item reached the write with no quantity allowance. An update sets "
            "a quantity, so it clamps rather than refuses, but the figure it clamps to is "
            "still the config's and still the chain's to supply."
        )
    requested = max(1, quantity)
    verdict = allowance.for_set(requested)
    async with cart_critical_section(session):
        cart = await backend.update_cart_item(session, product_id, verdict.allowed)
    return _written(
        f"Updated quantity{verdict.disclosure}. Cart now has {cart_summary(cart)}.", cart
    )


async def gated_remove_from_cart(
    *,
    backend: StorefrontBackend,
    session: ShoppingSessionContext,
    product_id: str,
    decision: CartGateDecision | None = None,
) -> ToolOutcome:
    answered = require_decision(decision, "remove_from_cart")
    if (held := answered.held_outcome()) is not None:
        return held
    async with cart_critical_section(session):
        cart = await backend.remove_from_cart(session, product_id)
    return _written(f"Removed. Cart now has {cart_summary(cart)}.", cart)


# -- the upstream order, as one named object ----------------------------------------


@dataclass(frozen=True)
class _Decided:
    """One call's answer. Satisfies :class:`CartGateDecision`."""

    held: ToolOutcome | None
    _allowance: QuantityAllowance | None

    def held_outcome(self) -> ToolOutcome | None:
        return self.held

    @property
    def allowance(self) -> QuantityAllowance | None:
        return self._allowance


class ReferenceOrdering:
    """The upstream precedence, as a replaceable object rather than an ``or``-chain.

    This is what ``gated_add_to_cart`` used to do in its own first line. It is kept so the
    library still works for a deployment that registers no chain — and it is kept *here*,
    named, rather than in the write, so that replacing it is a constructor argument and
    not an edit to the function that writes the cart.

    ``provenance_or_cart`` is the reference's own deliberate read-avoidance: update and
    remove accept a line already in the cart even with no session provenance, and the cart
    is fetched **only** when provenance alone fails.
    """

    async def decide(
        self,
        *,
        tool: str,
        backend: StorefrontBackend,
        config: ShoppingAgentConfig,
        session: ShoppingSessionContext,
        state: ShoppingSessionState,
        product_id: str,
    ) -> CartGateDecision:
        allowance = QuantityAllowance.from_config(config)
        if tool == "add_to_cart":
            held = check_provenance(state, product_id) or check_options(state, product_id)
            return _Decided(held, allowance)
        held = await self.provenance_or_cart(backend, session, state, product_id)
        # remove takes no quantity, so it needs no allowance; update clamps and does.
        return _Decided(held, allowance if tool == "update_cart_item" else None)

    @staticmethod
    async def provenance_or_cart(
        backend: StorefrontBackend,
        session: ShoppingSessionContext,
        state: ShoppingSessionState,
        product_id: str,
    ) -> ToolOutcome | None:
        if check_provenance(state, product_id) is None:
            return None
        current = await backend.get_cart(session)
        if any(item.product_id == product_id for item in current.items):
            return None
        return ToolOutcome.held(PROVENANCE_GATE, provenance_error(product_id))


#: The upstream order. Passed explicitly; never reached as a fallback.
REFERENCE_ORDERING = ReferenceOrdering()
