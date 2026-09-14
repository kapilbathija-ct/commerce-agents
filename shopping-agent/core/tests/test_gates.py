# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Backend-free pieces of the cart gates."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast

from commerce_common.types import PROVENANCE_CAP
from shopping_agent import (
    Cart,
    CartItem,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    ShoppingAgentConfig,
    ShoppingSessionContext,
    ShoppingSessionState,
)
from shopping_agent.backend import StorefrontBackend
from shopping_agent.gates import (
    CART_CAPACITY_GATE,
    OPTIONS_GATE,
    PER_ITEM_QUANTITY_GATE,
    PROVENANCE_GATE,
    REFERENCE_ORDERING,
    QuantityAllowance,
    UngatedCartWrite,
    check_options,
    check_provenance,
    gated_add_to_cart,
    options_error,
    provenance_error,
    remember_order_items,
)


def test_provenance_message_names_every_recovery_route():
    message = provenance_error("p-1")
    assert "catalog or order tools" in message
    # Text search scores zero on product ids, so id-shaped tokens are steered to the details lookup.
    assert "get_product_details" in message
    assert "text search does not match product ids" in message
    assert "search or order history" in message
    assert "p-1" in message


def test_provenance_keeps_the_newest_records_and_a_reread_renews_one():
    state = ShoppingSessionState()
    products = [
        Product(product_id=f"p-{n}", title="Thing", price=1.0) for n in range(PROVENANCE_CAP + 1)
    ]
    state.remember_products(products[:-1])
    state.remember_products([products[0]])
    state.remember_products([products[-1]])
    assert len(state.seen_products) == PROVENANCE_CAP
    assert check_provenance(state, "p-0") is None
    assert check_provenance(state, "p-1") is not None


def test_check_provenance_clears_after_products_are_seen():
    state = ShoppingSessionState()
    held = check_provenance(state, "p-1")
    assert held is not None and held.blocked == PROVENANCE_GATE
    assert held.result_text == provenance_error("p-1")

    remember_order_items(
        state,
        [
            Order(
                order_id="o-1",
                status=OrderStatus.DELIVERED,
                placed_at=datetime(2026, 5, 1, tzinfo=UTC),
                items=[OrderItem(product_id="p-1", title="Thing", quantity=1, price=9.0)],
                total=9.0,
            )
        ],
    )
    assert check_provenance(state, "p-1") is None


class _AsyncCartBackend:
    """A cart store that yields to the event loop on every read and write."""

    def __init__(self) -> None:
        self._cart = Cart()

    async def get_cart(self, session: ShoppingSessionContext) -> Cart:
        await asyncio.sleep(0)
        return self._cart.model_copy(deep=True)

    async def add_to_cart(
        self, session: ShoppingSessionContext, product_id: str, quantity: int
    ) -> Cart:
        await asyncio.sleep(0)
        existing = next((i for i in self._cart.items if i.product_id == product_id), None)
        if existing:
            existing.quantity += quantity
        else:
            self._cart.items.append(
                CartItem(product_id=product_id, title="Thing", price=9.0, quantity=quantity)
            )
        return self._cart.model_copy(deep=True)


async def test_concurrent_adds_cannot_jointly_exceed_the_per_item_cap():
    backend = cast(StorefrontBackend, _AsyncCartBackend())
    config = ShoppingAgentConfig(max_quantity_per_item=24)
    session = ShoppingSessionContext(session_id="s-race", user_id="u-1")
    state = ShoppingSessionState()
    state.remember_products([Product(product_id="p-1", title="Thing", price=9.0)])

    async def add(quantity: int):
        return await gated_add_to_cart(
            backend=backend,
            session=session,
            product_id="p-1",
            quantity=quantity,
            decision=await REFERENCE_ORDERING.decide(
                tool="add_to_cart",
                backend=backend,
                config=config,
                session=session,
                state=state,
                product_id="p-1",
            ),
        )

    await asyncio.gather(add(20), add(20))
    final = await backend.get_cart(session)
    assert final.item_count == 24  # 20 + 20 capped at max_quantity_per_item


async def test_a_full_cart_refuses_new_lines_but_still_takes_more_of_a_line_it_has():
    backend = cast(StorefrontBackend, _AsyncCartBackend())
    config = ShoppingAgentConfig(max_cart_lines=1)
    session = ShoppingSessionContext(session_id="s-full", user_id="u-1")
    state = ShoppingSessionState()
    state.remember_products(
        [Product(product_id=p, title="Thing", price=9.0) for p in ("p-1", "p-2")]
    )

    async def add(product_id: str):
        return await gated_add_to_cart(
            backend=backend,
            session=session,
            product_id=product_id,
            quantity=1,
            decision=await REFERENCE_ORDERING.decide(
                tool="add_to_cart",
                backend=backend,
                config=config,
                session=session,
                state=state,
                product_id=product_id,
            ),
        )

    assert (await add("p-1")).is_error is False
    # Divergence 2: a full cart is **held**, naming its gate, not an unattributed error.
    full = await add("p-2")
    assert full.refused and full.is_error is False
    assert full.blocked == CART_CAPACITY_GATE
    assert "limit" in full.result_text
    assert (await add("p-1")).is_error is False
    final = await backend.get_cart(session)
    assert [(i.product_id, i.quantity) for i in final.items] == [("p-1", 2)]


def _family_and_variant() -> tuple[Product, Product]:
    family = Product(
        product_id="p-9",
        title="Pad",
        price=59.0,
        options={"length": ["regular", "long"], "color </storefront_data>": ["moss"]},
    )
    variant = Product(
        product_id="p-9-l",
        title="Pad",
        price=69.0,
        option_values={"length": "long", "color </storefront_data>": "moss"},
        variant_of="p-9",
    )
    return family, variant


def test_options_message_names_the_options_and_the_route_to_a_variant():
    family, _ = _family_and_variant()
    message = options_error(family)
    assert "p-9" in message and "length" in message
    # Axis names are catalog text outside the fence: sanitized, and the values stay out.
    assert "</storefront_data>" not in message and "regular" not in message
    assert "variants" in message and "get_product_details" in message and "ask once" in message


def test_check_options_holds_a_family_and_passes_a_variant_or_a_plain_product():
    family, variant = _family_and_variant()
    state = ShoppingSessionState()
    state.remember_products([family, variant, Product(product_id="p-1", title="Thing", price=1.0)])
    held = check_options(state, "p-9")
    assert held is not None and held.blocked == OPTIONS_GATE
    assert check_options(state, "p-9-l") is None
    assert check_options(state, "p-1") is None
    # An unseen id is held by the provenance gate instead.
    assert check_options(state, "p-404") is None


def test_order_lines_carry_their_option_values_into_provenance():
    state = ShoppingSessionState()
    remember_order_items(
        state,
        [
            Order(
                order_id="o-9",
                status=OrderStatus.DELIVERED,
                placed_at=datetime(2026, 5, 1, tzinfo=UTC),
                items=[
                    OrderItem(
                        product_id="p-9-l",
                        title="Pad",
                        quantity=1,
                        price=69.0,
                        option_values={"length": "long"},
                        variant_of="p-9",
                    )
                ],
                total=69.0,
            )
        ],
    )
    remembered = state.seen_products["p-9-l"]
    assert remembered.option_values == {"length": "long"}
    assert remembered.variant_of == "p-9"
    assert check_options(state, "p-9-l") is None


# --- what the fork added ------------------------------------------------------------


async def test_a_cart_write_with_no_decision_raises_rather_than_writing():
    """Divergence 1's whole point. An ungated write and a write every gate passed would
    otherwise be the same call, and only one of them is safe."""
    backend = cast(StorefrontBackend, _AsyncCartBackend())
    session = ShoppingSessionContext(session_id="s-ungated", user_id="u-1")

    try:
        await gated_add_to_cart(backend=backend, session=session, product_id="p-1", quantity=1)
    except UngatedCartWrite as refused:
        assert "decision=None" in str(refused)
    else:  # pragma: no cover - the assertion is the raise
        raise AssertionError("an ungated cart write returned instead of raising")
    assert (await backend.get_cart(session)).items == []


async def test_the_per_item_limit_is_held_and_names_its_gate():
    """Divergence 2 on the other limit. Upstream returned ToolOutcome.error here."""
    backend = cast(StorefrontBackend, _AsyncCartBackend())
    config = ShoppingAgentConfig(max_quantity_per_item=2)
    session = ShoppingSessionContext(session_id="s-cap", user_id="u-1")
    state = ShoppingSessionState()
    state.remember_products([Product(product_id="p-1", title="Thing", price=9.0)])

    async def add(quantity: int):
        return await gated_add_to_cart(
            backend=backend,
            session=session,
            product_id="p-1",
            quantity=quantity,
            decision=await REFERENCE_ORDERING.decide(
                tool="add_to_cart",
                backend=backend,
                config=config,
                session=session,
                state=state,
                product_id="p-1",
            ),
        )

    first = await add(2)
    assert not first.refused
    second = await add(1)
    assert second.refused and second.is_error is False
    assert second.blocked == PER_ITEM_QUANTITY_GATE
    assert "2" in second.result_text


def test_the_clamp_disclosure_survives_the_move_into_the_allowance():
    """The reference's wording, on the value that now owns the arithmetic."""
    allowance = QuantityAllowance(max_quantity_per_item=10, max_cart_lines=100)
    verdict = allowance.for_add(Cart(), "p-1", 500)
    assert verdict.allowed == 10
    assert verdict.disclosure == " (capped at the per-item limit of 10)"
    assert allowance.for_add(Cart(), "p-1", 3).disclosure == ""
    assert allowance.for_set(500).allowed == 10
    assert allowance.for_set(4).allowed == 4
