# Divergences from `anthropics/commerce-agents`

Branch `ct-gate-chain`, forked at `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`.

This is the list an upstream contribution reads from. Every behavioural difference between
this branch and upstream is recorded here, newest last, each one with the requirement that
forced it and the signal that proves it. A change to this branch that is not in this list
is a defect in the list.

**Scope.** One package moves: `shopping-agent-core`. The four sibling packages pin their
dependencies by version (`shopping-agent-core==0.1.0.dev0`) and never by direct reference,
and PEP 440 makes such a specifier ignore a candidate's local label — so upstream builds of
the siblings resolve against this fork unchanged.

---

## 1 — The quantity caps return a held outcome, not an error (2026-09-14)

**Upstream.** `gated_add_to_cart` returns `ToolOutcome.error("The cart is full.")` when the
cart is at `max_cart_lines`, and `ToolOutcome.error("This item is already at the per-item
limit of N.")` when there is no headroom left under `max_quantity_per_item`.

**Here.** Both return `ToolOutcome.held(gate, reason)` — `cart_capacity` and
`per_item_quantity` respectively. `ToolOutcome.held` is upstream's own constructor; it sets
`blocked` to the gate's name and makes `refused` true.

**Why.** Errors and holds are relayed to the model differently, and they mean different
things: an error is "this tool did not work", which invites a retry, while a hold is "this
call was refused, by this gate, for this reason", which does not. Both caps are policy
decisions with a reason a shopper can act on, so both are holds. Upstream's own text could
not be a hold as written for a structural reason worth naming: `ToolOutcome.held` requires a
gate name and these two limits had none — they were anonymous strings inside the body of the
function that writes the cart. So the conversion needed the limits to be named first, which
is what `CART_CAPACITY_GATE` and `PER_ITEM_QUANTITY_GATE` are.

**Signal.** `shopping-agent/core/tests/test_executor.py::test_add_to_cart_cap_applies_across_repeated_adds`
asserts `third.refused and not third.is_error` and `third.blocked == PER_ITEM_QUANTITY_GATE`.
It asserted `third.is_error` upstream and failed on this branch before it was updated, which
is how the divergence was found rather than assumed.
`test_gates.py::test_a_full_cart_refuses_new_lines_but_still_takes_more_of_a_line_it_has`
carries the same conversion for `cart_capacity`.

**Side effect, deliberate.** Both reasons now state the limit rather than only the fact.
"The cart is full." is not a reason, and a held outcome is required to give one.

---

## 2 — Precedence is the caller's, not the write's (2026-09-14)

**Upstream.** Ordering lives in one expression inside `gated_add_to_cart`:

```python
if held := check_provenance(state, product_id) or check_options(state, product_id):
```

and in `_check_provenance_or_cart` for update and remove. There is nowhere to place a third
gate ahead of those two except by editing that line.

**Here.** The three `gated_*` writes take a **required** `decision` — a `CartGateDecision`,
which is two members and no inheritance, so a deployment's own registry satisfies it without
importing this package. `require_decision` raises `UngatedCartWrite` when it is absent.
There is no default: an ungated write and a write every gate passed would otherwise be the
same call.

The predicates are **unchanged and still exported** — `check_provenance`, `check_options`,
`QuantityAllowance` — because a chain wraps them rather than re-implementing them. Nothing
upstream gated has been re-expressed in different words.

`ShoppingToolExecutor.cart_decision(tool, product_id)` is the seam a deployment overrides.
Its default is `REFERENCE_ORDERING`, which is upstream's order kept as one named,
replaceable object. It is not a second authority: exactly one decision source answers any
single call, and a deployment that supplies its own never reaches this one.

**Why.** A new gate must be placeable ahead of all existing ones without editing any
service. Two consequences upstream cannot reach from the `or`-chain: the ordered list cannot
be read at runtime, so a refusal cannot be reconstructed from a decision record; and a
lower-priority concern cannot be prevented from contradicting a higher-priority one, because
there are no priorities.

**Signal.** `test_gates.py::test_a_cart_write_with_no_decision_raises_rather_than_writing`
— the write raises and the cart is still empty afterwards.

**Side effect, deliberate.** `config` and `state` leave the `gated_*` signatures. The writes
no longer read session state or configuration; they apply a decision. A caller that still
had to pass both would look like it was still deciding.

---

## 3 — The gate phase and the cart's critical section are mutually exclusive (2026-09-14)

**Upstream.** `gated_update_cart_item` and `gated_remove_from_cart` call
`_check_provenance_or_cart` **inside** `async with _cart_lock(session)`, and that function
reads the cart on one branch. So a gate's platform read sits in a section serialized per
session.

**Here.** `gate_phase()` and `cart_critical_section()` are named context managers that
refuse to nest, in either direction — `GatePhaseInsideCartLock` and
`CartLockInsideGatePhase`. The chain is evaluated above the lock; the cart read and the
write stay inside it.

**Why.** A read inside the critical section serializes every write for that session behind
it, against a drawer latency budget that does not accommodate a round trip. And a gate that
could write from inside the phase would land an effect before a higher-priority gate had
spoken, which is the ordering the chain exists to guarantee. Both were previously matters of
review; they are now runtime errors.

**Side effect, deliberate.** The in-cart fallback for update and remove — a line already in
the cart passes even with no session provenance, and the cart is fetched *only* when
provenance alone fails — is preserved exactly, including the read-avoidance. It moved above
the lock with the rest of the phase, which is strictly better: the read it may perform no
longer serializes.

**Note on what the lock still does.** `QuantityAllowance` is applied twice for an add: once
in the phase, above the lock, where a refusal costs no write; and once inside the critical
section against the cart as it actually is. Two applications of one computation, never two
computations — which is what keeps two concurrent adds in one `gather` from jointly
exceeding the per-item cap.
`test_gates.py::test_concurrent_adds_cannot_jointly_exceed_the_per_item_cap` is upstream's
own test of that property and passes unchanged.
