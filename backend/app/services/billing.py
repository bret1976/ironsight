"""Stripe billing for IronSight subscriptions."""
from __future__ import annotations

from typing import Any, Optional

from app.core.config import get_settings
from app.services import users


def stripe_configured() -> bool:
    s = get_settings()
    return bool(s.stripe_secret_key and s.stripe_price_pro)


def _stripe():
    import stripe

    s = get_settings()
    stripe.api_key = s.stripe_secret_key
    return stripe


def ensure_customer(user: dict[str, Any]) -> str:
    if user.get("stripe_customer_id"):
        return user["stripe_customer_id"]
    stripe = _stripe()
    cust = stripe.Customer.create(
        email=user["email"],
        name=user.get("name") or user["email"],
        metadata={"user_id": user["id"]},
    )
    users.update_user(user["id"], stripe_customer_id=cust.id)
    return cust.id


def create_checkout_session(user: dict[str, Any], plan_id: str, success_url: str, cancel_url: str) -> dict[str, Any]:
    s = get_settings()
    if not stripe_configured():
        raise RuntimeError("Stripe not configured. Set STRIPE_SECRET_KEY and STRIPE_PRICE_PRO / STRIPE_PRICE_TEAM")
    price_map = {
        "pro": s.stripe_price_pro,
        "team": s.stripe_price_team or s.stripe_price_pro,
    }
    if plan_id not in price_map or not price_map[plan_id]:
        raise ValueError(f"Unknown plan or missing price id: {plan_id}")
    stripe = _stripe()
    customer_id = ensure_customer(user)
    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_map[plan_id], "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"user_id": user["id"], "plan": plan_id},
        subscription_data={"metadata": {"user_id": user["id"], "plan": plan_id}},
        allow_promotion_codes=True,
    )
    return {"checkout_url": session.url, "session_id": session.id}


def create_portal_session(user: dict[str, Any], return_url: str) -> dict[str, Any]:
    if not user.get("stripe_customer_id"):
        raise ValueError("No Stripe customer")
    stripe = _stripe()
    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=return_url,
    )
    return {"portal_url": portal.url}


def handle_webhook(payload: bytes, sig_header: str) -> dict[str, Any]:
    s = get_settings()
    if not s.stripe_webhook_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET not set")
    stripe = _stripe()
    event = stripe.Webhook.construct_event(payload, sig_header, s.stripe_webhook_secret)

    etype = event["type"]
    data = event["data"]["object"]

    if etype == "checkout.session.completed":
        user_id = (data.get("metadata") or {}).get("user_id")
        plan = (data.get("metadata") or {}).get("plan") or "pro"
        sub_id = data.get("subscription")
        if user_id:
            users.update_user(
                user_id,
                plan=plan,
                stripe_subscription_id=sub_id,
                stripe_customer_id=data.get("customer") or None,
            )
        return {"handled": etype, "user_id": user_id, "plan": plan}

    if etype in ("customer.subscription.updated", "customer.subscription.created"):
        meta = data.get("metadata") or {}
        user_id = meta.get("user_id")
        plan = meta.get("plan")
        status = data.get("status")
        if not user_id and data.get("customer"):
            u = users.get_user_by_stripe_customer(data["customer"])
            user_id = u["id"] if u else None
        if user_id:
            if status in ("active", "trialing"):
                users.update_user(
                    user_id,
                    plan=plan or "pro",
                    stripe_subscription_id=data.get("id"),
                )
            elif status in ("canceled", "unpaid", "incomplete_expired"):
                users.update_user(user_id, plan="free", stripe_subscription_id=None)
        return {"handled": etype, "user_id": user_id, "status": status}

    if etype == "customer.subscription.deleted":
        cust = data.get("customer")
        u = users.get_user_by_stripe_customer(cust) if cust else None
        if u:
            users.update_user(u["id"], plan="free", stripe_subscription_id=None)
        return {"handled": etype, "downgraded": bool(u)}

    return {"handled": False, "type": etype}
