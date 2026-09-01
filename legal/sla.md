# IronSight Service Level Agreement (Team)

**Version:** 2026-07  
**Applies to:** Team plan customers

## 1. Availability target
We target **99.0%** monthly uptime for the API and web app, excluding scheduled maintenance and third-party outages (Stripe, vision providers, your host).

## 2. Pipeline reliability
We target **≥95%** successful completion rate for queued analyze jobs when input video is valid and under plan limits. Failed jobs can be retried from the studio.

## 3. Support response
Team plan: initial response within **48 business hours** via in-app Support.

## 4. Priority processing
Team plan jobs are ordered **ahead of Free/Pro** in the shared worker queue (higher `priority`). This is not a dedicated private cluster SLA.

## 5. Data
Session media retained per `SESSION_RETENTION_DAYS` (default 90). Daily backups are operator-configured.

## 6. Accuracy disclaimer
Hit/miss AI remains assistive. Coach lock / Fix score is the contractual path to “coach-ready” debriefs.

## 7. Acceptance
Org owners accept this SLA version in the Team Range dashboard. Live metrics: `GET /api/sla`.
