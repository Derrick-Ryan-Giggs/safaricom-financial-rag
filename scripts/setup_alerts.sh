#!/bin/sh
# scripts/setup_alerts.sh
#
# Creates two Cloud Monitoring alert policies for rag-app (roadmap item 6):
# memory utilization approaching the configured limit, and elevated error
# rate. Both notify by email. Run once -- re-running creates duplicate
# policies rather than updating existing ones, since these are simple
# `create` calls, not `apply`.
#
# Usage: EMAIL=you@example.com ./scripts/setup_alerts.sh

set -e

PROJECT="${PROJECT:-safaricom-intelligence}"
SERVICE="${SERVICE:-rag-app}"
EMAIL="${EMAIL:?Set EMAIL=you@example.com before running}"

# Notification channel (email). Reuses an existing one with the same email
# if found, so re-running this script doesn't spam duplicate channels --
# alert POLICIES themselves aren't deduplicated the same way, hence the
# one-time-run note above.
CHANNEL_ID=$(gcloud alpha monitoring channels list \
  --project="$PROJECT" \
  --filter="type=email AND labels.email_address=$EMAIL" \
  --format="value(name)" | head -n1)

if [ -z "$CHANNEL_ID" ]; then
  CHANNEL_ID=$(gcloud alpha monitoring channels create \
    --project="$PROJECT" \
    --display-name="rag-app alerts" \
    --type=email \
    --channel-labels="email_address=$EMAIL" \
    --format="value(name)")
fi

echo "Using notification channel: $CHANNEL_ID"

# Memory utilization approaching the configured limit (docker-compose.yml
# / your Cloud Run deploy set --memory=2Gi -- this fires before an OOM
# kill, not after, matching item 6's goal of finding out before checking
# the Errors tab by chance).
gcloud alpha monitoring policies create \
  --project="$PROJECT" \
  --display-name="rag-app: memory utilization high" \
  --notification-channels="$CHANNEL_ID" \
  --condition-display-name="Memory > 85% of limit" \
  --condition-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"" \
  --condition-threshold-value=0.85 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=60s \
  --aggregation="{\"alignmentPeriod\": \"60s\", \"perSeriesAligner\": \"ALIGN_MEAN\"}"

# Elevated error rate (5xx responses).
gcloud alpha monitoring policies create \
  --project="$PROJECT" \
  --display-name="rag-app: elevated error rate" \
  --notification-channels="$CHANNEL_ID" \
  --condition-display-name="5xx response count > 5 in 5 min" \
  --condition-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\"" \
  --condition-threshold-value=5 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=300s \
  --aggregation="{\"alignmentPeriod\": \"300s\", \"perSeriesAligner\": \"ALIGN_SUM\"}"

echo "Done. Check console.cloud.google.com/monitoring/alerting/policies to confirm."