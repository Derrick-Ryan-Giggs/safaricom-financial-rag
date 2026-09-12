#!/bin/sh
# scripts/setup_alerts.sh
#
# Creates two Cloud Monitoring alert policies for rag-app (roadmap item 6):
# memory utilization approaching the configured limit, and elevated error
# rate. Both notify by email. Run once -- re-running creates duplicate
# POLICIES (not duplicate notification channels; the channel lookup below
# is idempotent), since these are simple `create` calls, not `apply`.
#
# Usage: EMAIL=you@example.com ./scripts/setup_alerts.sh
#
# Uses --policy-from-file (a JSON policy definition) rather than building
# the condition via imperative flags -- gcloud's flag-based condition
# builder for `monitoring policies create` doesn't actually support
# --condition-threshold-value/-comparison/-duration the way an earlier
# version of this script assumed; --policy-from-file maps directly to the
# documented AlertPolicy REST resource schema instead, which is the
# stable interface here.

set -e

PROJECT="${PROJECT:-safaricom-intelligence}"
SERVICE="${SERVICE:-rag-app}"
EMAIL="${EMAIL:?Set EMAIL=you@example.com before running}"

# Notification channel (email). Reuses an existing one with the same email
# if found. Both sides of the filter need quoting -- gcloud's filter
# parser treats an unquoted value as ambiguous between a literal string
# and a field reference (this is what broke on the first run).
CHANNEL_ID=$(gcloud alpha monitoring channels list \
  --project="$PROJECT" \
  --filter="type=\"email\" AND labels.email_address=\"$EMAIL\"" \
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

# Memory utilization approaching the configured limit (whatever --memory
# your Cloud Run deploy set -- 2Gi per your earlier deploy command). Fires
# before an OOM kill, not after, matching item 6's goal of finding out
# before checking the Errors tab by chance.
cat > /tmp/rag-app-memory-policy.json <<EOF
{
  "displayName": "rag-app: memory utilization high",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "Memory > 85% of limit",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.85,
        "duration": "60s",
        "aggregations": [
          { "alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_PERCENTILE_99" }
         ]
      }
    }
  ],
  "notificationChannels": ["$CHANNEL_ID"]
}
EOF

gcloud alpha monitoring policies create \
  --project="$PROJECT" \
  --policy-from-file=/tmp/rag-app-memory-policy.json

# Elevated error rate (5xx responses).
cat > /tmp/rag-app-error-policy.json <<EOF
{
  "displayName": "rag-app: elevated error rate",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "5xx response count > 5 in 5 min",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\"",
        "comparison": "COMPARISON_GT",
        "thresholdValue": 5,
        "duration": "300s",
        "aggregations": [
          { "alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM" }
        ]
      }
    }
  ],
  "notificationChannels": ["$CHANNEL_ID"]
}
EOF

gcloud alpha monitoring policies create \
  --project="$PROJECT" \
  --policy-from-file=/tmp/rag-app-error-policy.json

echo "Done. Check console.cloud.google.com/monitoring/alerting/policies to confirm."