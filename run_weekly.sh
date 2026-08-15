#!/usr/bin/env bash
# run_weekly.sh — TLCA v3 | Mac M4 edition
# Equivalent of run_weekly.ps1 for macOS
#
# SETUP (first time only):
#   chmod +x run_weekly.sh
#   pip3 install requests   (or: pip3 install -r requirements.txt)
#   export ANTHROPIC_API_KEY="sk-ant-..."   (or add to ~/.zshrc)
#
# CRON (every Monday 07:30 Zürich time — UTC+1 winter, UTC+2 summer):
#   crontab -e
#   30 5 * * 1 cd /path/to/tlca && bash run_weekly.sh >> logs/weekly.log 2>&1
#   (adjust UTC offset: 05:30 UTC = 07:30 CET | 05:30 UTC = 07:30 CEST in summer → use 5 for both, DST handled by macOS)
#
# MANUAL RUN:
#   bash run_weekly.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=================================================="
echo " TLCA — Weekly Competitor Intelligence Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=================================================="

# ---------- helpers ----------
set_env_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "$name"="$value"
    echo "[env] $name=$value (default)"
  else
    echo "[env] $name=${!name} (kept)"
  fi
}

run_step() {
  local label="$1"
  local script="$2"
  echo ""
  echo "==> $label"
  if python3 "$script"; then
    echo "[OK] $label"
  else
    echo "[WARN] $label exited with error (continuing pipeline)"
  fi
}

# ---------- env defaults ----------
set_env_default INTEL_DB_PATH            "intel.db"
set_env_default INTEL_SOURCES_PATH       "sources.json"
set_env_default INTEL_VERIFY_SSL         "1"
set_env_default ALLOW_INSECURE_SSL_FALLBACK "1"
set_env_default INTEL_HTTP_TIMEOUT       "25"
set_env_default INTEL_WINDOW_DAYS        "7"
set_env_default INTEL_MAX_SUMMARIZE      "30"
set_env_default INTEL_REQUIRE_TAG_MATCH  "1"
set_env_default INTEL_DEDUP_ON_CANONICAL "1"
set_env_default INTEL_ALLOWED_TAGS       "GBS,GCC,Agentic_AI,Operating_Model,Client_Signal"

set_env_default ENABLE_LLM               "1"
set_env_default CLAUDE_MODEL             "claude-haiku-4-5-20251001"
set_env_default CLAUDE_TIMEOUT_SECONDS   "45"

set_env_default INTEL_OUT_DIR            "output"
set_env_default INTEL_TEMPLATE_PATH      "templates/outlook_newsletter_template.html"
set_env_default INTEL_EDITION            "1"
set_env_default INTEL_WINDOW_DAYS        "7"

set_env_default INTEL_BTN_BG             "#FFC72C"
set_env_default INTEL_BTN_TXT            "#1A1A24"
set_env_default INTEL_BTN_LABEL          "Read the source"
set_env_default INTEL_SUBJECT_PREFIX     "Global Competitor Intelligence Brief - GBS | GCC | Agentic AI"
set_env_default GMAIL_USER               "morichter97@gmail.com"
set_env_default INTEL_RECIPIENTS         "moritz.richter@ch.ey.com"

# Guard: ANTHROPIC_API_KEY must be set
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo ""
  echo "[ERROR] ANTHROPIC_API_KEY is not set."
  echo "        Export it before running:"
  echo "        export ANTHROPIC_API_KEY='sk-ant-...'"
  echo "        Or add it to ~/.zshrc and restart your terminal."
  exit 1
fi

# ---------- pipeline ----------
mkdir -p output logs

# Auto-increment edition
EDITION=$(python3 edition_counter.py)
export INTEL_EDITION="$EDITION"
echo "[info] Edition: $EDITION"

run_step "1/3 Ingest feeds"         "intel_ingest_stdlib.py"
run_step "2/3 Summarize new links"  "summarize_new_links.py"
run_step "3/3 Write newsletter"     "weekly_newsletter_output.py"
run_step "4/4 Build explorer"        "generate_dashboard.py"
run_step "5/5 Send newsletter"       "send_newsletter.py"

echo ""
echo "=================================================="
echo " Done. $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " Output → output/"
echo "=================================================="
