#!/bin/zsh
# DeepSeek-V4.1 Phase 3 battery: pack load + generate + engine compare.
#
# The sushi model server (~233 GB resident) shares this 512 GB machine with
# a 196 GB EXL3 pack + 200 GB engram tables — coexistence is impossible.
# This script frees the machine by unloading com.steven.sushi-server, runs
# the battery, and ALWAYS restores the server on exit (trap covers normal
# end, Ctrl-C, and SIGTERM), so the runtime serving this session comes back
# on its own even if the battery is interrupted.
#
# Logs: /tmp/dsv41_battery.{restore,1,2,3}
set -u
LABEL=com.steven.sushi-server
PLIST="$HOME/Library/LaunchAgents/com.steven.sushi-server.plist"
DOMAIN="gui/$(id -u)"
BASE=/tmp/dsv41_battery
MD="$HOME/src/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
PROMPT="What is the capital of France? Answer in one short sentence."

restore() {
  print -u2 "[$(date +%T)] restoring $LABEL"
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null
  launchctl kickstart "$DOMAIN/$LABEL" 2>/dev/null
  echo "[$(date +%T)] restored: $(launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 && echo yes || echo NO)" \
    >> "$BASE.restore"
}
trap restore EXIT INT TERM

echo "[$(date +%T)] stopping $LABEL (sushi server)" > "$BASE.restore"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
sleep 5

cd "$HOME/src/PonyExl3" || exit 9
export PONYEXL3_ENGRAM_DIR="$HOME/src/models/engram-src"

echo "=== [1/3] generate, engine=exl3, engram ON ===" > "$BASE.1"
time .venv/bin/python -u tools/dev/dsv41_run.py "$MD" -p "$PROMPT" -n 32 \
  >> "$BASE.1" 2>&1
echo "EXIT=$?" >> "$BASE.1"

echo "=== [2/3] generate, engine=exl3, engram OFF (ablation) ===" > "$BASE.2"
time env DSV41_SKIP_ENGRAM=1 .venv/bin/python -u tools/dev/dsv41_run.py "$MD" \
  -p "$PROMPT" -n 32 >> "$BASE.2" 2>&1
echo "EXIT=$?" >> "$BASE.2"

echo "=== [3/3] fold16 vs exl3 (greedy agreement + logit drift) ===" > "$BASE.3"
time .venv/bin/python -u -m ponyexl3.cli.compare_engines "$MD" \
  -p "Explain why the sky is blue." -n 64 --engines fold16 >> "$BASE.3" 2>&1
echo "EXIT=$?" >> "$BASE.3"

echo "[$(date +%T)] battery done" >> "$BASE.restore"
