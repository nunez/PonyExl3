#!/bin/zsh
# Run one command with the sushi model server's RAM freed, then ALWAYS
# restore the server (trap on EXIT/INT/TERM). Needed for any real-pack
# ponyexl3 run: 196 GB bundles + the ~230 GB sushi server do not coexist
# on this 512 GB machine (jetsam picks whichever grew — including the
# server itself, i.e. the session's own runtime).
set -u
LABEL=com.steven.sushi-server
PLIST="$HOME/Library/LaunchAgents/com.steven.sushi-server.plist"
DOMAIN="gui/$(id -u)"
LOG=/tmp/sushi_yield.log

restore() {
  print -u2 "[$(date +%T)] restoring $LABEL"
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null
  launchctl kickstart "$DOMAIN/$LABEL" 2>/dev/null
  echo "[$(date +%T)] restored: $(launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 && echo yes || echo NO)" >> "$LOG"
}
trap restore EXIT INT TERM

echo "[$(date +%T)] stopping $LABEL for: $*" >> "$LOG"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
sleep 5
"$@"
rc=$?
echo "[$(date +%T)] command exited rc=$rc" >> "$LOG"
exit $rc
