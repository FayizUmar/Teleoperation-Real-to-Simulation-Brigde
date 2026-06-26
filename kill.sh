#!/usr/bin/env bash
# Force-kill any running so101-lite teleop processes.

PATTERNS=("so101-lite" "so101_lite" "viewer.py")

pids=()
for pattern in "${PATTERNS[@]}"; do
    found=$(pgrep -f "$pattern" 2>/dev/null)
    for pid in $found; do
        pids+=("$pid")
    done
done

# Deduplicate
unique_pids=($(printf '%s\n' "${pids[@]}" | sort -u))

if [ ${#unique_pids[@]} -eq 0 ]; then
    echo "No so101-lite processes found."
    exit 0
fi

echo "Killing PIDs: ${unique_pids[*]}"
for pid in "${unique_pids[@]}"; do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || echo "<gone>")
    echo "  [$pid] $cmd"
    kill -9 "$pid" 2>/dev/null && echo "  → killed" || echo "  → already gone"
done
