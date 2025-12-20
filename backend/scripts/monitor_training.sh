#!/bin/bash
while ps aux | grep -q "[t]rain_timeseries"; do
  cpu=$(ps aux | grep "[t]rain_timeseries" | awk '{print $3}')
  time=$(ps aux | grep "[t]rain_timeseries" | awk '{print $10}')
  echo "$(date '+%H:%M:%S') - Training running - CPU: ${cpu}%, Time: ${time}"
  sleep 30
done
echo "$(date '+%H:%M:%S') - Training completed!"
cat /tmp/claude/tasks/b12e11a.output 2>/dev/null | tail -50
