#!/bin/bash
# Run data collection worker in background that survives lid closure

cd "$(dirname "$0")/../backend" || exit 1

# Prevent Mac from sleeping while this script runs
# -d prevents display sleep
# -i prevents idle sleep (system sleep)
# Runs the command and prevents sleep until it exits
caffeinate -d -i python3 worker.py
