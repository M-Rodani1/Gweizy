#!/bin/bash

echo "=== RUNTIME DIAGNOSTIC ==="
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Searching for libgomp..."

# Search for libgomp in nix store
find /nix/store -name 'libgomp.so.1' 2>/dev/null | head -5

# Find and set library path
for d in /nix/store/*/lib; do
    if [ -f "$d/libgomp.so.1" ]; then
        echo "Found libgomp in: $d"
        export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
        break
    fi
done

echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start gunicorn
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
