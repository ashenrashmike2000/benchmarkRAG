#!/bin/bash
set -e

echo "Starting Benchmark System..."

# Allow Qdrant to start if selected
sleep 3

exec "$@"
