#!/bin/sh
set -eu

test_name="${1:-playground}"
runner_log="$(mktemp)"
trap 'rm -f "$runner_log"' EXIT HUP INT TERM

if ! odin test /home/dpooria/Documents/code/opaw/omat/test/ \
        -define:ODIN_TEST_NAMES="test.${test_name}" \
        -define:ODIN_TEST_THREADS=1 \
        -define:ODIN_TEST_FANCY=false \
        -define:ODIN_TEST_SHORT_LOGS=true \
        -define:ODIN_TEST_TRACK_MEMORY=false \
        2>"$runner_log"; then
    cat "$runner_log" >&2
    exit 1
fi
