#!/usr/bin/env bash

set -xe

odin run center_columns.odin -file
odin run solve_system.odin -file
odin run fft_peak.odin -file
odin run integrate_samples.odin -file

# odin run openblas_custom_threading_callback.odin -file -define:OMAT_OPENBLAS=system:openblasp64 -define:OMAT_USE_OPENBLAS_THREADS=true -o:speed
