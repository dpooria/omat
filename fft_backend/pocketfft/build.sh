#!/bin/sh
set -eu
cd "$(dirname "$0")"
build_dir="${POCKETFFT_BUILD_DIR:-.}"
${CXX:-c++} -std=c++11 -O3 -fPIC ${CXXFLAGS:-} "$@" -c pocketfft.cc -o "$build_dir/pocketfft.o"
ar rcs "$build_dir/pocketfft.a" "$build_dir/pocketfft.o"
