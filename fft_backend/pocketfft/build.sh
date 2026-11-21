#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-c++} -std=c++11 -O3 -fPIC ${CXXFLAGS:-} "$@" -c pocketfft.cc -o pocketfft.o
ar rcs pocketfft.a pocketfft.o
