#!/bin/bash
# Build scripts/dsl_v2/registry_dump.cpp with the exact compile / link commands of bin/triton-opt of the local official
# build (ninja -t commands), replacing only triton-opt.cpp(.o) by registry_dump.cpp(.o); output in .cache/dsl_v2.
set -e
R=/data1/tzh/kernel-analyzer
B=$R/.cache/upstream/triton/build/cmake.linux-x86_64-cpython-3.11
O=$R/.cache/dsl_v2/registry_dump
mkdir -p $O
cd $B
N=/data1/tzh/envs/triton_main/bin/ninja
COMPILE=$($N -t commands bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o | tail -1)
LINK=$($N -t commands bin/triton-opt | tail -1)
COMPILE=${COMPILE// -MD -MT bin\/CMakeFiles\/triton-opt.dir\/triton-opt.cpp.o -MF bin\/CMakeFiles\/triton-opt.dir\/triton-opt.cpp.o.d/}
COMPILE=${COMPILE//-o bin\/CMakeFiles\/triton-opt.dir\/triton-opt.cpp.o/-o $O\/registry_dump.o}
COMPILE=${COMPILE//-c $R\/.cache\/upstream\/triton\/bin\/triton-opt.cpp/-c $R\/scripts\/dsl_v2\/registry_dump.cpp -I$R\/.cache\/upstream\/triton\/bin}
LINK=${LINK//bin\/CMakeFiles\/triton-opt.dir\/triton-opt.cpp.o -o bin\/triton-opt/$O\/registry_dump.o -o $O\/registry_dump}
eval "$COMPILE"
eval "$LINK"
sha256sum $O/registry_dump
