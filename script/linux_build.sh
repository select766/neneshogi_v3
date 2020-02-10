#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Linux バイナリのビルド
MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

cd `dirname $0`
cd ../source

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7
COMPILER=clang++-7
BUILDDIR=../build/user
mkdir -p ${BUILDDIR}
EDITION=USER_ENGINE
TARGET=YaneuraOu-user-linux-clang
declare -A TGTAIL=([avx2]=-avx2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done
