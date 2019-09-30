#export KALDI_ROOT=`pwd`/../../..
export KALDI_ROOT=/auto/dr-std/pg3/softwares/kaldi/
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sox/sox-14.4.2/src/
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

PATH=/usr/bin/${PATH:+:$PATH}
PATH=$PATH:/auto/dr-std/pg3/softwares/bin
PATH=$PATH:/usr/usc/gnu/gcc/4.9.3/bin


INCLUDE_PATH=/usr/usc/gnu/gcc/4.9.3/include${INCLUDE_PATH:+:$INCLUDE_PATH}



LD_LIBRARY_PATH=/auto/dr-std/pg3/softwares/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/auto/dr-std/pg3/softwares/kaldi/tools/openfst-1.6.7/lib/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/share/gdb/auto-load/usr/lib64/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/usc/gnu/gcc/4.9.3/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/usc/gnu/gcc/4.9.3/lib64/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/usc/cuda/9.2/lib64/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/auto/dr-std/pg3/softwares/kaldi/tools/sox/sox-14.4.2/src/.libs

LIB_DIR=/auto/dr-std/pg3/softwares/lib${LIB_DIR:+:$LIB_DIR}
LIB_DIR=$LIB_DIR:/auto/dr-std/pg3/softwares/kaldi/tools/openfst-1.6.7/lib/
LIB_DIR=$LIB_DIR:/usr/usc/gnu/gcc/4.9.3/lib64/
LIB_DIR=$LIB_DIR:/usr/usc/cuda/9.2/lib64/


LIBRARY_PATH=/usr/usc/gnu/gcc/4.9.3/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
LIBRARY_PATH=$LIBRARY_PATH:/usr/usc/gnu/gcc/4.9.3/lib64
LIBRARY_PATH=$LIBRARY_PATH:/usr/usc/cuda/9.2/lib64/


export PATH LIB_DIR LD_LIBRARY_PATH INCLUDE_PATH

export CXX=/usr/usc/gnu/gcc/4.9.3/bin/g++


