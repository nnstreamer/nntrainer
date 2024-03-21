set -e

BASEDIR=$(dirname $0)

pushd $BASEDIR
    BiQGEMM_REPO_DIR=BiQGEMM
    rm -rf $BiQGEMM_REPO_DIR
    git clone https://github.sec.samsung.net/AIP/BiQGEMM.git

    NNTRAINER_BiQGEMM_DIR=nntrainer/tensor/BiQGEMM
    rm -rf $NNTRAINER_BiQGEMM_DIR
    [ -e nntrainer/tensor/BiQGEMM.h ] && rm nntrainer/tensor/BiQGEMM.h

    mkdir $NNTRAINER_BiQGEMM_DIR
    cp -r $BiQGEMM_REPO_DIR/src $NNTRAINER_BiQGEMM_DIR
    cp $BiQGEMM_REPO_DIR/BiQGEMM.h nntrainer/tensor
    sed -i 's\src\BiQGEMM/src\g' nntrainer/tensor/BiQGEMM.h
    sed -i '3 i #define __ARM_NEON__ 1' $NNTRAINER_BiQGEMM_DIR/src/Core/Weights/Weights.h
    sed -i 's/num_threads = 1/num_threads = '"$1"'/g' $NNTRAINER_BiQGEMM_DIR/src/Core/setting/Setting.h

    BLAS_NEON_SETTING=nntrainer/tensor/blas_neon_setting.h
    sed -i '27 d' $BLAS_NEON_SETTING
    sed -i '27 i   static size_t num_threads = '"$2"';' $BLAS_NEON_SETTING
    sed -i '48 d' $BLAS_NEON_SETTING
    sed -i '48 i   static size_t num_threads = '"$3"';' $BLAS_NEON_SETTING
    rm -rf $BiQGEMM_REPO_DIR
popd