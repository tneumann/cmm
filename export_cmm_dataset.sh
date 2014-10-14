#!/bin/sh
# computes and exports the supplementary data

set -e

BASEDIR=cmm_dataset
OPTIONS="-off -ply"

mkdir -p $BASEDIR/horse
python compute_cmm.py meshes/horse_976.obj 30 100 -maxiter 5000 -s -o $BASEDIR/horse $OPTIONS

mkdir -p $BASEDIR/bumpy_cube
python compute_cmm.py meshes/bumpy_cube6.obj 26 20 -o $BASEDIR/bumpy_cube $OPTIONS

mkdir -p $BASEDIR/armadillo
python compute_cmm.py meshes/armadillo_25479.obj 15 100 -s -o $BASEDIR/armadillo $OPTIONS

mkdir -p $BASEDIR/bunny
python compute_cmm.py meshes/bunny_fixed.obj 20 100 -maxiter 10000 -s -o $BASEDIR/bunny $OPTIONS

mkdir -p $BASEDIR/hand
python compute_cmm.py meshes/hand_868.obj 6 2 -o $BASEDIR/hand $OPTIONS

mkdir -p $BASEDIR/hand_holes
python compute_cmm.py meshes/hand_868_holes2.obj 6 2 -o $BASEDIR/hand_holes $OPTIONS

# these meshes are not bundled right now
#mkdir -p $BASEDIR/dragon
#python compute_cmm.py meshes/stanford/xyzrgb_dragon_50k.off 20 100 -s -o $BASEDIR/dragon $OPTIONS

#mkdir -p $BASEDIR/elephant
#python compute_cmm.py meshes/aimshape/elephant_50000faces_edited.zip/elephant_50000faces_edited.obj 10 100 -s -o $BASEDIR/elephant $OPTIONS

