#!/bin/sh
    
# Activating the related conda environment
#/home/nest/miniconda3/bin/conda init
#exec bash
# conda activate cobel-spike
    
# NEST is installed here. When you relocate NEST, change this variable.
export NEST_INSTALL_DIR=/builds/cns/1-frameworks/cobel-spike/packages/nest-install
    
# MUSIC is installed here.
export MUSIC_INSTALL_DIR=/builds/cns/1-frameworks/cobel-spike/packages/music-install
    
# MUSIC adapters installed here.
export MUSIC_AD_INSTALL_DIR=/builds/cns/1-frameworks/cobel-spike/packages/music-adapters-install
    
# NEST finds standard *.sli files $NEST_DATA_DIR/sli
export NEST_DATA_DIR=$NEST_INSTALL_DIR/share/nest
    
# NEST finds help files $NEST_DOC_DIR/help
export NEST_DOC_DIR=$NEST_INSTALL_DIR/share/doc/nest
    
# The path where NEST looks for user modules.
export NEST_MODULE_PATH=$NEST_INSTALL_DIR/lib/nest
    
# The path where the PyNEST bindings are installed.
export NEST_PYTHON_PREFIX=$NEST_INSTALL_DIR/lib/python3.6/site-packages
    
# The path where python-music bindings are installed.
export MUSIC_PYTHON=$MUSIC_INSTALL_DIR/lib/python3.6/site-packages
    
export PYTHONPATH=$NEST_PYTHON_PREFIX:$MUSIC_PYTHON:$PYTHONPATH
    
export LD_LIBRARY_PATH=$MUSIC_INSTALL_DIR/lib:$MUSIC_AD_INSTALL_DIR/lib
    
# Make nest / sli /... executables visible.
export PATH=$NEST_INSTALL_DIR/bin:$MUSIC_INSTALL_DIR/bin:$MUSIC_AD_INSTALL_DIR/bin:$PATH
