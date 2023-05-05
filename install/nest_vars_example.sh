#!/bin/sh

# NEST is installed here. When you relocate NEST, change this variable.
export NEST_INSTALL_DIR=<root>/nest-simulator-2.20.0_install

# NEST finds standard *.sli files $NEST_DATA_DIR/sli
export NEST_DATA_DIR=$NEST_INSTALL_DIR/share/nest

# NEST finds help files $NEST_DOC_DIR/help
export NEST_DOC_DIR=$NEST_INSTALL_DIR/share/doc/nest

# The path where NEST looks for user modules.
export NEST_MODULE_PATH=$NEST_INSTALL_DIR/lib/nest

# The path where the PyNEST bindings are installed.
export NEST_PYTHON_PREFIX=$NEST_INSTALL_DIR/lib/python3.8/site-packages

# Prepend NEST to PYTHONPATH in a safe way even if PYTHONPATH is undefined
export PYTHONPATH=$NEST_PYTHON_PREFIX${PYTHONPATH:+:$PYTHONPATH}

# MUSIC is installed here.
export MUSIC_INSTALL_DIR=<root>/MUSIC_install

# MUSIC adapters installed here.
export MUSIC_AD_INSTALL_DIR=<root>/music-adapters_install

# The path where python-music bindings are installed.
export MUSIC_PYTHON=$MUSIC_INSTALL_DIR/lib/python3.8/site-packages

export LD_LIBRARY_PATH=$MUSIC_INSTALL_DIR/lib:$MUSIC_AD_INSTALL_DIR/lib


# Make nest / sli /... executables visible.
export PATH=$NEST_INSTALL_DIR/bin:$MUSIC_INSTALL_DIR/bin:$MUSIC_AD_INSTALL_DIR/bin:$PATH

