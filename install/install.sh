#!/bin/sh

# there is a problem when activating python3 envs from a shell script when using sh.
# the solution is to either run the script with bash (adding the #!/bin/bash shebang line) or use "source install.sh"
# or replace "source"  with "." in the script, as I have done

# make working folder
mkdir ../../packages
# make python environment
cd ../
REPOSDIR=$(pwd)
cd ../packages
BASEDIR=$(pwd)
ESCAPED=$(pwd | sed 's/\//\\\//g')

# IMPORTANT :
# source ./install.sh to run the script
# the script requires the nest_vars_example file to be in the same folder
# In cse your python version is different than 3.8, update the value inside nest_vars_example file

echo "installing the cobel toolchain in $BASEDIR"

mkdir cobel
python3.8 -m venv ./cobel/
. $BASEDIR/cobel/bin/activate
pip install pip==20.0.2 wheel==0.37.1 setuptools==44.0.0 # installed beforhand so all the dependencies can install without error
pip install --no-cache-dir -r $REPOSDIR/install/environment.txt


# install MUSIC 
git clone https://github.com/INCF/MUSIC.git 
mkdir MUSIC_install
cd $BASEDIR/MUSIC
. ./autogen.sh
./configure --prefix=$BASEDIR/MUSIC_install 
make -j4  
make install 
cd $BASEDIR
rm -rf $BASEDIR/ltmain.sh

# install nest
wget https://github.com/nest/nest-simulator/archive/v2.20.0.tar.gz 
tar -xzvf v2.20.0.tar.gz 
mkdir nest-simulator-2.20.0_install
mkdir nest-simulator-2.20.0_build
cd nest-simulator-2.20.0_build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/nest-simulator-2.20.0_install $BASEDIR/nest-simulator-2.20.0 -Dwith-mpi=ON -Dwith-music=$BASEDIR/MUSIC_install 
make -j4 
make install 
cd $BASEDIR

# install music adapters
git clone https://github.com/bghazinouri/music-adapters.git
mkdir music-adapters_install
mkdir music-adapters_build
cd music-adapters_build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/music-adapters_install -DMUSIC_ROOT_DIR=$BASEDIR/MUSIC_install $BASEDIR/music-adapters 
make -j4 
make install 
cd $BASEDIR

# install gym dependencies
cd $BASEDIR
git clone https://github.com/bghazinouri/gym_env
cd gym_env/gym-openfield
pip install -e .
cd $BASEDIR
git clone https://github.com/bghazinouri/python-gymz.git 
cd python-gymz/
pip install -e . 

# configure nest_vars.sh
cp $REPOSDIR/install/nest_vars_example.sh $BASEDIR/nest-simulator-2.20.0_install/bin/nest_vars.sh 
sed -i "s/<root>/$ESCAPED/g" $BASEDIR/nest-simulator-2.20.0_install/bin/nest_vars.sh
echo $BASEDIR
. $BASEDIR/nest-simulator-2.20.0_install/bin/nest_vars.sh
cd $REPOSDIR

# save paths to enviornment and nest_vars as script
# running the output script can be added to the run_simulation.sh script
echo "#!/bin/sh" > $REPOSDIR/simulator/source_paths.sh
echo ". $BASEDIR/cobel/bin/activate" >> $REPOSDIR/simulator/source_paths.sh
echo ". $BASEDIR/nest-simulator-2.20.0_install/bin/nest_vars.sh" >> $REPOSDIR/simulator/source_paths.sh
chmod +x $REPOSDIR/simulator/source_paths.sh

# remove unnessecary directories
rm -rf $BASEDIR/music-adapters_build
rm -rf $BASEDIR/nest-simulator-2.20.0_build
rm -rf $BASEDIR/nest-simulator-2.20.0
