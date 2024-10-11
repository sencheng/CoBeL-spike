## Downgrading cmake in ubuntu 24.04

The correct version of cmake is crucial for installing MUSIC and MUSIC-adapters. While you can install the correct version of cmake easily with apt in ubuntu 20.04 and ubuntu 22.04, this is unfortunately not possible with ubuntu 24.04. One solution that worked on our test system suggests to delete the current version of cmake and install an earlier version of cmake from source. This approach has downsides like potential dependency issues, package manager conflicts or problems with the system integration but is as far as we know the only way to install CoBeL-spike in ubuntu 24.04! Installing the correct version (cmake 3.22.1) involves following steps:
```bash
apt purge cmake
apt install libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz
tar zxvf cmake-3.22.1.tar.gz
cd cmake-3.22.1
./bootstrap
make
make install
```
Keep in mind that you need root access to install cmake from source. Furthermore, we recommend deleting the cmake version installed from source and re-installing cmake with apt again. Therefore, you need to go to the build directory *cmake-3.22.1* and run:
```bash
make uninstall
apt install cmake
```
Keep in mind that these commands need root access too. The build directory can be deleted afterwards.

## Installing python 3.8 in ubuntu 22.04 and ubuntu 24.04
The last supported python version of NEST-simulator 2.20 is 3.8, which is not pre-installed in ubuntu 22.04 and ubuntu 24.04. Our solution suggests to add the PPA [deadsnakes](https://launchpad.net/%7Edeadsnakes/+archive/ubuntu/ppa) and install python3.8 from this repository. Installing python3.8 involves following steps:
```bash
apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.8 python3.8-venv python3.8-dev
```
To execute these commands you need root access.