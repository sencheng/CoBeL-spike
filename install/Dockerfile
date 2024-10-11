FROM ubuntu:20.04 as build

#Install required ubuntu packages
ADD 	. /cobel-spike
RUN	TZ=Europe/Berlin &&\
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt-get update -y &&\
	apt-get install --no-install-recommends -y python3-dev python3-wheel python3-pip cmake git unzip wget automake cython3 libgsl-dev libltdl-dev libncurses-dev libreadline-dev openmpi-bin libopenmpi-dev ffmpeg pkg-config libjsoncpp-dev libzmq3-dev libblas-dev python-is-python3 gcc libtool g++ make  &&\
#Preparations
	cd cobel-spike &&\
	pip install pip==20.0.2 wheel==0.37.1 setuptools==44.0.0  &&\
	pip install --no-cache-dir -r install/environment.txt &&\
	mkdir packages &&\
	cd packages &&\
	git clone https://github.com/INCF/MUSIC.git &&\
	git clone https://github.com/bghazinouri/python-gymz.git &&\
	git clone https://github.com/bghazinouri/music-adapters.git &&\
	git clone https://github.com/bghazinouri/gym_env &&\
	wget https://github.com/nest/nest-simulator/archive/v2.20.0.tar.gz &&\
	tar -xzvf v2.20.0.tar.gz nest-simulator-2.20.0 &&\
#Install MUSIC
	mkdir music-install &&\
	cd MUSIC &&\
	./autogen.sh &&\
	./configure --prefix=/cobel-spike/packages/music-install &&\
	make -j4 &&\
	make install &&\
	cd .. &&\
#Install NEST
	mkdir nest-build &&\ 
	mkdir nest-install &&\	
	cd nest-build &&\
	cmake -DCMAKE_INSTALL_PREFIX:PATH=/cobel-spike/packages/nest-install /cobel-spike/packages/nest-simulator-2.20.0 -Dwith-mpi=ON -Dwith-music=/cobel-spike/packages/music-install &&\
	make -j4 &&\
	make install &&\
	cd .. &&\
#Install MUSIC Adapter
	mkdir music-adapters-build &&\
	mkdir music-adapters-install &&\
	cd music-adapters-build &&\
	cmake -DCMAKE_INSTALL_PREFIX:PATH=/cobel-spike/packages/music-adapters-install -DMUSIC_ROOT_DIR=/cobel-spike/packages/music-install /cobel-spike/packages/music-adapters &&\
	make -j4 &&\ 
	make install &&\
#Install gym-open field and gymz
	cd ../gym_env/gym-openfield &&\
	pip install -e . &&\
	cd ../.. &&\
	cd python-gymz &&\
	pip install -e . &&\
	cd .. &&\
#Clean packages folder
	rm -r MUSIC nest-simulator-2.20.0 nest-build music-adapters music-adapters-build &&\
	rm v2.20.0.tar.gz && \
#Permissions
	chmod +x /cobel-spike/simulator/run_simulation.sh


FROM ubuntu:20.04 as prod

#Create variables
ENV 	NEST_INSTALL_DIR=/cobel-spike/packages/nest-install\
	MUSIC_INSTALL_DIR=/cobel-spike/packages/music-install\
	MUSIC_AD_INSTALL_DIR=/cobel-spike/packages/music-adapters-install\
	NEST_DATA_DIR=/cobel-spike/packages/nest-install/share/nest\
	NEST_DOC_DIR=/cobel-spike/packages/nest-install/share/doc/nest\
	NEST_MODULE_PATH=/cobel-spike/packages/nest-install/lib/nest\
	NEST_PYTHON_PREFIX=/cobel-spike/packages/nest-install/lib/python3.8/site-packages\
	MUSIC_PYTHON=/cobel-spike/packages/music-install/lib/python3.8/site-packages\
	PYTHONPATH=/cobel-spike/packages/nest-install/lib/python3.8/site-packages:/cobel-spike/packages/music-install/lib/python3.8/site-packages:$PYTHONPATH\
	LD_LIBRARY_PATH=/cobel-spike/packages/music-install/lib:/cobel-spike/packages/music-adapters-install/lib\
	PATH=/cobel-spike/packages/nest-install/bin:/cobel-spike/packages/music-install/bin:/cobel-spike/packages/music-adapters-install/bin:$PATH\
	OMPI_ALLOW_RUN_AS_ROOT=1\
	OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


#Get ubuntu packages
RUN	TZ=Europe/Berlin &&\
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt-get update -y &&\
	apt-get install --no-install-recommends -y python3-dev python-is-python3 cython3 libgsl-dev libltdl-dev libncurses-dev libreadline-dev openmpi-bin libopenmpi-dev ffmpeg pkg-config libjsoncpp-dev libzmq3-dev libblas-dev python3-pyqt5 &&\
#Cleanup
	apt-get clean &&\
	apt-get -y autoremove &&\
	rm -rf /var/lib/apt/lists/*


#Copy from the build image
COPY 	--from=build /cobel-spike /cobel-spike
COPY 	--from=build /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY 	--from=build usr/local/bin/gymz-controller usr/local/bin/gymz-controller
