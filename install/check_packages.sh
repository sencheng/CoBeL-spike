#!/bin/bash


# Function to check if a package is installed
check_package() {
    package_name="$1"
    print_output="$2"

    if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "install ok installed"; then
        if "$print_output"; then
            echo -e "\e[32mPackage '$package_name' is installed.\e[0m"
        fi
        return 1
    fi

    if "$print_output"; then
        echo -e "\e[31mPackage '$package_name' is NOT installed with apt. Either install '$package_name' with apt or make sure that you have installed '$package_name' with other installation methods.\e[0m"
    fi
    return 0
}


# Checks if the package is in the desired range and outputs a warning if the package is not in the range
check_version() {
    package_name="$1"
    lower_bound="$2"
    upper_bound="$3"
    
    check_package "$package_name" false
    package_available=$?

    if [ "$package_available" -eq 1 ]; then
        check_version_range
        correct_version=$?

        if [ "$correct_version" -eq 0 ]; then
            echo -e "\e[31mPackage '$package_name' is NOT in the range of {$lower_bound, $upper_bound}. Check the README in cobel-spike/install for help.\e[0m"
        fi
    else
        echo "$package_name is not available"
    fi
}


# Helper function for check_version(). Checks if the package is in the desired range
check_version_range() {
    # Get the installed version
    installed_version=$(dpkg-query -W -f='${Version}' "$package_name")

    # Check if installed version is within the range
    if dpkg --compare-versions "$installed_version" ge "$lower_bound" && dpkg --compare-versions "$installed_version" le "$upper_bound"; then
        return 1
    else
        return 0
    fi
}


UBUNTU_VERSION=$(lsb_release -rs)

if [ "$UBUNTU_VERSION" = "20.04" ]; then
    echo "Running script for Ubuntu 20.04"
    declare -a packages=("python3.8" "python3.8-venv" "python3.8-dev" "python3-wheel" "python3-pip" "python3-pyqt5" "cmake" "git" "unzip" "wget" "automake" "cython" "libgsl-dev" "libltdl-dev" "libncurses-dev" "libreadline-dev" "openmpi-bin" "libopenmpi-dev" "ffmpeg" "pkg-config" "libjsoncpp-dev" "libzmq3-dev" "libblas-dev" "gcc" "libtool" "g++" "make")

    for package in "${packages[@]}"; do
        check_package "$package" true
    done


elif [ "$UBUNTU_VERSION" = "22.04" ]; then
    echo "Running script for Ubuntu 22.04"
    declare -a packages=("python3.8" "python3.8-venv" "python3.8-dev" "python3-wheel" "python3-pip" "python3-pyqt5" "cmake" "git" "unzip" "wget" "automake" "cython3" "libgsl-dev" "libltdl-dev" "libncurses-dev" "libreadline-dev" "openmpi-bin" "libopenmpi-dev" "ffmpeg" "pkg-config" "libjsoncpp-dev" "libzmq3-dev" "libblas-dev" "gcc" "libtool" "g++" "make")

    for package in "${packages[@]}"; do
        check_package "$package" true
    done

elif [ "$UBUNTU_VERSION" = "24.04" ]; then
    echo "Running script for Ubuntu 24.04"
    declare -a packages=("python3.8" "python3.8-venv" "python3.8-dev" "python3-wheel" "python3-pip" "python3-pyqt5" "cmake" "git" "unzip" "wget" "automake" "cython3" "libgsl-dev" "libltdl-dev" "libncurses-dev" "libreadline-dev" "openmpi-bin" "libopenmpi-dev" "ffmpeg" "pkg-config" "libjsoncpp-dev" "libzmq3-dev" "cppzmq-dev" "libblas-dev" "gcc" "libtool" "g++" "make")

    for package in "${packages[@]}"; do
        check_package "$package" true
    done

    check_version "cmake" "3.16.3" "3.22.1"

else
    echo "Checking for you version $UBUNTU_VERSION is currently not available!"
fi


