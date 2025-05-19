echo "Initializing environment and installing dependencies"

apt-get update && apt-get upgrade -y && apt-get install -y wget unzip cmake

if [ -d "../miniconda" ]
then
  echo "miniconda already installed"
else
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ../miniconda.sh
  bash ../miniconda.sh -b -p ../miniconda
fi

source ../miniconda/etc/profile.d/conda.sh

conda init bash

make initialize_environment

# for some reason this needs to be run again
source ../miniconda/etc/profile.d/conda.sh
conda activate gnn

# hack to fix a bug when running in GCE
export BOTO_CONFIG=null

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

make install_dev_deps

source ~/.profile
