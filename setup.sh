# create new conda environment
yes | conda create -q --name image-gen-prj -c conda-forge python=3.11.10

# activate the environment
conda activate image-gen-prj

# install the required packages
yes | conda install pip3
pip3 install --upgrade pip
pip3 install -r requirements.txt   