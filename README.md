# Autonomous Perception

## Installation
You should install
sudo -H pip3 install googledrivedownloader

sudo -H pip3 install tensorflow-gpu==2.1.0 
sudo -H pip3 install googledrivedownloader2

cuda-10.1
libcudnn7
libcudnn7-dev

## Usage

### Compare PSPNET
To compare with PSPNet you can use our converted pspnet_model.pb ou generate it yourself.

To generate you should clone the repository and set the environment variable.

```bash
    git clone https://github.com/hellochick/PSPNet-tensorflow.git
    export PSPNET_HOME='/home/nelson/projects/PSPNet-tensorflow'
```

Download the models, unzip and copy to "PSPNet-tensorflow" folder.

```bash
wget https://modeldepot.io/assets/uploads/models/models/8a738312-b8dd-4c8e-8db7-1d6f6a0ff9f8_model.zip -O PSPNet_model.zip
unzip PSPNet_model.zip
mv model PSPNet-tensorflow
```

Then, you should run generate_pspnet_pb.py to generate the .pb model. As this project use tensorflow 2, you shuold install tf1 to generate it.

```bash
sudo -H pip3 install -U virtualenv
virtualenv env
pip3 install tensorflow
pip3 install spyder==4.0.0b5
pip3 install scipy
pip3 install matplotlib
sudo -H pip3 install ipywidgets
sudo -H pip3 install  imgaug==0.4.0
```
``` python
python3 ./src/scripts/generate_pspnet_pb.py
```

You can also just download model.pb and copy it in ./data/psp_cityscape/

```bash
cd ./data/psp_cityscape/
wget https://**/model.pb

```

To run the comparison




## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History




