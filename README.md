# Project: Comparison of Image Generation Approaches

This project is the result of a laboratory study exploring the primary approaches to image generation using neural networks. The developed framework provides functionality for training, comparing quality metrics, and conducting inference with two major strategies: `Generative Adversarial Networks (GAN)` and `Denoising Diffusion Probabilistic Models (DDPM)`.

Alongside the main training pipelines, the project includes several modifications of vanilla methods aimed at improving final results. A detailed report of all experiments conducted based on these modifications is available in the `./docs` directory.

![example image](example.png)

## Setup workflow

The project is compatible with various environment managers. Examples using `pipenv` and `conda` are provided below:

**pipenv**
```
pipenv install
pipenv shell
pip install -r requirements.txt
```

**conda**
```
conda create -q --name image-gen-prj -c conda-forge python=3.11.10
conda activate image-gen-prj

# Install dependencies
conda install pip
pip install --upgrade pip
pip install -r requirements.txt
```

An automated setup script is also available:

```
sh setup.sh
```

## Example Inference

The project provides pretrained models for inference:

1. **Downloading pretrained models**
```
from huggingface_hub import snapshot_download
snapshot_download('Artem-fm/gen-ai-task', local_dir='./checkpoints')
```

2. **Running inference** Examples can be found in the `./scripts/` directory.



## Results and Reports

Detailed results and comparisons are documented in the following reports:

* [GAN Report](docs/gan-report.pdf)
* [DDPM Report](docs/ddpm-report.pdf)

## Contact

For questions or further information, please contact: 

**email:** fedorov.am@list.ru
**telegram:** t.me/fedorov_AMfm