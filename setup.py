import os
from setuptools import setup, find_packages

setup(
    name='2023_agaudin_jchavas_folding_supervised',
    version='0.0.1',
    packages=find_packages(
        exclude=['tests*', 'notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning models '
                'to analyze anterior cingulate sulcus patterns',
    long_description=open('README.rst').read(),
    install_requires=['pandas',
                      'scipy',
		              'psutil',
		              'orca',
                      'matplotlib',
                      'torch',
                      'tqdm',
                      'pqdm',
                      'torchvision',
                      'torch-summary',
                      'tensorboard',
                      'hydra-core',
                      'hydra-joblib-launcher',
                      'dataclasses',
                      'OmegaConf',
                      'scikit-learn',
                      'scikit-image',
                      'pytorch-lightning>=1.8.0',
                      'lightly',
                      'plotly',
                      'toolz',
		              'ipykernel',
                      'kaleido',
                      'pytorch_ssim',
                      'seaborn',
                      'statsmodels',
                      'umap-learn',
                      'numpy',
                      'plotly',
                      'pqdm',
                      'wandb'
                      ],
    extras_require={"anatomist": ['deep_folding @ \
                        git+https://git@github.com/neurospin/deep_folding',
                      ],
    },
    url='https://github.com/neurospin-projects/2023_agaudin_jchavas_folding_supervised',
    author='Joël Chavas, Aymeric Gaudin, Julien Laval',
    author_email='joel.chavas@cea.fr, aymeric.gaudin@cea.fr, julien.laval@cea.fr'
)
