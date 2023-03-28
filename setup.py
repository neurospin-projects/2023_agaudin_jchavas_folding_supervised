import os
from setuptools import setup, find_packages

setup(
    name='2022_jchavas_cingulate_inhibitory_control',
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
                      'tqdm==4.51.0',
                      'torchvision',
                      'torch-summary',
                      'tensorboard',
                      'hydra',
                      'hydra.core',
                      'dataclasses',
                      'OmegaConf',
                      'scikit-learn==0.24.2',
                      'scikit-image',
                      'pytorch-lightning',
                      'lightly',
                      'toolz',
		              'ipykernel',
                      'kaleido',
                      'pytorch_ssim',
                      'seaborn',
                      'statsmodels',
                      'umap-learn'
                      ],
    extras_require={"anatomist": ['deep_folding @ \
                        git+https://git@github.com/neurospin/deep_folding',
                      ],
    },
    url='https://github.com/neurospin-projects/2022_jchavas_cingulate_inhibitory_control',
    author='Joël Chavas, Aymeric Gaudin',
    author_email='joel.chavas@cea.fr, aymeric.gaudin@cea.fr'
)
