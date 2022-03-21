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
                      'matplotlib',
                      'torch',
                      'tqdm',
                      'torchvision',
                      'torch-summary',
                      'hydra',
                      'hydra.core',
                      'dataclasses',
                      'OmegaConf',
                      'sklearn',
                      'scikit-image',
                      'pytorch-lightning',
                      'lightly',
                      'toolz',
		                  'ipykernel',
                      'pytorch_ssim',
                      ],
    extras_require={
      "anatomist": ['deep_folding @ \
                        git+https://git@github.com/neurospin/deep_folding',
                      ],
    },
    url='https://github.com/neurospin-projects/2022_jchavas_cingulate_inhibitory_control',
    author='Joël Chavas',
    author_email='joel.chavas@cea.fr'
)
