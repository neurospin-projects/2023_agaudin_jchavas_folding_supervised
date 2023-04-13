
Unsupervised Representation Learning of Cingulate Cortical Folding Patterns
###########################################################################

Official Pytorch implementation for Unsupervised Learning and Cortical Folding (`paper <https://openreview.net/forum?id=ueRZzvQ_K6u>`_).
The project aims to study cortical folding patterns thanks to unsupervised deep learning methods.


Dependencies
------------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3


Set up the work environment
---------------------------
First, the repository can be cloned thanks to:

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2023_agaudin_jchavas_folding_supervised
    cd 2023_agaudin_jchavas_folding_supervised

Then, install the a virtual environment through the following command lines:

.. code-block:: shell

    python3 -m venv venv
    . venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .

Note that you might need a `BrainVISA <https://brainvisa.info>`_ environment to run
some of the functions or notebooks.