===================================================
Conditional Generation Using Polynomial Expansions
===================================================

Chainer implementation of the conditional image generation experiments as described on the NeurIPS'21 paper "**Conditional Generation Using Polynomial Expansions**".

Specifically, we include the code for image generation on class-conditional generation with CIFAR10.



Browsing the folders
====================
The folder structure is the following:

*    ``gen_models``: The folder for the generator models; this is the primary folder with the polynomial generator(s).

*    ``dis_models``: The folder for the discriminator models; do not modify the optional arguments, unless you know what you are doing.

*    ``updater``: The folder that contains the core code for the updater, i.e. the chunk of code that runs in every iteration to update the modules.

*    ``jobs``: It contains a) the yml with the hyper-parameter setting, b) the main command to load the models and run the training (train_mn*.py).

*    ``source``: It contains auxiliary code, you should probably not modify any of that code.

*    ``evaluations``: It contains the code for validation (either during training or offline).

*    ``datasets``: Each file in the folder describes a parser for a dataset; normally you should not modify it for provided dataset(s).



Train the network
=================

To train the network, you can execute the following command::

   python jobs/cope_neurips21/train_mn_cifar.py --label my_experiment

The yml file describes the modules/dataset to train on. The default dataset used is 
`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_; other datasets can 
be used by modifying the respective paths in the yml.

The hyper-parameters are included in the yml, no need to hardcode them in the files. 

The results are exported in the directory defined in `train_mn_cifar.py` (default `results_2var_neurips`). 
The directory with the results includes folders named as [start_date][label]; inside each respective
folder the pretrained weights and few indicative images are exported. 

By default, the code is using evaluation of the IS and FID every 2,000 iterations, so please make sure to either
include the related model in the path dictated in the `train_mn_cifar.py <https://github.com/grigorisg9gr/polynomial_nets_for_conditional_generation/blob/master/conditional_generation_with_gan/jobs/cope_neurips21/train_mn_cifar.py#L46>`_ or to change the frequency for the
evaluation in the yml (line `evaluation_interval <https://github.com/grigorisg9gr/polynomial_nets_for_conditional_generation/blob/master/conditional_generation_with_gan/jobs/cope_neurips21/cifar10_cnn_2var_prodpoly_linear_conditional.yml#L8>`_). 


Requirements
============

The code requires the chainer library.

Tested on a Linux machine with:

* chainer=4.0.0, chainercv=0.9.0,

* chainer=5.2.0, chainercv=0.12.0.

* chainer=6.1.0, chainercv=0.13.1.


The code is highly influenced by [1]_.

Apart from Chainer, the code depends on Pyaml [2]_ and [3]_ (for the evaluation). 


References
==========

.. [1] https://github.com/grigorisg9gr/polynomial_nets/

.. [2] https://pypi.org/project/pyaml/

.. [3] https://github.com/djsutherland/skl-groups

