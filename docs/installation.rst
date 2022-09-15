.. highlight:: shell

============
Installation
============


Stable release
--------------

To install ``tmmc-lnpy``, run this command in your terminal:

.. code-block:: console

    $ pip install tmmc-lnpy


Alternatively, install with conda or mamba:

.. code-block:: console

   $ conda install -c wpk-nist tmmc-lnpy



From sources
------------

The sources for lnpy can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone {the repo}

To create a conda environment with the needed dependencies, use:

.. conda-block:: console

   $ conda env create -n {optional-env-name (defaults to lnpy-env)} -f environment.yaml


To install an editable version of of ``tmmc-lnpy``, use to following:

.. code-block:: console

    $ pip install -e . --no-deps





.. _Github repo: https://github.com/usnistgov/tmmc-lnpy
