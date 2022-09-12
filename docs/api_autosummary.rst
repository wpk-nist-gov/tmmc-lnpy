#############
API Reference
#############


Masked lnPi objects
===================
.. currentmodule:: lnPi

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   MaskedData
   lnPiData

Collection of lnPi
==================


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   MaskedDataCollection


Ensemble properties
===================

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   xGrandCanonical
   xCanonical

===============
Multiple phases
===============

Local free energy calculation
=============================

.. currentmodule:: lnPi

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   wFreeEnergy
   wFreeEnergyCollection
   wFreeEnergyPhases


.. autosummary::
   :toctree: generated/

   merge_regions



Segmentation
============


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Segmenter
   PhaseCreator

   BuildPhases_mu
   BuildPhases_dmu

.. autosummary::
   :toctree: generated/

   peak_local_max_adaptive


Stability
=========

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Spinodals
   Binodals


.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   MaskedDataCollection.binodal
   MaskedDataCollection.spinodal


Utilities
=========

.. currentmodule:: lnPi

.. autosummary::
   :toctree: generated/

   utils.mask_change_convention
   utils.masks_change_convention
   utils.labels_to_masks
   utils.masks_to_labels
   utils.distance_matrix
