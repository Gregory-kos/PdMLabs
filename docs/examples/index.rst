💡 Examples
===========

Runnable notebooks live in the `example/ <https://github.com/PdM-Labs/PdMLabs/tree/main/example>`_
directory of the repository. Clone the repo and open them locally with Jupyter, or browse them
on GitHub.

Implement your own method
-------------------------

`Implement_your_own_method.ipynb <https://github.com/PdM-Labs/PdMLabs/blob/main/example/Implement_your_own_method.ipynb>`_

Walks through subclassing :class:`~pdmlabs.method.semi_supervised_method.SemiSupervisedMethodInterface`
to plug a custom anomaly detection method into a PdMLabs pipeline, including how to raise
:class:`~pdmlabs.exceptions.exception.NotFitForSourceException` when ``predict`` is called
before ``fit``.

Use a custom dataset
--------------------

`example_custom_dataset.ipynb <https://github.com/PdM-Labs/PdMLabs/blob/main/example/example_custom_dataset.ipynb>`_

Shows how to shape your own time-series data into the dataset dictionary that the experiment
flavors expect, using :class:`~pdmlabs.utils.dataset.Dataset`.

.. seealso::

   :doc:`../getting-started/quickstart` for a 5-minute end-to-end run, and
   :doc:`../user-guide/index` for step-by-step guidance.
