=====
LStar
=====

LStar is a variant implemetation of Dana Angluin's classic L\* algorithm for inference of regular languages.
It is loosely based on the following three papers:

* Angluin, Dana. "Learning regular sets from queries and counterexamples." Information and computation 75.2 (1987): 87-106
* Rivest, Ronald L., and Robert E. Schapire. "Inference of finite automata using homing sequences." Information and Computation 103.2 (1993): 299-347.
* Maler, Oded, and Irini-Eleftheria Mens. "Learning regular languages over large alphabets."
  International Conference on Tools and Algorithms for the Construction and Analysis of Systems. Springer, Berlin, Heidelberg, 2014.

It's all a bit prototypey at the moment, but I've found myself lacking a high quality implementation of this concept in the past,
and the aim is for this to eventually become that.
For now, use with extreme caution.

-----------
Basic Usage
-----------

In lieu of documentation, here is some really embarrassingly basic example usage.
More documentation to follow shortly, honest.

.. code-block:: pycon
    >>> from lstar import LStarLearner
    >>> # Create a learner for a given classifier
    >>> x = LStarLearner(lambda x: len(x) <= 2 and sum(x) <= 10)
    >>> # Starts out as the trivial automaton accepting everything
    >>> x.learned_automaton.debug()
    LearnedAutomaton:
      State s0, labelled True
        0+ -> s0
    >>> # Give it two counter-examples and tell it to make its
    >>> # automaton consistent with them.
    >>> x.make_consistent((0,) * 3)
    True
    >>> x.make_consistent((6,) * 2)
    True
    >>> # Now we have a much more involved automaton.
    >>> x.learned_automaton.debug()
    LearnedAutomaton:
      State s0, labelled True
        0-4 -> s1
        5+ -> s4
      State s1, labelled True
        0+ -> s2
      State s4, labelled True
        0-5 -> s2
        6+ -> s3
      State s2, labelled True
        0+ -> s3
      State s3, labelled False
        0+ -> s3

The automata generated support a reasonable set of basic methods for navigating the automata.
It's all available via the class documentation, I just haven't got around to setting up sphinx for this yet.
