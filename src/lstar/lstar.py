#  LStar, a library for inference of regular classifiers
#  Copyright (C) 2019 David R. MacIver

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


from functools import lru_cache
from array import array
from collections import deque
import bisect

ARRAY_CODES = ["B", "H", "I", "L", "Q", "O"]
NEXT_ARRAY_CODE = dict(zip(ARRAY_CODES, ARRAY_CODES[1:]))


def array_or_list(ls):
    best = list(ls)
    hi = len(ARRAY_CODES) - 1

    def is_good(i):
        if i == hi:
            return True
        nonlocal best
        try:
            best = array(ARRAY_CODES[i], ls)
            return True
        except OverflowError:
            return False

    if not is_good(0):
        lo = 0
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if is_good(mid):
                hi = mid
            else:
                lo = mid
    return best


class IntString(object):
    """An IntString represents an immutable sequence of non-negative integers,
    stored in a compact form."""

    __slots__ = ("__values", "__hash")

    def __init__(self, values=()):
        if isinstance(values, IntString):
            self.__values = values.__values
            self.__hash = values.__hash
        else:
            self.__values = array_or_list(values)
            self.__hash = None

    def __len__(self):
        return len(self.__values)

    def __iter__(self):
        return iter(self.__values)

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(tuple(self.__values))
        return self.__hash

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.__values[i]
        else:
            return IntString(self.__values[i])

    def __add__(self, other):
        if not isinstance(other, IntString):
            other = IntString(other)
        try:
            return IntString(self.__values + other.__values)
        except TypeError:
            return IntString(list(self.__values) + list(other.__values))

    def __radd__(self, other):
        return IntString(other) + self

    @classmethod
    def singleton(self, c):
        return IntString((c,))

    def __repr__(self):
        return "IntString(%r)" % (list(self.__values),)


IntString.null = IntString()


class RefinableIntegerPartition(object):
    """A refinable partition of {0, 1, ...,}"""

    def __init__(self):
        self.starts = array("Q", [0])

    def canonical(self, n):
        """Returns the beginning of the partition element containing `n`"""
        return self.starts[bisect.bisect_right(self.starts, n) - 1]

    def make_canonical(self, n):
        """Mark `n` as the beginning of a partition."""
        assert self.canonical(n) < n
        bisect.insort_left(self.starts, n)


class LearningState(object):
    """Internal state shared between an LStarLearner and a LearnedAutomaton."""

    def __init__(self, classifier):
        self.classify = lru_cache(typed=True, maxsize=1024)(classifier)
        self.generation = 0
        self.experiments = []
        self.node_labels = IndexedList()
        self.node_vecs = IndexedList()
        self.node_partitions = []
        self.__known_transitions = {}

        self.start_node = self.add_node(IntString.null)
        self.add_experiment(IntString.null)

    def transition(self, node, character):
        if character < 0:
            raise ValueError("Invalid character %r" % (character,))
        key = (node, character)
        try:
            return self.__known_transitions[key]
        except KeyError:
            pass

        canon = self.node_partitions[node].canonical(character)
        if canon != character:
            assert canon < character
            result = self.transition(node, canon)
        else:
            destination = self.node_labels[node] + IntString.singleton(character)
            try:
                result = self.node_labels.index_of(destination)
            except KeyError:
                destination_vec = self.__vec(destination)
                try:
                    result = self.node_vecs.index_of(destination_vec)
                except KeyError:
                    result = self.add_node(destination, vec=destination_vec)
        self.__known_transitions[key] = result
        return result

    def add_node(self, label, vec=None):
        label = IntString(label)
        assert label not in self.node_labels
        i = len(self.node_labels)
        self.node_labels.append(label)
        if vec is None:
            vec = self.__vec(label)
        self.node_vecs.append(vec)
        self.node_partitions.append(RefinableIntegerPartition())
        assert (
            len(self.node_labels)
            == len(self.node_vecs)
            == len(self.node_partitions)
            == i + 1
        )
        return i

    def add_experiment(self, ex):
        ex = IntString(ex)
        self.experiments.append(ex)
        new_vecs = []
        for (l, v) in zip(self.node_labels, self.node_vecs):
            new_vecs.append(v + (self.classify(l + ex),))
        self.node_vecs = IndexedList(new_vecs)
        self.__change_made()

    def make_canonical(self, node, c):
        self.node_partitions[node].make_canonical(c)
        self.__change_made()

    def make_consistent(self, string):
        correct = self.classify(string)
        string = IntString(string)
        if self.classify(string) != correct:
            raise ValueError("Classifier depends on type of string.")
        changed = False
        prev = -1

        # We loop until we achieve consistency. The loop is required because
        # we only make one patch at a time: Each time we pass through the
        # loop we either determine we are already consistent and stop, or we
        # find something we did wrong and patch it up.
        while True:
            assert prev < self.generation
            prev = self.generation

            automaton = LearnedAutomaton(self)

            if automaton.classify(string) == correct:
                break

            changed = True

            # Following the automaton resulted in us getting the wrong answer.
            # This means that we must at some point have made a "wrong transition".
            # That is, we went from state s1 to s2 via character c, but we shouldn't
            # have.
            #
            # We now begin a "diagnostic procedure" to identify s1, s2, and c and
            # to update our inference in some way so as to prevent that.

            states = [automaton.start()]
            for c in string:
                states.append(automaton.transition(states[-1], c))

            # states[i] is the state we are in after reading i characters from
            # string.
            assert len(states) == len(string) + 1

            # Our goal is to identify a bad transition and add an experiment that
            # would distinguish that transition.

            def is_good(i):
                return (
                    self.classify(automaton.label_of(states[i]) + string[i:]) == correct
                )

            # Equivalent to self.classify(string) == correct
            assert is_good(0)
            # Equivalent to automaton.classify(string) == correct
            assert not is_good(len(string))

            lo = 0
            hi = len(string)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if is_good(mid):
                    lo = mid
                else:
                    hi = mid

            # We've successfully diagnosed the problem. We made the transition
            # s1 -[c]-> s2, but string[hi:] is an experiment distinguishing the
            # two.
            s1 = states[lo]
            s2 = states[hi]
            c = string[lo]

            # Now we know that the transition from s1 -> s2 via c was bad.
            #
            # There are two ways this can happen:
            #
            #   * we might not have the experiment we needed to distinguish the
            #     destination nodes.
            #   * we might have treated the character string[lo] as if it were
            #     equivalent to a character which it was not equivalent to.
            #
            # We check the latter first because refining a partition is much
            # cheaper in the long run than adding an experiment.
            canon_c = self.node_partitions[s1].canonical(c)
            if canon_c != c:

                def vec_of(d):
                    return self.__vec(self.node_labels[s1] + IntString.singleton(d))

                canon_vec = self.node_vecs[s2]
                correct_vec = vec_of(c)

                # Our existing experiments were enough to distinguish this from
                # its canonical version, we just didn't check before now. We
                # need to split the partition here. In order to do this we find
                # some small value in the range [canonical, c] which our current
                # experiments consider equivalent to c. The reason we do this
                # search is so that if we have a long sequence of equivalent
                # characters we always treat them as canonising to the first
                # element of that sequence.
                if canon_vec != correct_vec:
                    bad_c = canon_c
                    good_c = c
                    while bad_c + 1 < good_c:
                        mid_c = (bad_c + good_c) // 2
                        if vec_of(mid_c) == correct_vec:
                            good_c = mid_c
                        else:
                            bad_c = mid_c
                    self.make_canonical(s1, good_c)
                    continue
            self.add_experiment(string[hi:])
        return changed

    def __change_made(self):
        self.generation += 1
        self.__known_transitions.clear()

    def __vec(self, string):
        return tuple([self.classify(string + e) for e in self.experiments])


class LearnedAutomaton(object):
    def __init__(self, learning_state):
        self.__state = learning_state
        self.__generation = learning_state.generation

    def __check_still_valid(self):
        if self.__generation != self.__state.generation:
            raise ValueError("Using stale automaton")

    def start(self):
        self.__check_still_valid()
        return self.__state.start_node

    def label_of(self, node):
        self.__check_still_valid()
        return self.__state.node_labels[node]

    def classify_node(self, node):
        self.__check_still_valid()
        return self.__state.node_vecs[node][0]

    def transition(self, node, character):
        self.__check_still_valid()
        return self.__state.transition(node, character)

    def classify(self, string):
        current = self.start()
        for c in string:
            current = self.transition(current, c)
        return self.classify_node(current)

    def canonical_alphabet(self, node):
        return self.__state.node_partitions[node].starts

    def iter_nodes(self):
        start = self.start()

        queue = deque([start])
        seen = {start}

        while queue:
            n = queue.popleft()
            yield n
            for c in self.canonical_alphabet(n):
                n2 = self.transition(n, c)
                if n2 not in seen:
                    queue.append(n2)
                    seen.add(n2)

    def debug(self):
        print("LearnedAutomaton:")
        for n in self.iter_nodes():
            print("  State s%d, labelled %r" % (n, self.classify_node(n)))
            canon = self.canonical_alphabet(n)
            for i, c1 in enumerate(canon):

                if i + 1 == len(canon):
                    label = "%d+" % (c1,)
                else:
                    c2 = canon[i + 1] - 1
                    if c1 == c2:
                        label = str(c1)
                    else:
                        label = "%d-%d" % (c1, c2)

                print("    %s -> s%d" % (label, self.transition(n, c1)))


class LStarLearner(object):
    """A learner that takes a classifier for sequences of integers and learns finite automata approximating it.
    If the classifier is regular then this will converge on a correct finite automaton for it after at most
    ``N`` corrections, where N is the size of the minimum DFA recognising it.

    Loosely based on the following papers:

    * Angluin, Dana. "Learning regular sets from queries and counterexamples." Information and computation 75.2 (1987): 87-106
    * Rivest, Ronald L., and Robert E. Schapire. "Inference of finite automata using homing sequences." Information and Computation 103.2 (1993): 299-347.
    * Maler, Oded, and Irini-Eleftheria Mens. "Learning regular languages over large alphabets."
      International Conference on Tools and Algorithms for the Construction and Analysis of Systems. Springer, Berlin, Heidelberg, 2014.

    How loosely varies by paper. In particular for the last paper the approach taken should be more considered "inspired by".

    The main novelty of the approach in this implementation is that we deliberately construct the automaton lazily as it
    is walked rather than having an expensive completion step. Under some workflows this allows for handling much
    larger DFAs than would otherwise be possible with L*. This is currently not what you might call well tested.
    """

    def __init__(self, classifier):
        """Construct a learner for a given ``classifier`` function.

        A classifier should take any sequence of integers and return an arbitrary hashable value.
        Note that the classifier *must not* depend on the type of the sequence, only its contents.
        """
        self.__state = LearningState(classifier)

    def classify(self, string):
        """Returns the underlying classification of this string."""
        return self.__state.classify(string)

    @property
    def learned_automaton(self):
        """Returns the current best learned automaton."""
        return LearnedAutomaton(self.__state)

    def make_consistent(self, string):
        """Updates the learner's internal model so that ``self.learned_automaton.classify(string) == self.classify(string)``
        after this call. Returns True if this makes any changes to the model"""
        return self.__state.make_consistent(string)

    def is_consistent(self, string):
        return self.learned_automaton.classify(string) == self.classify(string)


class IndexedList(object):
    __slots__ = ("__values", "__index")

    def __init__(self, values=()):
        self.__values = []
        self.__index = {}
        for v in values:
            self.append(v)

    def append(self, v):
        i = len(self.__values)
        if v in self.__index:
            raise ValueError("%r already in collection" % (v,))
        self.__values.append(v)
        self.__index[v] = i

    def index_of(self, v):
        return self.__index[v]

    def __getitem__(self, i):
        return self.__values[i]

    def __len__(self):
        return len(self.__values)

    def __repr__(self):
        return "IndexedList(%r)" % (self.__values,)

    def __contains__(self, value):
        return value in self.__index
