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

from lstar import LStarLearner
import pytest


def experiments(learner):
    return learner._LStarLearner__experiments


@pytest.mark.parametrize("label", [False, True, 42])
def test_can_learn_trivial_language_instantly(label):
    aut = LStarLearner(classifier=lambda x: label).learned_automaton

    assert aut.classify(b"") == label
    assert aut.classify(b"foo") == label

    assert aut.transition(aut.start(), 10) == aut.start()


def test_can_be_made_consistent():
    foo = list(b"foo")

    learner = LStarLearner(classifier=lambda x: list(x) == foo)

    # Initially we have not learned enough of the automaton to identify this string.
    assert learner.learned_automaton.classify(b"foo") == False

    learner.make_consistent(b"foo")

    assert learner.learned_automaton.classify(b"foo") == True
    assert len(list(learner.learned_automaton.iter_nodes())) == 4


def test_can_handle_very_large_alphabets():
    learner = LStarLearner(lambda ls: len(ls) == 3 and sum(ls) > 10 ** 4)

    values = [(10 ** 4,) * k for k in (3, 6, 7)] + [(10 ** 6, 0, 0)]

    for v in values:
        learner.make_consistent(v)

    learner.learned_automaton.debug()

    for v in values:
        assert learner.is_consistent(v)


@pytest.mark.parametrize('n', [1, 2, 10, 100])
def test_can_learn_length_languages(n):
    learner = LStarLearner(lambda ls: len(ls) == n)

    learner.make_consistent([0] * n)
    assert len(list(learner.learned_automaton.iter_nodes())) == n + 1
