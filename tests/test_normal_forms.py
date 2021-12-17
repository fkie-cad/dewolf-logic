"""Module testing the functionality transforming into normal forms."""
from simplifier.visitor import ToCnfVisitor
from simplifier.world.world import World


class TestCNF:
    def test_z3_to_cnf_negation(self):
        w = World()
        w.define(var := w.variable("v", 8), w.bitwise_negate(w.variable("x", 1)))
        ToCnfVisitor(var)
        assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_negate(w.variable("x", 1)))

    def test_is_in_cnf(self):
        w = World()
        w.define(
            var := w.variable("v", 8),
            w.bitwise_and(w.bitwise_or(w.variable("x", 8), w.variable("a", 8)), w.bitwise_or(w.variable("x", 8), w.variable("b", 8))),
        )
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 8)),
            w.bitwise_and(w.bitwise_or(w.variable("x", 8), w.variable("a", 8)), w.bitwise_or(w.variable("x", 8), w.variable("b", 8))),
        )

    def test_is_in_nested_cnf(self):
        w = World()
        w.define(
            var := w.variable("v", 8),
            w.bitwise_and(
                w.bitwise_and(w.variable("z", 8), w.bitwise_and(w.variable("y", 8), w.variable("u", 8))),
                w.bitwise_or(w.variable("x", 8), w.variable("a", 8)),
                w.bitwise_or(w.variable("x", 8), w.variable("b", 8)),
            ),
        )
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 8)),
            w.bitwise_and(
                w.bitwise_or(w.variable("x", 8), w.variable("a", 8)),
                w.bitwise_or(w.variable("x", 8), w.variable("b", 8)),
                w.variable("z", 8),
                w.variable("y", 8),
                w.variable("u", 8),
            ),
        )

    def test_is_in_cnf_contains_negation(self):
        w = World()
        w.define(var := w.variable("v", 8), w.from_string("(& (| (~x1@8) x2@8) (|x3@8 (~x1@8)))"))
        ToCnfVisitor(var)
        assert World.compare(w.get_definition(w.variable("v", 8)), w.from_string("(& (| (~x1@8) x2@8) (|x3@8 (~x1@8)))"))

    def test_in_dnf(self):
        w = World()
        w.define(
            var := w.variable("v", 8),
            w.bitwise_or(w.bitwise_and(w.variable("x", 8), w.variable("y", 8)), w.bitwise_and(w.variable("u", 8), w.variable("z", 8))),
        )
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 8)),
            w.bitwise_and(
                w.bitwise_or(w.variable("u", 8), w.variable("x", 8)),
                w.bitwise_or(w.variable("z", 8), w.variable("x", 8)),
                w.bitwise_or(w.variable("u", 8), w.variable("y", 8)),
                w.bitwise_or(w.variable("z", 8), w.variable("y", 8)),
            ),
        )

    def test_one_wrong_operand(self):
        w = World()
        w.define(var := w.variable("v", 8), w.from_string("(&(& (| (~x1@8) x2@8) (|x3@8 (~x1@8))) (| x4@8 (& x2@8 x3@8)))"))
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 8)),
            w.bitwise_and(
                w.bitwise_or(w.bitwise_negate(w.variable("x1", 8)), w.variable("x2", 8)),
                w.bitwise_or(w.variable("x3", 8), w.bitwise_negate(w.variable("x1", 8))),
                w.bitwise_or(w.variable("x2", 8), w.variable("x4", 8)),
                w.bitwise_or(w.variable("x3", 8), w.variable("x4", 8)),
            ),
        )

    def test_in_dnf_negation(self):
        w = World()
        w.define(var := w.variable("v", 8), w.from_string("(| (& x2@1 (~x1@1)) (& x3@1 (~x1@1)) )"))
        ToCnfVisitor(var)
        assert World.compare(w.get_definition(w.variable("v", 8)), w.from_string("(& (~x1@1) (| x3@1 x2@1) )"))

    def test_with_negation_of_operand(self):
        w = World()
        w.define(
            var := w.variable("v", 1),
            w.from_string("(|(| x1@1 (& x2@1 (~x1@1)) ) (| (& x3@1 (~(| x1@1 x2@1)) ) (& x5@1 (& x4@1 (~x1@1) ))))"),
        )
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 1)),
            w.bitwise_and(
                w.bitwise_or(w.variable("x5", 1), w.variable("x3", 1), w.variable("x2", 1), w.variable("x1", 1)),
                w.bitwise_or(w.variable("x4", 1), w.variable("x3", 1), w.variable("x2", 1), w.variable("x1", 1)),
            ),
        )

    def test_mix_formula(self):
        w = World()
        w.define(
            var := w.variable("v", 1),
            w.from_string("(| (& (| x2@1 x4@1) (~x1@1)) (& (| x3@1 x4@1) (| x5@1 (~x1@1))) )"),
        )
        ToCnfVisitor(var)
        assert World.compare(
            w.get_definition(w.variable("v", 1)),
            w.bitwise_and(
                w.bitwise_or(w.variable("x2", 1), w.variable("x4", 1), w.variable("x3", 1)),
                w.bitwise_or(w.bitwise_negate(w.variable("x1", 1)), w.variable("x3", 1), w.variable("x4", 1)),
                w.bitwise_or(w.bitwise_negate(w.variable("x1", 1)), w.variable("x5", 1)),
            ),
        )

    def test_negated_formula(self):
        w = World()
        w.define(var := w.variable("v", 8), w.from_string("(~ (& (| (~x1@8) x2@8) (|x3@8 (~x1@8)) ))"))
        ToCnfVisitor(var)
        assert World.compare(w.get_definition(w.variable("v", 8)), w.from_string("(& x1@8 (| (~x3@8) (~x2@8)))"))

    def test_partial_and_simplify_changes_and(self):
        """Start at a sub-formula and simplify changes and to or."""
        w = World()
        formula = w.from_string("(&  1@1 (| a@1 (& (~b@1) c@1)) )")
        w.define(var := w.variable("v", 1), w.bitwise_negate(formula))
        ToCnfVisitor(formula)
        assert World.compare(w.get_definition(var), w.from_string("(~ (& (| (~b@1) a@1) (| c@1 a@1)))"))

    def test_partial_and_simplify_changes_or(self):
        """Start at a sub-formula and simplify changes or to and."""
        w = World()
        formula = w.from_string("(|  0@1 (& a@1 (| (~b@1) c@1)) )")
        w.define(var := w.variable("v", 1), w.bitwise_negate(formula))
        ToCnfVisitor(formula)
        assert World.compare(w.get_definition(var), w.from_string("(~ (& a@1 (| (~b@1) c@1)) )"))


class TestDNF:
    pass
