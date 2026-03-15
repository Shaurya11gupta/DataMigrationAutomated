import re

from constraint_similarity_engine import ColumnConstraints, ConstraintCompatibilityJoin
from value_similarity_engine import ColumnStats, ValueSimilarity


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    if not condition:
        raise AssertionError(name)


def run_value_tests() -> None:
    # Categorical positive
    a = ColumnStats(["us", "in", "uk", "us", "in", "ca"] * 20)
    b = ColumnStats(["US", "IN", "UK", "US", "IN", "CA"] * 15)
    s1 = ValueSimilarity(a, b).compute_score()["final"]
    check("categorical overlap should be high", s1 >= 0.8)

    # Categorical negative
    c = ColumnStats(["dog", "cat", "fish"] * 50)
    d = ColumnStats(["red", "green", "blue"] * 50)
    s2 = ValueSimilarity(c, d).compute_score()["final"]
    check("categorical disjoint should be low", s2 <= 0.2)

    # Numeric positive
    n1 = ColumnStats(list(range(1, 1001)))
    n2 = ColumnStats(list(range(100, 1100)))
    s3 = ValueSimilarity(n1, n2).compute_score()["final"]
    check("numeric overlapping distributions should be high", s3 >= 0.7)

    # Numeric negative
    n3 = ColumnStats(list(range(1, 1001)))
    n4 = ColumnStats(list(range(5000, 6000)))
    s4 = ValueSimilarity(n3, n4).compute_score()["final"]
    check("numeric disjoint ranges should be low", s4 <= 0.2)

    print("Value similarity tests complete.")


def run_constraint_tests() -> None:
    pk = ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True, min_value=1, max_value=1000)
    fk = ColumnConstraints(nullable=False, is_foreign_key=True, min_value=1, max_value=1200)
    s1 = ConstraintCompatibilityJoin.score(pk, fk)["final"]
    check("pk-fk compatible should be high", s1 >= 0.75)

    e1 = ColumnConstraints(allowed_values={"us", "in", "uk"})
    e2 = ColumnConstraints(allowed_values={"fr", "de", "es"})
    s2 = ConstraintCompatibilityJoin.score(e1, e2)["final"]
    check("enum disjoint should be zero", s2 == 0.0)

    r1 = ColumnConstraints(min_value=1, max_value=100)
    r2 = ColumnConstraints(min_value=1000, max_value=2000)
    s3 = ConstraintCompatibilityJoin.score(r1, r2)["final"]
    check("range disjoint should be zero", s3 == 0.0)

    rg1 = ColumnConstraints(regex_pattern=re.compile(r"^[A-Z]{2}\d{4}$"))
    rg2 = ColumnConstraints(regex_pattern=re.compile(r"^[A-Z]{2}\d{4}$"))
    s4 = ConstraintCompatibilityJoin.score(rg1, rg2)["final"]
    check("same regex should be above baseline", s4 >= 0.55)

    print("Constraint similarity tests complete.")


if __name__ == "__main__":
    run_value_tests()
    run_constraint_tests()
    print("All sanity checks passed.")
