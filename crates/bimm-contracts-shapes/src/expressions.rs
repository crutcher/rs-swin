use crate::bindings::StackMap;
use crate::shape_patterns::TryMatchResult;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeExpr<'a> {
    Param(&'a str),
    Negate(&'a SizeExpr<'a>),
    Pow(&'a SizeExpr<'a>, usize),
    Sum(&'a [SizeExpr<'a>]),
    Prod(&'a [SizeExpr<'a>]),
}

impl Display for SizeExpr<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            SizeExpr::Param(param) => write!(f, "{}", param),
            SizeExpr::Negate(negate) => write!(f, "(-{})", negate),
            SizeExpr::Pow(base, exponent) => {
                write!(f, "({})^{}", base, exponent)
            }
            SizeExpr::Sum(values) => {
                write!(f, "(")?;
                for (idx, expr) in values.iter().enumerate() {
                    if idx > 0 {
                        write!(f, "+")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            SizeExpr::Prod(values) => {
                write!(f, "(")?;
                for (idx, expr) in values.iter().enumerate() {
                    if idx > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<'a> SizeExpr<'a> {
    /// Evaluate an expression.
    ///
    /// ## Arguments
    ///
    /// - `env` - the binding environment.
    ///
    /// ## Returns
    ///
    /// - `Right(value)` - the value of the expression.
    /// - `Left(count)` - the count of unbound parameters.
    #[must_use]
    fn try_evaluate<B>(
        &self,
        bindings: &B,
    ) -> EvalResult
    where
        B: StackMap<'a, usize>,
    {
        #[inline(always)]
        fn reduce_children<'a, B>(
            exprs: &'a [SizeExpr<'a>],
            bindings: &B,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> EvalResult
        where
            B: StackMap<'a, usize>,
        {
            let mut value = zero;
            let mut unbound_count = 0;
            for expr in exprs {
                match expr.try_evaluate(bindings) {
                    EvalResult::Value(v) => op(&mut value, v),
                    EvalResult::UnboundParams(c) => unbound_count += c,
                }
            }
            if unbound_count == 0 {
                EvalResult::Value(value)
            } else {
                EvalResult::UnboundParams(unbound_count)
            }
        }

        match self {
            SizeExpr::Param(name) => match bindings.lookup(name) {
                Some(value) => EvalResult::Value(value as isize),
                None => EvalResult::UnboundParams(1),
            },
            SizeExpr::Negate(expr) => match expr.try_evaluate(bindings) {
                EvalResult::Value(value) => EvalResult::Value(-value),
                x => x,
            },
            SizeExpr::Pow(base, exp) => match base.try_evaluate(bindings) {
                EvalResult::Value(value) => EvalResult::Value(value.pow(*exp as u32)),
                x => x,
            },
            SizeExpr::Sum(children) => {
                reduce_children(children, bindings, 0, |tmp, value| *tmp += value)
            }
            SizeExpr::Prod(children) => {
                reduce_children(children, bindings, 1, |tmp, value| *tmp *= value)
            }
        }
    }

    /// Reconcile an expression against a target value.
    ///
    /// ## Arguments
    ///
    /// * `target`: The target value to match.
    /// * `env`: The environment containing bindings for parameters.
    ///
    /// ## Returns
    ///
    /// * `Ok(MatchResult::Match)` if the expression matches the target.
    /// * `Ok(MatchResult::MissMatch)` if the expression does not match the target.
    /// * `Ok(MatchResult::Constraint(name, value))` if the expression can be solved for a single unbound parameter.
    /// * `Ok(MatchResult::UnderConstrained)` if the expression cannot be solved with the current bindings.
    #[must_use]
    pub fn try_match<B>(
        &'a self,
        target: isize,
        bindings: &B,
    ) -> Result<TryMatchResult<'a>, String>
    where
        B: StackMap<'a, usize>,
    {
        #[inline(always)]
        fn reduce_children<'a, B>(
            exprs: &'a [SizeExpr<'a>],
            bindings: &B,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> Result<(isize, Option<&'a SizeExpr<'a>>), String>
        where
            B: StackMap<'a, usize>,
        {
            let mut partial_value: isize = zero;
            let mut rem_expr = None;

            // At most one child can be unbound, and by only one parameter.
            for expr in exprs {
                match expr.try_evaluate(bindings) {
                    EvalResult::Value(value) => op(&mut partial_value, value),
                    EvalResult::UnboundParams(count) => {
                        if count == 1 && rem_expr.is_none() {
                            rem_expr = Some(expr);
                        } else {
                            return Err("Too many unbound params".to_string());
                        }
                    }
                }
            }
            // If the monoid is fully bound, then return (value, None);
            // Otherwise, return (partial_value, expr).
            Ok((partial_value, rem_expr))
        }

        match self {
            SizeExpr::Param(name) => {
                if let Some(value) = bindings.lookup(name) {
                    if value as isize == target {
                        Ok(TryMatchResult::TargetMatch)
                    } else {
                        Ok(TryMatchResult::ValueMissMatch)
                    }
                } else {
                    Ok(TryMatchResult::ParamConstraint(name, target))
                }
            }
            SizeExpr::Negate(expr) => expr.try_match(-target, bindings),
            SizeExpr::Pow(base, exp) => match crate::math_util::maybe_iroot(target, *exp) {
                Some(root) => base.try_match(root, bindings),
                None => Err("No integer solution.".to_string()),
            },
            SizeExpr::Sum(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_children(exprs, bindings, 0, |acc, value| *acc += value)?;
                if let Some(expr) = rem_expr {
                    let target = target - partial_value;
                    expr.try_match(target, bindings)
                } else if partial_value == target {
                    Ok(TryMatchResult::TargetMatch)
                } else {
                    Ok(TryMatchResult::ValueMissMatch)
                }
            }
            SizeExpr::Prod(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_children(exprs, bindings, 1, |acc, value| *acc *= value)?;
                if let Some(expr) = rem_expr {
                    if target % partial_value != 0 {
                        // Non-integer solution
                        return Err("No integer solution.".to_string());
                    }
                    let target = target / partial_value;
                    expr.try_match(target, bindings)
                } else if partial_value == target {
                    Ok(TryMatchResult::TargetMatch)
                } else {
                    Ok(TryMatchResult::ValueMissMatch)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::{MutableStackEnvironment, MutableStackMap};

    #[test]
    fn test_simple_sum() {
        // x + 5, target = 8, should solve x = 3
        let expr = SizeExpr::Sum(&[
            SizeExpr::Param("x"),
            SizeExpr::Negate(&SizeExpr::Param("y")),
        ]);
        let env = MutableStackEnvironment::new(&[("y", 5)]);

        assert_eq!(
            expr.try_match(8, &env),
            Ok(TryMatchResult::ParamConstraint("x", 13))
        );
    }

    #[test]
    fn test_simple_product() {
        // 2 * x, target = 6, should solve x = 3
        let expr = SizeExpr::Prod(&[SizeExpr::Param("y"), SizeExpr::Param("x")]);
        let env = MutableStackEnvironment::new(&[("y", 2)]);

        assert_eq!(
            expr.try_match(6, &env),
            Ok(TryMatchResult::ParamConstraint("x", 3))
        );
    }

    #[test]
    fn test_quadratic_case() {
        // p * p, target = 9
        // This becomes: p * p = 9, so we solve the first p for 9/p
        // But since both factors are the same unbound param, this should work
        let expr = SizeExpr::Prod(&[SizeExpr::Param("p"), SizeExpr::Param("p")]);
        let env = MutableStackEnvironment::new(&[]);

        // This should solve: first p gets target/second_p, but second_p is unbound too
        // Actually, this should fail because we have the same param multiple times
        assert_eq!(
            expr.try_match(25, &env),
            Err("Too many unbound params".to_string())
        );
    }

    #[test]
    fn test_complex_expression() {
        // w * p * h * p + c * z
        // With w=2, h=3, c=1, z=4, and p unbound, target=25
        // This should fail because p appears twice in the product
        static EXPR: SizeExpr = SizeExpr::Sum(&[
            SizeExpr::Prod(&[
                SizeExpr::Param("w"),
                SizeExpr::Param("p"),
                SizeExpr::Param("h"),
                SizeExpr::Param("p"),
            ]),
            SizeExpr::Prod(&[SizeExpr::Param("c"), SizeExpr::Param("z")]),
        ]);

        let mut env = MutableStackEnvironment::new(&[]);
        env.bind("w", 2);
        env.bind("h", 3);
        env.bind("c", 1);
        env.bind("z", 4);

        // p appears multiple times in the first product term, so this should fail
        assert_eq!(
            EXPR.try_match(25, &env),
            Err("Too many unbound params".to_string())
        );
    }

    #[test]
    fn test_nested_structure() {
        // 2 * (x + 3) - 1 == 9
        // 2 * (x + 3) == 10
        // x + 3 == 5
        // x == 2
        let expr = SizeExpr::Sum(&[
            SizeExpr::Prod(&[
                SizeExpr::Param("a"),
                SizeExpr::Sum(&[SizeExpr::Param("x"), SizeExpr::Param("b")]),
            ]),
            SizeExpr::Negate(&SizeExpr::Param("c")),
        ]);
        let env = MutableStackEnvironment::new(&[("a", 2), ("b", 3), ("c", 1)]);

        assert_eq!(
            expr.try_match(9, &env),
            Ok(TryMatchResult::ParamConstraint("x", 2))
        );
    }

    #[test]
    fn test_all_bound_success() {
        // x + 5 with x = 3, target = 8
        let expr = SizeExpr::Sum(&[SizeExpr::Param("x"), SizeExpr::Param("a")]);

        let mut env = MutableStackEnvironment::new(&[("a", 5)]);
        env.bind("x", 3);

        assert_eq!(expr.try_match(8, &env), Ok(TryMatchResult::TargetMatch));
    }

    #[test]
    fn test_multiple_unbound() {
        // x + y, multiple unbound params
        let expr = SizeExpr::Sum(&[SizeExpr::Param("x"), SizeExpr::Param("y")]);
        let env = MutableStackEnvironment::new(&[]);

        assert_eq!(
            expr.try_match(25, &env),
            Err("Too many unbound params".to_string())
        );
    }

    #[test]
    fn test_no_integer_solution() {
        // 2 * x == 3
        // x = 3/2, which is not an integer
        let expr = SizeExpr::Prod(&[SizeExpr::Param("a"), SizeExpr::Param("x")]);
        let env = MutableStackEnvironment::new(&[("a", 2)]);

        assert_eq!(
            expr.try_match(3, &env),
            Err("No integer solution.".to_string())
        );
    }

    #[test]
    fn test_power_solve_base() {
        // x^3 = 8, should solve x = 2
        let expr = SizeExpr::Pow(&SizeExpr::Param("x"), 3);
        let env = MutableStackEnvironment::new(&[]);

        assert_eq!(
            expr.try_match(8, &env),
            Ok(TryMatchResult::ParamConstraint("x", 2))
        );
    }

    #[test]
    fn test_power_no_solution() {
        // 2^3 = x where x != 8 (no solution when fully bound and doesn't match)
        let expr = SizeExpr::Pow(&(SizeExpr::Param("a")), 3);
        let env = MutableStackEnvironment::new(&[("a", 2)]);

        // 2^3 = 8, so target 5 should fail
        assert_eq!(
            expr.try_match(5, &env),
            Err("No integer solution.".to_string())
        );
    }

    #[test]
    fn test_power_in_sum() {
        // x^2 + 1 = 10, should solve x = 3 (since 3^2 + 1 = 10)
        let expr = SizeExpr::Sum(&[
            SizeExpr::Pow(&SizeExpr::Param("x"), 2),
            SizeExpr::Param("a"),
        ]);
        let env = MutableStackEnvironment::new(&[("a", 1)]);

        assert_eq!(
            expr.try_match(10, &env),
            Ok(TryMatchResult::ParamConstraint("x", 3))
        );
    }

    #[test]
    fn test_power_cube_root() {
        // x^3 = 27, should solve x = 3
        let expr = SizeExpr::Pow(&(SizeExpr::Param("x")), 3);
        let env = MutableStackEnvironment::new(&[]);

        assert_eq!(
            expr.try_match(27, &env),
            Ok(TryMatchResult::ParamConstraint("x", 3))
        );
    }

    #[test]
    fn test_power_negative_base_odd_exp() {
        // x^3 = -8, should solve x = -2 (since (-2)^3 = -8)
        let expr = SizeExpr::Pow(&SizeExpr::Param("x"), 3);
        let env = MutableStackEnvironment::new(&[]);

        assert_eq!(
            expr.try_match(-8, &env),
            Ok(TryMatchResult::ParamConstraint("x", -2))
        );
    }

    #[test]
    fn test_power_negative_base_even_exp() {
        // x^2 = -4 (no real solution for even exponent and negative target)
        let expr = SizeExpr::Pow(&SizeExpr::Param("x"), 2);
        let env = MutableStackEnvironment::new(&[]);

        assert_eq!(
            expr.try_match(-4, &env),
            Err("No integer solution.".to_string())
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EvalResult {
    Value(isize),
    UnboundParams(usize),
}
