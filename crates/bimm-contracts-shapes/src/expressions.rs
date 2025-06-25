use crate::bindings::StackMap;
use std::fmt::{Display, Formatter};

/// A stack/static expression algebra for dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSizeExpr<'a> {
    /// A parameter reference.
    Param(&'a str),

    /// Negation of an expression.
    Negate(&'a DimSizeExpr<'a>),

    /// Exponentiation of an expression.
    Pow(&'a DimSizeExpr<'a>, usize),

    /// Sum of expressions.
    Sum(&'a [DimSizeExpr<'a>]),

    /// Product of expressions.
    Prod(&'a [DimSizeExpr<'a>]),
}

impl Display for DimSizeExpr<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO: with some lifting, we could elide more of the parentheses.
        match self {
            DimSizeExpr::Param(param) => write!(f, "{}", param),
            DimSizeExpr::Negate(negate) => write!(f, "(-{})", negate),
            DimSizeExpr::Pow(base, exponent) => {
                write!(f, "({})^{}", base, exponent)
            }
            DimSizeExpr::Sum(values) => {
                write!(f, "(")?;
                for (idx, expr) in values.iter().enumerate() {
                    if idx > 0 {
                        write!(f, "+")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            DimSizeExpr::Prod(values) => {
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum TryEvalResult {
    /// The evaluated value of the expression.
    Value(isize),

    /// The count of unbound parameters in the expression.
    UnboundParams(usize),
}

/// Result of `SizeExpr::try_match()`.
///
/// All values are borrowed from the original expression,
/// so they are valid as long as the expression is valid.
///
/// Runtime errors (malformed expressions, too-many unbound parameters, etc.)
/// are not represented here; and are returned as `Err(String)` from `try_match`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryMatchResult<'a> {
    /// All params bound and expression equals target.
    Match,

    /// Expression value does not match the target.
    Conflict,

    /// Expression can be solved for a single unbound param.
    ParamConstraint(&'a str, isize),
}

impl<'a> DimSizeExpr<'a> {
    /// Evaluate an expression.
    ///
    /// ## Arguments
    ///
    /// - `env` - the binding environment.
    ///
    /// ## Returns
    ///
    /// A TryEvalResult:
    /// * `Value(value)` - the evaluated value of the expression.
    /// * `UnboundParams(count)` - the count of unbound parameters.
    #[must_use]
    fn try_eval<E>(
        &self,
        env: &E,
    ) -> TryEvalResult
    where
        E: StackMap<'a, usize>,
    {
        #[inline(always)]
        fn reduce_children<'a, B>(
            exprs: &'a [DimSizeExpr<'a>],
            bindings: &B,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> TryEvalResult
        where
            B: StackMap<'a, usize>,
        {
            let mut value = zero;
            let mut unbound_count = 0;
            for expr in exprs {
                match expr.try_eval(bindings) {
                    TryEvalResult::Value(v) => op(&mut value, v),
                    TryEvalResult::UnboundParams(c) => unbound_count += c,
                }
            }
            if unbound_count == 0 {
                TryEvalResult::Value(value)
            } else {
                TryEvalResult::UnboundParams(unbound_count)
            }
        }

        match self {
            DimSizeExpr::Param(name) => match env.lookup(name) {
                Some(value) => TryEvalResult::Value(value as isize),
                None => TryEvalResult::UnboundParams(1),
            },
            DimSizeExpr::Negate(expr) => match expr.try_eval(env) {
                TryEvalResult::Value(value) => TryEvalResult::Value(-value),
                x => x,
            },
            DimSizeExpr::Pow(base, exp) => match base.try_eval(env) {
                TryEvalResult::Value(value) => TryEvalResult::Value(value.pow(*exp as u32)),
                x => x,
            },
            DimSizeExpr::Sum(children) => {
                reduce_children(children, env, 0, |tmp, value| *tmp += value)
            }
            DimSizeExpr::Prod(children) => {
                reduce_children(children, env, 1, |tmp, value| *tmp *= value)
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
    pub fn try_match<E>(
        &'a self,
        target: isize,
        env: &E,
    ) -> Result<TryMatchResult<'a>, String>
    where
        E: StackMap<'a, usize>,
    {
        #[inline(always)]
        fn reduce_children<'a, E>(
            exprs: &'a [DimSizeExpr<'a>],
            env: &E,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> Result<(isize, Option<&'a DimSizeExpr<'a>>), String>
        where
            E: StackMap<'a, usize>,
        {
            let mut partial_value: isize = zero;
            let mut rem_expr = None;
            // At most one child can be unbound, and by only one parameter.
            for expr in exprs {
                match expr.try_eval(env) {
                    TryEvalResult::Value(value) => op(&mut partial_value, value),
                    TryEvalResult::UnboundParams(count) => {
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
            DimSizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    if value as isize == target {
                        Ok(TryMatchResult::Match)
                    } else {
                        Ok(TryMatchResult::Conflict)
                    }
                } else {
                    Ok(TryMatchResult::ParamConstraint(name, target))
                }
            }
            DimSizeExpr::Negate(expr) => expr.try_match(-target, env),
            DimSizeExpr::Pow(base, exp) => match crate::math_util::maybe_iroot(target, *exp) {
                Some(root) => base.try_match(root, env),
                None => Err("No integer solution.".to_string()),
            },
            DimSizeExpr::Sum(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_children(exprs, env, 0, |acc, value| *acc += value)?;
                if let Some(expr) = rem_expr {
                    let target = target - partial_value;
                    expr.try_match(target, env)
                } else if partial_value == target {
                    Ok(TryMatchResult::Match)
                } else {
                    Ok(TryMatchResult::Conflict)
                }
            }
            DimSizeExpr::Prod(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_children(exprs, env, 1, |acc, value| *acc *= value)?;
                if let Some(expr) = rem_expr {
                    if target % partial_value != 0 {
                        // Non-integer solution
                        return Err("No integer solution.".to_string());
                    }
                    let target = target / partial_value;
                    expr.try_match(target, env)
                } else if partial_value == target {
                    Ok(TryMatchResult::Match)
                } else {
                    Ok(TryMatchResult::Conflict)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::{StackEnvironment};

    #[test]
    fn test_try_eval() {
        let env: StackEnvironment = &[("a", 5), ("b", 3)];

        let expr = DimSizeExpr::Param("a");
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(5));
        assert_eq!(expr.try_match(5, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(42, &env).unwrap(), TryMatchResult::Conflict);

        let expr = DimSizeExpr::Param("x");
        assert_eq!(expr.try_eval(&env), TryEvalResult::UnboundParams(1));
        assert_eq!(expr.try_match(5, &env).unwrap(), TryMatchResult::ParamConstraint("x", 5));

        let expr = DimSizeExpr::Negate(&DimSizeExpr::Param("a"));
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(-5));
        assert_eq!(expr.try_match(-5, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(42, &env).unwrap(), TryMatchResult::Conflict);

        let expr = DimSizeExpr::Negate(&DimSizeExpr::Param("x"));
        assert_eq!(expr.try_eval(&env), TryEvalResult::UnboundParams(1));
        assert_eq!(expr.try_match(-5, &env).unwrap(), TryMatchResult::ParamConstraint("x", 5));

        let expr = DimSizeExpr::Pow(&DimSizeExpr::Param("a"), 3);
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(5 * 5 * 5));
        assert_eq!(expr.try_match(5 * 5 * 5, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(4 * 4 * 4, &env).unwrap(), TryMatchResult::Conflict);

        let expr = DimSizeExpr::Pow(&DimSizeExpr::Param("x"), 3);
        assert_eq!(expr.try_eval(&env), TryEvalResult::UnboundParams(1));
        assert_eq!(expr.try_match(27, &env).unwrap(), TryMatchResult::ParamConstraint("x", 3));

        let expr = DimSizeExpr::Sum(&[DimSizeExpr::Param("a"), DimSizeExpr::Param("b")]);
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(8));
        assert_eq!(expr.try_match(8, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(42, &env).unwrap(), TryMatchResult::Conflict);

        let expr = DimSizeExpr::Sum(&[DimSizeExpr::Param("x"), DimSizeExpr::Param("b")]);
        assert_eq!(expr.try_eval(&env), TryEvalResult::UnboundParams(1));
        assert_eq!(expr.try_match(8, &env).unwrap(), TryMatchResult::ParamConstraint("x", 5));

        let expr = DimSizeExpr::Prod(&[DimSizeExpr::Param("a"), DimSizeExpr::Param("b")]);
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(15));
        assert_eq!(expr.try_match(15, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(42, &env).unwrap(), TryMatchResult::Conflict);

        let expr = DimSizeExpr::Prod(&[DimSizeExpr::Param("x"), DimSizeExpr::Param("b")]);
        assert_eq!(expr.try_eval(&env), TryEvalResult::UnboundParams(1));
        assert_eq!(expr.try_match(15, &env).unwrap(), TryMatchResult::ParamConstraint("x", 5));

        // Fancy
        let expr =
        DimSizeExpr::Sum(&[
            DimSizeExpr::Prod(&[DimSizeExpr::Param("a"), DimSizeExpr::Param("b")]),
            DimSizeExpr::Negate(&DimSizeExpr::Pow(&DimSizeExpr::Param("a"), 2))
        ]);
        let target = 5 * 3 - 5 * 5; // 15 - 25 = -10
        assert_eq!(expr.try_eval(&env), TryEvalResult::Value(target));
        assert_eq!(expr.try_match(target, &env).unwrap(), TryMatchResult::Match);
        assert_eq!(expr.try_match(42, &env).unwrap(), TryMatchResult::Conflict);
    }
}
