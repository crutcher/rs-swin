use std::fmt::{Display, Formatter};

pub enum OpTree {
    Leaf(String),
    Node { op: String, children: Vec<OpTree> },
}

pub trait StaticBindings<'a> {
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<usize>;

    fn contains_key(
        &self,
        key: &'a str,
    ) -> bool {
        self.lookup(key).is_some()
    }
}

pub trait MutBindings<'a>: StaticBindings<'a> {
    fn insert(
        &mut self,
        key: &'a str,
        value: usize,
    );
}

pub type Bindings<'a> = &'a [(&'a str, usize)];

impl<'a> StaticBindings<'a> for Bindings<'a> {
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<usize> {
        self.iter()
            .find_map(|(k, v)| if k == &key { Some(*v) } else { None })
    }

    fn contains_key(
        &self,
        key: &'a str,
    ) -> bool {
        self.iter().any(|(k, _)| k == &key)
    }
}

pub struct Env<'a> {
    bindings: Bindings<'a>,
    local: Vec<(&'a str, usize)>,
}

impl<'a> StaticBindings<'a> for Env<'a> {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        let local_bindings = &self.local[..];
        local_bindings.lookup(key).or_else(|| {
            // If not found in local, check the static bindings
            self.bindings.lookup(key)
        })
    }

    fn contains_key(
        &self,
        key: &str,
    ) -> bool {
        let local_bindings = &self.local[..];
        local_bindings.contains_key(key) || self.bindings.contains_key(key)
    }
}

impl<'a> MutBindings<'a> for Env<'a> {
    fn insert(
        &mut self,
        key: &'a str,
        value: usize,
    ) {
        self.local.push((key, value))
    }
}

impl<'a> Env<'a> {
    fn new(bindings: Bindings<'a>) -> Self {
        Env {
            bindings,
            local: Vec::new(),
        }
    }

    fn export<const K: usize>(
        &self,
        keys: &[&str; K],
    ) -> [usize; K] {
        let mut values = [0; K];
        let local_bindings = &self.local[..];
        for i in 0..K {
            let key = keys[i];
            values[i] = match local_bindings.lookup(key) {
                Some(value) => value,
                None => panic!("No value for key \"{}\"", key),
            };
        }
        values
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeExpr<'a> {
    Fixed(isize),
    Param(&'a str),
    Negate(Box<SizeExpr<'a>>),
    Pow(Box<SizeExpr<'a>>, usize),
    Sum(Vec<SizeExpr<'a>>),
    Prod(Vec<SizeExpr<'a>>),
}

impl Display for SizeExpr<'_> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            SizeExpr::Fixed(fixed) => write!(f, "{}", fixed),
            SizeExpr::Param(param) => write!(f, "{}", param),
            SizeExpr::Negate(negate) => write!(f, "-({})", negate),
            SizeExpr::Pow(base, exponent) => {
                write!(f, "({})^({})", base, exponent)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternTerm<'a> {
    Any,
    Ellipsis,
    Expr(SizeExpr<'a>),
}

impl Display for PatternTerm<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            PatternTerm::Any => write!(f, "_"),
            PatternTerm::Ellipsis => write!(f, "..."),
            PatternTerm::Expr(expr) => write!(f, "{}", expr),
        }
    }
}

fn format_shape_pattern<'a>(pattern: &[PatternTerm<'a>]) -> String {
    // TODO: make this a `fmt` method.
    let mut result = String::new();
    result.push('[');
    for (idx, expr) in pattern.iter().enumerate() {
        if idx > 0 {
            result.push_str(", ");
        }
        result.push_str(&format!("{}", expr));
    }
    result.push(']');
    result
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
    TargetMatch,

    /// Expression value does not match the target.
    ValueMissMatch,

    /// Expression can be solved for a single unbound param.
    ParamConstraint(&'a str, isize),
}

/// Find the exact integer nth root if it exists.
///
/// ## Arguments
///
/// - `value` - the value.
/// - `n` - the root power.
///
/// ## Returns
///
/// Either the exact root (positive if possible) or None.
fn maybe_nth_integer_root(
    value: isize,
    n: usize,
) -> Option<isize> {
    match value {
        0 => Some(0),
        1 => Some(1),
        v if v > 1 => {
            let value = value as usize;
            let n = n as u32;
            for x in 2usize.. {
                let y: usize = x.saturating_pow(n);
                if value < y {
                    break;
                }
                if value == y {
                    return Some(x as isize);
                }
            }
            None
        }
        v => {
            // Negative values: only possible for odd exponents
            if n % 2 == 1 {
                // For odd n, (-x)^n = -(x^n), so we can find the positive root and negate
                maybe_nth_integer_root(-v, n).map(|root| -root)
            } else {
                None // No real root for negative value with even exponent
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EvalResult {
    Value(isize),
    UnboundParams(usize),
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
    fn try_evaluate(
        &self,
        env: &Env<'a>,
    ) -> EvalResult {
        #[inline(always)]
        fn reduce_monoid<'a>(
            exprs: &'a [SizeExpr<'a>],
            env: &'a Env,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> EvalResult {
            let mut tmp = zero;
            let mut count = 0;
            for expr in exprs {
                match expr.try_evaluate(env) {
                    EvalResult::Value(v) => op(&mut tmp, v),
                    EvalResult::UnboundParams(c) => count += c,
                }
            }
            if count == 0 {
                EvalResult::Value(tmp)
            } else {
                EvalResult::UnboundParams(count)
            }
        }
        match self {
            SizeExpr::Fixed(value) => EvalResult::Value(*value),
            SizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    EvalResult::Value(value as isize)
                } else {
                    EvalResult::UnboundParams(1) // Single unbound param
                }
            }
            SizeExpr::Negate(expr) => match expr.try_evaluate(env) {
                EvalResult::Value(value) => EvalResult::Value(-value),
                x => x,
            },
            SizeExpr::Pow(base, exp) => {
                match base.try_evaluate(env) {
                    EvalResult::Value(value) => {
                        // Any number to the power of 0 is 1.
                        if *exp == 0 {
                            EvalResult::Value(1)
                        } else {
                            EvalResult::Value(value.pow(*exp as u32))
                        }
                    }
                    x => x,
                }
            }
            SizeExpr::Sum(exprs) => reduce_monoid(exprs, env, 0, |tmp, value| *tmp += value),
            SizeExpr::Prod(exprs) => reduce_monoid(exprs, env, 1, |tmp, value| *tmp *= value),
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
    ///
    /// ## Errors
    ///
    /// Returns an error string if the expression cannot be reconciled with the target value, such as when an exponent is invalid or a sum/product does not match the target.
    pub fn try_match(
        &'a self,
        target: isize,
        env: &Env<'a>,
    ) -> Result<TryMatchResult<'a>, String> {
        #[inline(always)]
        fn reduce_monoid<'a>(
            exprs: &'a [SizeExpr<'a>],
            env: &Env,
            zero: isize,
            op: fn(&mut isize, isize),
        ) -> Result<(isize, Option<&'a SizeExpr<'a>>), String> {
            let mut partial_value: isize = zero;
            let mut rem_expr = None;

            for expr in exprs {
                match expr.try_evaluate(env) {
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
            Ok((partial_value, rem_expr))
        }

        match self {
            SizeExpr::Fixed(value) => {
                if *value == target {
                    Ok(TryMatchResult::TargetMatch)
                } else {
                    Ok(TryMatchResult::ValueMissMatch)
                }
            }
            SizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    if value as isize == target {
                        Ok(TryMatchResult::TargetMatch)
                    } else {
                        Ok(TryMatchResult::ValueMissMatch)
                    }
                } else {
                    Ok(TryMatchResult::ParamConstraint(name, target))
                }
            }
            SizeExpr::Negate(expr) => expr.try_match(-target, env),
            SizeExpr::Pow(base, exp) => match maybe_nth_integer_root(target, *exp) {
                Some(root) => base.try_match(root, env),
                None => Err("No integer solution.".to_string()),
            },
            SizeExpr::Sum(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_monoid(exprs, env, 0, |acc, value| *acc += value)?;
                if let Some(expr) = rem_expr {
                    expr.try_match(target - partial_value, env)
                } else if partial_value == target {
                    Ok(TryMatchResult::TargetMatch)
                } else {
                    Ok(TryMatchResult::ValueMissMatch)
                }
            }
            SizeExpr::Prod(exprs) => {
                let (partial_value, rem_expr) =
                    reduce_monoid(exprs, env, 1, |acc, value| *acc *= value)?;
                if target % partial_value != 0 {
                    // Non-integer solution
                    Err("No integer solution.".to_string())
                } else if let Some(expr) = rem_expr {
                    expr.try_match(target / partial_value, env)
                } else if partial_value == target {
                    Ok(TryMatchResult::TargetMatch)
                } else {
                    Ok(TryMatchResult::ValueMissMatch)
                }
            }
        }
    }
}

pub trait ShapePattern<'a> {
    fn unpack_shape<const K: usize>(
        &'a self,
        shape: &[usize],
        keys: &[&'a str; K],
        bindings: Bindings<'a>,
    ) -> Result<[usize; K], String>;
}

impl<'a, const D: usize> ShapePattern<'a> for [PatternTerm<'a>; D] {
    fn unpack_shape<const K: usize>(
        &'a self,
        shape: &[usize],
        keys: &[&'a str; K],
        bindings: Bindings<'a>,
    ) -> Result<[usize; K], String> {
        fn invalid_pattern<'a>(
            error: String,
            pattern: &[PatternTerm<'a>],
        ) -> String {
            format!(
                "Invalid Pattern: {}\nPattern: {}\n",
                error,
                format_shape_pattern(pattern)
            )
        }

        fn describe_missmatch<'a>(
            pattern: &[PatternTerm<'a>],
            shape: &[usize],
            bindings: Bindings<'a>,
        ) -> String {
            format!(
                "Shape Match Failure:\nShape: {:#?}\nPattern: {:#?}\nBindings: {:#?}\n",
                shape, pattern, bindings
            )
        }

        let mut env: Env<'a> = Env::new(bindings);

        let mut ellipsis_pos: Option<usize> = None;
        for (idx, term) in self.iter().enumerate() {
            if let PatternTerm::Ellipsis = term {
                if ellipsis_pos.is_some() {
                    return Err(invalid_pattern(
                        "Multiple ellipses in pattern".to_string(),
                        self,
                    ));
                }
                ellipsis_pos = Some(idx);
            }
        }

        let ellipsis_len = if ellipsis_pos.is_some() {
            let shape_n = shape.len();
            let non_ellipsis_terms = self.len() - 1; // Exclude the ellipsis term itself

            if shape_n < non_ellipsis_terms {
                return Err(describe_missmatch(self, shape, bindings));
            } else {
                shape_n - non_ellipsis_terms
            }
        } else {
            0
        };

        if ellipsis_len == 0 {
            ellipsis_pos = None;
        }

        let marker = match ellipsis_pos {
            Some(pos) => pos,
            None => self.len(),
        };

        for pattern_idx in 0..marker {
            let term = &self[pattern_idx];
            let shape_idx = pattern_idx;

            (match term {
                PatternTerm::Any => {
                    // Any term, no action needed
                    Ok(())
                }
                PatternTerm::Ellipsis => Err(invalid_pattern(
                    "INTERNAL ERROR: mishandled Ellipsis".to_string(),
                    self,
                )),
                PatternTerm::Expr(expr) => {
                    let target = shape[shape_idx] as isize;
                    match expr.try_match(target, &env)? {
                        TryMatchResult::TargetMatch => Ok(()),
                        TryMatchResult::ValueMissMatch => {
                            Err(describe_missmatch(self, shape, bindings))
                        }
                        TryMatchResult::ParamConstraint(param_name, value) => {
                            env.insert(param_name, value as usize);
                            Ok(())
                        }
                    }
                }
            })?;
        }

        if ellipsis_len > 0 {
            for pattern_idx in (ellipsis_pos.unwrap() + 1)..shape.len() {
                let term = &self[pattern_idx];
                let shape_idx = pattern_idx + ellipsis_len;
                (match term {
                    PatternTerm::Any => {
                        // Any term, no action needed
                        Ok(())
                    }
                    PatternTerm::Ellipsis => Err("INTERNAL ERROR: mishandled Ellipsis".to_string()),
                    PatternTerm::Expr(expr) => {
                        let target = shape[shape_idx] as isize;
                        match expr.try_match(target, &env)? {
                            TryMatchResult::TargetMatch => Ok(()),
                            TryMatchResult::ValueMissMatch => {
                                Err(describe_missmatch(self, shape, bindings))
                            }
                            TryMatchResult::ParamConstraint(param_name, value) => {
                                env.insert(param_name, value as usize);
                                Ok(())
                            }
                        }
                    }
                })?;
            }
        }

        Ok(env.export(keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_sum() {
        // x + 5, target = 8, should solve x = 3
        let expr1 = SizeExpr::Fixed(5);
        let exprs = vec![SizeExpr::Param("x"), SizeExpr::Negate(Box::new(expr1))];
        let expr = SizeExpr::Sum(exprs);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(8, &env),
            Ok(TryMatchResult::ParamConstraint("x", 13))
        );
    }

    #[test]
    fn test_simple_product() {
        // 2 * x, target = 6, should solve x = 3
        let exprs = vec![SizeExpr::Fixed(2), SizeExpr::Param("x")];
        let expr = SizeExpr::Prod(exprs);
        let env = Env::new(&[]);

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
        let exprs = vec![SizeExpr::Param("p"), SizeExpr::Param("p")];
        let expr = SizeExpr::Prod(exprs);
        let env = Env::new(&[]);

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
        let exprs1 = vec![SizeExpr::Param("c"), SizeExpr::Param("z")];
        let exprs2 = vec![
            SizeExpr::Param("w"),
            SizeExpr::Param("p"),
            SizeExpr::Param("h"),
            SizeExpr::Param("p"),
        ];
        let exprs = vec![SizeExpr::Prod(exprs2), SizeExpr::Prod(exprs1)];
        let expr = SizeExpr::Sum(exprs);

        let mut env = Env::new(&[]);
        env.insert("w", 2);
        env.insert("h", 3);
        env.insert("c", 1);
        env.insert("z", 4);

        // p appears multiple times in the first product term, so this should fail
        assert_eq!(
            expr.try_match(25, &env),
            Err("Too many unbound params".to_string())
        );
    }

    #[test]
    fn test_nested_structure() {
        // 2 * (x + 3) - 1 == 9
        // 2 * (x + 3) == 10
        // x + 3 == 5
        // x == 2
        let exprs = vec![SizeExpr::Param("x"), SizeExpr::Fixed(3)];
        let exprs2 = vec![SizeExpr::Fixed(2), SizeExpr::Sum(exprs)];
        let exprs1 = vec![SizeExpr::Prod(exprs2), SizeExpr::Fixed(-1)];
        let expr = SizeExpr::Sum(exprs1);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(9, &env),
            Ok(TryMatchResult::ParamConstraint("x", 2))
        );
    }

    #[test]
    fn test_all_bound_success() {
        // x + 5 with x = 3, target = 8
        let exprs = vec![SizeExpr::Param("x"), SizeExpr::Fixed(5)];
        let expr = SizeExpr::Sum(exprs);

        let mut env = Env::new(&[]);
        env.insert("x", 3);

        assert_eq!(expr.try_match(8, &env), Ok(TryMatchResult::TargetMatch));
    }

    #[test]
    fn test_multiple_unbound() {
        // x + y, multiple unbound params
        let exprs = vec![SizeExpr::Param("x"), SizeExpr::Param("y")];
        let expr = SizeExpr::Sum(exprs);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(25, &env),
            Err("Too many unbound params".to_string())
        );
    }

    #[test]
    fn test_no_integer_solution() {
        // 2 * x == 3
        // x = 3/2, which is not an integer
        let exprs = vec![SizeExpr::Fixed(2), SizeExpr::Param("x")];
        let expr = SizeExpr::Prod(exprs);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(3, &env),
            Err("No integer solution.".to_string())
        );
    }

    #[test]
    fn test_power_solve_base() {
        // x^3 = 8, should solve x = 2
        let base = SizeExpr::Param("x");
        let expr = SizeExpr::Pow(Box::new(base), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(8, &env),
            Ok(TryMatchResult::ParamConstraint("x", 2))
        );
    }

    #[test]
    fn test_power_no_solution() {
        // 2^3 = x where x != 8 (no solution when fully bound and doesn't match)
        let base = SizeExpr::Fixed(2);
        let expr = SizeExpr::Pow(Box::new(base), 3);
        let env = Env::new(&[]);

        // 2^3 = 8, so target 5 should fail
        assert_eq!(
            expr.try_match(5, &env),
            Err("No integer solution.".to_string())
        );
    }

    #[test]
    fn test_power_in_sum() {
        // x^2 + 1 = 10, should solve x = 3 (since 3^2 + 1 = 10)
        let base = SizeExpr::Param("x");
        let exprs = vec![SizeExpr::Pow(Box::new(base), 2), SizeExpr::Fixed(1)];
        let expr = SizeExpr::Sum(exprs);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(10, &env),
            Ok(TryMatchResult::ParamConstraint("x", 3))
        );
    }

    #[test]
    fn test_power_cube_root() {
        // x^3 = 27, should solve x = 3
        let base = SizeExpr::Param("x");
        let expr = SizeExpr::Pow(Box::new(base), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(27, &env),
            Ok(TryMatchResult::ParamConstraint("x", 3))
        );
    }

    #[test]
    fn test_power_negative_base_odd_exp() {
        // x^3 = -8, should solve x = -2 (since (-2)^3 = -8)
        let base = SizeExpr::Param("x");
        let expr = SizeExpr::Pow(Box::new(base), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(-8, &env),
            Ok(TryMatchResult::ParamConstraint("x", -2))
        );
    }

    #[test]
    fn test_power_negative_base_even_exp() {
        // x^2 = -4 (no real solution for even exponent and negative target)
        let base = SizeExpr::Param("x");
        let expr = SizeExpr::Pow(Box::new(base), 2);
        let env = Env::new(&[]);

        assert_eq!(
            expr.try_match(-4, &env),
            Err("No integer solution.".to_string())
        );
    }

    #[test]
    fn test_format_pattern() {
        let base = SizeExpr::Param("h");
        let exprs = vec![SizeExpr::Param("a"), SizeExpr::Param("b")];
        let exprs1 = vec![SizeExpr::Param("h"), SizeExpr::Sum(exprs)];
        let pattern = [
            PatternTerm::Expr(SizeExpr::Param("b")),
            PatternTerm::Expr(SizeExpr::Prod(exprs1)),
            PatternTerm::Expr(SizeExpr::Pow(Box::new(base), 2)),
        ];

        assert_eq!(format_shape_pattern(&pattern), "[b, (h*(a+b)), (h)^(2)]");
    }

    #[test]
    fn test_unpack_shape() {
        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;

        let shape = [b, h * p, w * p, c];
        // println!("shape: {:?}", shape);

        let pattern = [
            PatternTerm::Expr(SizeExpr::Param("b")),
            PatternTerm::Expr(SizeExpr::Prod(vec![
                SizeExpr::Param("h"),
                SizeExpr::Param("p"),
            ])),
            PatternTerm::Expr(SizeExpr::Prod(vec![
                SizeExpr::Param("w"),
                SizeExpr::Param("p"),
            ])),
            PatternTerm::Expr(SizeExpr::Param("c")),
        ];

        let bindings = [("p", p), ("c", c)];

        let [u_b, u_h, u_w] = pattern
            .unpack_shape(&shape, &["b", "h", "w"], &bindings)
            .unwrap();

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
    }
}
