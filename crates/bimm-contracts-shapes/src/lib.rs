pub trait StaticBindings {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize>;

    fn contains_key(
        &self,
        key: &str,
    ) -> bool {
        self.lookup(key).is_some()
    }
}

pub type Bindings<'a> = &'a [(&'a str, usize)];

impl<'a> StaticBindings for Bindings<'a> {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.iter()
            .find_map(|(k, v)| if k == &key { Some(*v) } else { None })
    }

    fn contains_key(
        &self,
        key: &str,
    ) -> bool {
        self.iter().any(|(k, _)| k == &key)
    }
}

pub struct Env<'a> {
    bindings: Bindings<'a>,
    local: Vec<(&'a str, usize)>,
}

impl<'a> StaticBindings for Env<'a> {
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

impl<'a> Env<'a> {
    pub fn insert(
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternTerm<'a> {
    Any,
    Ellipsis,
    Expr(SizeExpr<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchResult<'a> {
    /// All params bound and expression equals target.
    Match,

    /// Expression value does not match target.
    MissMatch,

    /// Expression can be solved for a single unbound param.
    Constraint(&'a str, isize),

    /// Expression cannot be solved with current bindings
    UnderConstrained,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalValue {
    Value(isize),
    UnboundCount(usize),
}

impl<'a> SizeExpr<'a> {
    pub fn fixed(value: isize) -> Self {
        SizeExpr::Fixed(value)
    }

    pub fn param(name: &'a str) -> Self {
        SizeExpr::Param(name)
    }

    pub fn negate(expr: SizeExpr<'a>) -> Self {
        SizeExpr::Negate(Box::new(expr))
    }

    pub fn pow(
        base: SizeExpr<'a>,
        exp: usize,
    ) -> Self {
        assert!(exp > 0, "Exponents must be > 0: {exp}");
        SizeExpr::Pow(Box::new(base), exp)
    }

    pub fn sum(exprs: Vec<SizeExpr<'a>>) -> Self {
        SizeExpr::Sum(exprs)
    }

    pub fn prod(exprs: Vec<SizeExpr<'a>>) -> Self {
        SizeExpr::Prod(exprs)
    }

    pub fn maybe_eval(
        &self,
        env: &Env<'a>,
    ) -> EvalValue {
        fn monoid<'a>(
            exprs: &'a [SizeExpr<'a>],
            env: &'a Env,
            init: isize,
            op: fn(&mut isize, isize),
        ) -> EvalValue {
            let mut accum = init;
            let mut unbound_count = 0;

            for expr in exprs {
                match expr.maybe_eval(env) {
                    EvalValue::Value(value) => op(&mut accum, value),
                    EvalValue::UnboundCount(count) => unbound_count += count,
                }
            }
            if unbound_count > 0 {
                EvalValue::UnboundCount(unbound_count)
            } else {
                EvalValue::Value(accum)
            }
        }

        match self {
            SizeExpr::Fixed(value) => EvalValue::Value(*value),
            SizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    EvalValue::Value(value as isize)
                } else {
                    EvalValue::UnboundCount(1) // Single unbound param
                }
            }
            SizeExpr::Negate(expr) => match expr.maybe_eval(env) {
                EvalValue::Value(value) => EvalValue::Value(-value),
                ubc => ubc,
            },
            SizeExpr::Pow(base, exp) => {
                match base.maybe_eval(env) {
                    EvalValue::Value(value) => {
                        if *exp == 0 {
                            // Any number to the power of 0 is 1.
                            EvalValue::Value(1_isize)
                        } else {
                            EvalValue::Value(value.pow(*exp as u32))
                        }
                    }
                    ubc => ubc,
                }
            }
            SizeExpr::Sum(exprs) => monoid(exprs, env, 0, |accum, value| *accum += value),
            SizeExpr::Prod(exprs) => monoid(exprs, env, 1, |accum, value| *accum *= value),
        }
    }

    /// Find exact integer nth root if it exists (no search needed)
    fn exact_nth_root(
        value: isize,
        n: usize,
    ) -> Option<isize> {
        if n == 0 {
            return None; // Invalid exponent
        }
        if n == 1 {
            return Some(value);
        }

        match value {
            0 => Some(0),
            1 => Some(1),
            v if v > 0 => {
                // For positive values, compute approximate root and check exact match
                let approx = (v as f64).powf(1.0 / n as f64).round() as i32;

                // Check candidates around the approximation
                for candidate in (approx.saturating_sub(1))..=(approx + 1) {
                    let candidate = candidate as isize;
                    if candidate >= 0 {
                        if let Some(power) = candidate.checked_pow(n as u32) {
                            if power == value {
                                return Some(candidate);
                            }
                        }
                    }
                }
                None
            }
            v => {
                // Negative values: only possible for odd exponents
                if n % 2 == 1 {
                    // For odd n, (-x)^n = -(x^n), so we can find the positive root and negate
                    Self::exact_nth_root(-v, n).map(|root| -root)
                } else {
                    None // No real root for negative value with even exponent
                }
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
    ///
    /// ## Errors
    ///
    /// Returns an error string if the expression cannot be reconciled with the target value, such as when an exponent is invalid or a sum/product does not match the target.
    fn reconcile_expression_value(
        &'a self,
        target: isize,
        env: &Env<'a>,
    ) -> Result<MatchResult<'a>, String> {
        match self {
            SizeExpr::Fixed(value) => {
                if *value == target {
                    Ok(MatchResult::Match)
                } else {
                    Err(format!(
                        "Fixed value {} does not match target {}",
                        value, target
                    ))
                }
            }
            SizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    if value as isize == target {
                        Ok(MatchResult::Match)
                    } else {
                        Err(format!(
                            "Parameter {} has value {}, does not match target {}",
                            name, value, target
                        ))
                    }
                } else {
                    Ok(MatchResult::Constraint(name, target))
                }
            }
            SizeExpr::Negate(expr) => expr.reconcile_expression_value(-target, env),
            SizeExpr::Pow(base, exp) => {
                let exp = *exp;

                if exp == 0 {
                    // 0 is forbidden, because we cannot solve for base in base^0 = target.
                    return Err("Exponent must be greater than 0".to_string());
                }

                // base^exp = target, solve for base
                // base = target^(1/exp
                let root = match Self::exact_nth_root(target, exp) {
                    Some(root) => root,
                    None => return Ok(MatchResult::MissMatch), // No exact integer root
                };
                base.reconcile_expression_value(root, env)
            }
            SizeExpr::Sum(exprs) => {
                let mut unbound_expr = None;
                let mut accumulator: isize = 0;

                for expr in exprs {
                    match expr.maybe_eval(env) {
                        EvalValue::Value(value) => accumulator += value,
                        EvalValue::UnboundCount(count) => {
                            if count == 1 && unbound_expr.is_none() {
                                unbound_expr = Some(expr);
                            } else {
                                return Ok(MatchResult::MissMatch);
                            }
                        }
                    }
                }
                if let Some(expr) = unbound_expr {
                    expr.reconcile_expression_value(target - accumulator, env)
                } else if accumulator == target {
                    Ok(MatchResult::Match)
                } else {
                    Err(format!(
                        "Sum of bound expressions {} does not match target {}",
                        accumulator, target
                    ))
                }
            }
            SizeExpr::Prod(exprs) => {
                let mut unbound_expr = None;
                let mut accumulator: isize = 1;

                for expr in exprs {
                    match expr.maybe_eval(env) {
                        EvalValue::Value(value) => accumulator *= value,
                        EvalValue::UnboundCount(count) => {
                            if count == 1 && unbound_expr.is_none() {
                                unbound_expr = Some(expr);
                            } else {
                                return Ok(MatchResult::MissMatch);
                            }
                        }
                    }
                }

                if target % accumulator != 0 {
                    // Non-integer solution
                    return Ok(MatchResult::MissMatch);
                }

                if let Some(expr) = unbound_expr {
                    expr.reconcile_expression_value(target / accumulator, env)
                } else if accumulator == target {
                    Ok(MatchResult::Match)
                } else {
                    Err(format!(
                        "Prod of bound expressions {} does not match target {}",
                        accumulator, target
                    ))
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
        fn invalid_pattern(
            error: String,
            pattern: &[PatternTerm],
        ) -> String {
            format!("Invalid Pattern: {}\nPattern: {:#?}\n", error, pattern)
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
                    match expr.reconcile_expression_value(target, &env)? {
                        MatchResult::Match => Ok(()),
                        MatchResult::MissMatch => Err(describe_missmatch(self, shape, bindings)),
                        MatchResult::Constraint(param_name, value) => {
                            let param_name: &'a str = param_name;
                            env.insert(param_name, value as usize);
                            Ok(())
                        }
                        MatchResult::UnderConstrained => {
                            Err(describe_missmatch(self, shape, bindings))
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
                        match expr.reconcile_expression_value(target, &env)? {
                            MatchResult::Match => Ok(()),
                            MatchResult::MissMatch => {
                                Err(describe_missmatch(self, shape, bindings))
                            }
                            MatchResult::Constraint(param_name, value) => {
                                let param_name: &'a str = param_name;
                                env.insert(param_name, value as usize);
                                Ok(())
                            }
                            MatchResult::UnderConstrained => {
                                Err(describe_missmatch(self, shape, bindings))
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
        let expr = SizeExpr::sum(vec![
            SizeExpr::param("x"),
            SizeExpr::negate(SizeExpr::fixed(5)),
        ]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(8, &env),
            Ok(MatchResult::Constraint("x", 13))
        );
    }

    #[test]
    fn test_simple_product() {
        // 2 * x, target = 6, should solve x = 3
        let expr = SizeExpr::prod(vec![SizeExpr::fixed(2), SizeExpr::param("x")]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(6, &env),
            Ok(MatchResult::Constraint("x", 3))
        );
    }

    #[test]
    fn test_quadratic_case() {
        // p * p, target = 9
        // This becomes: p * p = 9, so we solve the first p for 9/p
        // But since both factors are the same unbound param, this should work
        let expr = SizeExpr::prod(vec![SizeExpr::param("p"), SizeExpr::param("p")]);
        let env = Env::new(&[]);

        // This should solve: first p gets target/second_p, but second_p is unbound too
        // Actually, this should fail because we have the same param multiple times
        assert_eq!(
            expr.reconcile_expression_value(9, &env),
            Ok(MatchResult::MissMatch)
        );
    }

    #[test]
    fn test_complex_expression() {
        // w * p * h * p + c * z
        // With w=2, h=3, c=1, z=4, and p unbound, target=25
        // This should fail because p appears twice in the product
        let expr = SizeExpr::sum(vec![
            SizeExpr::prod(vec![
                SizeExpr::param("w"),
                SizeExpr::param("p"),
                SizeExpr::param("h"),
                SizeExpr::param("p"),
            ]),
            SizeExpr::prod(vec![SizeExpr::param("c"), SizeExpr::param("z")]),
        ]);

        let mut env = Env::new(&[]);
        env.insert("w", 2);
        env.insert("h", 3);
        env.insert("c", 1);
        env.insert("z", 4);

        // p appears multiple times in the first product term, so this should fail
        assert_eq!(
            expr.reconcile_expression_value(25, &env),
            Ok(MatchResult::MissMatch)
        );
    }

    #[test]
    fn test_nested_structure() {
        // 2 * (x + 3) - 1 == 9
        // 2 * (x + 3) == 10
        // x + 3 == 5
        // x == 2
        let expr = SizeExpr::sum(vec![
            SizeExpr::prod(vec![
                SizeExpr::fixed(2),
                SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::fixed(3)]),
            ]),
            SizeExpr::fixed(-1),
        ]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(9, &env),
            Ok(MatchResult::Constraint("x", 2))
        );
    }

    #[test]
    fn test_all_bound_success() {
        // x + 5 with x = 3, target = 8
        let expr = SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::fixed(5)]);

        let mut env = Env::new(&[]);
        env.insert("x", 3);

        assert_eq!(
            expr.reconcile_expression_value(8, &env),
            Ok(MatchResult::Match)
        );
    }

    #[test]
    fn test_multiple_unbound() {
        // x + y, multiple unbound params
        let expr = SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::param("y")]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(5, &env),
            Ok(MatchResult::MissMatch)
        );
    }

    #[test]
    fn test_no_integer_solution() {
        // 2 * x == 3
        // x = 3/2, which is not an integer
        let expr = SizeExpr::prod(vec![SizeExpr::fixed(2), SizeExpr::param("x")]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(3, &env),
            Ok(MatchResult::MissMatch)
        );
    }

    #[test]
    fn test_power_solve_base() {
        // x^3 = 8, should solve x = 2
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(8, &env),
            Ok(MatchResult::Constraint("x", 2))
        );
    }

    #[test]
    fn test_power_no_solution() {
        // 2^3 = x where x != 8 (no solution when fully bound and doesn't match)
        let expr = SizeExpr::pow(SizeExpr::fixed(2), 3);
        let env = Env::new(&[]);

        // 2^3 = 8, so target 5 should fail
        assert_eq!(
            expr.reconcile_expression_value(5, &env),
            Ok(MatchResult::MissMatch)
        );
    }

    #[test]
    fn test_power_in_sum() {
        // x^2 + 1 = 10, should solve x = 3 (since 3^2 + 1 = 10)
        let expr = SizeExpr::sum(vec![
            SizeExpr::pow(SizeExpr::param("x"), 2),
            SizeExpr::fixed(1),
        ]);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(10, &env),
            Ok(MatchResult::Constraint("x", 3))
        );
    }

    #[test]
    fn test_power_cube_root() {
        // x^3 = 27, should solve x = 3
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(27, &env),
            Ok(MatchResult::Constraint("x", 3))
        );
    }

    #[test]
    fn test_power_negative_base_odd_exp() {
        // x^3 = -8, should solve x = -2 (since (-2)^3 = -8)
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(-8, &env),
            Ok(MatchResult::Constraint("x", -2))
        );
    }

    #[test]
    fn test_power_negative_base_even_exp() {
        // x^2 = -4 (no real solution for even exponent and negative target)
        let expr = SizeExpr::pow(SizeExpr::param("x"), 2);
        let env = Env::new(&[]);

        assert_eq!(
            expr.reconcile_expression_value(-4, &env),
            Ok(MatchResult::MissMatch)
        );
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
            PatternTerm::Expr(SizeExpr::param("b")),
            PatternTerm::Expr(SizeExpr::prod(vec![
                SizeExpr::param("h"),
                SizeExpr::param("p"),
            ])),
            PatternTerm::Expr(SizeExpr::prod(vec![
                SizeExpr::param("w"),
                SizeExpr::param("p"),
            ])),
            PatternTerm::Expr(SizeExpr::param("c")),
        ];
        // println!("pattern: {:#?}", pattern);

        let bindings = [("p", p), ("c", c)];

        println!();

        let [u_b, u_h, u_w] = pattern
            .unpack_shape(&shape, &["b", "h", "w"], &bindings)
            .unwrap();

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
    }
}
