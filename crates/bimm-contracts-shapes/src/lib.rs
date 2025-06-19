use std::collections::HashMap;
use std::ops::{AddAssign, MulAssign};

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
    local: HashMap<&'a str, usize>,
    bindings: Bindings<'a>,
}

impl<'a> StaticBindings for Env<'a> {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.local
            .get(key)
            .cloned()
            .or_else(|| self.bindings.lookup(key))
    }

    fn contains_key(
        &self,
        key: &str,
    ) -> bool {
        self.local.contains_key(key) || self.bindings.contains_key(key)
    }
}

impl<'a> Env<'a> {
    pub fn insert(
        &mut self,
        key: &'a str,
        value: usize,
    ) {
        self.local.insert(key, value);
    }
}

impl<'a> Env<'a> {
    fn new(bindings: Bindings<'a>) -> Self {
        Env {
            local: HashMap::new(),
            bindings,
        }
    }

    fn export<const K: usize>(
        &self,
        keys: &[&str; K],
    ) -> [usize; K] {
        let mut values = [0; K];
        for i in 0..K {
            let key = keys[i];
            values[i] = match self.lookup(key) {
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
pub enum MatchResult<'a> {
    Success,               // All params bound and expression equals target
    Failure,               // Multiple unbound params or other failure
    Solve(&'a str, isize), // Single unbound param and its required value
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

    // TODO(crutcher): Consider Result<<volue>, <unbound_count>> ?

    /// Evaluate expression with given bindings (assumes all params are bound)
    pub fn evaluate(
        &self,
        env: &Env<'a>,
    ) -> Option<isize> {
        match self {
            SizeExpr::Fixed(value) => Some(*value),
            SizeExpr::Param(name) => env.lookup(name).map(|v| v as isize),
            SizeExpr::Negate(expr) => expr.evaluate(env).map(|x| -x),
            SizeExpr::Pow(base, exp) => {
                let base_val = base.evaluate(env)?;
                let exp = *exp;
                base_val.checked_pow(exp as u32)
            }
            SizeExpr::Sum(exprs) => {
                let mut sum = 0;
                for expr in exprs {
                    sum += expr.evaluate(env)?;
                }
                Some(sum)
            }
            SizeExpr::Prod(exprs) => {
                let mut prod = 1;
                for expr in exprs {
                    prod *= expr.evaluate(env)?;
                }
                Some(prod)
            }
        }
    }

    pub fn value_or_count(
        &self,
        env: &Env<'a>,
    ) -> Result<isize, usize> {
        match self {
            SizeExpr::Fixed(value) => Ok(*value),
            SizeExpr::Param(name) => {
                if let Some(value) = env.lookup(name) {
                    Ok(value as isize)
                } else {
                    Err(1) // Single unbound param
                }
            }
            SizeExpr::Negate(expr) => expr.value_or_count(env).map(|x| -x),
            SizeExpr::Pow(base, exp) => {
                let base_val = base.value_or_count(env)?;
                if *exp == 0 {
                    Ok(1) // Any number to the power of 0 is 1
                } else {
                    base_val.checked_pow(*exp as u32).ok_or(1)
                }
            }
            SizeExpr::Sum(exprs) => {
                let mut sum = 0;
                let mut unbound_count = 0;

                for expr in exprs {
                    match expr.value_or_count(env) {
                        Ok(value) => sum += value,
                        Err(count) => unbound_count += count,
                    }
                }

                if unbound_count > 0 {
                    Err(unbound_count)
                } else {
                    Ok(sum)
                }
            }
            SizeExpr::Prod(exprs) => {
                let mut prod = 1;
                let mut unbound_count = 0;

                for expr in exprs {
                    match expr.value_or_count(env) {
                        Ok(value) => prod *= value,
                        Err(count) => unbound_count += count,
                    }
                }

                if unbound_count > 0 {
                    Err(unbound_count)
                } else {
                    Ok(prod)
                }
            }
        }
    }

    /// Match this expression against a target value with given bindings
    pub fn match_target(
        &self,
        target: isize,
        env: &Env<'a>,
    ) -> MatchResult {
        // TODO(crutcher): Result<?, ?> type; remove binding_state.
        match self.binding_state(env) {
            BindingState::FullyBound => {
                if self.evaluate(env).unwrap() == target {
                    MatchResult::Success
                } else {
                    MatchResult::Failure
                }
            }
            BindingState::MultipleUnbound => MatchResult::Failure,
            BindingState::SingleUnbound(param_name) => {
                match self.solve_for_target(param_name, target, env) {
                    Ok(value) => MatchResult::Solve(param_name, value),
                    Err(_) => MatchResult::Failure,
                }
            }
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

    /// Determine the binding state of this expression
    fn binding_state(
        &self,
        env: &Env<'a>,
    ) -> BindingState {
        match self {
            SizeExpr::Param(name) => {
                if env.contains_key(name) {
                    BindingState::FullyBound
                } else {
                    BindingState::SingleUnbound(name)
                }
            }
            SizeExpr::Fixed(_) => BindingState::FullyBound,
            SizeExpr::Negate(expr) => expr.binding_state(env),
            SizeExpr::Pow(base, _) => base.binding_state(env),
            SizeExpr::Sum(exprs) | SizeExpr::Prod(exprs) => {
                let mut unbound_params = Vec::new();

                for expr in exprs {
                    match expr.binding_state(env) {
                        BindingState::FullyBound => {}
                        BindingState::SingleUnbound(param) => {
                            if !unbound_params.contains(&param) {
                                unbound_params.push(param);
                            }
                        }
                        BindingState::MultipleUnbound => return BindingState::MultipleUnbound,
                    }
                }

                match unbound_params.len() {
                    0 => BindingState::FullyBound,
                    1 => BindingState::SingleUnbound(unbound_params[0]),
                    _ => BindingState::MultipleUnbound,
                }
            }
        }
    }

    fn count_unbound_params(
        &self,
        env: &Env<'a>,
    ) -> usize {
        match self {
            SizeExpr::Param(name) => {
                if env.contains_key(name) {
                    0 // Fully bound
                } else {
                    1 // Single unbound param
                }
            }
            SizeExpr::Fixed(_) => 0, // Fixed values are always bound
            SizeExpr::Negate(expr) => expr.count_unbound_params(env),
            SizeExpr::Pow(base, _) => base.count_unbound_params(env),
            SizeExpr::Sum(exprs) | SizeExpr::Prod(exprs) => exprs
                .iter()
                .map(|expr| expr.count_unbound_params(env))
                .sum(),
        }
    }

    /// Solve for a parameter by unpeeling the expression structure
    fn solve_for_target(
        &self,
        // TODO: remove the need for this parameter.
        target_param: &str,
        target: isize,
        env: &Env<'a>,
    ) -> Result<isize, String> {
        if self.count_unbound_params(env) > 1 {
            return Err("Multiple unbound parameters, cannot solve".to_string());
        }

        #[inline(always)]
        fn partially_evaluate_children<'a>(
            exprs: &'a [SizeExpr<'a>],
            env: &Env,
            init: isize,
            op: fn(&mut isize, isize),
        ) -> Result<(&'a SizeExpr<'a>, isize), String> {
            let mut unbound_expr = None;
            let mut accumulator: isize = init;

            // TODO(crutcher): sum, prod should handle children in a common way.

            for expr in exprs {
                match expr.evaluate(env) {
                    Some(value) => {
                        op(&mut accumulator, value);
                    }
                    None => {
                        if unbound_expr.is_some() {
                            // More than one unbound expression, cannot solve
                            return Err("Multiple unbound expressions in sum".to_string());
                        } else {
                            // Found the unbound expression
                            unbound_expr = Some(expr);
                        }
                    }
                }
            }
            if let Some(expr) = unbound_expr {
                Ok((expr, accumulator))
            } else {
                Err("No unbound expression found".to_string())
            }
        }

        // TODO: Crutcher; migrate to Result<isize, String> for better error handling.
        match self {
            SizeExpr::Param(name) => {
                if *name == target_param {
                    Ok(target)
                } else {
                    // This param should be bound, but we're solving for a different param
                    Err(format!(
                        "INTERNAL ERROR: solving for {} but found param {}",
                        target_param, name
                    ))
                }
            }
            SizeExpr::Fixed(value) => {
                // Fixed value should equal target
                if *value == target {
                    Ok(target)
                } else {
                    Err(format!(
                        "Fixed value {} does not match target {}",
                        value, target
                    ))
                }
            }
            SizeExpr::Negate(expr) => {
                // neg(expr) = target  =>  expr = -target
                expr.solve_for_target(target_param, -target, env)
            }
            SizeExpr::Pow(base, exp) => {
                // Exponent must be fixed and positive
                let exp = *exp;

                if exp == 0 {
                    // Invalid exponent
                    return Err("Exponent must be greater than 0".to_string());
                }

                // base^exp = target, solve for base
                // base = target^(1/exp
                let root = Self::exact_nth_root(target, exp)
                    .map_or_else(|| Err("No integer nth root found".to_string()), Ok)?;
                base.solve_for_target(target_param, root, env)
            }
            SizeExpr::Sum(exprs) => {
                // sum(exprs) = target
                partially_evaluate_children(exprs, env, 0, |acc, value| acc.add_assign(value))
                    .and_then(|(unbound_expr, bound_sum)| {
                        // unbound_expr + bound_sum = target  =>  unbound_expr = target - bound_sum
                        unbound_expr.solve_for_target(target_param, target - bound_sum, env)
                    })
            }
            SizeExpr::Prod(exprs) => {
                // prod(exprs) = target
                partially_evaluate_children(exprs, env, 1, |acc, value| acc.mul_assign(value))
                    .and_then(|(unbound_expr, bound_product)| {
                        // unbound_expr * bound_product = target  =>  unbound_expr = target / bound_product
                        if bound_product == 0 {
                            return Err("Product of bound expressions is zero".to_string());
                        }

                        if target % bound_product != 0 {
                            return Err("Target is not divisible by product of bound expressions"
                                .to_string());
                        }

                        unbound_expr.solve_for_target(target_param, target / bound_product, env)
                    })
            }
        }
    }
}

#[derive(Debug)]
pub enum BindingState<'a> {
    FullyBound,
    SingleUnbound(&'a str),
    MultipleUnbound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternTerm<'a> {
    Any,
    Ellipsis,
    Expr(SizeExpr<'a>),
}

pub trait ShapePattern<'a> {
    fn unpack_shape<const K: usize>(
        &self,
        shape: &[usize],
        keys: [&'a str; K],
        bindings: Bindings<'a>,
    ) -> Result<[usize; K], String>;
}

impl<'a, const D: usize> ShapePattern<'a> for [PatternTerm<'a>; D] {
    fn unpack_shape<const K: usize>(
        &self,
        shape: &[usize],
        keys: [&'a str; K],
        bindings: Bindings<'a>,
    ) -> Result<[usize; K], String> {
        let mut env = Env::new(bindings);

        let mut ellipsis_pos: Option<usize> = None;
        for (idx, term) in self.iter().enumerate() {
            if let PatternTerm::Ellipsis = term {
                if ellipsis_pos.is_some() {
                    return Err(format!("Multiple ellipses in pattern: {:#?}", self));
                }
                ellipsis_pos = Some(idx);
            }
        }

        let ellipsis_len = if ellipsis_pos.is_some() {
            let shape_n = shape.len();
            let non_ellipsis_terms = self.len() - 1; // Exclude the ellipsis term itself

            if shape_n < non_ellipsis_terms {
                return Err(format!(
                    "Shape too short for pattern: pattern:{:#?}, shape:{:#?}",
                    self, shape
                ));
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

            println!();
            println!("pattern_idx: {:?}", pattern_idx);
            println!("shape_idx: {:?}", shape_idx);
            println!("term: {:?}", term);

            match term {
                PatternTerm::Any => {
                    // Any term, no action needed
                }
                PatternTerm::Ellipsis => {
                    return Err(format!(
                        "INTERNAL ERROR: Ellipsis term miss-handled pattern: {:#?}",
                        self
                    ));
                }
                PatternTerm::Expr(expr) => {
                    let target = shape[pattern_idx];
                    match expr.match_target(target as isize, &env) {
                        MatchResult::Success => {}
                        MatchResult::Failure => {
                            return Err(format!(
                                "Pattern term {:#?} does not match target {}",
                                expr, target
                            ));
                        }
                        MatchResult::Solve(param_name, value) => {
                            env.insert(param_name, value as usize);
                        }
                    }
                }
            }
        }

        if ellipsis_len > 0 {
            for pattern_idx in (ellipsis_pos.unwrap() + 1)..shape.len() {
                let shape_idx = pattern_idx + ellipsis_len;
                let term = &self[pattern_idx];

                println!();
                println!("pattern_idx: {:?}", pattern_idx);
                println!("shape_idx: {:?}", shape_idx);
                println!("term: {:?}", term);

                match term {
                    PatternTerm::Any => {
                        // Any term, no action needed
                    }
                    PatternTerm::Ellipsis => {
                        return Err(format!(
                            "INTERNAL ERROR: Ellipsis term miss-handled pattern: {:#?}",
                            self
                        ));
                    }
                    PatternTerm::Expr(expr) => {
                        let target = shape[shape_idx];
                        match expr.match_target(target as isize, &env) {
                            MatchResult::Success => {}
                            MatchResult::Failure => {
                                return Err(format!(
                                    "Pattern term {:#?} does not match target {}",
                                    expr, target
                                ));
                            }
                            MatchResult::Solve(param_name, value) => {
                                env.insert(param_name, value as usize);
                            }
                        }
                    }
                }
            }
        }

        Ok(env.export(&keys))
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

        assert_eq!(expr.match_target(8, &env), MatchResult::Solve("x", 13));
    }

    #[test]
    fn test_simple_product() {
        // 2 * x, target = 6, should solve x = 3
        let expr = SizeExpr::prod(vec![SizeExpr::fixed(2), SizeExpr::param("x")]);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(6, &env), MatchResult::Solve("x", 3));
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
        assert_eq!(expr.match_target(9, &env), MatchResult::Failure);
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
        assert_eq!(expr.match_target(25, &env), MatchResult::Failure);
    }

    #[test]
    fn test_nested_structure() {
        // 2 * (x + 3) - 1 with target = 9
        // Should solve: 2 * (x + 3) - 1 = 9
        //              2 * (x + 3) = 10
        //              x + 3 = 5
        //              x = 2
        let expr = SizeExpr::sum(vec![
            SizeExpr::prod(vec![
                SizeExpr::fixed(2),
                SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::fixed(3)]),
            ]),
            SizeExpr::fixed(-1),
        ]);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(9, &env), MatchResult::Solve("x", 2));
    }

    #[test]
    fn test_all_bound_success() {
        // x + 5 with x = 3, target = 8
        let expr = SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::fixed(5)]);

        let mut env = Env::new(&[]);
        env.insert("x", 3);

        assert_eq!(expr.match_target(8, &env), MatchResult::Success);
    }

    #[test]
    fn test_multiple_unbound() {
        // x + y, multiple unbound params
        let expr = SizeExpr::sum(vec![SizeExpr::param("x"), SizeExpr::param("y")]);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(5, &env), MatchResult::Failure);
    }

    #[test]
    fn test_no_integer_solution() {
        // 2 * x, target = 3 (no integer solution)
        let expr = SizeExpr::prod(vec![SizeExpr::fixed(2), SizeExpr::param("x")]);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(3, &env), MatchResult::Failure);
    }

    #[test]
    fn test_power_solve_base() {
        // x^3 = 8, should solve x = 2
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(8, &env), MatchResult::Solve("x", 2));
    }

    #[test]
    fn test_power_no_solution() {
        // 2^3 = x where x != 8 (no solution when fully bound and doesn't match)
        let expr = SizeExpr::pow(SizeExpr::fixed(2), 3);
        let env = Env::new(&[]);

        // 2^3 = 8, so target 5 should fail
        assert_eq!(expr.match_target(5, &env), MatchResult::Failure);
    }

    #[test]
    fn test_power_in_sum() {
        // x^2 + 1 = 10, should solve x = 3 (since 3^2 + 1 = 10)
        let expr = SizeExpr::sum(vec![
            SizeExpr::pow(SizeExpr::param("x"), 2),
            SizeExpr::fixed(1),
        ]);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(10, &env), MatchResult::Solve("x", 3));
    }

    #[test]
    fn test_power_cube_root() {
        // x^3 = 27, should solve x = 3
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(27, &env), MatchResult::Solve("x", 3));
    }

    #[test]
    fn test_power_negative_base_odd_exp() {
        // x^3 = -8, should solve x = -2 (since (-2)^3 = -8)
        let expr = SizeExpr::pow(SizeExpr::param("x"), 3);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(-8, &env), MatchResult::Solve("x", -2));
    }

    #[test]
    fn test_power_negative_base_even_exp() {
        // x^2 = -4 (no real solution for even exponent and negative target)
        let expr = SizeExpr::pow(SizeExpr::param("x"), 2);
        let env = Env::new(&[]);

        assert_eq!(expr.match_target(-4, &env), MatchResult::Failure);
    }

    #[test]
    fn test_unpack_shape() {
        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;

        let shape = [b, h * p, w * p, c];
        println!("shape: {:?}", shape);

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
        println!("pattern: {:#?}", pattern);

        let bindings = [("p", p), ("c", c)];

        println!();

        let [u_b, u_h, u_w] = pattern
            .unpack_shape(&shape, ["b", "h", "w"], &bindings)
            .unwrap();

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
    }
}
