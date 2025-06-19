use std::collections::HashMap;

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

    /// Match this expression against a target value with given bindings
    pub fn match_target(
        &self,
        target: isize,
        env: &Env<'a>,
    ) -> MatchResult {
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
                    Some(value) => MatchResult::Solve(param_name, value),
                    None => MatchResult::Failure,
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

    /// Solve for a parameter by unpeeling the expression structure
    fn solve_for_target(
        &self,
        target_param: &str,
        target: isize,
        env: &Env<'a>,
    ) -> Option<isize> {
        // TODO: Crutcher; migrate to Result<isize, String> for better error handling.
        match self {
            SizeExpr::Param(name) => {
                if *name == target_param {
                    Some(target)
                } else {
                    // This param should be bound, but we're solving for a different param
                    None
                }
            }
            SizeExpr::Fixed(value) => {
                // Fixed value should equal target
                if *value == target {
                    Some(target)
                } else {
                    None
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
                    return None;
                }

                // base^exp = target, solve for base
                // base = target^(1/exp)
                let root = Self::exact_nth_root(target, exp)?;
                base.solve_for_target(target_param, root, env)
            }
            SizeExpr::Sum(exprs) => {
                // sum(exprs) = target
                // Find the one unbound expr, evaluate all others, subtract from target
                let mut unbound_expr = None;
                let mut bound_sum: isize = 0;

                for expr in exprs {
                    match expr.binding_state(env) {
                        BindingState::FullyBound => {
                            bound_sum += expr.evaluate(env)?;
                        }
                        BindingState::SingleUnbound(param) if param == target_param => {
                            if unbound_expr.is_some() {
                                return None; // Multiple unbound expressions
                            }
                            unbound_expr = Some(expr);
                        }
                        _ => return None, // Wrong param or multiple unbound
                    }
                }

                if let Some(expr) = unbound_expr {
                    // unbound_expr + bound_sum = target  =>  unbound_expr = target - bound_sum
                    expr.solve_for_target(target_param, target - bound_sum, env)
                } else {
                    None
                }
            }
            SizeExpr::Prod(exprs) => {
                // prod(exprs) = target
                // Find the one unbound expr, evaluate all others, divide target by their product
                let mut unbound_expr = None;
                let mut bound_product = 1;

                for expr in exprs {
                    match expr.binding_state(env) {
                        BindingState::FullyBound => {
                            let value = expr.evaluate(env)?;
                            if value == 0 {
                                // Product is zero, but we need non-zero target
                                if target != 0 {
                                    return None;
                                }
                                // If target is also 0, the unbound param can be anything
                                // We'll return 0 as "closest to zero"
                                return Some(0);
                            }
                            bound_product *= value;
                        }
                        BindingState::SingleUnbound(param) if param == target_param => {
                            if unbound_expr.is_some() {
                                return None; // Multiple unbound expressions
                            }
                            unbound_expr = Some(expr);
                        }
                        _ => return None, // Wrong param or multiple unbound
                    }
                }

                if let Some(expr) = unbound_expr {
                    // unbound_expr * bound_product = target  =>  unbound_expr = target / bound_product
                    if bound_product == 0 {
                        // This shouldn't happen given our check above
                        return None;
                    }

                    if target % bound_product != 0 {
                        // No integer solution
                        return None;
                    }

                    expr.solve_for_target(target_param, target / bound_product, env)
                } else {
                    None
                }
            }
        }
    }

    /// Evaluate expression with given bindings (assumes all params are bound)
    fn evaluate(
        &self,
        env: &Env<'a>,
    ) -> Option<isize> {
        match self {
            SizeExpr::Param(name) => env.lookup(name).map(|v| v as isize),
            SizeExpr::Fixed(value) => Some(*value),
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
