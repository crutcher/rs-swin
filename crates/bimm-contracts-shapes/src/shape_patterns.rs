use crate::bindings::{MutableStackEnvironment, MutableStackMap, StackEnvironment, StackMap};
use crate::expressions::{DimSizeExpr, TryMatchResult};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternTerm<'a> {
    Any,
    Ellipsis,
    Expr(DimSizeExpr<'a>),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapePattern<'a> {
    pub terms: &'a [PatternTerm<'a>],
    pub ellipsis_pos: Option<usize>,
}

impl Display for ShapePattern<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "[")?;
        for (idx, expr) in self.terms.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", expr)?;
        }
        write!(f, "]")
    }
}

impl<'a> ShapePattern<'a> {
    pub const fn new(terms: &'a [PatternTerm<'a>]) -> Self {
        let mut i = 0;
        let mut ellipsis_pos: Option<usize> = None;

        while i < terms.len() {
            if matches!(terms[i], PatternTerm::Ellipsis) {
                match ellipsis_pos {
                    Some(_) => panic!("Multiple ellipses in pattern"),
                    None => ellipsis_pos = Some(i),
                }
            }
            i += 1;
        }

        ShapePattern {
            terms,
            ellipsis_pos,
        }
    }

    #[inline(always)]
    #[must_use]
    fn check_ellipsis_split(
        &self,
        size: usize,
    ) -> Result<(usize, usize), String> {
        let k = self.terms.len();
        match self.ellipsis_pos {
            None => {
                if size != k {
                    Err(format!(
                        "Pattern size {} does not match the number of terms {}",
                        size, k
                    ))
                } else {
                    Ok((k, 0))
                }
            }
            Some(pos) => {
                let non_ellipsis_terms = k - 1;
                if size < non_ellipsis_terms {
                    return Err(format!(
                        "Pattern size {} is less than the number of terms {} (without ellipsis)",
                        size, non_ellipsis_terms
                    ));
                }
                Ok((pos, size - non_ellipsis_terms))
            }
        }
    }

    /// Match a shape to the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `bindings`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// Either success, or an error.
    #[must_use]
    pub fn match_shape(
        &'a self,
        shape: &[usize],
        bindings: StackEnvironment<'a>,
    ) -> Result<(), String> {
        self.extract_keys(shape, &[], bindings).map(|_| ())
    }

    /// Match a shape to the pattern, and extract keys.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `bindings`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// Either the list of key values; or an error.
    #[must_use]
    pub fn extract_keys<const K: usize>(
        &'a self,
        shape: &[usize],
        keys: &[&'a str; K],
        bindings: StackEnvironment<'a>,
    ) -> Result<[usize; K], String> {
        let fail = |msg: String| {
            format!(
                "Shape Match Error: {msg}\nShape: {:?}\nPattern: {self}\nBindings: {:?}",
                shape, bindings
            )
        };

        let mut env: MutableStackEnvironment<'a> = MutableStackEnvironment::new(bindings);

        let (e_start, e_size) = match self.check_ellipsis_split(shape.len()) {
            Ok((e_start, e_size)) => (e_start, e_size),
            Err(msg) => return Err(fail(msg)),
        };

        for (shape_idx, &dim_size) in shape.iter().enumerate() {
            let term_idx = if shape_idx < e_start {
                shape_idx
            } else if shape_idx < (e_start + e_size) {
                continue;
            } else {
                shape_idx + 1 - e_size
            };

            let expr = match &self.terms[term_idx] {
                PatternTerm::Any => continue,
                PatternTerm::Ellipsis => {
                    return Err(fail("INTERNAL ERROR: out-of-place Ellipsis".to_string()));
                }
                PatternTerm::Expr(expr) => expr,
            };

            match expr.try_match(dim_size as isize, &bindings) {
                Err(msg) => return Err(fail(msg)),
                Ok(TryMatchResult::Match) => continue,
                Ok(TryMatchResult::Conflict) => {
                    return Err(fail("Value MissMatch".to_string()));
                }
                Ok(TryMatchResult::ParamConstraint(param_name, value)) => {
                    match env.lookup(param_name) {
                        None => env.bind(param_name, value as usize),
                        Some(v) => {
                            return Err(fail(format!(
                                "Constraint miss-match: {} {} != {}",
                                param_name, value, v
                            )));
                        }
                    }
                }
            }
        }

        Ok(env.export_key_values(keys))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::shape_patterns::{PatternTerm, ShapePattern};

    #[test]
    fn test_format_pattern() {
        let pattern = ShapePattern::new(&[
            PatternTerm::Any,
            PatternTerm::Ellipsis,
            PatternTerm::Expr(DimSizeExpr::Param("b")),
            PatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("h"),
                DimSizeExpr::Sum(&[
                    DimSizeExpr::Param("a"),
                    DimSizeExpr::Negate(&DimSizeExpr::Param("b")),
                ]),
            ])),
            PatternTerm::Expr(DimSizeExpr::Pow(&DimSizeExpr::Param("h"), 2)),
        ]);

        assert_eq!(pattern.to_string(), "[_, ..., b, (h*(a+(-b))), (h)^2]");
    }

    #[test]
    fn test_unpack_shape() {
        static PATTERN: ShapePattern = ShapePattern::new(&[
            PatternTerm::Any,
            PatternTerm::Expr(DimSizeExpr::Param("b")),
            PatternTerm::Ellipsis,
            PatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("h"),
                DimSizeExpr::Param("p"),
            ])),
            PatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("w"),
                DimSizeExpr::Param("p"),
            ])),
            PatternTerm::Expr(DimSizeExpr::Pow(&DimSizeExpr::Param("z"), 3)),
            PatternTerm::Expr(DimSizeExpr::Param("c")),
        ]);

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, z * z * z, c];

        let [u_b, u_h, u_w, u_z] = PATTERN
            .extract_keys(&shape, &["b", "h", "w", "z"], &[("p", p), ("c", c)])
            .unwrap();

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }
}
