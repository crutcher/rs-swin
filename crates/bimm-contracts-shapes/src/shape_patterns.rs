use crate::bindings::{MutableStackEnvironment, MutableStackMap, StackEnvironment, StackMap};
use crate::expressions::{DimSizeExpr, TryMatchResult};
use std::fmt::{Display, Formatter};

/// A term in a shape pattern, which can be:
/// - `Any`: matches any dimension size.
/// - `Ellipsis`: matches a variable number of dimensions.
/// - `Expr`: a dimension size expression that must match a specific value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapePatternTerm<'a> {
    Any,
    Ellipsis,
    Expr(DimSizeExpr<'a>),
}

impl Display for ShapePatternTerm<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ShapePatternTerm::Any => write!(f, "_"),
            ShapePatternTerm::Ellipsis => write!(f, "..."),
            ShapePatternTerm::Expr(expr) => write!(f, "{}", expr),
        }
    }
}

/// A shape pattern, which is a sequence of terms that can match a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapePattern<'a> {
    /// The terms in the pattern.
    pub terms: &'a [ShapePatternTerm<'a>],

    /// The position of the ellipsis in the pattern, if any.
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
    /// Create a new shape pattern from a slice of terms.
    ///
    /// ## Arguments
    ///
    /// - `terms`: a slice of `ShapePatternTerm` that defines the pattern.
    ///
    /// ## Returns
    ///
    /// A new `ShapePattern` instance.
    pub const fn new(terms: &'a [ShapePatternTerm<'a>]) -> Self {
        let mut i = 0;
        let mut ellipsis_pos: Option<usize> = None;

        while i < terms.len() {
            if matches!(terms[i], ShapePatternTerm::Ellipsis) {
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

    /// Check if the pattern has an ellipsis.
    ///
    /// ## Arguments
    ///
    /// - `size`: the size of the shape to match.
    ///
    /// ## Returns
    ///
    /// - `Ok((usize, usize))`: the position of the ellipsis and the number of dimensions it matches.
    /// - `Err(String)`: an error message if the pattern does not match the expected size.
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
        self.extract_dims(shape, &[], bindings).map(|_| ())
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
    pub fn extract_dims<const K: usize>(
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
                ShapePatternTerm::Any => continue,
                ShapePatternTerm::Ellipsis => {
                    return Err(fail("INTERNAL ERROR: out-of-place Ellipsis".to_string()));
                }
                ShapePatternTerm::Expr(expr) => expr,
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
    use crate::shape_patterns::{ShapePattern, ShapePatternTerm};

    #[test]
    fn test_format_pattern() {
        let pattern = ShapePattern::new(&[
            ShapePatternTerm::Any,
            ShapePatternTerm::Ellipsis,
            ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
            ShapePatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("h"),
                DimSizeExpr::Sum(&[
                    DimSizeExpr::Param("a"),
                    DimSizeExpr::Negate(&DimSizeExpr::Param("b")),
                ]),
            ])),
            ShapePatternTerm::Expr(DimSizeExpr::Pow(&DimSizeExpr::Param("h"), 2)),
        ]);

        assert_eq!(pattern.to_string(), "[_, ..., b, (h*(a+(-b))), (h)^2]");
    }

    #[test]
    fn test_unpack_shape() {
        static PATTERN: ShapePattern = ShapePattern::new(&[
            ShapePatternTerm::Any,
            ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
            ShapePatternTerm::Ellipsis,
            ShapePatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("h"),
                DimSizeExpr::Param("p"),
            ])),
            ShapePatternTerm::Expr(DimSizeExpr::Prod(&[
                DimSizeExpr::Param("w"),
                DimSizeExpr::Param("p"),
            ])),
            ShapePatternTerm::Expr(DimSizeExpr::Pow(&DimSizeExpr::Param("z"), 3)),
            ShapePatternTerm::Expr(DimSizeExpr::Param("c")),
        ]);

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, z * z * z, c];

        let [u_b, u_h, u_w, u_z] = PATTERN
            .extract_dims(&shape, &["b", "h", "w", "z"], &[("p", p), ("c", c)])
            .unwrap();

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }
}
