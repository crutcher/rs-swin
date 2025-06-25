use crate::bindings::{MutableStackEnvironment, MutableStackMap, StackEnvironment, StackMap};
use crate::expressions::{DimSizeExpr, TryMatchResult};
use std::fmt::{Display, Formatter};

/// A term in a shape pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapePatternTerm<'a> {
    /// Matches any dimension size.
    Any,

    /// Matches a variable number of dimensions (ellipsis).
    Ellipsis,

    /// A dimension size expression that must match a specific value.
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

    /// Match a shape pattern.
    ///
    /// Wraps `unpack_shape`, without extracting keys.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    #[inline(always)]
    pub fn assert_shape(
        &'a self,
        shape: &[usize],
        env: StackEnvironment<'a>,
    ) {
        let _ignored = self.unpack_shape(shape, &[], env);
    }

    /// Match and unpack a shape pattern.
    ///
    /// Wraps `maybe_unpack_shape` and panics if the shape does not match.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// The list of key values.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    #[must_use]
    pub fn unpack_shape<const K: usize>(
        &'a self,
        shape: &[usize],
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> [usize; K] {
        self.maybe_unpack_shape(shape, keys, env).unwrap()
    }

    /// Match and unpack a shape pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// Either the list of key values; or an error.
    #[must_use]
    pub fn maybe_unpack_shape<const K: usize>(
        &'a self,
        shape: &[usize],
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> Result<[usize; K], String> {
        let fail = |msg: String| -> String {
            format!(
                "Shape Error:: {}\n shape:\n  {:?}\n expected:\n  {self}\n  {:?}",
                msg, shape, env
            )
        };
        let fail_at = |shape_idx: usize, term_idx: usize, msg: String| -> String {
            fail(format!(
                "{} !~ {} :: {}",
                shape[shape_idx], self.terms[term_idx], msg
            ))
        };

        let rank = shape.len();

        let mut mut_env: MutableStackEnvironment<'a> = MutableStackEnvironment::new(env);

        let (e_start, e_size) = match self.check_ellipsis_split(rank) {
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
                    unreachable!("Ellipsis should have been handled before");
                }
                ShapePatternTerm::Expr(expr) => expr,
            };

            match expr.try_match(dim_size as isize, &env) {
                Err(msg) => return Err(fail_at(shape_idx, term_idx, msg)),
                Ok(TryMatchResult::Match) => continue,
                Ok(TryMatchResult::Conflict) => {
                    return Err(fail_at(shape_idx, term_idx, "Value MissMatch".to_string()));
                }
                Ok(TryMatchResult::ParamConstraint(param_name, value)) => {
                    match mut_env.lookup(param_name) {
                        None => mut_env.bind(param_name, value as usize),
                        Some(v) => {
                            return Err(fail_at(
                                shape_idx,
                                term_idx,
                                format!("Constraint miss-match: {} {} != {}", param_name, v, value),
                            ));
                        }
                    }
                }
            }
        }

        Ok(mut_env.export_key_values(keys))
    }

    /// Check if the pattern has an ellipsis.
    ///
    /// ## Arguments
    ///
    /// - `rank`: the number of dims of the shape to match.
    ///
    /// ## Returns
    ///
    /// - `Ok((usize, usize))`: the position of the ellipsis and the number of dimensions it matches.
    /// - `Err(String)`: an error message if the pattern does not match the expected size.
    #[inline(always)]
    #[must_use]
    fn check_ellipsis_split(
        &self,
        rank: usize,
    ) -> Result<(usize, usize), String> {
        let k = self.terms.len();
        match self.ellipsis_pos {
            None => {
                if rank != k {
                    Err(format!("Shape rank {} != pattern dim count {}", rank, k,))
                } else {
                    Ok((k, 0))
                }
            }
            Some(pos) => {
                let non_ellipsis_terms = k - 1;
                if rank < non_ellipsis_terms {
                    return Err(format!(
                        "Shape rank {} < non-ellipsis pattern term count {}",
                        rank, non_ellipsis_terms,
                    ));
                }
                Ok((pos, rank - non_ellipsis_terms))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape_patterns::{ShapePattern, ShapePatternTerm};

    #[should_panic(expected = "Multiple ellipses in pattern")]
    #[test]
    fn test_bad_new() {
        // Multiple ellipses in pattern should panic.
        let _ = ShapePattern::new(&[
            ShapePatternTerm::Any,
            ShapePatternTerm::Ellipsis,
            ShapePatternTerm::Ellipsis,
        ]);
    }
    #[test]
    fn test_check_ellipsis_split() {
        {
            // With ellipsis.
            let pattern = ShapePattern::new(&[
                ShapePatternTerm::Any,
                ShapePatternTerm::Ellipsis,
                ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
            ]);

            assert_eq!(pattern.check_ellipsis_split(2), Ok((1, 0)));
            assert_eq!(pattern.check_ellipsis_split(3), Ok((1, 1)));
            assert_eq!(pattern.check_ellipsis_split(4), Ok((1, 2)));

            assert_eq!(
                pattern.check_ellipsis_split(1),
                Err("Shape rank 1 < non-ellipsis pattern term count 2".to_string())
            );
        }
        {
            // Without ellipsis.
            let pattern = ShapePattern::new(&[
                ShapePatternTerm::Any,
                ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
            ]);

            assert_eq!(pattern.check_ellipsis_split(2), Ok((2, 0)));

            assert_eq!(
                pattern.check_ellipsis_split(1),
                Err("Shape rank 1 != pattern dim count 2".to_string())
            );
        }
    }

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
    fn test_panic_msg() {
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

        let shape = [12, b, 1, 2, 3, h * p, w * p, 1 + z * z * z, c];

        let result =
            PATTERN.maybe_unpack_shape(&shape, &["b", "h", "w", "z"], &[("p", p), ("c", c)]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert_eq!(
            err_msg,
            "\
Shape Error:: 65 !~ (z)^3 :: No integer solution.
 shape:
  [12, 2, 1, 2, 3, 12, 8, 65, 5]
 expected:
  [_, b, ..., (h*p), (w*p), (z)^3, c]
  [(\"p\", 4), (\"c\", 5)]"
        );
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
        let env = [("p", p), ("c", c)];

        PATTERN.assert_shape(&shape, &env);

        let [u_b, u_h, u_w, u_z] = PATTERN.unpack_shape(&shape, &["b", "h", "w", "z"], &env);

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }

    #[should_panic(expected = "Shape rank 3 != pattern dim count 1")]
    #[test]
    fn test_shape_mismatch_no_ellipsis() {
        // This should panic because the shape does not match the pattern.
        let pattern = ShapePattern::new(&[ShapePatternTerm::Expr(DimSizeExpr::Param("a"))]);
        let shape = [1, 2, 3];
        pattern.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Shape rank 3 < non-ellipsis pattern term count 4")]
    #[test]
    fn test_shape_mismatch_with_ellipsis() {
        // This should panic because the shape does not match the pattern.
        let pattern = ShapePattern::new(&[
            ShapePatternTerm::Any,
            ShapePatternTerm::Any,
            ShapePatternTerm::Ellipsis,
            ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
            ShapePatternTerm::Expr(DimSizeExpr::Param("c")),
        ]);
        let shape = [1, 2, 3];
        pattern.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Value MissMatch")]
    #[test]
    fn test_shape_mismatch_value() {
        // This should panic because the value does not match the constraint.
        let pattern = ShapePattern::new(&[
            ShapePatternTerm::Expr(DimSizeExpr::Param("a")),
            ShapePatternTerm::Expr(DimSizeExpr::Param("b")),
        ]);
        let shape = [2, 3];
        pattern.assert_shape(&shape, &[("a", 2), ("b", 4)]);
    }

    #[should_panic(expected = "Constraint miss-match: a 2 != 3")]
    #[test]
    fn test_shape_mismatch_constraint() {
        // This should panic because the constraint does not match.
        let pattern = ShapePattern::new(&[
            ShapePatternTerm::Expr(DimSizeExpr::Param("a")),
            ShapePatternTerm::Expr(DimSizeExpr::Param("a")),
        ]);
        let shape = [2, 3];
        pattern.assert_shape(&shape, &[]);
    }
}
