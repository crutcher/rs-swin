//! # Shape Contracts.
//!
//! `bimm-contracts` is built around the [`ShapeContract`] interface.
//! - A [`ShapeContract`] is a sequence of [`DimMatcher`]s.
//! - A [`DimMatcher`] matches one or more dimensions of a shape:
//!   - [`DimMatcher::Any`] matches any dimension size.
//!   - [`DimMatcher::Ellipsis`] matches a variable number of dimensions (ellipsis).
//!   - [`DimMatcher::Expr`] matches a dimension expression that must match a specific value.
//!
//! A [`ShapeContract`] should usually be constructed using the [`crate::shape_contract`] macro.
//!
//! ## Example
//!
//! ```rust
//! use bimm_contracts::{shape_contract, ShapeContract};
//!
//! static CONTRACT : ShapeContract = shape_contract![
//!    ...,
//!    "height" = "h_wins" * "window",
//!    "width" = "w_wins" * "window",
//!    "channels",
//! ];
//!
//! let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
//!
//! // Assert the shape, given the bindings.
//! let [h_wins, w_wins] = CONTRACT.unpack_shape(
//!     &shape,
//!     &["h_wins", "w_wins"],
//!     &[("window", 8)]
//! );
//! assert_eq!(h_wins, 2);
//! assert_eq!(w_wins, 3);
//! ```

use crate::bindings::{MutableStackEnvironment, MutableStackMap, StackEnvironment, StackMap};
use crate::expressions::{DimExpr, TryMatchResult};
use crate::shape_argument::ShapeArgument;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};

/// A term in a shape pattern.
///
/// Users should generally use [`crate::shape_contract`] to construct patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMatcher<'a> {
    /// Matches any dimension size.
    Any {
        /// An optional label for the matcher.
        label: Option<&'a str>,
    },

    /// Matches a variable number of dimensions (ellipsis).
    Ellipsis {
        /// An optional label for the matcher.
        label: Option<&'a str>,
    },

    /// A dimension size expression that must match a specific value.
    Expr {
        /// An optional label for the matcher.
        label: Option<&'a str>,

        /// The dimension expression that must match a specific value.
        expr: DimExpr<'a>,
    },
}

impl<'a> DimMatcher<'a> {
    /// Create a new `DimMatcher` that matches any dimension size.
    pub const fn any() -> Self {
        DimMatcher::Any { label: None }
    }

    /// Create a new `DimMatcher` that matches a variable number of dimensions (ellipsis).
    pub const fn ellipsis() -> Self {
        DimMatcher::Ellipsis { label: None }
    }

    /// Create a new `DimMatcher` from a dimension expression.
    ///
    /// ## Arguments
    ///
    /// - `expr`: a dimension expression that must match a specific value.
    ///
    /// ## Returns
    ///
    /// A new `DimMatcher` that matches the given expression.
    pub const fn expr(expr: DimExpr<'a>) -> Self {
        DimMatcher::Expr { label: None, expr }
    }

    /// Get the label of the matcher, if any.
    pub const fn label(&self) -> Option<&'a str> {
        match self {
            DimMatcher::Any { label } => *label,
            DimMatcher::Ellipsis { label } => *label,
            DimMatcher::Expr { label, .. } => *label,
        }
    }

    /// Attach a label to the matcher.
    ///
    /// ## Arguments
    ///
    /// - `label`: an optional label to attach to the matcher.
    ///
    /// ## Returns
    ///
    /// A new `DimMatcher` with the label attached.
    pub const fn with_label(
        self,
        label: Option<&'a str>,
    ) -> Self {
        match self {
            DimMatcher::Any { .. } => DimMatcher::Any { label },
            DimMatcher::Ellipsis { .. } => DimMatcher::Ellipsis { label },
            DimMatcher::Expr { expr, .. } => DimMatcher::Expr { label, expr },
        }
    }
}

impl Display for DimMatcher<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> core::fmt::Result {
        if let Some(label) = self.label() {
            write!(f, "{label}=")?;
        }
        match self {
            DimMatcher::Any { label: _ } => write!(f, "_"),
            DimMatcher::Ellipsis { label: _ } => write!(f, "..."),
            DimMatcher::Expr { label: _, expr } => write!(f, "{expr}"),
        }
    }
}

/// A shape pattern, which is a sequence of terms that can match a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeContract<'a> {
    /// The terms in the pattern.
    pub terms: &'a [DimMatcher<'a>],

    /// The slot map for the pattern, if any.
    slot_map: Option<&'a [&'a str]>,

    /// The position of the ellipsis in the pattern, if any.
    pub ellipsis_pos: Option<usize>,
}

impl Display for ShapeContract<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (idx, expr) in self.terms.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{expr}")?;
        }
        write!(f, "]")
    }
}

impl<'a> ShapeContract<'a> {
    /// Create a new shape pattern from a slice of terms.
    ///
    /// ## Arguments
    ///
    /// - `terms`: a slice of `ShapePatternTerm` that defines the pattern.
    ///
    /// ## Returns
    ///
    /// A new `ShapePattern` instance.
    ///
    /// ## Macro Support
    ///
    /// Consider using the [`crate::shape_contract`] macro instead.
    ///
    /// ```
    /// use bimm_contracts::{shape_contract, ShapeContract};
    ///
    /// static CONTRACT : ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    /// ```
    pub const fn new(terms: &'a [DimMatcher<'a>]) -> Self {
        let mut i = 0;
        let mut ellipsis_pos: Option<usize> = None;

        while i < terms.len() {
            if matches!(terms[i], DimMatcher::Ellipsis { label: _ }) {
                match ellipsis_pos {
                    Some(_) => panic!("Multiple ellipses in pattern"),
                    None => ellipsis_pos = Some(i),
                }
            }
            i += 1;
        }

        ShapeContract {
            terms,
            slot_map: None,
            ellipsis_pos,
        }
    }

    /// Add a slot map to the contract.
    pub const fn with_slots(
        self,
        slot_map: &'a [&'a str],
    ) -> Self {
        // TODO: Verify that the slot map is valid.
        // 1. Implement visit for the matchers/expressions.
        // 2. Verify that the slots match the pattern.

        ShapeContract {
            terms: self.terms,
            slot_map: Some(slot_map),
            ellipsis_pos: self.ellipsis_pos,
        }
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bimm_contracts::{shape_contract, run_periodically, ShapeContract};
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Run under backoff amortization.
    /// run_periodically! {{
    ///     // Statically allocated contract.
    ///     static CONTRACT : ShapeContract = shape_contract![
    ///        ...,
    ///        "height" = "h_wins" * "window",
    ///        "width" = "w_wins" * "window",
    ///        "channels",
    ///     ];
    ///
    ///     // Assert the shape, given the bindings.
    ///     CONTRACT.assert_shape(
    ///         &shape,
    ///         &[("h_wins", 2), ("w_wins", 3), ("channels", 4)]
    ///     );
    /// }}
    /// ```
    #[inline(always)]
    pub fn assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) where
        S: ShapeArgument,
    {
        let result = self.try_assert_shape(shape, env);
        if result.is_err() {
            panic!("{}", result.unwrap_err());
        }
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern.
    /// - `Err(String)`: if the shape does not match the pattern, with an error message.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bimm_contracts::{shape_contract, run_periodically, ShapeContract};
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT : ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Assert the shape, given the bindings; or throw.
    /// CONTRACT.try_assert_shape(
    ///     &shape,
    ///     &[("h_wins", 2), ("w_wins", 3), ("channels", 4)]
    /// ).unwrap();
    /// ```
    #[inline(always)]
    pub fn try_assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let mut mut_env = MutableStackEnvironment::new(env);

        self.try_resolve_match(shape, &mut mut_env)
    }

    /// Match and unpack `K` keys from a shape pattern.
    ///
    /// Wraps `try_unpack_shape` and panics if the shape does not match.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// An `[usize; K]` of the unpacked `keys` values.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bimm_contracts::{shape_contract, run_periodically, ShapeContract};
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT : ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Unpack the shape, given the bindings.
    /// let [h, w, c] = CONTRACT.unpack_shape(
    ///     &shape,
    ///     &["h_wins", "w_wins", "channels"],
    ///     &[("window", 8)]
    /// );
    /// assert_eq!(h, 2);
    /// assert_eq!(w, 3);
    /// assert_eq!(c, 4);
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> [usize; K]
    where
        S: ShapeArgument,
    {
        match self.try_unpack_shape(shape, keys, env) {
            Ok(values) => values,
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Try and match and unpack `K` keys from a shape pattern.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// A `Result<[usize; K], String>` of the unpacked `keys` values.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bimm_contracts::{shape_contract, run_periodically, ShapeContract};
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT : ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Unpack the shape, given the bindings; or throw.
    /// let [h, w, c] = CONTRACT.try_unpack_shape(
    ///     &shape,
    ///     &["h_wins", "w_wins", "channels"],
    ///     &[("window", 8)]
    /// ).unwrap();
    /// assert_eq!(h, 2);
    /// assert_eq!(w, 3);
    /// assert_eq!(c, 4);
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn try_unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> Result<[usize; K], String>
    where
        S: ShapeArgument,
    {
        let mut mut_env = MutableStackEnvironment::new(env);

        self.try_resolve_match(shape, &mut mut_env)?;

        Ok(mut_env.export_key_values(keys))
    }

    /// Resolve the match for the shape against the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the mutable environment to bind parameters.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern.
    /// - `Err(String)`: if the shape does not match the pattern, with an error message.
    #[must_use]
    #[inline(always)]
    fn try_resolve_match<S>(
        &'a self,
        shape: S,
        env: &mut MutableStackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let shape = &shape.get_shape_vec();

        let fail = |msg| -> String {
            format!(
                "Shape Error:: {msg}\n shape:\n  {shape:?}\n expected:\n  {self}\n  {{{}}}",
                env.backing
                    .iter()
                    .map(|(k, v)| format!("\"{k}\": {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let fail_at = |shape_idx, term_idx, msg| -> String {
            fail(format!(
                "{} !~ {} :: {msg}",
                shape[shape_idx], self.terms[term_idx]
            ))
        };

        let rank = shape.len();

        let (e_start, e_size) = match self.try_ellipsis_split(rank) {
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

            let matcher = &self.terms[term_idx];
            if let Some(label) = matcher.label() {
                match env.lookup(label) {
                    Some(value) => {
                        if value != dim_size {
                            return Err(fail_at(
                                shape_idx,
                                term_idx,
                                "Value MissMatch.".to_string(),
                            ));
                        }
                    }
                    None => {
                        env.bind(label, dim_size);
                    }
                }
            }

            let expr = match matcher {
                DimMatcher::Any { label: _ } => continue,
                DimMatcher::Ellipsis { label: _ } => {
                    unreachable!("Ellipsis should have been handled before")
                }
                DimMatcher::Expr { label: _, expr } => expr,
            };

            match expr.try_match(dim_size as isize, env) {
                Ok(TryMatchResult::Match) => continue,
                Ok(TryMatchResult::Conflict) => {
                    return Err(fail_at(shape_idx, term_idx, "Value MissMatch.".to_string()));
                }
                Ok(TryMatchResult::ParamConstraint(param_name, value)) => {
                    env.bind(param_name, value as usize);
                }
                Err(msg) => return Err(fail_at(shape_idx, term_idx, msg)),
            }
        }

        Ok(())
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
    fn try_ellipsis_split(
        &self,
        rank: usize,
    ) -> Result<(usize, usize), String> {
        let k = self.terms.len();
        match self.ellipsis_pos {
            None => {
                if rank != k {
                    Err(format!("Shape rank {rank} != pattern dim count {k}",))
                } else {
                    Ok((k, 0))
                }
            }
            Some(pos) => {
                let non_ellipsis_terms = k - 1;
                if rank < non_ellipsis_terms {
                    return Err(format!(
                        "Shape rank {rank} < non-ellipsis pattern term count {non_ellipsis_terms}",
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
    use crate::contracts::{DimMatcher, ShapeContract};
    use alloc::string::ToString;
    use bimm_contracts_macros::shape_contract;
    use indoc::indoc;

    #[test]
    fn test_dim_matcher_builders() {
        assert_eq!(DimMatcher::any(), DimMatcher::Any { label: None });

        assert_eq!(DimMatcher::ellipsis(), DimMatcher::Ellipsis { label: None });

        assert_eq!(
            DimMatcher::expr(DimExpr::Param("a")),
            DimMatcher::Expr {
                label: None,
                expr: DimExpr::Param("a")
            }
        );
    }

    #[test]
    fn test_with_label() {
        assert_eq!(
            DimMatcher::any().with_label(Some("abc")),
            DimMatcher::Any { label: Some("abc") }
        );

        assert_eq!(
            DimMatcher::ellipsis().with_label(Some("abc")),
            DimMatcher::Ellipsis { label: Some("abc") }
        );

        assert_eq!(
            DimMatcher::expr(DimExpr::Param("a")).with_label(Some("abc")),
            DimMatcher::Expr {
                label: Some("abc"),
                expr: DimExpr::Param("a")
            }
        );
    }

    #[should_panic(expected = "Multiple ellipses in pattern")]
    #[test]
    fn test_bad_new() {
        // Multiple ellipses in pattern should panic.
        let _ = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::ellipsis(),
        ]);
    }

    #[test]
    fn test_shape_contract_macro() {
        /// For the shape_contract macro namespace.
        use crate as bimm_contracts;
        static CONTRACT: ShapeContract = shape_contract![
            ...,
            "height" = "hwins" * "window",
            "width" = "wwins" * "window",
            "color",
        ];

        let hwins = 2;
        let wwins = 3;
        let window = 4;
        let color = 3;

        let shape = [1, 2, 3, hwins * window, wwins * window, color];

        let [u_hwins, u_wwins, u_height] = CONTRACT.unpack_shape(
            &shape,
            &["hwins", "wwins", "height"],
            &[
                ("height", hwins * window),
                ("window", window),
                ("color", color),
            ],
        );
        assert_eq!(u_hwins, hwins);
        assert_eq!(u_wwins, wwins);
        assert_eq!(u_height, hwins * window);

        assert_eq!(
            CONTRACT
                .try_unpack_shape(
                    &shape,
                    &["hwins", "wwins"],
                    &[("window", window + 1), ("color", color),]
                )
                .unwrap_err(),
            indoc! {r#"
                Shape Error:: 8 !~ height=(hwins*window) :: No integer solution.
                 shape:
                  [1, 2, 3, 8, 12, 3]
                 expected:
                  [..., height=(hwins*window), width=(wwins*window), color]
                  {"window": 5, "color": 3}"#
            },
        );

        assert_eq!(
            CONTRACT
                .try_unpack_shape(
                    &shape,
                    &["hwins", "wwins"],
                    &[("height", 1), ("window", window), ("color", color),]
                )
                .unwrap_err(),
            indoc! {r#"
                Shape Error:: 8 !~ height=(hwins*window) :: Value MissMatch.
                 shape:
                  [1, 2, 3, 8, 12, 3]
                 expected:
                  [..., height=(hwins*window), width=(wwins*window), color]
                  {"height": 1, "window": 4, "color": 3}"#
            },
        );
    }

    #[test]
    fn test_check_ellipsis_split() {
        {
            // With ellipsis.
            static PATTERN: ShapeContract = ShapeContract::new(&[
                DimMatcher::any(),
                DimMatcher::ellipsis(),
                DimMatcher::expr(DimExpr::Param("b")),
            ]);

            assert_eq!(PATTERN.try_ellipsis_split(2), Ok((1, 0)));
            assert_eq!(PATTERN.try_ellipsis_split(3), Ok((1, 1)));
            assert_eq!(PATTERN.try_ellipsis_split(4), Ok((1, 2)));

            assert_eq!(
                PATTERN.try_ellipsis_split(1),
                Err("Shape rank 1 < non-ellipsis pattern term count 2".to_string())
            );
        }
        {
            // Without ellipsis.
            static PATTERN: ShapeContract =
                ShapeContract::new(&[DimMatcher::any(), DimMatcher::expr(DimExpr::Param("b"))]);

            assert_eq!(PATTERN.try_ellipsis_split(2), Ok((2, 0)));

            assert_eq!(
                PATTERN.try_ellipsis_split(1),
                Err("Shape rank 1 != pattern dim count 2".to_string())
            );
        }
    }

    #[test]
    fn test_format_pattern() {
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::expr(DimExpr::Prod(&[
                DimExpr::Param("h"),
                DimExpr::Sum(&[DimExpr::Param("a"), DimExpr::Negate(&DimExpr::Param("b"))]),
            ])),
            DimMatcher::expr(DimExpr::Pow(&DimExpr::Param("h"), 2)),
        ]);

        assert_eq!(PATTERN.to_string(), "[_, ..., b, (h*(a+(-b))), (h)^2]");
    }

    #[test]
    fn test_unpack_shape() {
        static CONTRACT: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("h"), DimExpr::Param("p")])),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("w"), DimExpr::Param("p")])),
            DimMatcher::expr(DimExpr::Pow(&DimExpr::Param("z"), 3)),
            DimMatcher::expr(DimExpr::Param("c")),
        ]);

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, z * z * z, c];
        let env = [("p", p), ("c", c)];

        CONTRACT.assert_shape(&shape, &env);

        let [u_b, u_h, u_w, u_z] = CONTRACT.unpack_shape(&shape, &["b", "h", "w", "z"], &env);

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }

    #[should_panic(expected = "Shape Error:: 1 !~ a :: Value MissMatch.")]
    #[test]
    fn test_unpack_shape_panic() {
        use crate as bimm_contracts;
        static CONTRACT: ShapeContract = shape_contract!["a", "b", "c"];
        let _ignore = CONTRACT.unpack_shape(&[1, 2, 3], &["a", "b", "c"], &[("a", 7)]);
    }

    #[should_panic(expected = "Shape rank 3 != pattern dim count 1")]
    #[test]
    fn test_shape_mismatch_no_ellipsis() {
        // This should panic because the shape does not match the pattern.
        static PATTERN: ShapeContract =
            ShapeContract::new(&[DimMatcher::expr(DimExpr::Param("a"))]);
        let shape = [1, 2, 3];
        PATTERN.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Shape rank 3 < non-ellipsis pattern term count 4")]
    #[test]
    fn test_shape_mismatch_with_ellipsis() {
        // This should panic because the shape does not match the pattern.
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::expr(DimExpr::Param("c")),
        ]);
        let shape = [1, 2, 3];
        PATTERN.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Value MissMatch")]
    #[test]
    fn test_shape_mismatch_value() {
        // This should panic because the value does not match the constraint.
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::expr(DimExpr::Param("a")),
            DimMatcher::expr(DimExpr::Param("b")),
        ]);
        let shape = [2, 3];
        PATTERN.assert_shape(&shape, &[("a", 2), ("b", 4)]);
    }
}
