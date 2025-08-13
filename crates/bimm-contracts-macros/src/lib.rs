#![warn(missing_docs)]
//! `proc_macro` support for BIMM Contracts.

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Result as SynResult;
use syn::parse::{Parse, ParseStream};
use syn::{LitStr, Token, parse_macro_input};

/// Parse shape contract from token stream.
fn parse_shape_contract_terms(input: ParseStream) -> SynResult<ShapeContract> {
    let mut terms = Vec::new();

    while !input.is_empty() {
        // Parse a dim term
        let term = parse_dim_matcher_tokens(input)?;
        terms.push(term);

        // Check for comma separator
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }

    Ok(ShapeContract { terms })
}

/// Parse a single contract dim term from tokens.
fn parse_dim_matcher_tokens(input: ParseStream) -> SynResult<DimMatcher> {
    let mut label = None;

    // peek 2: ["name" =]
    if input.peek(LitStr) && input.peek2(Token![=]) {
        let lit: LitStr = input.parse()?;
        label = Some(lit.value());
        input.parse::<Token![=]>()?;
    }

    // Check for "_" (underscore) for Any
    if input.peek(Token![_]) {
        input.parse::<Token![_]>()?;
        return Ok(DimMatcher::Any { label });
    }

    // Check for ellipsis "..."
    if input.peek(Token![...]) {
        input.parse::<Token![...]>()?;
        return Ok(DimMatcher::Ellipsis { label });
    }

    // Otherwise, parse as an expression.
    let expr = parse_expr_tokens(input)?;
    Ok(DimMatcher::Expr { label, expr })
}

/// Represents a node in a dimension expression.
#[derive(Debug, Clone, PartialEq)]
enum ExprNode {
    /// A parameter (string literal).
    Param(String),

    /// Negation of another expression.
    Negate(Box<ExprNode>),

    /// Power of an expression (base raised to an exponent).
    Pow(Box<ExprNode>, usize),

    /// Sum of multiple expressions (addition and subtraction).
    Sum(Vec<ExprNode>),

    /// Product of multiple expressions (multiplication).
    Prod(Vec<ExprNode>),
}

/// Represents a matcher for a dimension in a shape contract.
#[derive(Debug, Clone, PartialEq)]
enum DimMatcher {
    /// Matches any dimension, ignoring size.
    Any { label: Option<String> },

    /// Matches an ellipsis, allowing any number of dimensions.
    ///
    /// There can only be one ellipsis in a shape contract.
    Ellipsis { label: Option<String> },

    /// Matches a dimension based on an expression.
    Expr {
        label: Option<String>,
        expr: ExprNode,
    },
}

/// Represents a shape contract, which consists of multiple dimension matchers.
///
/// The `shape_contract!` macro allows you to define a shape contract
/// that can be used to match shapes in a type-safe manner.
#[derive(Debug, Clone, PartialEq)]
struct ShapeContract {
    /// The terms of the shape contract, each represented by a `DimMatcher`.
    pub terms: Vec<DimMatcher>,
}

/// Custom parser for shape contract syntax.
struct ContractSyntax {
    /// The parsed shape contract.
    contract: ShapeContract,
}

impl Parse for ContractSyntax {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let contract = parse_shape_contract_terms(input)?;
        Ok(ContractSyntax { contract })
    }
}

/// Parse expression from token stream.
fn parse_expr_tokens(input: ParseStream) -> SynResult<ExprNode> {
    parse_sum_expr(input)
}

/// Parse sum/difference (lowest precedence).
fn parse_sum_expr(input: ParseStream) -> SynResult<ExprNode> {
    let left = parse_prod_expr(input)?;
    let mut terms = vec![left.clone()];

    while input.peek(Token![+]) || input.peek(Token![-]) {
        if input.parse::<Token![+]>().is_ok() {
            terms.push(parse_prod_expr(input)?);
        } else if input.parse::<Token![-]>().is_ok() {
            terms.push(ExprNode::Negate(Box::new(parse_prod_expr(input)?)));
        }
    }

    Ok(if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        ExprNode::Sum(terms)
    })
}

/// Parse multiplication (higher precedence).
fn parse_prod_expr(input: ParseStream) -> SynResult<ExprNode> {
    let mut factors = vec![parse_power_expr(input)?];

    while input.peek(Token![*]) {
        input.parse::<Token![*]>()?;
        factors.push(parse_power_expr(input)?);
    }

    Ok(if factors.len() == 1 {
        factors.into_iter().next().unwrap()
    } else {
        ExprNode::Prod(factors)
    })
}

/// Parse power (highest precedence).
fn parse_power_expr(input: ParseStream) -> SynResult<ExprNode> {
    let base = parse_factor_expr(input)?;

    if input.peek(Token![^]) {
        input.parse::<Token![^]>()?;
        let exp: syn::LitInt = input.parse()?;
        let exp_value: usize = exp.base10_parse()?;
        Ok(ExprNode::Pow(Box::new(base), exp_value))
    } else {
        Ok(base)
    }
}

/// Parse factors (parameters, parentheses, negation).
fn parse_factor_expr(input: ParseStream) -> SynResult<ExprNode> {
    // Handle unary operators
    if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        let expr = parse_factor_expr(input)?;
        return Ok(ExprNode::Negate(Box::new(expr)));
    }

    if input.peek(Token![+]) {
        input.parse::<Token![+]>()?;
        // Unary plus is no-op
        return parse_factor_expr(input);
    }

    // Handle parentheses
    if input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        return parse_expr_tokens(&content);
    }

    // Handle string literals (parameters)
    if input.peek(LitStr) {
        let lit: LitStr = input.parse()?;
        return Ok(ExprNode::Param(lit.value()));
    }

    Err(syn::Error::new(
        input.span(),
        "Expected parameter, parentheses, or unary operator",
    ))
}

impl ExprNode {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            ExprNode::Param(name) => {
                quote! {
                    bimm_contracts::DimExpr::Param(#name)
                }
            }
            ExprNode::Negate(expr) => {
                let inner = expr.to_tokens();
                quote! {
                    bimm_contracts::DimExpr::Negate(&#inner)
                }
            }
            ExprNode::Pow(base, exp) => {
                let base_tokens = base.to_tokens();
                quote! {
                    bimm_contracts::DimExpr::Pow(&#base_tokens, #exp)
                }
            }
            ExprNode::Sum(terms) => {
                let term_tokens: Vec<_> = terms.iter().map(|t| t.to_tokens()).collect();
                quote! {
                    bimm_contracts::DimExpr::Sum(&[#(#term_tokens),*])
                }
            }
            ExprNode::Prod(factors) => {
                let factor_tokens: Vec<_> = factors.iter().map(|f| f.to_tokens()).collect();
                quote! {
                    bimm_contracts::DimExpr::Prod(&[#(#factor_tokens),*])
                }
            }
        }
    }
}

impl DimMatcher {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            DimMatcher::Any { label } => {
                let base = quote! { bimm_contracts::DimMatcher::any() };
                if label.is_some() {
                    quote! { #base.with_label(Some(#label)) }
                } else {
                    base
                }
            }
            DimMatcher::Ellipsis { label } => {
                let base = quote! { bimm_contracts::DimMatcher::ellipsis() };
                if label.is_some() {
                    quote! { #base.with_label(Some(#label)) }
                } else {
                    base
                }
            }
            DimMatcher::Expr { label, expr } => {
                let expr_tokens = expr.to_tokens();
                let base = quote! { bimm_contracts::DimMatcher::expr(#expr_tokens) };
                if label.is_some() {
                    quote! { #base.with_label(Some(#label)) }
                } else {
                    base
                }
            }
        }
    }
}

impl ShapeContract {
    fn to_tokens(&self) -> TokenStream2 {
        let term_tokens: Vec<_> = self.terms.iter().map(|t| t.to_tokens()).collect();
        quote! {
            bimm_contracts::ShapeContract::new(&[
                #(#term_tokens),*
            ])
        }
    }
}

/// Parse a shape contract at compile time and return the `ShapePattern` struct.
///
/// This macro generates `no_std` compatible code.
///
/// A shape pattern is made of one or more dimension matcher terms:
/// - `_`: for any shape; ignores the size, but requires the dimension to exist.,
/// - `...`: for ellipsis; matches any number of dimensions, only one ellipsis is allowed,
/// - a dim expression.
///
/// ```bnf
/// ShapeContract => <LabeledExpr> { ',' <LabeledExpr> }* ','?
/// LabeledExpr => {Param "="}? <Expr>
/// Expr => <Term> { <AddOp> <Term> }
/// Term => <Power> { <MulOp> <Power> }
/// Power => <Factor> [ ^ <usize> ]
/// Factor => <Param> | ( '(' <Expression> ')' ) | NegOp <Factor>
/// Param => '"' <identifier> '"'
/// identifier => { <alpha> | "_" } { <alphanumeric> | "_" }*
/// NegOp =>      '+' | '-'
/// AddOp =>      '+' | '-'
/// MulOp =>      '*'
/// ```
///
/// # Example
/// ```rust.norun
/// use bimm_contracts::{ShapeContract, shape_contract};
/// static CONTRACT: ShapeContract = shape_contract![_, "x" + "y", ..., "z" ^ 2];
/// ```
#[proc_macro]
pub fn shape_contract(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ContractSyntax);
    let tokens = parsed.contract.to_tokens();
    quote! {
        {
            extern crate alloc;
            #[allow(unused_imports)]
            use bimm_contracts::{ShapeContract, DimMatcher, DimExpr};
            use alloc::boxed::Box;
            #tokens
        }
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    struct ExprSyntax {
        expr: ExprNode,
    }

    impl Parse for ExprSyntax {
        fn parse(input: ParseStream) -> SynResult<Self> {
            let expr = parse_expr_tokens(input)?;
            Ok(ExprSyntax { expr })
        }
    }

    #[test]
    fn test_unary_add_op() {
        let tokens: proc_macro2::TokenStream = r#"+ "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(input.expr, ExprNode::Param("x".to_string()));

        assert_eq!(
            input.expr.to_tokens().to_string(),
            "bimm_contracts :: DimExpr :: Param (\"x\")"
        );
    }

    #[test]
    fn test_parse_simple_expression() {
        let tokens: proc_macro2::TokenStream = r#""x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(input.expr, ExprNode::Param("x".to_string()));

        assert_eq!(
            input.expr.to_tokens().to_string(),
            "bimm_contracts :: DimExpr :: Param (\"x\")"
        );
    }

    #[test]
    fn test_parse_unary_negation() {
        let tokens: proc_macro2::TokenStream = r#"-"x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprNode::Negate(Box::new(ExprNode::Param("x".to_string())))
        );

        assert_eq!(
            input.expr.to_tokens().to_string(),
            "bimm_contracts :: DimExpr :: Negate (& bimm_contracts :: DimExpr :: Param (\"x\"))"
        );
    }

    #[test]
    fn test_mixed_addition() {
        let tokens: proc_macro2::TokenStream = r#""a" + "b" - "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprNode::Sum(vec![
                ExprNode::Param("a".to_string()),
                ExprNode::Param("b".to_string()),
                ExprNode::Negate(Box::new(ExprNode::Param("x".to_string()))),
            ])
        );
    }

    #[test]
    fn test_mangled_addition() {
        let tokens: proc_macro2::TokenStream = r#""a" + - - "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprNode::Sum(vec![
                ExprNode::Param("a".to_string()),
                ExprNode::Negate(Box::new(ExprNode::Negate(Box::new(ExprNode::Param(
                    "x".to_string()
                ))))),
            ])
        );

        assert_eq!(
            input.expr.to_tokens().to_string(),
            "bimm_contracts :: DimExpr :: Sum (& [bimm_contracts :: DimExpr :: Param (\"a\") , bimm_contracts :: DimExpr :: Negate (& bimm_contracts :: DimExpr :: Negate (& bimm_contracts :: DimExpr :: Param (\"x\")))])"
        );
    }

    #[test]
    fn test_parse_power_precedence() {
        let tokens: proc_macro2::TokenStream = r#"-"x" ^ 3"#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprNode::Pow(
                Box::new(ExprNode::Negate(Box::new(ExprNode::Param("x".to_string())))),
                3
            )
        );

        assert_eq!(
            input.expr.to_tokens().to_string(),
            "bimm_contracts :: DimExpr :: Pow (& bimm_contracts :: DimExpr :: Negate (& bimm_contracts :: DimExpr :: Param (\"x\")) , 3usize)"
        );
    }

    #[test]
    fn test_parse_shape_contract_terms() {
        let tokens: proc_macro2::TokenStream =
            r#""any" = _, "x", ..., "y" + ("z" * "w") ^ 2"#.parse().unwrap();
        let input = syn::parse2::<ContractSyntax>(tokens).unwrap();
        let contract = input.contract;

        assert_eq!(contract.terms.len(), 4);
        assert_eq!(
            contract.terms[0],
            DimMatcher::Any {
                label: Some("any".to_string())
            }
        );
        assert_eq!(
            contract.terms[1],
            DimMatcher::Expr {
                label: None,
                expr: ExprNode::Param("x".to_string())
            }
        );
        assert_eq!(contract.terms[2], DimMatcher::Ellipsis { label: None });
        assert_eq!(
            contract.terms[3],
            DimMatcher::Expr {
                label: None,
                expr: ExprNode::Sum(vec![
                    ExprNode::Param("y".to_string()),
                    ExprNode::Pow(
                        Box::new(ExprNode::Prod(vec![
                            ExprNode::Param("z".to_string()),
                            ExprNode::Param("w".to_string())
                        ])),
                        2
                    ),
                ])
            }
        );

        assert_eq!(
            contract.to_tokens().to_string(),
            "bimm_contracts :: ShapeContract :: new (& [\
bimm_contracts :: DimMatcher :: any () . with_label (Some (\"any\")) , \
bimm_contracts :: DimMatcher :: expr (bimm_contracts :: DimExpr :: Param (\"x\")) , \
bimm_contracts :: DimMatcher :: ellipsis () , \
bimm_contracts :: DimMatcher :: expr (bimm_contracts :: DimExpr :: Sum (& [bimm_contracts :: DimExpr :: Param (\"y\") , \
bimm_contracts :: DimExpr :: Pow (& bimm_contracts :: DimExpr :: Prod (& [bimm_contracts :: DimExpr :: Param (\"z\") , \
bimm_contracts :: DimExpr :: Param (\"w\")\
]) , 2usize)]))])",
        );
    }
}
