use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Result as SynResult;
use syn::parse::{Parse, ParseStream};
use syn::{LitStr, Token, parse_macro_input};

/// Parse shape pattern from token stream.
fn parse_pattern_tokens(input: ParseStream) -> SynResult<ShapeContract> {
    let mut terms = Vec::new();

    while !input.is_empty() {
        // Parse a pattern term
        let term = parse_pattern_term_tokens(input)?;
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

/// Parse a single pattern term from tokens.
fn parse_pattern_term_tokens(input: ParseStream) -> SynResult<PatternTerm> {
    // Check for "_" (underscore) for Any
    if input.peek(Token![_]) {
        input.parse::<Token![_]>()?;
        return Ok(PatternTerm::Any);
    }

    // Check for ellipsis "..."
    if input.peek(Token![...]) {
        input.parse::<Token![...]>()?;
        return Ok(PatternTerm::Ellipsis);
    }

    // Otherwise, parse as expression
    let expr = parse_expr_tokens(input)?;
    Ok(PatternTerm::Expr(expr))
}

#[derive(Debug, Clone, PartialEq)]
enum ExprNode {
    Param(String),
    Negate(Box<ExprNode>),
    Pow(Box<ExprNode>, usize),
    Sum(Vec<ExprNode>),
    Prod(Vec<ExprNode>),
}

#[derive(Debug, Clone, PartialEq)]
enum PatternTerm {
    Any,
    Ellipsis,
    Expr(ExprNode),
}

#[derive(Debug, Clone, PartialEq)]
struct ShapeContract {
    pub terms: Vec<PatternTerm>,
}

/// Custom parser for shape pattern syntax.
struct PatternSyntax {
    pattern: ShapeContract,
}

impl Parse for PatternSyntax {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let pattern = parse_pattern_tokens(input)?;
        Ok(PatternSyntax { pattern })
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
            let right = parse_prod_expr(input)?;
            terms.push(right);
        } else if input.parse::<Token![-]>().is_ok() {
            let right = parse_prod_expr(input)?;
            terms.push(ExprNode::Negate(Box::new(right)));
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
        return parse_factor_expr(input); // Unary plus is no-op
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

impl PatternTerm {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            PatternTerm::Any => {
                quote! { bimm_contracts::DimMatcher::Any }
            }
            PatternTerm::Ellipsis => {
                quote! { bimm_contracts::DimMatcher::Ellipsis }
            }
            PatternTerm::Expr(expr) => {
                let expr_tokens = expr.to_tokens();
                quote! { bimm_contracts::DimMatcher::Expr(#expr_tokens) }
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

/// Parse a shape pattern at compile time and return the ShapePattern struct.
///
/// A shape pattern is made of one or more dimension matcher terms:
/// - `_`: for any shape; ignores the size, but requires the dimension to exist.,
/// - `...`: for ellipsis; matches any number of dimensions, only one ellipsis is allowed,
/// - a dim expression.
///
/// ```bnf
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
/// ```rust.no_run
/// use super::{ShapeContract, shape_contract};
/// static CONTRACT: ShapeContract = shape_contract!(_, "x" + "y", ..., "z" ^ 2);
/// ```
#[proc_macro]
pub fn shape_contract(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as PatternSyntax);

    let tokens = parsed.pattern.to_tokens();
    quote! {
        {
            #[allow(unused_imports)]
            use bimm_contracts::{ShapeContract, DimMatcher, DimExpr};
            use std::boxed::Box;
            #tokens
        }
    }
    .into()
}

#[cfg(test)]
mod tests {

    use crate::{ExprNode, PatternTerm, ShapeContract};

    #[test]
    fn test_macro_usage() {
        // Expression macro usage:
        // let ast1 = expr!("x");
        // let ast2 = expr!("a" + "b" * "c" ^ 2);
        // let ast3 = expr!(("x" + "y") * "z");

        // Shape pattern macro usage:
        // let pattern1 = shape_contract!(_);
        // let pattern2 = shape_contract!(_, "x", ...);
        // let pattern3 = shape_contract!("a" + "b", ..., "z" ^ 2);

        // For testing purposes, we'll show the expected output:
        let expected_expr = ExprNode::Sum(vec![
            ExprNode::Param("a".to_string()),
            ExprNode::Prod(vec![
                ExprNode::Param("b".to_string()),
                ExprNode::Pow(Box::new(ExprNode::Param("c".to_string())), 2),
            ]),
        ]);

        let expected_pattern = ShapeContract {
            terms: vec![
                PatternTerm::Any,
                PatternTerm::Expr(ExprNode::Param("x".to_string())),
                PatternTerm::Ellipsis,
            ],
        };

        println!("Expected AST: {expected_expr:#?}");
        println!("Expected Pattern: {expected_pattern:#?}");
    }
}
