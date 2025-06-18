mod alt;

#[allow(dead_code)]
#[allow(unused)]
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Atom<'a> {
    Fixed(usize),
    Param(&'a str),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term<'a> {
    Atom(Atom<'a>),
    Neg(Box<Term<'a>>),
    Prod(Vec<Term<'a>>),
    Sum(Vec<Term<'a>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalResult {
    Value(isize),
    UnboundCount(usize),
}

impl<'a> Term<'a> {
    pub fn fixed(value: usize) -> Self {
        Term::Atom(Atom::Fixed(value))
    }

    pub fn param(name: &'a str) -> Self {
        Term::Atom(Atom::Param(name))
    }

    pub fn neg(term: Term<'a>) -> Self {
        Term::Neg(Box::new(term))
    }

    pub fn prod(terms: Vec<Term<'a>>) -> Self {
        Term::Prod(terms)
    }

    pub fn sum(terms: Vec<Term<'a>>) -> Self {
        Term::Sum(terms)
    }

    fn eval<const D: usize>(
        &self,
        env: &Env<'a, D>,
    ) -> EvalResult {
        match self {
            Term::Atom(atom) => match atom {
                Atom::Fixed(val) => EvalResult::Value(*val as isize),
                Atom::Param(name) => match env.lookup(name) {
                    Some(value) => EvalResult::Value(value as isize),
                    None => EvalResult::UnboundCount(1),
                },
            },
            Term::Neg(term) => match term.eval(env) {
                EvalResult::Value(val) => EvalResult::Value(0 - val),
                EvalResult::UnboundCount(count) => EvalResult::UnboundCount(count),
            },
            Term::Prod(terms) => {
                let mut product = 1;
                let mut unbound_count = 0;
                for term in terms {
                    match term.eval(env) {
                        EvalResult::Value(val) => product *= val,
                        EvalResult::UnboundCount(count) => unbound_count += count,
                    }
                }

                if unbound_count > 0 {
                    EvalResult::UnboundCount(unbound_count)
                } else {
                    EvalResult::Value(product)
                }
            }
            Term::Sum(terms) => {
                let mut sum = 0;
                let mut unbound_count = 0;
                for term in terms {
                    match term.eval(env) {
                        EvalResult::Value(val) => sum += val,
                        EvalResult::UnboundCount(count) => unbound_count += count,
                    }
                }

                if unbound_count > 0 {
                    EvalResult::UnboundCount(unbound_count)
                } else {
                    EvalResult::Value(sum)
                }
            }
        }
    }
    
    fn match_target<const D: usize>(
        &self,
        target: usize,
        env: &mut Env<'a, D>,
    ) -> Result<(), String> {
        let target = target as isize;
        
        match self.eval(env) {
            EvalResult::Value(val) => {
                if val == target {
                    return Ok(());
                } else {
                    return Err(format!("Expected {}, got {}", target, val));
                }
            }
            EvalResult::UnboundCount(count) => {
                return Err(format!("Expected {}, got {}", target, count));
            }
        }
        
        unimplemented!()
    }
}

pub enum Pattern<'a> {
    Any,
    Ellipsis,
    Expr(Term<'a>),
}

type Bindings<'a, const D: usize> = &'a [(&'a str, usize); D];

fn lookup_binding<'a, const D: usize>(
    bindings: Bindings<'a, D>,
    key: &str,
) -> Option<usize> {
    bindings
        .iter()
        .find_map(|(k, v)| if k == &key { Some(*v) } else { None })
}

struct Env<'a, const B: usize> {
    local: HashMap<&'a str, usize>,
    bindings: Bindings<'a, B>,
}

impl<'a, const B: usize> Env<'a, B> {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.local
            .get(key)
            .cloned()
            .or_else(|| lookup_binding(self.bindings, key))
    }

    fn insert(
        &mut self,
        key: &'a str,
        value: usize,
    ) {
        self.local.insert(key, value);
    }
}

impl<'a, const B: usize> Env<'a, B> {
    fn new(bindings: Bindings<'a, B>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratch() {
        let _b = 2;
        let _h = 3;
        let _w = 2;
        let _p = 4;
        let _c = 5;

        let shape = [_b, _h * _p, _w * _p, _c];

        let _pattern = [
            Pattern::Expr(Term::param("b")),
            Pattern::Expr(Term::prod(vec![Term::param("h"), Term::param("p")])),
            Pattern::Expr(Term::prod(vec![Term::param("w"), Term::param("p")])),
            Pattern::Expr(Term::param("c")),
        ];
    }

    #[test]
    fn test_eval() {
        let bindings = [("a", 2), ("b", 3)];
        let mut env = Env::new(&bindings);
        env.insert("c", 4);

        assert_eq!(Term::fixed(42).eval(&env), EvalResult::Value(42));
        assert_eq!(
            Term::neg(Term::fixed(42)).eval(&env),
            EvalResult::Value(-42)
        );

        assert_eq!(Term::param("a").eval(&env), EvalResult::Value(2));
        assert_eq!(Term::param("c").eval(&env), EvalResult::Value(4));
        assert_eq!(Term::param("x").eval(&env), EvalResult::UnboundCount(1));

        assert_eq!(
            Term::neg(Term::param("c")).eval(&env),
            EvalResult::Value(-4)
        );
        assert_eq!(
            Term::neg(Term::param("x")).eval(&env),
            EvalResult::UnboundCount(1)
        );

        assert_eq!(
            Term::prod(vec![Term::param("a"), Term::param("b")]).eval(&env),
            EvalResult::Value(6)
        );
        assert_eq!(
            Term::prod(vec![Term::param("a"), Term::neg(Term::param("b"))]).eval(&env),
            EvalResult::Value(-6)
        );
        assert_eq!(
            Term::prod(vec![Term::param("a"), Term::param("x"), Term::param("y")]).eval(&env),
            EvalResult::UnboundCount(2)
        );

        assert_eq!(
            Term::sum(vec![Term::param("a"), Term::param("b")]).eval(&env),
            EvalResult::Value(5)
        );
        assert_eq!(
            Term::sum(vec![Term::param("a"), Term::neg(Term::param("b"))]).eval(&env),
            EvalResult::Value(-1)
        );
        assert_eq!(
            Term::sum(vec![Term::param("a"), Term::param("x"), Term::param("y")]).eval(&env),
            EvalResult::UnboundCount(2)
        );
    }
}
