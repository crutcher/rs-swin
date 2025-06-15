use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Atom<'a> {
    Value(usize),
    Bound(&'a str),
}

impl<'a> Atom<'a> {
    fn value(value: usize) -> Self {
        Atom::Value(value)
    }
    
    fn bound(name: &'a str) -> Self {
        Atom::Bound(name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term<'a> {
    Atom(Atom<'a>),
    Prod(&'a [Term<'a>]),
}

impl <'a> Term<'a> {
    fn atom(value: usize) -> Self {
        Term::Atom(Atom::value(value))
    }
    
    fn bound(name: &'a str) -> Self {
        Term::Atom(Atom::bound(name))
    }
    
    fn prod(terms: &'a [Term<'a>]) -> Self {
        Term::Prod(terms)
    }
}

pub enum Pattern<'a> {
    Any,
    Ellipsis,
    Term(Term<'a>),
}

impl<'a> Pattern<'a> {
    pub fn any() -> Self {
        Pattern::Any
    }

    pub fn ellipsis() -> Self {
        Pattern::Ellipsis
    }
    
    pub fn bound(name: &'a str) -> Self {
        Pattern::Term(Term::bound(name))
    }
    
    pub fn value(value: usize) -> Self {
        Pattern::Term(Term::atom(value))
    }

    pub fn prod(terms: &'a [Term<'a>]) -> Self {
        Pattern::Term(Term::prod(terms))
    }
}

#[derive(Debug)]
pub enum DimTerm<'a> {
    Named(&'a str),
    Const(usize),
    Prod(&'a [DimTerm<'a>]),
}

impl<'a> From<&'a str> for DimTerm<'a> {
    fn from(val: &'a str) -> Self {
        DimTerm::Named(val)
    }
}

impl<'a> From<&'a usize> for DimTerm<'a> {
    fn from(val: &'a usize) -> Self {
        DimTerm::Const(*val)
    }
}

impl<'a> From<&'a [DimTerm<'a>]> for DimTerm<'a> {
    fn from(val: &'a [DimTerm<'a>]) -> Self {
        DimTerm::Prod(val)
    }
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

trait DimPattern<'a, const D: usize> {
    fn unpack_shape<const K: usize, const B: usize>(
        &self,
        shape: &[usize],
        keys: &[&str; K],
        bindings: Bindings<B>,
    ) -> [usize; K];
}

impl<'a, const D: usize> DimPattern<'a, D> for [DimTerm<'a>; D] {
    fn unpack_shape<const K: usize, const B: usize>(
        &self,
        shape: &[usize],
        keys: &[&str; K],
        bindings: Bindings<B>,
    ) -> [usize; K] {
        // TODO: modeling `...` and `*` patterns;
        // probably need to not use the const D parameter.
        if shape.len() != D {
            panic!("Shape length mismatch: expected {}, got {}", D, shape.len());
        }

        let env = Env::new(bindings);

        let mut values = [0; K];
        for i in 0..K {
            let key = keys[i];
            values[i] = match env.lookup(key) {
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
            Pattern::bound("b"),
            Pattern::prod(&[Term::bound("h"), Term::bound("p")]),
            Pattern::prod(&[Term::bound("w"), Term::bound("p")]),
            Pattern::bound("c"),
        ];
    }
}
