use std::collections::HashMap;

#[derive(Debug)]
pub enum DimTerm<'a> {
    Named(&'a str),
    Const(usize),
    Prod(&'a [DimTerm<'a>]),
}

impl<'a> Into<DimTerm<'a>> for &'a str {
    fn into(self) -> DimTerm<'a> {
        DimTerm::Named(self)
    }
}

impl<'a> Into<DimTerm<'a>> for &'a usize {
    fn into(self) -> DimTerm<'a> {
        DimTerm::Const(*self)
    }
}

impl<'a> Into<DimTerm<'a>> for &'a [DimTerm<'a>] {
    fn into(self) -> DimTerm<'a> {
        DimTerm::Prod(self)
    }
}

type Bindings<'a, const D: usize> = &'a [(&'a str, usize); D];

fn lookup_binding<'a, const D: usize>(
    bindings: Bindings<'a, D>,
    key: &str,
) -> Option<usize> {
    bindings.iter().find_map(|(k, v)| if k == &key { Some(*v) } else { None })
}

struct Env<'a, const B: usize> {
    local: HashMap<&'a str, usize>,
    bindings: Bindings<'a, B>,
}

impl <'a, const B: usize> Env<'a, B> {
    fn lookup(&self, key: &str) -> Option<usize> {
        self.local.get(key).cloned().or_else(|| lookup_binding(self.bindings, key))
    }

    fn insert(&mut self, key: &'a str, value: usize) {
        self.local.insert(key, value);
    }
}

impl<'a, const B: usize> Env<'a, B> {
    fn new(
        bindings: Bindings<'a, B>,
    ) -> Self {
        Env { local: HashMap::new(), bindings }
    }
    
    fn export<const K: usize>(&self, keys: &[&str; K]) -> [usize; K] {
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
        
        let mut env = Env::new(bindings);
        
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
        
        let [b, h, w] = [
            DimTerm::Named("b"),
            DimTerm::Prod(&[
                DimTerm::Named("h"),
                DimTerm::Named("p"),
            ]),
            DimTerm::Prod(&[
                DimTerm::Named("w"),
                DimTerm::Named("p"),
            ]),
            DimTerm::Named("c"),
        ]
        .unpack_shape(
            &shape,
            &["b", "h", "w"],
            &[
                ("p", _p),
                ("c", _c),
            ],
        );
        
        assert_eq!(b, _b);
        assert_eq!(h, _h);
        assert_eq!(w, _w);
    }
}