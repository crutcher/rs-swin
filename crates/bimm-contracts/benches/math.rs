use bimm_contracts::math::maybe_iroot;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_maybe_iroot(c: &mut Criterion) {
    let mut idx = 2;
    c.bench_function("maybe_iroot", |b| {
        b.iter(|| {
            let _r = maybe_iroot(idx, 4);
            idx += 1;
        })
    });
}

criterion_group!(math, bench_maybe_iroot,);
criterion_main!(math);
