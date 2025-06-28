use crate::{DimExpr, DimMatcher, ShapeContract, run_every_nth};
use test::Bencher;

static PATTERN: ShapeContract = ShapeContract::new(&[
    DimMatcher::Any,
    DimMatcher::Expr(DimExpr::Param("b")),
    DimMatcher::Ellipsis,
    DimMatcher::Expr(DimExpr::Prod(&[DimExpr::Param("h"), DimExpr::Param("p")])),
    DimMatcher::Expr(DimExpr::Prod(&[DimExpr::Param("w"), DimExpr::Param("p")])),
    DimMatcher::Expr(DimExpr::Pow(&DimExpr::Param("z"), 3)),
    DimMatcher::Expr(DimExpr::Param("c")),
]);

static BATCH: usize = 2;
static HEIGHT: usize = 3;
static WIDTH: usize = 2;
static PADDING: usize = 4;
static CHANNELS: usize = 5;
static COLOR: usize = 4;

#[bench]
fn bench_unpack_shape(b: &mut Bencher) {
    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];

    b.iter(|| {
        let _ = PATTERN.unpack_shape(&shape, &["b", "h", "w", "z"], &env);
    });
}
#[bench]
fn bench_assert_shape(b: &mut Bencher) {
    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];
    
    b.iter(|| {
        PATTERN.assert_shape(&shape, &env);
    });
}

#[bench]
fn bench_assert_shape_every_nth(b: &mut Bencher) {
    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];

    b.iter(|| {
        run_every_nth!(PATTERN.assert_shape(&shape, &env));
    });
}
