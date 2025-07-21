use std::sync::Arc;
use crate::core::{BuildOperatorFactory, MapOperatorFactory, OperatorSpec};
use crate::define_reflexive_id;

// TODO:
// - check plan
// - spec/check only
// - parse config?
pub struct OpKey {
    pub operator: &'static str,
    pub get_spec: fn() -> OperatorSpec,
    pub get_builder: fn() -> Arc<dyn BuildOperatorFactory>,
}

inventory::collect!(OpKey);

impl OpKey {
    pub fn get_spec(&self) -> OperatorSpec {
        (self.get_spec)()
            .with_operator_id(self.operator)
    }

    pub fn get_builder(&self) -> Arc<dyn BuildOperatorFactory> {
        (self.get_builder)()
    }

}

pub fn autofactory() -> MapOperatorFactory {
    let mut factory = MapOperatorFactory::new();
    for opkey in inventory::iter::<OpKey>() {
        factory.add_operation(
            opkey.operator,
            opkey.get_builder()
        );
    }
    factory
}

#[macro_export]
macro_rules! register_op {
    ($name:ident, spec: $get_spec:expr, $get_builder:expr) => {
        define_reflexive_id!($name);

        inventory::submit! {
            OpKey {
                operator: $name,
                get_spec: || $get_spec,
                get_builder: $get_builder,
            }
        }
    };
}

#[cfg(test)]
mod tests {


}