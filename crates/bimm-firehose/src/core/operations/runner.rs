use crate::core::operations::environment::OpEnvironment;
use crate::core::operations::factory::OperatorInitializationContext;
use crate::core::operations::operator::FirehoseOperator;
use crate::core::rows::{AnyArc, Row};
use crate::core::schema::{BuildPlan, TableSchema};
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Context for applying a column operator.
pub struct OperatorApplyBatchContext {
    len: usize,
    inputs: BTreeMap<String, Vec<Option<AnyArc>>>,
    outputs: BTreeMap<String, Vec<Option<AnyArc>>>,
}

impl OperatorApplyBatchContext {
    /// Creates a new `ColumnBuildBatchContext` with the specified inputs.
    pub fn new(inputs: BTreeMap<String, Vec<Option<AnyArc>>>) -> Self {
        let len = inputs.values().map(|v| v.len()).max().unwrap_or(0);
        Self {
            len,
            inputs,
            outputs: Default::default(),
        }
    }

    /// Returns the number of rows in the batch context.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the batch context empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the input map.
    pub fn input_map(&self) -> &BTreeMap<String, Vec<Option<AnyArc>>> {
        &self.inputs
    }

    /// Returns the output map.
    pub fn output_map(&self) -> &BTreeMap<String, Vec<Option<AnyArc>>> {
        &self.outputs
    }

    /// Returns a mutable reference to the output map.
    pub fn mut_output_map(&mut self) -> &mut BTreeMap<String, Vec<Option<AnyArc>>> {
        &mut self.outputs
    }

    /// Get the input for a specific row and parameter name.
    pub fn get_row_input(
        &self,
        row: usize,
        name: &str,
    ) -> Option<&AnyArc> {
        self.inputs
            .get(name)
            .and_then(|input| input.get(row))
            .and_then(|value| value.as_ref())
    }

    /// Get the input for a specific row and parameter name, downcasting to a specific type.
    pub fn get_row_input_downcast<T: Any + 'static>(
        &self,
        row: usize,
        name: &str,
    ) -> Option<&T> {
        self.get_row_input(row, name)
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// Set an output value for a specific row and parameter name.
    pub fn set_row_output(
        &mut self,
        row: usize,
        name: &str,
        value: Option<AnyArc>,
    ) {
        self.outputs
            .entry(name.to_string())
            .or_default()
            .resize(self.len, None);
        if let Some(output) = self.outputs.get_mut(name) {
            if row < output.len() {
                output[row] = value;
            } else {
                panic!("Row index out of bounds for output '{name}'");
            }
        }
    }
}

/// Row build context; dependent on `ColumnBuildBatchContext`.
pub struct OperatorApplyRowContext<'a> {
    batch_context: &'a mut OperatorApplyBatchContext,
    index: usize,
}

impl<'a> OperatorApplyRowContext<'a> {
    /// Creates a new `ColumnBuildRowContext` for the specified index in the batch context.
    ///
    /// # Arguments
    ///
    /// * `batch_context` - A mutable reference to the `ColumnBuildBatchContext`.
    /// * `index` - The index of the row in the batch context.
    pub fn new(
        batch_context: &'a mut OperatorApplyBatchContext,
        index: usize,
    ) -> Self {
        Self {
            batch_context,
            index,
        }
    }

    /// Gets a reference to the input value for the specified parameter name.
    pub fn get_input(
        &self,
        name: &str,
    ) -> Option<&AnyArc> {
        self.batch_context.get_row_input(self.index, name)
    }

    /// Get the input value for the specified parameter name, downcasting to a specific type.
    pub fn get_input_downcast<T: Any + 'static>(
        &self,
        name: &str,
    ) -> Option<&T> {
        self.batch_context
            .get_row_input_downcast::<T>(self.index, name)
    }

    /// Sets an output value for the specified parameter name.
    pub fn set_output(
        &mut self,
        name: &str,
        value: Option<AnyArc>,
    ) {
        self.batch_context.set_row_output(self.index, name, value)
    }
}

/// Represents a schema + instantiated column operator for a particular build plan.
pub struct OperationRunner {
    /// The table schema that this operator is bound to.
    pub table_schema: TableSchema,

    /// A reference to the build plan that this operator is part of.
    pub build_plan: BuildPlan,

    /// Maps from input parameter names to their slot indices in the input row.
    input_slot_map: BTreeMap<String, usize>,

    /// Maps from output parameter names to their slot indices in the output row.
    output_slot_map: BTreeMap<String, usize>,

    /// The operator that this builder wraps.
    operator: Box<dyn FirehoseOperator>,
}

impl Debug for OperationRunner {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if !f.alternate() {
            f.debug_struct("ColumnBuilder")
                .field("id", &self.build_plan.operator_id)
                .field("inputs", &self.build_plan.inputs)
                .field("outputs", &self.build_plan.outputs)
                .finish()
        } else {
            f.debug_struct("ColumnBuilder")
                .field("build_plan", &self.build_plan)
                .finish()
        }
    }
}

impl OperationRunner {
    /// Create a new `BoundPlanBuilder` by binding a `BuildPlan` to a `BimmTableSchema`.
    ///
    /// # Arguments
    ///
    /// * `table_schema` - The schema of the table to which this plan is bound.
    /// * `build_plan` - The build plan that describes the operator and its inputs/outputs.
    /// * `env` - An environment that can create the operator based on the build plan.
    ///
    /// # Returns
    ///
    /// A result containing a `BoundPlanBuilder` if successful, or an error message if the binding fails.
    #[must_use]
    pub fn new_for_plan<E>(
        table_schema: &TableSchema,
        build_plan: &BuildPlan,
        env: &E,
    ) -> Result<OperationRunner, String>
    where
        E: OpEnvironment,
    {
        let table_schema = table_schema.clone();
        let build_plan = build_plan.clone();

        let input_types = build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let output_types = build_plan
            .outputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let context = OperatorInitializationContext::new(
            table_schema.clone(),
            build_plan.clone(),
            input_types.clone(),
            output_types.clone(),
        );
        let operator = env.build(&context)?;

        let input_slot_map = build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| (pname.clone(), table_schema.column_index(cname).unwrap()))
            .collect::<BTreeMap<_, _>>();

        let output_slot_map = build_plan
            .outputs
            .iter()
            .map(|(pname, cname)| (pname.clone(), table_schema.column_index(cname).unwrap()))
            .collect::<BTreeMap<_, _>>();

        Ok(OperationRunner {
            table_schema,
            build_plan,
            input_slot_map,
            output_slot_map,
            operator,
        })
    }

    /// Get the effective batch size for the operator.
    pub fn effective_batch_size(&self) -> usize {
        self.operator.effective_batch_size()
    }

    /// Apply the operator to a batch of rows.
    ///
    /// # Arguments
    ///
    /// * `rows` - A mutable slice of `BimmRow` instances that will be processed by the operator.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error message if the operation fails.
    pub fn apply_batch(
        &self,
        rows: &mut [Row],
    ) -> Result<(), String> {
        let mut context = OperatorApplyBatchContext::new(
            self.input_slot_map
                .iter()
                .map(|(pname, &index)| {
                    (
                        pname.clone(),
                        rows.iter()
                            .map(|row| row.get_slot_arc_any(index))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        );

        self.operator.apply_batch(&mut context)?;

        for (pname, values) in context.output_map().iter() {
            for idx in 0..rows.len() {
                let row = &mut rows[idx];
                row.set_slot(self.output_slot_map[pname], values[idx].clone());
            }
        }

        Ok(())
    }
}
