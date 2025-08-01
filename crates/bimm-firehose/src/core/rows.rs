use crate::core::ValueBox;
use crate::core::operations::signature::FirehoseOperatorSignature;
use crate::core::schema::{BuildPlan, FirehoseTableSchema};
use std::fmt::Debug;
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;

/// Represents a row in a Firehose table, containing values for each column.
pub struct FirehoseRow {
    /// The schema of the row, defining the columns and their data types.
    schema: Arc<FirehoseTableSchema>,

    /// The values in the row, where each value is an `Option<ValueBox>`.
    slots: Vec<Option<ValueBox>>,
}

impl Debug for FirehoseRow {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ValueRow ")?;
        if f.alternate() {
            self.inner_fmt_vert(f)
        } else {
            self.inner_fmt_horiz(f)
        }
    }
}

impl FirehoseRow {
    /// Creates a new `ValueRow` with the given schema and initializes all slots to `None`.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        let mut slots = Vec::with_capacity(schema.columns.len());
        slots.resize_with(schema.columns.len(), || None);
        FirehoseRow { schema, slots }
    }

    /// Vertical inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_vert(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for (idx, col) in self.schema.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }

            let name = col.name.as_str();
            let type_name = col.data_type.type_name.as_str();

            writeln!(f, "  # {type_name}")?;
            write!(f, "  {name}: ")?;

            let value = &self.slots[idx];
            match value {
                Some(v) => writeln!(f, "{v:?},"),
                None => writeln!(f, "None,"),
            }?;
        }
        write!(f, "}}")
    }

    /// Horizontal inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_horiz(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{{ ")?;
        for (idx, (name, value)) in self.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}: ")?;
            match value {
                Some(v) => write!(f, "{v:?}"),
                None => write!(f, "None"),
            }?;
        }
        write!(f, " }}")
    }
}

pub trait FirehoseRowReader {
    /// Returns the schema of the row.
    fn schema(&self) -> &Arc<FirehoseTableSchema>;

    /// Returns an iterator over the column names and their corresponding values.
    fn iter(&self) -> impl Iterator<Item = (&str, &Option<ValueBox>)>;

    /// Returns true if the row has a value for the specified column name.
    ///
    /// # Arguments
    ///
    /// - `column_name`: The name of the column to check for a value.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the row has a value for the specified column.
    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool;

    /// Gets a reference to the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// # Returns
    ///
    /// An `Option<&ValueBox>` containing a reference to the value of the specified column,
    /// or `None` if the column does not exist or has no value.
    fn get(
        &self,
        column_name: &str,
    ) -> Option<&ValueBox>;
}

pub trait FirehoseRowWriter {
    /// Sets the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to set the value for.
    /// * `value`: The value to set for the specified column, wrapped in a `ValueBox`.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    fn set(
        &mut self,
        column_name: &str,
        value: ValueBox,
    );

    /// Take the value of the column, setting it to `None`, and returning it as an `Option<ValueBox>`.
    fn take(
        &mut self,
        column_name: &str,
    ) -> Option<ValueBox>;
}

impl FirehoseRowReader for FirehoseRow {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &Option<ValueBox>)> {
        self.schema.names_iter().zip(self.slots.iter())
    }

    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index].is_some()
    }

    fn get(
        &self,
        column_name: &str,
    ) -> Option<&ValueBox> {
        let index = self.schema.column_index(column_name)?;
        self.slots.get(index)?.as_ref()
    }
}

impl FirehoseRowWriter for FirehoseRow {
    fn set(
        &mut self,
        column_name: &str,
        value: ValueBox,
    ) {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index] = Some(value);
    }

    fn take(
        &mut self,
        column_name: &str,
    ) -> Option<ValueBox> {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index].take()
    }
}

/// Represents a batch of `ValueRow`, with `FirehoseTableSchema` as its schema.
pub struct FirehoseRowBatch {
    /// The schema of the row batch.
    schema: Arc<FirehoseTableSchema>,

    /// The rows in the batch.
    pub(crate) rows: Vec<FirehoseRow>,
}

impl Debug for FirehoseRowBatch {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        writeln!(f, "ValueRowBatch {{")?;
        for (idx, row) in self.iter().enumerate() {
            write!(f, "  {idx}: ")?;
            row.inner_fmt_horiz(f)?;
            writeln!(f, ",")?;
        }
        write!(f, "}}")
    }
}

impl FirehoseRowBatch {
    /// Creates a new `ValueRowBatch` with the given schema and initializes an empty vector of rows.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        Self::new_with_size(schema, 0)
    }

    /// Creates a new `ValueRowBatch` with the given schema and row count.
    pub fn new_with_size(
        schema: Arc<FirehoseTableSchema>,
        count: usize,
    ) -> Self {
        let mut rows = Vec::with_capacity(count);
        rows.resize_with(count, || FirehoseRow::new(schema.clone()));
        FirehoseRowBatch { schema, rows }
    }

    /// The number of rows in the batch.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Checks if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns an iterator over the rows in the batch.
    pub fn iter(&self) -> impl Iterator<Item = &FirehoseRow> {
        self.rows.iter()
    }

    /// Returns a mutable iterator over the rows in the batch.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FirehoseRow> {
        self.rows.iter_mut()
    }

    pub fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    /// Adds a new row to the batch.
    ///
    /// # Arguments
    ///
    /// * `row`: The `ValueRow` to add to the batch.
    ///
    /// # Panics
    ///
    /// Panics if the row's schema does not match the batch's schema.
    pub fn add_row(
        &mut self,
        row: FirehoseRow,
    ) {
        if row.schema != self.schema {
            panic!("Cannot add row with different schema");
        }
        self.rows.push(row);
    }

    /// Creates a new row in the batch with the same schema as the batch.
    ///
    /// This initializes the row with all slots set to `None`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the newly created `ValueRow`.
    pub fn new_row(&mut self) -> &mut FirehoseRow {
        let row = FirehoseRow::new(self.schema.clone());
        self.rows.push(row);
        self.rows.last_mut().unwrap()
    }
}

impl Index<usize> for FirehoseRowBatch {
    type Output = FirehoseRow;

    /// Returns a reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for FirehoseRowBatch {
    /// Returns a mutable reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Index<Range<usize>> for FirehoseRowBatch {
    type Output = [FirehoseRow];

    /// Returns a slice of rows for the specified range.
    ///
    /// # Arguments
    ///
    /// * `range`: The range of indices to retrieve.
    fn index(
        &self,
        range: Range<usize>,
    ) -> &Self::Output {
        &self.rows[range]
    }
}

impl IndexMut<Range<usize>> for FirehoseRowBatch {
    /// Returns a mutable slice of rows for the specified range.
    ///
    /// # Arguments
    ///
    /// * `range`: The range of indices to retrieve.
    fn index_mut(
        &mut self,
        range: Range<usize>,
    ) -> &mut Self::Output {
        &mut self.rows[range]
    }
}

pub struct ParameterMap {
    build_plan: Arc<BuildPlan>,
}

impl ParameterMap {
    pub fn new(build_plan: Arc<BuildPlan>) -> Self {
        ParameterMap { build_plan }
    }

    pub fn assert_input_column(
        &self,
        parameter_name: &str,
    ) -> &str {
        self.build_plan.inputs[parameter_name].as_str()
    }

    pub fn assert_output_column(
        &self,
        parameter_name: &str,
    ) -> &str {
        self.build_plan.outputs[parameter_name].as_str()
    }
}

/// A transaction for a batch of rows processed by an operator.
pub struct FirehoseBatchTransaction<'a> {
    parameter_map: ParameterMap,

    original: &'a mut FirehoseRowBatch,

    updates: FirehoseRowBatch,

    /// The build plan that describes the operator and its inputs/outputs.
    build_plan: Arc<BuildPlan>,

    /// Signature of the operator being executed.
    signature: Arc<FirehoseOperatorSignature>,
}

impl<'a> FirehoseBatchTransaction<'a> {
    /// Creates a new `OperatorBatchTransaction` for the given row batch.
    pub fn new(
        original: &'a mut FirehoseRowBatch,
        build_plan: Arc<BuildPlan>,
        signature: Arc<FirehoseOperatorSignature>,
    ) -> FirehoseBatchTransaction<'a> {
        let updates = FirehoseRowBatch::new_with_size(original.schema().clone(), original.len());

        FirehoseBatchTransaction {
            parameter_map: ParameterMap::new(build_plan.clone()),
            original,
            updates,
            build_plan,
            signature,
        }
    }

    pub fn commit(mut self) -> anyhow::Result<()> {
        for (original, update) in self.original.iter_mut().zip(self.updates.iter_mut()) {
            for column_name in self.build_plan.outputs.values() {
                if let Some(value) = update.take(column_name) {
                    original.set(column_name, value)
                }
            }
        }
        Ok(())
    }

    /// The length of the row batch.
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Checks if the row batch is empty.
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    pub fn row_txn(
        &mut self,
        index: usize,
    ) -> FirehoseRowTransaction<'_> {
        let original = &mut self.original.rows[index];
        let updates = &mut self.updates.rows[index];
        FirehoseRowTransaction {
            parameter_map: &self.parameter_map,
            original,
            updates,
        }
    }
}

pub struct FirehoseRowTransaction<'a> {
    parameter_map: &'a ParameterMap,
    original: &'a mut FirehoseRow,
    updates: &'a mut FirehoseRow,
}

impl FirehoseRowReader for FirehoseRowTransaction<'_> {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        self.original.schema()
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &Option<ValueBox>)> {
        self.original.iter()
    }

    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool {
        let column_name = self.parameter_map.assert_input_column(column_name);
        self.original.has_column_value(column_name)
    }

    fn get(
        &self,
        column_name: &str,
    ) -> Option<&ValueBox> {
        let column_name = self.parameter_map.assert_input_column(column_name);
        self.original.get(column_name)
    }
}

impl FirehoseRowWriter for FirehoseRowTransaction<'_> {
    fn set(
        &mut self,
        column_name: &str,
        value: ValueBox,
    ) {
        let column_name = self.parameter_map.assert_output_column(column_name);
        self.updates.set(column_name, value);
    }

    fn take(
        &mut self,
        column_name: &str,
    ) -> Option<ValueBox> {
        let column_name = self.parameter_map.assert_output_column(column_name);
        self.updates.take(column_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::{ColumnSchema, FirehoseTableSchema};
    use std::sync::Arc;

    /// Ensures that `ValueRow` is `Send`, allowing it to be safely shared across threads.
    #[allow(dead_code)]
    const VALUE_ROW_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<FirehoseRow>();
    };

    #[test]
    fn test_value_row_creation() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let row = FirehoseRow::new(schema.clone());
        assert_eq!(row.slots.len(), 2);
        assert!(!row.has_column_value("foo"));
        assert!(!row.has_column_value("bar"));
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct MyStruct {
        pub value: i32,
    }

    #[test]
    fn test_row_mutation() -> anyhow::Result<()> {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<MyStruct>("bar"),
            ColumnSchema::new::<String>("qux"),
        ]));

        let mut row = FirehoseRow::new(schema.clone());
        assert!(!row.has_column_value("foo"));
        assert!(!row.has_column_value("bar"));

        row.set("foo", ValueBox::serializing(42)?);
        assert!(row.has_column_value("foo"));
        assert_eq!(row.get("foo").unwrap().deserializing::<f32>()?, 42_f32);

        let my_struct = MyStruct { value: 100 };
        row.set("bar", ValueBox::boxing(my_struct.clone()));
        assert!(row.has_column_value("bar"));

        assert_eq!(
            format!("{row:?}"),
            "ValueRow { foo: {42}, bar: [Any { .. }], qux: None }"
        );
        assert_eq!(
            format!("{row:#?}"),
            indoc::indoc! {r#"
              ValueRow {
                # i32
                foo: {42},

                # bimm_firehose::core::rows::tests::MyStruct
                bar: [Any { .. }],

                # alloc::string::String
                qux: None,
              }"#},
        );

        Ok(())
    }

    #[test]
    fn test_value_row_batch_debug() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = FirehoseRowBatch::new(schema.clone());
        let row = batch.new_row();
        row.set("foo", ValueBox::serializing(42).unwrap());

        let row = batch.new_row();
        row.set("bar", ValueBox::serializing("Hello").unwrap());

        assert_eq!(
            format!("{batch:#?}"),
            indoc::indoc! {r#"
                ValueRowBatch {
                  0: { foo: {42}, bar: None },
                  1: { foo: None, bar: {"Hello"} },
                }"#}
        );
    }
}
