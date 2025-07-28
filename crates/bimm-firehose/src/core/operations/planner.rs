use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A builder for constructing a call to an operator in a `BuildPlan`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationPlanner {
    /// The ID of the operator to be called.
    pub operator_id: String,

    /// A map from formal parameter names to column names for inputs.
    pub inputs: BTreeMap<String, String>,

    /// A map from formal parameter names to column names for outputs.
    pub outputs: BTreeMap<String, String>,

    /// A map from formal parameter names to their extensions, serialized as JSON.
    pub output_extensions: BTreeMap<String, serde_json::Value>,

    /// An optional configuration for the call, serialized as JSON.
    pub config: Option<serde_json::Value>,
}

impl OperationPlanner {
    /// Creates a new `CallBuilder` for the specified operator ID.
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the operator to be called.
    pub fn new(operator_id: &str) -> Self {
        Self {
            operator_id: operator_id.to_string(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            output_extensions: BTreeMap::new(),
            config: None,
        }
    }

    /// Adds an input parameter to the call builder.
    ///
    /// # Arguments
    ///
    /// * `pname` - The name of the input parameter.
    /// * `cname` - The column name in the input table.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the input parameter added.
    pub fn with_input(
        mut self,
        pname: &str,
        cname: &str,
    ) -> Self {
        if self.inputs.contains_key(pname) {
            panic!("Input parameter '{pname}' already exists.");
        }
        self.inputs.insert(pname.to_string(), cname.to_string());
        self
    }

    /// Adds an output parameter to the call builder.
    ///
    /// # Arguments
    ///
    /// * `pname` - The name of the output parameter.
    /// * `cname` - The column name in the output table.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the output parameter added.
    pub fn with_output(
        mut self,
        pname: &str,
        cname: &str,
    ) -> Self {
        if self.outputs.contains_key(pname) {
            panic!("Output parameter '{pname}' already exists.");
        }
        self.outputs.insert(pname.to_string(), cname.to_string());
        self
    }

    /// Adds an output extension to the call builder.
    pub fn with_output_extension<T>(
        mut self,
        pname: &str,
        extension: T,
    ) -> Self
    where
        T: Serialize,
    {
        if !self.outputs.contains_key(pname) {
            panic!("Output parameter '{pname}' must be defined before adding an extension.");
        }
        if self.output_extensions.contains_key(pname) {
            panic!("Output extension for '{pname}' already exists.");
        }
        self.output_extensions.insert(
            pname.to_string(),
            serde_json::to_value(extension).expect("Failed to serialize output extension"),
        );
        self
    }

    /// Adds a configuration to the call builder.
    ///
    /// The configuration is serialized to JSON and stored in the call.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to be added, which must implement `Serialize`.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the configuration added.
    pub fn with_config<T>(
        mut self,
        config: T,
    ) -> Self
    where
        T: Serialize,
    {
        self.config = Some(serde_json::to_value(config).expect("Failed to serialize config"));
        self
    }
}
