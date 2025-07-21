use crate::core::DataTypeDescription;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Defines the arity (requirement level) of a parameter
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterArity {
    /// The parameter is required and must be provided.
    #[default]
    Required,

    /// The parameter is optional and may be omitted.
    Optional,
}

/// Defines a single parameter specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// The name of the parameter.
    pub name: String,

    /// An optional description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The data type of the parameter.
    pub data_type: DataTypeDescription,

    /// The arity of the parameter, indicating whether it is required or optional.
    pub arity: ParameterArity,
}

impl ParameterSpec {
    /// Creates a new `ParameterSpec` with the given name and data type.
    ///
    /// The type `T` is used to infer the data type of the parameter.
    ///
    /// ## Parameters
    ///
    /// - `name`: The name of the parameter.
    /// - `T`: The type of the parameter, which is used to determine the data type description.
    ///
    /// ## Returns
    ///
    /// A new `ParameterSpec` instance with the specified name, data type, and required arity.
    pub fn new<T>(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            data_type: DataTypeDescription::new::<T>(),
            arity: ParameterArity::Required,
        }
    }

    /// Extends the parameter specification with a description.
    pub fn with_description(
        self,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: self.name,
            description: Some(description.into()),
            data_type: self.data_type,
            arity: self.arity,
        }
    }

    /// Extends the parameter specification with a data type.
    pub fn with_arity(
        self,
        arity: ParameterArity,
    ) -> Self {
        Self {
            name: self.name,
            description: self.description,
            data_type: self.data_type,
            arity,
        }
    }
}

/// Defines the complete input/output specification for an operator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSpec {
    /// The identifier for the operator.
    pub operator_id: Option<String>,

    /// Optional description.
    pub description: Option<String>,

    /// A list of input parameters for the operator.
    pub inputs: Vec<ParameterSpec>,

    /// A list of output parameters for the operator.
    pub outputs: Vec<ParameterSpec>,
}

impl Default for OperatorSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorSpec {
    /// Creates a new `OperatorSpec` with no inputs or outputs.
    pub const fn new() -> Self {
        Self {
            operator_id: None,
            description: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Extends the operator specification with an operator ID.
    pub fn with_operator_id(
        self,
        operator_id: &str,
    ) -> Self {
        Self {
            operator_id: Some(operator_id.into()),
            description: self.description,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Extends the operator specification with a description.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        Self {
            operator_id: self.operator_id,
            description: Some(description.to_string()),
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Extends the operator specification with an input parameter.
    pub fn with_input(
        self,
        spec: ParameterSpec,
    ) -> Self {
        if self.inputs.iter().any(|prev| prev.name == spec.name) {
            panic!("Duplicate parameter '{}'.", spec.name);
        }

        let mut inputs = self.inputs;
        inputs.push(spec);
        Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs,
            outputs: self.outputs,
        }
    }

    /// Generate an output plan suitable for `.add_build_plan_and_outputs()`.
    pub fn output_plan(&self) -> Vec<(String, DataTypeDescription, Option<String>)> {
        self.outputs
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    spec.data_type.clone(),
                    spec.description.clone(),
                )
            })
            .collect()
    }

    /// Extends the operator specification with an output parameter.
    pub fn with_output(
        self,
        spec: ParameterSpec,
    ) -> Self {
        if self.outputs.iter().any(|prev| prev.name == spec.name) {
            panic!("Duplicate parameter '{}'.", spec.name);
        }

        let mut outputs = self.outputs;
        outputs.push(spec);
        Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs: self.inputs,
            outputs,
        }
    }

    /// Validates the provided input types against the specification
    pub fn validate_inputs(
        &self,
        input_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_parameters("input", &self.inputs, input_types)
    }

    /// Validates the provided output types against the specification
    pub fn validate_outputs(
        &self,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_parameters("output", &self.outputs, output_types)
    }

    /// Validates both inputs and outputs
    pub fn validate(
        &self,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_inputs(input_types)?;
        self.validate_outputs(output_types)?;
        Ok(())
    }

    fn validate_parameters(
        &self,
        param_type: &str,
        specs: &[ParameterSpec],
        provided: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        // Check for required parameters
        let required_params: Vec<_> = specs
            .iter()
            .filter(|spec| spec.arity == ParameterArity::Required)
            .collect();

        for spec in &required_params {
            if !provided.contains_key(&spec.name) {
                return Err(format!(
                    "Missing required {} parameter '{}' of type {:?}",
                    param_type, spec.name, spec.data_type
                ));
            }

            if provided[&spec.name] != spec.data_type {
                return Err(format!(
                    "{} parameter '{}' expected type {:?}, but got {:?}",
                    param_type, spec.name, spec.data_type, provided[&spec.name]
                ));
            }
        }

        // Check for unknown parameters
        let expected_names: BTreeMap<String, &DataTypeDescription> = specs
            .iter()
            .map(|spec| (spec.name.clone(), &spec.data_type))
            .collect();

        for (name, data_type) in provided {
            match expected_names.get(name) {
                Some(expected_type) => {
                    if data_type != *expected_type {
                        return Err(format!(
                            "{param_type} parameter '{name}' expected type {expected_type:?}, but got {data_type:?}"
                        ));
                    }
                }
                None => {
                    return Err(format!(
                        "Unexpected {} parameter '{}'. Expected parameters: [{}]",
                        param_type,
                        name,
                        specs
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_spec_validation() {
        let spec = OperatorSpec::new()
            .with_input(ParameterSpec::new::<i32>("input1"))
            .with_input(ParameterSpec::new::<String>("input2").with_arity(ParameterArity::Optional))
            .with_output(ParameterSpec::new::<f64>("output"));

        let mut input_types = BTreeMap::new();
        input_types.insert("input1".to_string(), DataTypeDescription::new::<i32>());
        input_types.insert("input2".to_string(), DataTypeDescription::new::<String>());

        let mut output_types = BTreeMap::new();
        output_types.insert("output".to_string(), DataTypeDescription::new::<f64>());

        assert!(spec.validate(&input_types, &output_types).is_ok());
    }
}
