use anyhow::bail;
use regex::Regex;
use regex_macro::regex;

/// Regular expression pattern for a valid identifier.
static IDENT_PATTERN: &str = r"^[A-Za-z_][A-Za-z0-9_]*$";

/// Returns a static regex for matching valid identifiers.
pub fn ident_regex() -> &'static Regex {
    regex!(IDENT_PATTERN)
}

/// Checks if a string is a valid identifier.
///
/// # Arguments
///
/// - `s`: The string to check.
///
/// # Returns
///
/// `true` if the string is a valid identifier, `false` otherwise.
pub fn is_ident(s: &str) -> bool {
    ident_regex().is_match(s)
}

/// Checks if a string is a valid identifier.
///
/// # Arguments
///
/// - `s`: The string to check.
///
/// # Returns
///
/// Results in:
/// - `Ok(())` if the string is a valid identifier,
/// - `Err` with a message if it is not.
pub fn check_ident(s: &str) -> anyhow::Result<()> {
    if is_ident(s) {
        Ok(())
    } else {
        bail!("Invalid identifier: '{s}'")
    }
}

/// Regular expression pattern for a valid path identifier.
static PATH_IDENT_PATTERN: &str = r"^([A-Za-z_][A-Za-z0-9_]*::)+([A-Za-z_][A-Za-z0-9_]*)$";

/// Returns a static regex for matching valid path identifiers.
pub fn path_ident_regex() -> &'static Regex {
    regex!(PATH_IDENT_PATTERN)
}

/// Parses a path identifier into its components.
///
/// # Arguments
///
/// - `ident`: The path identifier string to parse.
///
/// # Returns
///
/// - `Ok(Vec<String>)` containing the components of the path identifier,
/// - `Err(String)` if the identifier is invalid.
pub fn parse_path_ident(ident: &str) -> anyhow::Result<Vec<String>> {
    match path_ident_regex().find(ident) {
        Some(m) => {
            let parts = m.as_str().split("::").map(String::from).collect();
            Ok(parts)
        }
        None => bail!("Invalid path identifier: '{ident}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ident() {
        assert!(is_ident("a"));
        assert!(is_ident("a_9"));
        assert!(is_ident("_abc9"));

        assert!(!is_ident(""));
        assert!(!is_ident("9a"));
    }

    #[test]
    fn test_check_ident() {
        assert!(check_ident("a").is_ok());
        assert!(check_ident("a_9").is_ok());
        assert!(check_ident("_abc9").is_ok());

        assert_eq!(
            check_ident("").unwrap_err().to_string(),
            "Invalid identifier: ''"
        );
        assert_eq!(
            check_ident("9a").unwrap_err().to_string(),
            "Invalid identifier: '9a'"
        );
    }

    #[test]
    fn test_parse_path_ident() {
        assert_eq!(
            parse_path_ident("a::b").unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(
            parse_path_ident("a::b2::c").unwrap(),
            vec!["a".to_string(), "b2".to_string(), "c".to_string()]
        );

        assert_eq!(
            parse_path_ident("a").unwrap_err().to_string(),
            "Invalid path identifier: 'a'"
        );
        assert_eq!(
            parse_path_ident("9a::x").unwrap_err().to_string(),
            "Invalid path identifier: '9a::x'"
        );
    }
}
