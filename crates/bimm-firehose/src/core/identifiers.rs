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
/// ## Arguments
///
/// - `s`: The string to check.
///
/// ## Returns
///
/// `true` if the string is a valid identifier, `false` otherwise.
pub fn is_ident(s: &str) -> bool {
    ident_regex().is_match(s)
}

/// Checks if a string is a valid identifier.
///
/// ## Arguments
///
/// - `s`: The string to check.
///
/// ## Returns
///
/// Results in:
/// - `Ok(())` if the string is a valid identifier,
/// - `Err` with a message if it is not.
pub fn check_ident(s: &str) -> Result<(), String> {
    if is_ident(s) {
        Ok(())
    } else {
        Err(format!("Invalid identifier: '{s}'"))
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
/// ## Arguments
///
/// - `ident`: The path identifier string to parse.
///
/// ## Returns
///
/// - `Ok(Vec<String>)` containing the components of the path identifier,
/// - `Err(String)` if the identifier is invalid.
pub fn parse_path_ident(ident: &str) -> Result<Vec<String>, String> {
    match path_ident_regex().find(ident) {
        Some(m) => {
            let parts = m.as_str().split("::").map(String::from).collect();
            Ok(parts)
        }
        None => Err(format!("Invalid path identifier: '{ident}'")),
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

        assert_eq!(check_ident(""), Err("Invalid identifier: ''".to_string()));
        assert_eq!(
            check_ident("9a"),
            Err("Invalid identifier: '9a'".to_string())
        );
    }

    #[test]
    fn test_parse_path_ident() {
        assert_eq!(
            parse_path_ident("a::b"),
            Ok(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            parse_path_ident("a::b2::c"),
            Ok(vec!["a".to_string(), "b2".to_string(), "c".to_string()])
        );

        assert_eq!(
            parse_path_ident("a"),
            Err("Invalid path identifier: 'a'".to_string())
        );
        assert_eq!(
            parse_path_ident("9a::x"),
            Err("Invalid path identifier: '9a::x'".to_string())
        );
    }
}
