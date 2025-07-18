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
    // This is to avoid using regex for perf and no-std compatibility.
    for (idx, c) in s.chars().enumerate() {
        if c.is_alphabetic() || c == '_' {
            continue;
        }
        if idx > 0 && c.is_numeric() {
            continue;
        }
        return false;
    }
    !s.is_empty()
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
}
