use std::fmt;

#[derive(Debug)]
pub enum SiggError {
    Io(String),
    Parse(String),
    Runtime(String),
}

impl SiggError {
    pub fn io(msg: impl Into<String>) -> Self {
        SiggError::Io(msg.into())
    }
    pub fn parse(msg: impl Into<String>) -> Self {
        SiggError::Parse(msg.into())
    }
    pub fn runtime(msg: impl Into<String>) -> Self {
        SiggError::Runtime(msg.into())
    }
}

impl fmt::Display for SiggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiggError::Io(s) => write!(f, "IO error: {s}"),
            SiggError::Parse(s) => write!(f, "Parse error: {s}"),
            SiggError::Runtime(s) => write!(f, "Runtime error: {s}"),
        }
    }
}

impl std::error::Error for SiggError {}
