use std::fmt;
use crate::span::Span;

#[derive(Debug)]
pub enum SiggError {
    Io(String),
    Parse { msg: String, span: Option<Span> },
    Runtime { msg: String, span: Option<Span>, stack_trace: Vec<String> },
}

impl SiggError {
    pub fn io(msg: impl Into<String>) -> Self {
        SiggError::Io(msg.into())
    }
    pub fn parse(msg: impl Into<String>) -> Self {
        SiggError::Parse { msg: msg.into(), span: None }
    }
    pub fn parse_with_span(msg: impl Into<String>, span: Span) -> Self {
        SiggError::Parse { msg: msg.into(), span: Some(span) }
    }
    pub fn runtime(msg: impl Into<String>) -> Self {
        SiggError::Runtime { msg: msg.into(), span: None, stack_trace: vec![] }
    }
    pub fn runtime_with_span(msg: impl Into<String>, span: Span) -> Self {
        SiggError::Runtime { msg: msg.into(), span: Some(span), stack_trace: vec![] }
    }
    pub fn runtime_with_stack(msg: impl Into<String>, stack_trace: Vec<String>) -> Self {
        SiggError::Runtime { msg: msg.into(), span: None, stack_trace }
    }
}

impl fmt::Display for SiggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiggError::Io(s) => write!(f, "IO error: {s}"),
            SiggError::Parse { msg, span } => {
                if let Some(span) = span {
                    write!(f, "Parse error at line {}: {msg}", span.line)
                } else {
                    write!(f, "Parse error: {msg}")
                }
            }
            SiggError::Runtime { msg, span, stack_trace } => {
                if let Some(span) = span {
                    write!(f, "Runtime error at line {}: {msg}", span.line)?;
                } else {
                    write!(f, "Runtime error: {msg}")?;
                }
                if !stack_trace.is_empty() {
                    write!(f, "\nStack trace:")?;
                    for frame in stack_trace {
                        write!(f, "\n  {}", frame)?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for SiggError {}
