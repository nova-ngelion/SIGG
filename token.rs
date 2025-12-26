
use crate::span::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // keywords
    Fn,
    Let,
    Repeat,
    Transition,
    Print,

    // identifiers / literals
    Ident(String),
    Number(f64),
    Str(String),

    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Eq,

    // operators
    Plus,
    Minus,
    Star,
    Slash,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: Tok,
    pub span: Span,
}
