use crate::span::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // keywords
    Fn,
    Let,
    Repeat,
    Transition,
    Print,
    If, // 新しいキーワード: if
    Else, // 新しいキーワード: else
    Return, // 新しいキーワード: return
    Import, // 新しいキーワード: import
    Struct, // struct
    Enum,   // enum
    Macro,  // macro
    As,     // as
    Event, // event
    When,  // when
    
    // identifiers / literals
    Ident(String),
    Number(f64),
    Str(String),

    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket, // [
    RBracket, // ]
    Comma,
    Semi,
    Colon, // :
    Dot,   // .
    Eq,
    EqEq, // 新しいトークン: ==
    Neq, // 新しいトークン: !=
    Lt, // 新しいトークン: <
    Gt, // 新しいトークン: >
    Le, // 新しいトークン: <=
    Ge, // 新しいトークン: >=
    Arrow, // ->

    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp, // 新しいトークン: &
    Pipe, // 新しいトークン: |
    And, // 新しいトークン: &&
    Or,  // 新しいトークン: ||
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: Tok,
    pub span: Span,
}
