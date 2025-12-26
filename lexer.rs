
use crate::error::SiggError;
use crate::span::Span;
use crate::token::{Tok, Token};

pub struct Lexer<'a> {
    src: &'a str,
    i: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, i: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.i..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => { self.bump(); }
                Some('/') => {
                    let rest = &self.src[self.i..];
                    if rest.starts_with("//") {
                        while let Some(c2) = self.bump() {
                            if c2 == '\n' { break; }
                        }
                    } else { break; }
                }
                _ => break,
            }
        }
    }

    fn lex_ident_or_kw(&mut self, start: usize) -> Token {
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.bump();
            } else {
                break;
            }
        }
        let end = self.i;
        let s = &self.src[start..end];
        let kind = match s {
            "fn" => Tok::Fn,
            "let" => Tok::Let,
            "repeat" => Tok::Repeat,
            "transition" => Tok::Transition,
            "print" => Tok::Print,
            _ => Tok::Ident(s.to_string()),
        };
        Token { kind, span: Span::new(start, end) }
    }

    fn lex_number(&mut self, start: usize) -> Result<Token, SiggError> {
        let mut saw_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.bump();
            } else if c == '.' && !saw_dot {
                saw_dot = true;
                self.bump();
            } else {
                break;
            }
        }
        let end = self.i;
        let s = &self.src[start..end];
        let n: f64 = s.parse().map_err(|_| SiggError::parse(format!("invalid number: {s}")))?;
        Ok(Token { kind: Tok::Number(n), span: Span::new(start, end) })
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, SiggError> {
        // opening " already consumed
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                '"' => {
                    let end = self.i;
                    return Ok(Token { kind: Tok::Str(out), span: Span::new(start, end) });
                }
                '\\' => {
                    let esc = self.bump().ok_or_else(|| SiggError::parse("unterminated string"))?;
                    out.push(match esc {
                        'n' => '\n',
                        't' => '\t',
                        '"' => '"',
                        '\\' => '\\',
                        other => other,
                    });
                }
                other => out.push(other),
            }
        }
        Err(SiggError::parse("unterminated string"))
    }

    pub fn next_token(&mut self) -> Result<Option<Token>, SiggError> {
        self.skip_ws();
        let start = self.i;
        let c = match self.bump() {
            None => return Ok(None),
            Some(c) => c,
        };

        let tok = match c {
            '(' => Token { kind: Tok::LParen, span: Span::new(start, self.i) },
            ')' => Token { kind: Tok::RParen, span: Span::new(start, self.i) },
            '{' => Token { kind: Tok::LBrace, span: Span::new(start, self.i) },
            '}' => Token { kind: Tok::RBrace, span: Span::new(start, self.i) },
            ',' => Token { kind: Tok::Comma, span: Span::new(start, self.i) },
            ';' => Token { kind: Tok::Semi, span: Span::new(start, self.i) },
            '=' => Token { kind: Tok::Eq, span: Span::new(start, self.i) },
            '+' => Token { kind: Tok::Plus, span: Span::new(start, self.i) },
            '-' => Token { kind: Tok::Minus, span: Span::new(start, self.i) },
            '*' => Token { kind: Tok::Star, span: Span::new(start, self.i) },
            '/' => Token { kind: Tok::Slash, span: Span::new(start, self.i) },
            '"' => return Ok(Some(self.lex_string(start)?)),
            c if c.is_ascii_digit() => return Ok(Some(self.lex_number(start)?)),
            c if c.is_alphabetic() || c == '_' => return Ok(Some(self.lex_ident_or_kw(start))),
            other => return Err(SiggError::parse(format!("unexpected char: {other:?}"))),
        };

        Ok(Some(tok))
    }
}
