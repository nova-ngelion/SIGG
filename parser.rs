
use crate::ast::*;
use crate::error::SiggError;
use crate::lexer::Lexer;
use crate::token::{Tok, Token};

pub struct Parser<'a> {
    lex: Lexer<'a>,
    cur: Option<Token>,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str) -> Self {
        let mut p = Self { lex: Lexer::new(src), cur: None };
        let _ = p.bump();
        p
    }

    fn bump(&mut self) -> Result<(), SiggError> {
        self.cur = self.lex.next_token()?;
        Ok(())
    }

    fn cur_kind(&self) -> Option<&Tok> {
        self.cur.as_ref().map(|t| &t.kind)
    }

    fn eat(&mut self, k: &Tok) -> Result<bool, SiggError> {
        if self.cur_kind() == Some(k) {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, k: Tok) -> Result<(), SiggError> {
        if self.cur_kind() == Some(&k) {
            self.bump()?;
            Ok(())
        } else {
            Err(SiggError::parse(format!(
                "expected {:?}, got {:?}",
                k,
                self.cur_kind()
            )))
        }
    }

    fn take_ident(&mut self) -> Result<String, SiggError> {
        match self.cur_kind() {
            Some(Tok::Ident(s)) => {
                let out = s.clone();
                self.bump()?;
                Ok(out)
            }
            _ => Err(SiggError::parse("expected identifier")),
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, SiggError> {
        let mut fns = vec![];
        while self.cur.is_some() {
            fns.push(self.parse_fn_def()?);
        }
        Ok(Program { fns })
    }

    fn parse_fn_def(&mut self) -> Result<FnDef, SiggError> {
        self.expect(Tok::Fn)?;
        let name = self.take_ident()?;
        self.expect(Tok::LParen)?;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(FnDef { name, body })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, SiggError> {
        match self.cur_kind() {
            Some(Tok::Let) => {
                self.bump()?;
                let pat = self.parse_pattern()?;
                self.expect(Tok::Eq)?;
                let expr = self.parse_expr_bp(0)?;
                self.expect(Tok::Semi)?;
                Ok(Stmt::Let { pat, expr })
            }
            Some(Tok::Repeat) => self.parse_repeat_stmt(),
            Some(Tok::Transition) => self.parse_transition_stmt(),
            Some(Tok::Ident(_)) => {
                // assignment or expr?
                let _save = self.cur.clone();
                let name = self.take_ident()?;
                if self.cur_kind() == Some(&Tok::Eq) {
                    self.bump()?;
                    let expr = self.parse_expr_bp(0)?;
                    self.expect(Tok::Semi)?;
                    Ok(Stmt::Assign { name, expr })
                } else {
                    // rollback: treat as expr starting with Var(name)
                    // (cheap: build Expr::Var then parse rest as binary/call)
                    // We'll reconstruct by parsing postfix on this "prefix" node.
                    let mut lhs = Expr::Var(name);
                    lhs = self.parse_postfix(lhs)?;
                    lhs = self.parse_bin_rhs(0, lhs)?;
                    self.expect(Tok::Semi)?;
                    Ok(Stmt::Expr(lhs))
                }
            }
            _ => {
                let e = self.parse_expr_bp(0)?;
                self.expect(Tok::Semi)?;
                Ok(Stmt::Expr(e))
            }
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern, SiggError> {
        match self.cur_kind() {
            Some(Tok::Ident(s)) if s == "_" => {
                self.bump()?;
                Ok(Pattern::Wildcard)
            }
            Some(Tok::Ident(_)) => Ok(Pattern::Name(self.take_ident()?)),
            Some(Tok::LParen) => {
                self.bump()?; // (
                let mut items = vec![];
                if self.cur_kind() != Some(&Tok::RParen) {
                    items.push(self.parse_pattern()?);
                    while self.cur_kind() == Some(&Tok::Comma) {
                        self.bump()?;
                        items.push(self.parse_pattern()?);
                    }
                }
                self.expect(Tok::RParen)?;
                Ok(Pattern::Tuple(items))
            }
            other => Err(SiggError::parse(format!("unexpected token in pattern: {other:?}"))),
        }
    }

    fn parse_repeat_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Repeat)?;
        self.expect(Tok::LParen)?;
        let count = self.parse_expr_bp(0)?;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::Repeat { count, body })
    }

    fn parse_transition_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Transition)?;
        // sugar: transition(steps){...}
        let count = if self.cur_kind() == Some(&Tok::LParen) {
            self.bump()?;
            let e = self.parse_expr_bp(0)?;
            self.expect(Tok::RParen)?;
            e
        } else {
            Expr::Number(1.0)
        };
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::Transition { count, body })
    }

    // ---------- expression parsing (Pratt) ----------
    fn precedence(op: &Tok) -> Option<(u8, BinOp)> {
        match op {
            Tok::Plus => Some((10, BinOp::Add)),
            Tok::Minus => Some((10, BinOp::Sub)),
            Tok::Star => Some((20, BinOp::Mul)),
            Tok::Slash => Some((20, BinOp::Div)),
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, SiggError> {
        let mut lhs = self.parse_prefix()?;
        lhs = self.parse_postfix(lhs)?;
        lhs = self.parse_bin_rhs(min_bp, lhs)?;
        Ok(lhs)
    }

    fn parse_bin_rhs(&mut self, min_bp: u8, mut lhs: Expr) -> Result<Expr, SiggError> {
        loop {
            let Some(op_tok) = self.cur_kind() else { break; };
            let Some((bp, op)) = Self::precedence(op_tok) else { break; };
            if bp < min_bp { break; }
            self.bump()?; // consume op
            let mut rhs = self.parse_prefix()?;
            rhs = self.parse_postfix(rhs)?;
            // right binding power: bp+1 for left-assoc
            rhs = self.parse_bin_rhs(bp + 1, rhs)?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, SiggError> {
        match self.cur_kind() {
            Some(Tok::Number(n)) => {
                let v = *n;
                self.bump()?;
                Ok(Expr::Number(v))
            }
            Some(Tok::Str(s)) => {
                let v = s.clone();
                self.bump()?;
                Ok(Expr::Str(v))
            }
            Some(Tok::Minus) => {
                self.bump()?;
                let rhs = self.parse_expr_bp(100)?;
                Ok(Expr::Unary { op: UnOp::Neg, rhs: Box::new(rhs) })
            }
            Some(Tok::LParen) => {
                self.bump()?;
                // tuple or grouped
                if self.cur_kind() == Some(&Tok::RParen) {
                    self.bump()?;
                    return Ok(Expr::Tuple(vec![]));
                }
                let first = self.parse_expr_bp(0)?;
                if self.cur_kind() == Some(&Tok::Comma) {
                    self.bump()?;
                    let mut items = vec![first];
                    items.push(self.parse_expr_bp(0)?);
                    while self.cur_kind() == Some(&Tok::Comma) {
                        self.bump()?;
                        items.push(self.parse_expr_bp(0)?);
                    }
                    self.expect(Tok::RParen)?;
                    Ok(Expr::Tuple(items))
                } else {
                    self.expect(Tok::RParen)?;
                    Ok(Expr::Group(Box::new(first)))
                }
            }
            Some(Tok::Print) => {
                // print(expr) as normal call "print"
                self.bump()?;
                self.expect(Tok::LParen)?;
                let mut args = vec![];
                if self.cur_kind() != Some(&Tok::RParen) {
                    args.push(self.parse_expr_bp(0)?);
                    while self.cur_kind() == Some(&Tok::Comma) {
                        self.bump()?;
                        args.push(self.parse_expr_bp(0)?);
                    }
                }
                self.expect(Tok::RParen)?;
                Ok(Expr::Call { callee: "print".to_string(), args })
            }
            Some(Tok::Ident(_)) => {
                let name = self.take_ident()?;
                Ok(Expr::Var(name))
            }
            other => Err(SiggError::parse(format!("unexpected token in expr: {other:?}"))),
        }
    }

    fn parse_postfix(&mut self, mut lhs: Expr) -> Result<Expr, SiggError> {
        loop {
            if self.cur_kind() == Some(&Tok::LParen) {
                // call
                let callee = match lhs {
                    Expr::Var(ref s) => s.clone(),
                    _ => return Err(SiggError::parse("call target must be identifier")),
                };
                self.bump()?; // (
                let mut args = vec![];
                if self.cur_kind() != Some(&Tok::RParen) {
                    args.push(self.parse_expr_bp(0)?);
                    while self.cur_kind() == Some(&Tok::Comma) {
                        self.bump()?;
                        args.push(self.parse_expr_bp(0)?);
                    }
                }
                self.expect(Tok::RParen)?;
                lhs = Expr::Call { callee, args };
                continue;
            }
            break;
        }
        Ok(lhs)
    }
}
