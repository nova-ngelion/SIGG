use crate::ast::*;
use crate::error::SiggError;
use crate::lexer::Lexer;
use crate::span::Span;
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

    #[allow(dead_code)]
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

    fn take_str(&mut self) -> Result<String, SiggError> {
        match self.cur_kind() {
            Some(Tok::Str(s)) => {
                let out = s.clone();
                self.bump()?;
                Ok(out)
            }
            _ => Err(SiggError::parse("expected string")),
        }
    }
    // ★ parse_program を名前空間対応に修正
    pub fn parse_program(&mut self) -> Result<Program, SiggError> {
        let mut imports = vec![];
        let mut namespaces = vec![];
        let mut fns = vec![];
        
        while self.cur.is_some() {
            match self.cur_kind() {
                Some(Tok::Import) => {
                    let stmt = self.parse_import_stmt()?;
                    if let Stmt::Import { path, .. } = stmt {
                        imports.push(path);
                    }
                }
                Some(Tok::Ident(name)) if name == "namespace" => {
                    self.bump()?; // consume "namespace"
                    namespaces.push(self.parse_namespace_def()?);
                }
                Some(Tok::Fn) => {
                    fns.push(self.parse_fn_def()?);
                }
                _ => return Err(SiggError::parse("expected import, namespace, or fn")),
            }
        }
        
        Ok(Program { imports, namespaces, fns })
    }

    fn parse_fn_def(&mut self) -> Result<FnDef, SiggError> {
        let start = self.cur.as_ref().unwrap().span;
        self.expect(Tok::Fn)?;
        let name = self.take_ident()?;
        self.expect(Tok::LParen)?;
        
        // パラメータリストを解析
        let mut params = Vec::new();
        if self.cur_kind() != Some(&Tok::RParen) {
            loop {
                let param_name = self.take_ident()?;
                let mut param_type = None;
                if self.cur_kind() == Some(&Tok::Colon) {
                    self.bump()?;
                    // ★ 修正: TypeAnnotation を返す
                    param_type = Some(self.parse_type_annotation()?);
                }
                params.push((param_name, param_type));
                
                if self.cur_kind() == Some(&Tok::Comma) {
                    self.bump()?;
                } else {
                    break;
                }
            }
        }
        
        self.expect(Tok::RParen)?;

        let mut ret_type = None;
        if self.cur_kind() == Some(&Tok::Arrow) {
            self.bump()?;
            // ★ 修正: TypeAnnotation を返す
            ret_type = Some(self.parse_type_annotation()?);
        }

        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        let end = self.cur.as_ref().unwrap().span;
        self.expect(Tok::RBrace)?;
        
        Ok(FnDef { 
            namespace: None,  // ★ 追加
            name, 
            params, 
            ret_type, 
            body, 
            span: Span::combine(start, end) 
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, SiggError> {
        match self.cur_kind() {
            Some(Tok::Let) => {
                self.bump()?;
                let pat = self.parse_pattern()?;
                
                let mut type_ann = None;
                if self.cur_kind() == Some(&Tok::Colon) {
                    self.bump()?;
                    type_ann = Some(self.parse_type_annotation()?);
                }
                
                self.expect(Tok::Eq)?;
                let expr = self.parse_expr_bp(0)?;
                self.expect(Tok::Semi)?;
                Ok(Stmt::Let { pat, type_ann, expr })
            }
            Some(Tok::Repeat) => self.parse_repeat_stmt(),
            Some(Tok::Transition) => self.parse_transition_stmt(),
            Some(Tok::If) => self.parse_if_stmt(), // 新しい文: if
            Some(Tok::Return) => self.parse_return_stmt(), // 新しい文: return
            Some(Tok::Import) => self.parse_import_stmt(), // 新しい文: import
            Some(Tok::Struct) => self.parse_struct_def(),
            Some(Tok::Enum) => self.parse_enum_def(),
            Some(Tok::Macro) => self.parse_macro_def(),
            Some(Tok::LBrace) => self.parse_block_stmt(),
            Some(Tok::Event) => self.parse_event_stmt(),
            Some(Tok::When)  => self.parse_when_stmt(),

            _ => {
                // Expr or Assignment
                let expr = self.parse_expr_bp(0)?;
                if self.cur_kind() == Some(&Tok::Eq) {
                    self.bump()?;
                    let rhs = self.parse_expr_bp(0)?;
                    self.expect(Tok::Semi)?;
                    Ok(Stmt::Assign { lhs: expr, expr: rhs })
                } else {
                    self.expect(Tok::Semi)?;
                    Ok(Stmt::Expr(expr))
                }
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

    fn parse_if_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::If)?;
        self.expect(Tok::LParen)?;
        let cond = self.parse_expr_bp(0)?;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let mut then_branch = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            then_branch.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        let else_branch = if self.cur_kind() == Some(&Tok::Else) {
            self.bump()?;
            if self.cur_kind() == Some(&Tok::If) {
                // else if
                let else_if_stmt = self.parse_if_stmt()?;
                Some(vec![else_if_stmt])
            } else {
                self.expect(Tok::LBrace)?;
                let mut else_stmts = vec![];
                while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
                    else_stmts.push(self.parse_stmt()?);
                }
                self.expect(Tok::RBrace)?;
                Some(else_stmts)
            }
        } else {
            None
        };
        Ok(Stmt::If { cond, then_branch, else_branch })
    }
    // ---------- expression parsing (Pratt) ----------
    fn precedence(op: &Tok) -> Option<(u8, BinOp)> {
        match op {
            Tok::Plus => Some((10, BinOp::Add)),
            Tok::Minus => Some((10, BinOp::Sub)),
            Tok::Star => Some((20, BinOp::Mul)),
            Tok::Slash => Some((20, BinOp::Div)),
            Tok::Percent => Some((20, BinOp::Mod)), // %
            Tok::Amp => Some((15, BinOp::BitAnd)), // &
            Tok::EqEq => Some((5, BinOp::Eq)), // 新しい演算子: ==
            Tok::Neq => Some((5, BinOp::Ne)), // 新しい演算子: !=
            Tok::Lt => Some((5, BinOp::Lt)), // 新しい演算子: <
            Tok::Gt => Some((5, BinOp::Gt)), // 新しい演算子: >
            Tok::Le => Some((5, BinOp::Le)), // 新しい演算子: <=
            Tok::Ge => Some((5, BinOp::Ge)), // 新しい演算子: >=
            Tok::And => Some((4, BinOp::And)), // &&
            Tok::Or => Some((3, BinOp::Or)),   // ||
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, SiggError> {
        let mut lhs = self.parse_prefix()?;
        lhs = self.parse_postfix(lhs)?;
        lhs = self.parse_bin_rhs(min_bp, lhs)?;
        Ok(lhs)
    }

    fn parse_postfix(&mut self, mut lhs: Expr) -> Result<Expr, SiggError> {
        loop {
            match self.cur_kind() {
                Some(Tok::LBracket) => {
                    self.bump()?;
                    let index = self.parse_expr_bp(0)?;
                    self.expect(Tok::RBracket)?;
                    lhs = Expr::Index { expr: Box::new(lhs), index: Box::new(index) };
                }
                Some(Tok::Dot) => {
                    self.bump()?;
                    let field = self.take_ident()?;
                    lhs = Expr::Field { expr: Box::new(lhs), name: field };
                }
                Some(Tok::LParen) => {
                    // call
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
                    lhs = Expr::Call { callee: Box::new(lhs), args };
                }
                _ => break,
            }
        }
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
            Some(Tok::If) => self.parse_if_expr(), // 新しい式: if
            Some(Tok::Pipe) => self.parse_lambda_expr(), // 新しい式: lambda
            Some(Tok::Print) => {
                self.bump()?;
                Ok(Expr::Var("print".to_string()))
            }
            Some(Tok::Ident(_)) => {
                let name = self.take_ident()?;
                // Check for Struct Init: Name { ... }
                if self.cur_kind() == Some(&Tok::LBrace) {
                    self.bump()?;
                    let mut fields = vec![];
                    if self.cur_kind() != Some(&Tok::RBrace) {
                        loop {
                            let fname = self.take_ident()?;
                            self.expect(Tok::Colon)?;
                            let fexpr = self.parse_expr_bp(0)?;
                            fields.push((fname, fexpr));
                            if self.cur_kind() == Some(&Tok::Comma) { 
                                self.bump()?; 
                            } else { 
                                break; 
                            }
                        }
                    }
                    self.expect(Tok::RBrace)?;
                    Ok(Expr::StructInit { 
                        namespace: None,  // ★ 追加
                        name, 
                        fields 
                    })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(SiggError::parse(format!("unexpected token in expr: {other:?}"))),
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, SiggError> {
        self.expect(Tok::If)?;
        self.expect(Tok::LParen)?;
        let cond = self.parse_expr_bp(0)?;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let then_branch = self.parse_expr_bp(0)?;
        self.expect(Tok::RBrace)?;
        let else_branch = if self.cur_kind() == Some(&Tok::Else) {
            self.bump()?;
            if self.cur_kind() == Some(&Tok::If) {
                // else if
                Some(Box::new(self.parse_if_expr()?))
            } else {
                self.expect(Tok::LBrace)?;
                let expr = self.parse_expr_bp(0)?;
                self.expect(Tok::RBrace)?;
                Some(Box::new(expr))
            }
        } else {
            None
        };
        Ok(Expr::If { cond: Box::new(cond), then_branch: Box::new(then_branch), else_branch })
    }
    // ★ 修正: Lambda のパースで型注釈対応
    fn parse_lambda_expr(&mut self) -> Result<Expr, SiggError> {
        self.expect(Tok::Pipe)?;
        let mut params = vec![];
        if self.cur_kind() != Some(&Tok::Pipe) {
            loop {
                let param_name = self.take_ident()?;
                let mut param_type = None;
                if self.cur_kind() == Some(&Tok::Colon) {
                    self.bump()?;
                    param_type = Some(self.parse_type_annotation()?);
                }
                params.push((param_name, param_type));
                
                if self.cur_kind() == Some(&Tok::Comma) {
                    self.bump()?;
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::Pipe)?;
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Expr::Lambda { params, body })
    }
    fn parse_return_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Return)?;
        let expr = self.parse_expr_bp(0)?;
        self.expect(Tok::Semi)?;
        Ok(Stmt::Return { expr })
    }

    fn parse_import_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Import)?;
        let path = self.take_str()?;
        let mut alias = None;
        if self.cur_kind() == Some(&Tok::As) {
            self.bump()?;
            alias = Some(self.take_ident()?);
        }
        self.expect(Tok::Semi)?;
        Ok(Stmt::Import { path, alias })
    }
    // ★ 既存の parse_struct_def を修正
    fn parse_struct_def(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Struct)?;
        let name = self.take_ident()?;
        self.expect(Tok::LBrace)?;
        let mut fields = vec![];
        while self.cur_kind() != Some(&Tok::RBrace) {
            let fname = self.take_ident()?;
            self.expect(Tok::Colon)?;
            let ftype = self.parse_type_annotation()?;
            fields.push((fname, ftype));
            if self.cur_kind() == Some(&Tok::Comma) { 
                self.bump()?; 
            } else { 
                break; 
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::StructDef { name, fields })
    }
    // ★ 既存の parse_enum_def を修正
    // ★ parse_enum_def の修正（data_type を使用）
    fn parse_enum_def(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Enum)?;
        let name = self.take_ident()?;
        self.expect(Tok::LBrace)?;
        let mut variants = vec![];
        while self.cur_kind() != Some(&Tok::RBrace) {
            let variant_name = self.take_ident()?;
            let data_type = if self.cur_kind() == Some(&Tok::LParen) {
                self.bump()?;
                let ty = self.parse_type_annotation()?;
                self.expect(Tok::RParen)?;
                Some(ty)
            } else {
                None
            };
            // ★ 修正: data -> data_type
            variants.push(EnumVariant { name: variant_name, data_type });
            if self.cur_kind() == Some(&Tok::Comma) { 
                self.bump()?; 
            } else { 
                break; 
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::EnumDef { name, variants })
    }

    fn parse_macro_def(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Macro)?;
        let name = self.take_ident()?;
        self.expect(Tok::LParen)?;
        let mut params = vec![];
        while self.cur_kind() != Some(&Tok::RParen) {
            params.push(self.take_ident()?);
            if self.cur_kind() == Some(&Tok::Comma) { self.bump()?; } else { break; }
        }
        self.expect(Tok::RParen)?;
        self.expect(Tok::Eq)?; // macro name(args) = expr;
        let body = self.parse_expr_bp(0)?;
        self.expect(Tok::Semi)?;
        Ok(Stmt::MacroDef { name, params, body })
    }

    // ★ Stmt::Let のパースで型注釈対応
    fn parse_let_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.bump()?; // consume 'let'
        let pat = self.parse_pattern()?;
        
        let mut type_ann = None;
        if self.cur_kind() == Some(&Tok::Colon) {
            self.bump()?;
            type_ann = Some(self.parse_type_annotation()?);
        }
        
        self.expect(Tok::Eq)?;
        let expr = self.parse_expr_bp(0)?;
        self.expect(Tok::Semi)?;
        
        Ok(Stmt::Let { pat, type_ann, expr })
    }
    // ★ 追加: TypeAnnotation のパース
    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, SiggError> {
        let name = self.take_ident()?;
        
        // ジェネリック型のサポート（例: Vec<T>）
        if self.cur_kind() == Some(&Tok::Lt) {
            self.bump()?;
            let mut type_params = vec![];
            type_params.push(self.parse_type_annotation()?);
            while self.cur_kind() == Some(&Tok::Comma) {
                self.bump()?;
                type_params.push(self.parse_type_annotation()?);
            }
            self.expect(Tok::Gt)?;
            // ★ 修正: Generic はタプルバリアント
            Ok(TypeAnnotation::Generic(name, type_params))
        } else {
            Ok(TypeAnnotation::Simple(name))
        }
    }

    // ★ 追加: Namespace 定義のパース
    fn parse_namespace_def(&mut self) -> Result<NamespaceDef, SiggError> {
        let name = self.take_ident()?;
        self.expect(Tok::LBrace)?;
        
        let mut items = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            match self.cur_kind() {
                Some(Tok::Fn) => {
                    let mut fn_def = self.parse_fn_def()?;
                    fn_def.namespace = Some(name.clone());
                    items.push(NamespaceItem::Function(fn_def));
                }
                Some(Tok::Struct) => {
                    let struct_def = self.parse_struct_def_for_namespace()?;
                    items.push(NamespaceItem::Struct(struct_def));
                }
                Some(Tok::Enum) => {
                    let enum_def = self.parse_enum_def_for_namespace()?;
                    items.push(NamespaceItem::Enum(enum_def));
                }
                _ => return Err(SiggError::parse("expected fn, struct, or enum in namespace")),
            }
        }
        
        self.expect(Tok::RBrace)?;
        Ok(NamespaceDef { name, items })
    }
    // ★ 修正: Struct 定義のパース（TypeAnnotation を返す）
    fn parse_struct_def_for_namespace(&mut self) -> Result<StructDef, SiggError> {
        self.expect(Tok::Struct)?;
        let name = self.take_ident()?;
        self.expect(Tok::LBrace)?;
        let mut fields = vec![];
        while self.cur_kind() != Some(&Tok::RBrace) {
            let fname = self.take_ident()?;
            self.expect(Tok::Colon)?;
            let ftype = self.parse_type_annotation()?;
            fields.push((fname, ftype));
            if self.cur_kind() == Some(&Tok::Comma) { 
                self.bump()?; 
            } else { 
                break; 
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(StructDef { name, fields })
    }

    // ★ 修正: Enum 定義のパース（EnumVariant を返す）
    fn parse_enum_def_for_namespace(&mut self) -> Result<EnumDef, SiggError> {
        self.expect(Tok::Enum)?;
        let name = self.take_ident()?;
        self.expect(Tok::LBrace)?;
        let mut variants = vec![];
        while self.cur_kind() != Some(&Tok::RBrace) {
            let variant_name = self.take_ident()?;
            let data_type = if self.cur_kind() == Some(&Tok::LParen) {
                self.bump()?;
                let ty = self.parse_type_annotation()?;
                self.expect(Tok::RParen)?;
                Some(ty)
            } else {
                None
            };
            // ★ 修正: data -> data_type
            variants.push(EnumVariant { name: variant_name, data_type });
            if self.cur_kind() == Some(&Tok::Comma) { 
                self.bump()?; 
            } else { 
                break; 
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(EnumDef { name, variants })
    }
    fn parse_block_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::Block { body })
    }
    fn parse_when_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::When)?;

        // when (cond) { ... } と when cond { ... } の両対応
        let cond = if self.cur_kind() == Some(&Tok::LParen) {
            self.bump()?;
            let c = self.parse_expr_bp(0)?;
            self.expect(Tok::RParen)?;
            c
        } else {
            self.parse_expr_bp(0)?
        };

        // 本体は { ... } の「ブロック」
        let then_branch = self.parse_braced_stmt_list()?;

        // 任意: else { ... }
        let else_branch = if self.cur_kind() == Some(&Tok::Else) {
            self.bump()?;
            Some(self.parse_braced_stmt_list()?)
        } else {
            None
        };

        Ok(Stmt::When { cond, then_branch, else_branch })
    }

    fn parse_event_stmt(&mut self) -> Result<Stmt, SiggError> {
        self.expect(Tok::Event)?;
        self.expect(Tok::LBrace)?;

        let mut arms: Vec<(Expr, Vec<Stmt>)> = vec![];
        let mut else_branch: Option<Vec<Stmt>> = None;

        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            match self.cur_kind() {
                Some(Tok::When) => {
                    self.bump()?; // consume When

                    let cond = if self.cur_kind() == Some(&Tok::LParen) {
                        self.bump()?;
                        let c = self.parse_expr_bp(0)?;
                        self.expect(Tok::RParen)?;
                        c
                    } else {
                        self.parse_expr_bp(0)?
                    };

                    let body = self.parse_braced_stmt_list()?;
                    arms.push((cond, body));
                }
                Some(Tok::Else) => {
                    self.bump()?;
                    else_branch = Some(self.parse_braced_stmt_list()?);
                }
                _ => {
                    return Err(SiggError::parse(
                        "event { ... } 内は when/else だけが書けます".to_string()
                    ));                    
                }
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Stmt::Event { arms, else_branch })
    }

    /// { stmt* } の中身だけ Vec<Stmt> として読む（Stmt::Block を返さない）
    fn parse_braced_stmt_list(&mut self) -> Result<Vec<Stmt>, SiggError> {
        self.expect(Tok::LBrace)?;
        let mut body = vec![];
        while self.cur.is_some() && self.cur_kind() != Some(&Tok::RBrace) {
            body.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(body)
    }
}
