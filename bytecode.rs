use std::collections::HashMap;
use std::fs;

use crate::ast::*;
use crate::error::SiggError;
use crate::value::Value;
use crate::span::Span;
// use crate::builtins;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId(pub u16);

#[derive(Clone, Debug)]
pub enum Op {
    Const(u16),
    LoadLocal(u16),
    StoreLocal(u16),

    // dynamic ops
    Add,
    Sub,
    Mul,
    Div,
    Mod, // %
    BitAnd, // &
    Not,
    Eq, // 新しい演算子: ==
    Ne, // 新しい演算子: !=
    Lt, // 新しい演算子: <
    Gt, // 新しい演算子: >
    Le, // 新しい演算子: <=
    Ge, // 新しい演算子: >=
    And, // 新しい演算子: &&
    Or,  // 新しい演算子: ||
    Neg,
    Visualize,
    StoreElement, // スタックから [Tensor, Index, Value] をポップして代入するs
    StoreIndex,

    MakeTuple(u16),
    UnpackTuple(u16), // pop tuple -> push elements (0..n-1)

    CallId { id: FnId, argc: u16 },
    CallLambda { argc: u16 }, // lambda call
    CallDynamic { argc: u16 }, // dynamic call
    Pop,
    Return,

    RepeatInit(u16),
    RepeatCheckJump { slot: u16, off: i32 },
    RepeatDecJump { slot: u16, off: i32 },

    JumpIfFalse { off: i32 }, // 新しい命令: if用
    Jump { off: i32 }, // 新しい命令: if用

    PushScope, // ブロックスコープ開始
    PopScope,  // ブロックスコープ終了

    Index, // generic index: stack [.., expr, index] -> [.., value]

    MakeLambda(FnId), // lambda作成

    DefGlobal(usize), // グローバル変数を定義 (let a = 10;)
    GetGlobal(usize), // グローバル変数を取得 (print(a);)
    SetGlobal(usize), // グローバル変数を更新 (a = 20;)
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub ops: Vec<Op>,
    pub consts: Vec<Value>,
    pub local_count: u16,
}

#[derive(Clone, Debug)]
pub struct CompiledProgram {
    pub fns: HashMap<FnId, (Chunk, Span)>,
    pub table: FnTable,
    pub main_id: FnId,
    pub namespaces: HashMap<String, NamespaceInfo>,
    pub type_info: HashMap<FnId, FunctionTypeInfo>,
}

#[derive(Clone, Debug)]
pub struct NamespaceInfo {
    pub name: String,
    pub functions: HashMap<String, FnId>,
    pub structs: HashMap<String, StructDef>,
    pub enums: HashMap<String, EnumDef>,
}

#[derive(Clone, Debug)]
pub struct FunctionTypeInfo {
    pub params: Vec<TypeAnnotation>,
    pub ret_type: Option<TypeAnnotation>,
}

impl Chunk {
    pub fn new() -> Self {
        Self { ops: vec![], consts: vec![], local_count: 0 }
    }
    pub fn add_const(&mut self, v: Value) -> u16 {
        self.consts.push(v);
        (self.consts.len() - 1) as u16
    }
    pub fn emit(&mut self, op: Op) { self.ops.push(op); }
}

#[derive(Clone, Debug)]
pub struct FnTable {
    pub id_to_name: Vec<String>,
    pub name_to_id: HashMap<String, FnId>,
}
impl FnTable {
    pub fn new() -> Self {
        Self { id_to_name: vec![], name_to_id: HashMap::new() }
    }
    pub fn intern(&mut self, name: &str) -> FnId {
        if let Some(id) = self.name_to_id.get(name) { return *id; }
        let id = FnId(self.id_to_name.len() as u16);
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }
    pub fn name(&self, id: FnId) -> &str { &self.id_to_name[id.0 as usize] }
    pub fn default() -> Self {
        Self::new()
    }
}

pub struct FnCompiler<'a> {
    pub locals: HashMap<String, u16>,
    pub chunk: Chunk,
    pub table: &'a mut FnTable,
    pub next_repeat_slot: u16,
    pub lambda_fns: Vec<(FnId, Chunk, Span)>,
    pub scope_depth: usize, // ★ 追加
}

impl<'a> FnCompiler<'a> {
    pub fn new(table: &'a mut FnTable) -> Self {
        Self {
            locals: HashMap::new(),
            chunk: Chunk::new(),
            table,
            next_repeat_slot: 0,
            lambda_fns: vec![],
            scope_depth: 0,
        }
    }
    pub fn resolve_local(&self, name: &str) -> Option<u16> {
        self.locals.get(name).copied()
    }

    fn alloc_repeat_slot(&mut self) -> u16 {
        let s = self.next_repeat_slot;
        self.next_repeat_slot = self.next_repeat_slot.wrapping_add(1);
        s
    }

    fn local_index(&mut self, name: &str) -> u16 {
        if let Some(&i) = self.locals.get(name) { return i; }
        let i = self.locals.len() as u16;
        self.locals.insert(name.to_string(), i);
        if i + 1 > self.chunk.local_count { self.chunk.local_count = i + 1; }
        i
    }

    // ★ compile_store_pattern の修正（重複パターンを削除）
    fn compile_store_pattern(&mut self, p: &Pattern) -> Result<(), SiggError> {
        match p {
            Pattern::Wildcard => {
                self.chunk.emit(Op::Pop);
                Ok(())
            }
            Pattern::Name(name) => {
                let idx = self.local_index(name);
                self.chunk.emit(Op::StoreLocal(idx));
                Ok(())
            }
            Pattern::Tuple(items) => {
                let n = items.len() as u16;
                self.chunk.emit(Op::UnpackTuple(n));
                // after UnpackTuple: stack has [.., v0, v1, .., v_{n-1}]
                // store patterns from last to first (stack pop order)
                for it in items.iter().rev() {
                    self.compile_store_pattern(it)?;
                }
                Ok(())
            }
            Pattern::Struct { name: _, fields } => {
                // 構造体のアンパック
                // TODO: Op::UnpackStruct を実装
                for (_field_name, field_pat) in fields {
                    // 簡易実装: フィールドアクセス命令が必要
                    self.compile_store_pattern(field_pat)?;
                }
                Ok(())
            }
        }
    }

    pub fn compile_stmt(&mut self, s: &Stmt) -> Result<(), SiggError> {
        match s {
            // ★ 修正: type_ann フィールドを追加
            Stmt::Let { pat, type_ann, expr } => {
                let _ = type_ann;
                // 1. まず右辺の式を評価してスタックに積む
                self.compile_expr(expr)?;

                // 2. スコープによって処理を分岐
                if self.scope_depth > 0 {
                    // --- ローカル変数の場合 ---
                    // 既存のロジック (compile_store_pattern が StoreLocal を発行する)
                    self.compile_store_pattern(pat)?;
                } else {
                    // --- グローバル変数の場合 ---
                    // パターンが単純な変数名であることを確認
                    if let Pattern::Name(name) = pat {
                        // 変数名をテーブルに登録してIDを取得
                        // (FnTableの実装に合わせて intern か add_string を使ってください)
                        let name_idx = self.table.intern(name); 
                        self.chunk.emit(Op::DefGlobal(name_idx.0 as usize));
                    } else {
                        // 複雑なパターン（タプル分解など）はグローバルでは一旦非対応にするか、個別に実装が必要
                        return Err(SiggError::runtime("Global destructing definitions not supported yet"));
                    }
                }
                Ok(())
            }
            Stmt::Block { body } => {
                self.scope_depth += 1; // ★ スコープイン
                self.chunk.emit(Op::PushScope);
                for st in body { self.compile_stmt(st)?; }
                self.chunk.emit(Op::PopScope);
                self.scope_depth -= 1; // ★ スコープアウト
                Ok(())
            }
            Stmt::Assign { lhs, expr: rhs_expr } => {
                match lhs {
                    Expr::Index { expr: base, index } => {
                        self.compile_expr(base)?;
                        self.compile_expr(index)?;
                        self.compile_expr(rhs_expr)?;
                        
                        self.chunk.emit(Op::StoreIndex); 
            
                        // 重要：リスト（値型）の場合、StoreIndexがスタックに積んだ「新しいリスト」を元の変数に書き戻す
                        if let Expr::Var(name) = &**base {
                            // 変数名からスロット番号を解決する（メソッド名は既存のコードに合わせてください）
                            if let Some(slot) = self.resolve_local(name) {
                                self.chunk.emit(Op::StoreLocal(slot as u16));
                            } else {
                                let name_idx = self.table.intern(name);
                                self.chunk.emit(Op::SetGlobal(name_idx.0 as usize));
                            }
                        }
                        Ok(())
                    }
                    Expr::Var(name) => {
                        self.compile_expr(rhs_expr)?; // 値をスタックへ

                        // 1. ローカルにあれば StoreLocal
                        if let Some(slot) = self.resolve_local(name) {
                            self.chunk.emit(Op::StoreLocal(slot as u16));
                        } else {
                            // 2. なければグローバルとして SetGlobal
                            let name_idx = self.table.intern(name);
                            self.chunk.emit(Op::SetGlobal(name_idx.0 as usize));
                        }
                        Ok(())
                    }
                    _ => Err(SiggError::runtime(format!("Invalid assignment target: {:?}", lhs)))
                }
            }
            Stmt::Expr(e) => {
                self.compile_expr(e)?;
                self.chunk.emit(Op::Pop);
                Ok(())
            }
            Stmt::Repeat { count, body } => self.compile_repeat_like(count, body),
            Stmt::Transition { count, body } => self.compile_repeat_like(count, body),
            Stmt::If { cond, then_branch, else_branch } => {
                self.compile_expr(cond)?;
                let jump_if_false_pos = self.chunk.ops.len();
                self.chunk.emit(Op::JumpIfFalse { off: 0 });
                self.chunk.emit(Op::PushScope);
                for st in then_branch { self.compile_stmt(st)?; }
                self.chunk.emit(Op::PopScope);
                let jump_to_end_pos = if else_branch.is_some() {
                    let pos = self.chunk.ops.len();
                    self.chunk.emit(Op::Jump { off: 0 });
                    pos
                } else { 0 };
                let else_start = self.chunk.ops.len();
                if let Some(else_stmts) = else_branch {
                    self.chunk.emit(Op::PushScope);
                    for st in else_stmts { self.compile_stmt(st)?; }
                    self.chunk.emit(Op::PopScope);
                }
                let end = self.chunk.ops.len();
                if let Op::JumpIfFalse { off } = &mut self.chunk.ops[jump_if_false_pos] {
                    *off = (else_start as i32) - (jump_if_false_pos as i32 + 1);
                }
                if else_branch.is_some() {
                    if let Op::Jump { off } = &mut self.chunk.ops[jump_to_end_pos] {
                        *off = (end as i32) - (jump_to_end_pos as i32 + 1);
                    }
                }
                Ok(())
            }
            Stmt::Return { expr } => {
                self.compile_expr(expr)?;
                self.chunk.emit(Op::Return);
                Ok(())
            }
            Stmt::Import { .. } => Ok(()), // importはコンパイル時に処理済み
            Stmt::StructDef { .. } => Ok(()), // 型定義はコンパイル時に処理済み
            Stmt::EnumDef { .. } => Ok(()), // 型定義はコンパイル時に処理済み
            Stmt::MacroDef { .. } => Ok(()), // マクロはコンパイル時に処理済み
            Stmt::NamespaceDef(_) => Ok(()), // 名前空間はコンパイル時に処理済み
            Stmt::When { cond, then_branch, else_branch } => {
                // 実質 If と同じ
                self.compile_expr(cond)?;
                let jif_pos = self.chunk.ops.len();
                self.chunk.emit(Op::JumpIfFalse { off: 0 });

                self.chunk.emit(Op::PushScope);
                for st in then_branch { self.compile_stmt(st)?; }
                self.chunk.emit(Op::PopScope);

                let jend_pos = if else_branch.is_some() {
                    let p = self.chunk.ops.len();
                    self.chunk.emit(Op::Jump { off: 0 });
                    p
                } else { 0 };

                let else_start = self.chunk.ops.len();
                if let Some(else_stmts) = else_branch {
                    self.chunk.emit(Op::PushScope);
                    for st in else_stmts { self.compile_stmt(st)?; }
                    self.chunk.emit(Op::PopScope);
                }
                let end = self.chunk.ops.len();

                if let Op::JumpIfFalse { off } = &mut self.chunk.ops[jif_pos] {
                    *off = (else_start as i32) - (jif_pos as i32 + 1);
                }
                if else_branch.is_some() {
                    if let Op::Jump { off } = &mut self.chunk.ops[jend_pos] {
                        *off = (end as i32) - (jend_pos as i32 + 1);
                    }
                }
                Ok(())
            }

            Stmt::Event { arms, else_branch } => {
                // event は “else-if チェーン（first match）”
                // 実装: 각 arm を順に
                //  cond -> false なら次へ, true なら body 実行して end へジャンプ
                let mut jump_to_end_positions: Vec<usize> = vec![];
                let mut pending_jif_positions: Vec<usize> = vec![];

                for (cond, body) in arms {
                    self.compile_expr(cond)?;
                    let jif_pos = self.chunk.ops.len();
                    self.chunk.emit(Op::JumpIfFalse { off: 0 });
                    pending_jif_positions.push(jif_pos);

                    self.chunk.emit(Op::PushScope);
                    for st in body { self.compile_stmt(st)?; }
                    self.chunk.emit(Op::PopScope);

                    let jend_pos = self.chunk.ops.len();
                    self.chunk.emit(Op::Jump { off: 0 });
                    jump_to_end_positions.push(jend_pos);

                    // 次の arm の開始位置へ飛ぶように jif をパッチ
                    let next_start = self.chunk.ops.len();
                    if let Op::JumpIfFalse { off } = &mut self.chunk.ops[jif_pos] {
                        *off = (next_start as i32) - (jif_pos as i32 + 1);
                    }
                }

                // どれも当たらなかった場合（ここに落ちる）
                let _else_start = self.chunk.ops.len();
                if let Some(else_stmts) = else_branch {
                    self.chunk.emit(Op::PushScope);
                    for st in else_stmts { self.compile_stmt(st)?; }
                    self.chunk.emit(Op::PopScope);
                }

                let end = self.chunk.ops.len();
                for jend_pos in jump_to_end_positions {
                    if let Op::Jump { off } = &mut self.chunk.ops[jend_pos] {
                        *off = (end as i32) - (jend_pos as i32 + 1);
                    }
                }
                Ok(())
            }
        }
    }

    fn compile_repeat_like(&mut self, count: &Expr, body: &Vec<Stmt>) -> Result<(), SiggError> {
        self.compile_expr(count)?;
        let slot = self.alloc_repeat_slot();
        self.chunk.emit(Op::RepeatInit(slot));

        let check_ip = self.chunk.ops.len();
        let j_pos = self.chunk.ops.len();
        self.chunk.emit(Op::RepeatCheckJump { slot, off: 0 });

        self.chunk.emit(Op::PushScope); // repeat body scope
        for st in body { self.compile_stmt(st)?; }
        self.chunk.emit(Op::PopScope);

        let jmp_pos = self.chunk.ops.len();
        let back = check_ip as i32 - (jmp_pos as i32 + 1);
        self.chunk.emit(Op::RepeatDecJump { slot, off: back });

        let end_ip = self.chunk.ops.len();
        let fwd = end_ip as i32 - (j_pos as i32 + 1);
        if let Op::RepeatCheckJump { slot: s2, off } = &mut self.chunk.ops[j_pos] {
            let _ = *s2;
            *off = fwd;
        }
        Ok(())
    }

    fn compile_expr(&mut self, e: &Expr) -> Result<(), SiggError> {
        match e {
            Expr::Var(name) => {
                // 1) ローカルなら読む
                if let Some(idx) = self.lookup_local(name) {
                    self.chunk.emit(Op::LoadLocal(idx));
                    return Ok(());
                }
                // 2) 関数名なら関数値として参照（= Lambda(FnId) を積む）
                // if let Some(id) = self.table.name_to_id.get(name).copied() {
                //     let ci = self.chunk.add_const(Value::Lambda(id));
                //     self.chunk.emit(Op::Const(ci));
                //     return Ok(());
                // }
                // 3) 未定義はエラー（Unit を作らない）
                let name_idx = self.table.intern(name);
                self.chunk.emit(Op::GetGlobal(name_idx.0 as usize));
                return Ok(());
            }
            Expr::NamespacedVar { namespace, name } => {
                let qualified = format!("{}::{}", namespace, name);
                // 1) ローカルに同名があればローカル（特殊ケース）
                if let Some(idx) = self.lookup_local(&qualified) {
                    self.chunk.emit(Op::LoadLocal(idx));
                    return Ok(());
                }
                // 2) 名前空間関数なら関数値
                if let Some(id) = self.table.name_to_id.get(&qualified).copied() {
                    let ci = self.chunk.add_const(Value::Lambda(id));
                    self.chunk.emit(Op::Const(ci));
                    return Ok(());
                }
                Err(SiggError::runtime(format!("undefined namespaced value: {}", qualified)))
            }
            Expr::Number(n) => {
                let ci = self.chunk.add_const(Value::Number(*n));
                self.chunk.emit(Op::Const(ci));
                Ok(()) // ★ 追加
            }
            Expr::Str(s) => {
                let ci = self.chunk.add_const(Value::Str(s.clone()));
                self.chunk.emit(Op::Const(ci));
                Ok(()) // ★ 追加
            }
            Expr::Group(inner) => {
                self.compile_expr(inner)?;
                Ok(()) // ★ 追加
            }
            Expr::Unary { op, rhs } => {
                self.compile_expr(rhs)?;
                match op {
                    UnOp::Neg => self.chunk.emit(Op::Neg),
                    UnOp::Not => {
                        self.chunk.emit(Op::Not);
                    }
                }
                Ok(()) // ★ 追加
            }
            Expr::Binary { op, lhs, rhs } => {
                match op {
                    BinOp::And => {
                        // (truthy(lhs) && truthy(rhs)) を Bool で返し、rhs は短絡
                        // 1) lhs
                        self.compile_expr(lhs)?;
                        let j_lhs_false = self.chunk.ops.len();
                        self.chunk.emit(Op::JumpIfFalse { off: 0 }); // pop(lhs); falseなら飛ぶ
            
                        // 2) rhs
                        self.compile_expr(rhs)?;
                        let j_rhs_false = self.chunk.ops.len();
                        self.chunk.emit(Op::JumpIfFalse { off: 0 }); // pop(rhs); falseなら飛ぶ
            
                        // 3) 両方 truthy -> true
                        self.emit_bool(true);
                        let j_end = self.chunk.ops.len();
                        self.chunk.emit(Op::Jump { off: 0 });
            
                        // rhs false ラベル
                        let rhs_false_ip = self.chunk.ops.len();
                        self.emit_bool(false);
                        let j_end2 = self.chunk.ops.len();
                        self.chunk.emit(Op::Jump { off: 0 });
            
                        // lhs false ラベル
                        let lhs_false_ip = self.chunk.ops.len();
                        self.emit_bool(false);
            
                        // end ラベル
                        let end_ip = self.chunk.ops.len();
            
                        // パッチ
                        if let Op::JumpIfFalse { off } = &mut self.chunk.ops[j_lhs_false] {
                            *off = (lhs_false_ip as i32) - (j_lhs_false as i32 + 1);
                        }
                        if let Op::JumpIfFalse { off } = &mut self.chunk.ops[j_rhs_false] {
                            *off = (rhs_false_ip as i32) - (j_rhs_false as i32 + 1);
                        }
                        if let Op::Jump { off } = &mut self.chunk.ops[j_end] {
                            *off = (end_ip as i32) - (j_end as i32 + 1);
                        }
                        if let Op::Jump { off } = &mut self.chunk.ops[j_end2] {
                            *off = (end_ip as i32) - (j_end2 as i32 + 1);
                        }
                        Ok(())
                    }
            
                    BinOp::Or => {
                        // (truthy(lhs) || truthy(rhs)) を Bool で返し、rhs は短絡
                        // 1) lhs
                        self.compile_expr(lhs)?;
                        let j_lhs_false = self.chunk.ops.len();
                        self.chunk.emit(Op::JumpIfFalse { off: 0 }); // pop(lhs); falseなら rhs 評価へ
            
                        // lhs が truthy -> その時点で true（rhs 評価しない）
                        self.emit_bool(true);
                        let j_end = self.chunk.ops.len();
                        self.chunk.emit(Op::Jump { off: 0 });
            
                        // rhs 評価ラベル
                        let rhs_eval_ip = self.chunk.ops.len();
                        self.compile_expr(rhs)?;
                        let j_rhs_false = self.chunk.ops.len();
                        self.chunk.emit(Op::JumpIfFalse { off: 0 }); // pop(rhs); falseなら false
            
                        // rhs truthy -> true
                        self.emit_bool(true);
                        let j_end2 = self.chunk.ops.len();
                        self.chunk.emit(Op::Jump { off: 0 });
            
                        // rhs false ラベル
                        let rhs_false_ip = self.chunk.ops.len();
                        self.emit_bool(false);
            
                        // end
                        let end_ip = self.chunk.ops.len();
            
                        // パッチ
                        if let Op::JumpIfFalse { off } = &mut self.chunk.ops[j_lhs_false] {
                            *off = (rhs_eval_ip as i32) - (j_lhs_false as i32 + 1);
                        }
                        if let Op::JumpIfFalse { off } = &mut self.chunk.ops[j_rhs_false] {
                            *off = (rhs_false_ip as i32) - (j_rhs_false as i32 + 1);
                        }
                        if let Op::Jump { off } = &mut self.chunk.ops[j_end] {
                            *off = (end_ip as i32) - (j_end as i32 + 1);
                        }
                        if let Op::Jump { off } = &mut self.chunk.ops[j_end2] {
                            *off = (end_ip as i32) - (j_end2 as i32 + 1);
                        }
                        Ok(())
                    }
                    // それ以外は従来通り（両辺評価）
                    _ => {
                        self.compile_expr(lhs)?;
                        self.compile_expr(rhs)?;
                        match op {
                            BinOp::Add => self.chunk.emit(Op::Add),
                            BinOp::Sub => self.chunk.emit(Op::Sub),
                            BinOp::Mul => self.chunk.emit(Op::Mul),
                            BinOp::Div => self.chunk.emit(Op::Div),
                            BinOp::Mod => self.chunk.emit(Op::Mod),
                            BinOp::BitAnd => self.chunk.emit(Op::BitAnd),
                            BinOp::Eq => self.chunk.emit(Op::Eq),
                            BinOp::Ne => self.chunk.emit(Op::Ne),
                            BinOp::Lt => self.chunk.emit(Op::Lt),
                            BinOp::Gt => self.chunk.emit(Op::Gt),
                            BinOp::Le => self.chunk.emit(Op::Le),
                            BinOp::Ge => self.chunk.emit(Op::Ge),
                            BinOp::And => self.chunk.emit(Op::And),
                            BinOp::Or => self.chunk.emit(Op::Or),
                        }
                        Ok(()) // ★ 追加
                    }
                }
            }
            Expr::Tuple(items) => {
                for it in items { self.compile_expr(it)?; }
                self.chunk.emit(Op::MakeTuple(items.len() as u16));
                Ok(()) // ★ 追加
            }
            Expr::Call { callee, args } => {
                match &**callee {
                    Expr::Var(name) => {
                        if is_builtin(name) {
                            for a in args { self.compile_expr(a)?; }
                            let id = self.table.intern(name);
                            self.chunk.emit(Op::CallId { id, argc: args.len() as u16 });
                            return Ok(());
                        }
            
                        if let Some(fid) = self.table.name_to_id.get(name).copied() {
                            let ci = self.chunk.add_const(Value::Lambda(fid));
                            self.chunk.emit(Op::Const(ci));
                            for a in args { self.compile_expr(a)?; }
                            self.chunk.emit(Op::CallLambda { argc: args.len() as u16 });
                            return Ok(());
                        }
            
                        // fallthrough: 値としての関数（変数に入ってる等）
                        self.compile_expr(&**callee)?;
                        for a in args { self.compile_expr(a)?; }
                        self.chunk.emit(Op::CallDynamic { argc: args.len() as u16 });
                        Ok(())
                    }
                    _ => {
                        // callee が式の場合は常に動的呼び出し
                        self.compile_expr(&**callee)?;
                        for a in args { self.compile_expr(a)?; }
                        self.chunk.emit(Op::CallDynamic { argc: args.len() as u16 });
                        Ok(())
                    }
                }
            }
            
            Expr::If { cond, then_branch, else_branch } => {
                self.compile_expr(cond)?;
                let jump_if_false_pos = self.chunk.ops.len();
                self.chunk.emit(Op::JumpIfFalse { off: 0 });
                self.compile_expr(then_branch)?;
                let jump_to_end_pos = if else_branch.is_some() {
                    let pos = self.chunk.ops.len();
                    self.chunk.emit(Op::Jump { off: 0 });
                    pos
                } else { 0 };
                let else_start = self.chunk.ops.len();
                if let Some(else_expr) = else_branch {
                    self.compile_expr(else_expr)?;
                }
                let end = self.chunk.ops.len();
                if let Op::JumpIfFalse { off } = &mut self.chunk.ops[jump_if_false_pos] {
                    *off = (else_start as i32) - (jump_if_false_pos as i32 + 1);
                }
                if else_branch.is_some() {
                    if let Op::Jump { off } = &mut self.chunk.ops[jump_to_end_pos] {
                        *off = (end as i32) - (jump_to_end_pos as i32 + 1);
                    }
                }
                Ok(()) // ★ 追加
            }
            Expr::Index { expr, index } => {
                // 1. 対象（TensorやListなど）を評価してスタックに積む
                self.compile_expr(expr)?;
                // 2. インデックスを評価してスタックに積む
                self.compile_expr(index)?;
                
                // 3. 値を取り出す命令を発行
                self.chunk.emit(Op::Index); 
                Ok(())
            }
            Expr::Field { expr, name } => {
                self.compile_expr(expr)?;
                // TODO: Op::Field を実装
                let _ = name;
                Err(SiggError::runtime("field access not yet implemented"))
            }
            Expr::Lambda { params, body } => {
                let lambda_name = format!("lambda_{}", self.table.id_to_name.len());
                let id = self.table.intern(&lambda_name);
                let mut lambda_compiler = FnCompiler::new(self.table);
                
                for (param_name, _param_type) in params {
                    lambda_compiler.local_index(param_name);
                }
                
                for stmt in body {
                    lambda_compiler.compile_stmt(stmt)?;
                }
                lambda_compiler.chunk.emit(Op::Return);
                let span = Span::new(0, 0, 0);
                self.lambda_fns.push((id, lambda_compiler.chunk, span));
                self.chunk.emit(Op::MakeLambda(id));
                Ok(()) // ★ 追加
            }
            Expr::StructInit { namespace, name, fields } => {
                // TODO: Op::MakeStruct を実装
                let _ = (namespace, name, fields);
                Err(SiggError::runtime("struct initialization not yet implemented"))
            }
            Expr::EnumInit { namespace, enum_name, variant, data } => {
                // TODO: Op::MakeEnum を実装
                let _ = (namespace, enum_name, variant, data);
                Err(SiggError::runtime("enum initialization not yet implemented"))
            }
            Expr::List(items) => {
                // 1. 中身の式を順番にコンパイルしてスタックに積む
                for item in items {
                    self.compile_expr(item)?;
                }
    
                // 2. "list" という名前の関数IDを取得（なければ登録）
                let name = "list";
                if !self.table.name_to_id.contains_key(name) {
                    self.table.intern(name);
                }
                let id = *self.table.name_to_id.get(name).unwrap();
    
                // 3. 関数呼び出し命令を発行
                self.chunk.emit(Op::CallId {
                    id,
                    argc: items.len() as u16, // ※ u8 か u16 かは Op の定義に合わせてください
                });
                Ok(())
            }
        }
    }
    fn lookup_local(&self, name: &str) -> Option<u16> {
        self.locals.get(name).copied()
    }
    fn emit_bool(&mut self, b: bool) {
        let ci = self.chunk.add_const(Value::Bool(b));
        self.chunk.emit(Op::Const(ci));
    }
}

fn is_builtin(name: &str) -> bool {
    matches!(name, "print" | "vec_get" | "vec_set" | "vec_new" | "vec_push" | "vec_len" | "map_get" | "map_set" | "map_new" | "map_len" | "noise2" | "rand2" | "mix" | "clamp" | "project" | "reaction_gs" | "world_step_gs" | "pocket_open" | "atlas_new" | "pocket_read" | "pocket_write" | "pocket_persist" | "atlas_query_topk" | "pocket_trigger_extract_auto" | "pocket_atlas_update" | "ai_pocket_open" | "ai_pocket_write_f32" | "ai_pocket_read_f32" | "ai_create" | "ai_tick" | "ai_get_score" | "ai_get_mode" | "ai_get_env_period" | "ai_set_env_period" | "ai_get_mu" | "ai_now_tick")
}

// ★ compile 関数の修正
pub fn compile(p: &Program) -> Result<CompiledProgram, SiggError> {
    let mut all_fns = p.fns.clone();
    for import_path in &p.imports {
        let content = fs::read_to_string(import_path)
            .map_err(|e| SiggError::io(format!("failed to read {}: {}", import_path, e)))?;
        let mut parser = crate::parser::Parser::new(&content);
        let imported_prog = parser.parse_program()
            .map_err(|e| SiggError::parse(format!("in {}: {}", import_path, e)))?;
        all_fns.extend(imported_prog.fns);
    }
    for b in crate::builtins::builtins() {
        let dummy_fn = FnDef {
            name: b.name.to_string(),
            params: vec![],
            ret_type: None,
            body: vec![],
            span: Span::new(0, 0, 0), // Span::default() がなければ new(0,0,0)
            namespace: None,
        };
        all_fns.push(dummy_fn);
    }

    let mut table = FnTable::new();
    for f in &all_fns { table.intern(&f.name); }

    let mut fns_map: HashMap<FnId, (Chunk, Span)> = HashMap::new();
    let mut all_lambdas = vec![];

    // ★ 1. ループに入る前に「組み込み関数の名前リスト」を作っておきます
    let builtin_names: std::collections::HashSet<String> = crate::builtins::builtins()
        .into_iter()
        .map(|b| b.name.to_string())
        .collect();

    for f in &all_fns {
        // ★ 2. まず最初に関数名 (fn_name) を確定させます
        let fn_name = if let Some(ns) = &f.namespace {
            format!("{}::{}", ns, f.name)
        } else {
            f.name.clone()
        };

        // ★ 3. fn_name ができた後で、チェックを行います
        // 組み込み関数なら、バイトコードの生成をスキップします
        if builtin_names.contains(&fn_name) {
            continue;
        }

        // 以下、通常のコンパイル処理
        let id = *table.name_to_id.get(&fn_name).unwrap();
        let mut fc = FnCompiler::new(&mut table);
        
        for (param_name, _) in &f.params {
            fc.local_index(param_name);
        }
        
        for st in &f.body {
            fc.compile_stmt(st)?;
        }
        fc.chunk.emit(Op::Return);
        fns_map.insert(id, (fc.chunk, f.span));
        all_lambdas.extend(fc.lambda_fns);
    }

    for (id, chunk, span) in all_lambdas {
        fns_map.insert(id, (chunk, span));
    }

    let main_id = *table.name_to_id.get("main")
        .ok_or_else(|| SiggError::parse("missing fn main()"))?;
    // ★ 修正: 必須フィールドを追加
    Ok(CompiledProgram { 
        fns: fns_map, 
        table, 
        main_id,
        namespaces: HashMap::new(),  // ★ 追加
        type_info: HashMap::new(),   // ★ 追加
    })
}

pub fn compile_with_namespaces(p: &Program) -> Result<CompiledProgram, SiggError> {
    let mut all_fns = p.fns.clone();
    let mut namespaces: HashMap<String, NamespaceInfo> = HashMap::new();
    
    // 名前空間の処理
    for ns_def in &p.namespaces {
        let mut ns_info = NamespaceInfo {
            name: ns_def.name.clone(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
        };
        
        for item in &ns_def.items {
            match item {
                NamespaceItem::Function(fn_def) => {
                    ns_info.functions.insert(fn_def.name.clone(), FnId(0)); // 後で更新
                    all_fns.push(fn_def.clone());
                }
                NamespaceItem::Struct(struct_def) => {
                    ns_info.structs.insert(struct_def.name.clone(), struct_def.clone());
                }
                NamespaceItem::Enum(enum_def) => {
                    ns_info.enums.insert(enum_def.name.clone(), enum_def.clone());
                }
            }
        }
        
        namespaces.insert(ns_def.name.clone(), ns_info);
    }
    
    // インポートの処理
    for import_path in &p.imports {
        let content = std::fs::read_to_string(import_path)
            .map_err(|e| SiggError::io(format!("failed to read {}: {}", import_path, e)))?;
        let mut parser = crate::parser::Parser::new(&content);
        let imported_prog = parser.parse_program()
            .map_err(|e| SiggError::parse(format!("in {}: {}", import_path, e)))?;
        all_fns.extend(imported_prog.fns);
    }
    
    let mut table = FnTable::new();
    let mut type_info: HashMap<FnId, FunctionTypeInfo> = HashMap::new();
    
    // 関数テーブルの構築
    for f in &all_fns {
        let fn_name = if let Some(ns) = &f.namespace {
            format!("{}::{}", ns, f.name)
        } else {
            f.name.clone()
        };
        
        let id = table.intern(&fn_name);
        
        // 型情報の保存
        if !f.params.is_empty() || f.ret_type.is_some() {
            type_info.insert(id, FunctionTypeInfo {
                params: f.params.iter()
                    .filter_map(|(_, ty)| ty.clone())
                    .collect(),
                ret_type: f.ret_type.clone(),
            });
        }
    }
    
    let mut fns_map: HashMap<FnId, (Chunk, Span)> = HashMap::new();
    let mut all_lambdas = vec![];
    
    for f in &all_fns {
        let fn_name = if let Some(ns) = &f.namespace {
            format!("{}::{}", ns, f.name)
        } else {
            f.name.clone()
        };
        
        let id = *table.name_to_id.get(&fn_name).unwrap();
        let mut fc = FnCompiler::new(&mut table);
        
        // パラメータを登録
        for (param_name, _) in &f.params {
            fc.local_index(param_name);
        }
        
        for st in &f.body {
            fc.compile_stmt(st)?;
        }
        fc.chunk.emit(Op::Return);
        fns_map.insert(id, (fc.chunk, f.span));
        all_lambdas.extend(fc.lambda_fns);
    }
    
    for (id, chunk, span) in all_lambdas {
        fns_map.insert(id, (chunk, span));
    }
    
    let main_id = *table.name_to_id.get("main")
        .ok_or_else(|| SiggError::parse("missing fn main()"))?;
    
    Ok(CompiledProgram {
        fns: fns_map,
        table,
        main_id,
        namespaces,
        type_info,
    })
}