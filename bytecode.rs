
use std::collections::HashMap;

use crate::ast::*;
use crate::error::SiggError;
use crate::value::Value;

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
    Neg,

    MakeTuple(u16),
    UnpackTuple(u16), // pop tuple -> push elements (0..n-1)

    CallId { id: FnId, argc: u16 },
    Pop,
    Return,

    RepeatInit(u16),
    RepeatCheckJump { slot: u16, off: i32 },
    RepeatDecJump { slot: u16, off: i32 },
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub ops: Vec<Op>,
    pub consts: Vec<Value>,
    pub local_count: u16,
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
}

#[derive(Clone, Debug)]
pub struct CompiledProgram {
    pub fns: HashMap<FnId, Chunk>,
    pub table: FnTable,
    pub main_id: FnId,
}

struct FnCompiler<'a> {
    locals: HashMap<String, u16>,
    chunk: Chunk,
    table: &'a mut FnTable,
    next_repeat_slot: u16,
}

impl<'a> FnCompiler<'a> {
    fn new(table: &'a mut FnTable) -> Self {
        Self { locals: HashMap::new(), chunk: Chunk::new(), table, next_repeat_slot: 0 }
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

    fn compile_store_pattern(&mut self, p: &Pattern) -> Result<(), SiggError> {
        match p {
            Pattern::Wildcard => { self.chunk.emit(Op::Pop); Ok(()) }
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
        }
    }

    fn compile_stmt(&mut self, s: &Stmt) -> Result<(), SiggError> {
        match s {
            Stmt::Let { pat, expr } => {
                self.compile_expr(expr)?;
                self.compile_store_pattern(pat)?;
                Ok(())
            }
            Stmt::Assign { name, expr } => {
                self.compile_expr(expr)?;
                let idx = self.local_index(name);
                self.chunk.emit(Op::StoreLocal(idx));
                Ok(())
            }
            Stmt::Expr(e) => {
                self.compile_expr(e)?;
                self.chunk.emit(Op::Pop);
                Ok(())
            }
            Stmt::Repeat { count, body } => self.compile_repeat_like(count, body),
            Stmt::Transition { count, body } => self.compile_repeat_like(count, body),
        }
    }

    fn compile_repeat_like(&mut self, count: &Expr, body: &Vec<Stmt>) -> Result<(), SiggError> {
        self.compile_expr(count)?;
        let slot = self.alloc_repeat_slot();
        self.chunk.emit(Op::RepeatInit(slot));

        let check_ip = self.chunk.ops.len();
        let j_pos = self.chunk.ops.len();
        self.chunk.emit(Op::RepeatCheckJump { slot, off: 0 });

        for st in body { self.compile_stmt(st)?; }

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
            Expr::Number(n) => {
                let ci = self.chunk.add_const(Value::Number(*n));
                self.chunk.emit(Op::Const(ci));
            }
            Expr::Str(s) => {
                let ci = self.chunk.add_const(Value::Str(s.clone()));
                self.chunk.emit(Op::Const(ci));
            }
            Expr::Var(name) => {
                let idx = self.local_index(name);
                self.chunk.emit(Op::LoadLocal(idx));
            }
            Expr::Group(inner) => self.compile_expr(inner)?,
            Expr::Unary { op, rhs } => {
                self.compile_expr(rhs)?;
                match op {
                    UnOp::Neg => self.chunk.emit(Op::Neg),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                self.compile_expr(lhs)?;
                self.compile_expr(rhs)?;
                match op {
                    BinOp::Add => self.chunk.emit(Op::Add),
                    BinOp::Sub => self.chunk.emit(Op::Sub),
                    BinOp::Mul => self.chunk.emit(Op::Mul),
                    BinOp::Div => self.chunk.emit(Op::Div),
                }
            }
            Expr::Tuple(items) => {
                for it in items { self.compile_expr(it)?; }
                self.chunk.emit(Op::MakeTuple(items.len() as u16));
            }
            Expr::Call { callee, args } => {
                for a in args { self.compile_expr(a)?; }
                let id = self.table.intern(callee);
                self.chunk.emit(Op::CallId { id, argc: args.len() as u16 });
            }
        }
        Ok(())
    }
}

pub fn compile(p: &Program) -> Result<CompiledProgram, SiggError> {
    let mut table = FnTable::new();
    for f in &p.fns { table.intern(&f.name); }

    let mut fns_map: HashMap<FnId, Chunk> = HashMap::new();

    for f in &p.fns {
        let id = *table.name_to_id.get(&f.name).unwrap();
        let mut fc = FnCompiler::new(&mut table);
        for st in &f.body { fc.compile_stmt(st)?; }
        fc.chunk.emit(Op::Return);
        fns_map.insert(id, fc.chunk);
    }

    let main_id = *table.name_to_id.get("main").ok_or_else(|| SiggError::parse("missing fn main()"))?;
    Ok(CompiledProgram { fns: fns_map, table, main_id })
}
