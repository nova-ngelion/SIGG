
use std::collections::HashMap;
use std::sync::Arc;

use crate::builtins;
use crate::bytecode::{CompiledProgram, FnId, Op};
use crate::error::SiggError;
use crate::value::{Grid, GridRef, Value};

type BuiltinFn = fn(Vec<Value>) -> Result<Value, SiggError>;

#[derive(Clone)]
struct Frame {
    fn_id: FnId,
    ip: usize,
    locals: Vec<Value>,
    repeat_u32: Vec<u32>,
}

pub struct VM {
    builtins: HashMap<String, BuiltinFn>,
}

impl VM {
    pub fn new() -> Self {
        let mut m: HashMap<String, BuiltinFn> = HashMap::new();
        for b in builtins::builtins() {
            m.insert(b.name.to_string(), b.f);
        }
        Self { builtins: m }
    }

    pub fn exec_compiled(&mut self, prog: &CompiledProgram) -> Result<(), SiggError> {
        let mut stack: Vec<Value> = Vec::new();
        let mut callstack: Vec<Frame> = Vec::new();

        // main frame
        {
            let main_chunk = prog
                .fns
                .get(&prog.main_id)
                .ok_or_else(|| SiggError::runtime("missing main chunk"))?;
            callstack.push(Frame {
                fn_id: prog.main_id,
                ip: 0,
                locals: vec![Value::Unit; main_chunk.local_count as usize],
                repeat_u32: Vec::new(),
            });
        }

        loop {
            let Some(frame) = callstack.last_mut() else { break; };
            let chunk = prog
                .fns
                .get(&frame.fn_id)
                .ok_or_else(|| SiggError::runtime("missing function chunk"))?;

            if frame.ip >= chunk.ops.len() {
                callstack.pop();
                if callstack.is_empty() { break; }
                continue;
            }

            let op = chunk.ops[frame.ip].clone();
            frame.ip += 1;

            match op {
                Op::Const(ci) => {
                    let v = chunk
                        .consts
                        .get(ci as usize)
                        .cloned()
                        .ok_or_else(|| SiggError::runtime("const index out of range"))?;
                    stack.push(v);
                }
                Op::LoadLocal(i) => {
                    let v = frame
                        .locals
                        .get(i as usize)
                        .cloned()
                        .ok_or_else(|| SiggError::runtime("local index out of range"))?;
                    stack.push(v);
                }
                Op::StoreLocal(i) => {
                    let v = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow"))?;
                    let idx = i as usize;
                    if idx >= frame.locals.len() { frame.locals.resize(idx + 1, Value::Unit); }
                    frame.locals[idx] = v;
                }
                Op::Pop => { let _ = stack.pop(); }

                Op::Neg => {
                    let a = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow"))?;
                    stack.push(neg(a)?);
                }
                Op::Add => binop(&mut stack, add)?,
                Op::Sub => binop(&mut stack, sub)?,
                Op::Mul => binop(&mut stack, mul)?,
                Op::Div => binop(&mut stack, div)?,

                Op::MakeTuple(n) => {
                    let n = n as usize;
                    if stack.len() < n { return Err(SiggError::runtime("stack underflow (tuple)")); }
                    let mut xs = Vec::with_capacity(n);
                    for _ in 0..n { xs.push(stack.pop().unwrap()); }
                    xs.reverse();
                    stack.push(Value::Tuple(xs));
                }
                Op::UnpackTuple(n) => {
                    let n = n as usize;
                    let v = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow (unpack)"))?;
                    match v {
                        Value::Tuple(xs) => {
                            if xs.len() != n { return Err(SiggError::runtime("tuple arity mismatch")); }
                            for it in xs { stack.push(it); }
                        }
                        _ => return Err(SiggError::runtime("unpack expects tuple")),
                    }
                }

                Op::CallId { id, argc } => {
                    let argc = argc as usize;
                    if stack.len() < argc { return Err(SiggError::runtime("stack underflow (call)")); }
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc { args.push(stack.pop().unwrap()); }
                    args.reverse();

                    let name = prog.table.name(id).to_string();

                    // builtin?
                    if let Some(f) = self.builtins.get(&name).cloned() {
                        let out = f(args)?;
                        stack.push(out);
                        continue;
                    }

                    // user function
                    let chunk2 = prog
                        .fns
                        .get(&id)
                        .ok_or_else(|| SiggError::runtime(format!("unknown fn: {name}")))?;
                    callstack.push(Frame {
                        fn_id: id,
                        ip: 0,
                        locals: vec![Value::Unit; chunk2.local_count as usize],
                        repeat_u32: Vec::new(),
                    });
                }

                Op::RepeatInit(slot) => {
                    let v = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow (repeat init)"))?;
                    let n = as_u32(&v)?;
                    let s = slot as usize;
                    if s >= frame.repeat_u32.len() { frame.repeat_u32.resize(s + 1, 0); }
                    frame.repeat_u32[s] = n;
                }
                Op::RepeatCheckJump { slot, off } => {
                    let s = slot as usize;
                    let c = *frame.repeat_u32.get(s).unwrap_or(&0);
                    if c == 0 {
                        let ip = frame.ip as i32 + off;
                        frame.ip = ip as usize;
                    }
                }
                Op::RepeatDecJump { slot, off } => {
                    let s = slot as usize;
                    let c = *frame.repeat_u32.get(s).unwrap_or(&0);
                    let c2 = c.saturating_sub(1);
                    if s >= frame.repeat_u32.len() { frame.repeat_u32.resize(s + 1, 0); }
                    frame.repeat_u32[s] = c2;
                    let ip = frame.ip as i32 + off;
                    frame.ip = ip as usize;
                }

                Op::Return => {
                    callstack.pop();
                    if callstack.is_empty() { break; }
                }
            }
        }

        Ok(())
    }
}

fn as_f64(v: &Value) -> Result<f64, SiggError> {
    match v {
        Value::Number(n) => Ok(*n),
        Value::F32(x) => Ok(*x as f64),
        _ => Err(SiggError::runtime("expected number")),
    }
}
fn as_u32(v: &Value) -> Result<u32, SiggError> {
    let n = as_f64(v)?;
    if n < 0.0 { return Err(SiggError::runtime("expected non-negative integer")); }
    Ok(n as u32)
}

fn binop(stack: &mut Vec<Value>, f: fn(Value, Value) -> Result<Value, SiggError>) -> Result<(), SiggError> {
    let b = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow"))?;
    let a = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow"))?;
    stack.push(f(a, b)?);
    Ok(())
}

fn neg(a: Value) -> Result<Value, SiggError> {
    match a {
        Value::Number(n) => Ok(Value::Number(-n)),
        Value::F32(x) => Ok(Value::F32(-x)),
        Value::Grid(g) => {
            let gg = g.as_ref();
            let mut out = Grid::new(gg.dims.clone(), 0.0, gg.boundary);
            for i in 0..gg.data.len() { out.data[i] = -gg.data[i]; }
            Ok(Value::Grid(Arc::new(out)))
        }
        _ => Err(SiggError::runtime("neg: unsupported types")),
    }
}

fn add(a: Value, b: Value) -> Result<Value, SiggError> { elem_bin(a, b, |x,y| x+y, |x,y| x+y) }
fn sub(a: Value, b: Value) -> Result<Value, SiggError> { elem_bin(a, b, |x,y| x-y, |x,y| x-y) }
fn mul(a: Value, b: Value) -> Result<Value, SiggError> { elem_bin(a, b, |x,y| x*y, |x,y| x*y) }
fn div(a: Value, b: Value) -> Result<Value, SiggError> { elem_bin(a, b, |x,y| x/y, |x,y| x/y) }

fn elem_bin(
    a: Value,
    b: Value,
    nf: fn(f64,f64)->f64,
    gf: fn(f32,f32)->f32
) -> Result<Value, SiggError> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Number(nf(x,y))),
        (Value::F32(x), Value::F32(y)) => Ok(Value::F32(gf(x,y))),
        (Value::Number(x), Value::F32(y)) => Ok(Value::F32(gf(x as f32,y))),
        (Value::F32(x), Value::Number(y)) => Ok(Value::F32(gf(x,y as f32))),

        // grid op scalar (cell op s)
        (Value::Grid(g), Value::Number(n)) => grid_map_scalar_lr(g, n as f32, gf),
        (Value::Grid(g), Value::F32(s)) => grid_map_scalar_lr(g, s, gf),

        // scalar op grid (s op cell)
        (Value::Number(n), Value::Grid(g)) => grid_map_scalar_rl(g, n as f32, gf),
        (Value::F32(s), Value::Grid(g)) => grid_map_scalar_rl(g, s, gf),

        // grid op grid
        (Value::Grid(g1), Value::Grid(g2)) => grid_zip(g1, g2, gf),

        _ => Err(SiggError::runtime("binary op: unsupported types")),
    }
}

fn grid_map_scalar_lr(g: GridRef, s: f32, f: fn(f32,f32)->f32) -> Result<Value, SiggError> {
    let gg = g.as_ref();
    let mut out = Grid::new(gg.dims.clone(), 0.0, gg.boundary);
    for i in 0..gg.data.len() { out.data[i] = f(gg.data[i], s); }
    Ok(Value::Grid(Arc::new(out)))
}
fn grid_map_scalar_rl(g: GridRef, s: f32, f: fn(f32,f32)->f32) -> Result<Value, SiggError> {
    let gg = g.as_ref();
    let mut out = Grid::new(gg.dims.clone(), 0.0, gg.boundary);
    for i in 0..gg.data.len() { out.data[i] = f(s, gg.data[i]); }
    Ok(Value::Grid(Arc::new(out)))
}
fn grid_zip(g1: GridRef, g2: GridRef, f: fn(f32,f32)->f32) -> Result<Value, SiggError> {
    let a = g1.as_ref();
    let b = g2.as_ref();
    if a.dims != b.dims { return Err(SiggError::runtime("grid shape mismatch")); }
    let mut out = Grid::new(a.dims.clone(), 0.0, a.boundary);
    for i in 0..a.data.len() { out.data[i] = f(a.data[i], b.data[i]); }
    Ok(Value::Grid(Arc::new(out)))
}
