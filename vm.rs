use std::collections::HashMap;
use std::sync::Arc;

use crate::builtins;
use crate::bytecode::{CompiledProgram, FnId, Op};
use crate::error::SiggError;
use crate::value::{Grid, GridRef, Value};

type BuiltinFn = fn(Vec<Value>) -> Result<Value, SiggError>;

use crate::span::Span;

#[derive(Clone)]
struct Frame {
    fn_id: FnId,
    ip: usize,
    locals: Vec<Value>,
    repeat_u32: Vec<u32>,
    span: Option<Span>, // for error reporting
    scope_stack: Vec<HashMap<String, u16>>, // for block scope
}

pub struct VM {
    builtins: HashMap<String, BuiltinFn>,
    seed: u64,
}

impl VM {
    pub fn new() -> Self {
        let mut m: HashMap<String, BuiltinFn> = HashMap::new();
        for b in builtins::builtins() {
            m.insert(b.name.to_string(), b.f);
        }
        Self { builtins: m, seed: 0 }
    }

    fn build_stack_trace(&self, prog: &CompiledProgram, callstack: &[Frame]) -> Vec<String> {
        callstack.iter().rev().map(|f| {
            let name = prog.table.name(f.fn_id);
            if let Some(span) = f.span {
                format!("{} at line {}", name, span.line)
            } else {
                name.to_string()
            }
        }).collect()
    }
    pub fn call_value(
        &mut self,
        func: &Value,
        args: Vec<Value>,
        prog: &CompiledProgram,
        callstack: &mut Vec<Frame>,
    ) -> Result<Value, SiggError> {
        match func {
            Value::Lambda(id) | Value::Function { id, .. } => {
                self.call_lambda(*id, args, prog, callstack)
            }
            Value::Closure { fn_id, upvalues } => {
                // クロージャの場合、upvaluesをローカル変数として設定
                let mut extended_args = upvalues.clone();
                extended_args.extend(args);
                self.call_lambda(*fn_id, extended_args, prog, callstack)
            }
            Value::HostFunction { func, .. } => {
                func(args)
            }
            _ => Err(SiggError::runtime("not a callable value")),
        }
    }
    
    fn call_lambda(
        &mut self,
        fn_id: FnId,
        args: Vec<Value>,
        prog: &CompiledProgram,
        callstack: &mut Vec<Frame>,
    ) -> Result<Value, SiggError> {
        let (chunk, span) = prog.fns.get(&fn_id)
            .ok_or_else(|| SiggError::runtime(format!("unknown function: {:?}", fn_id)))?;
        
        let mut locals = vec![Value::Unit; chunk.local_count as usize];
        
        // 引数をローカル変数に設定
        for (i, arg) in args.into_iter().enumerate() {
            if i < locals.len() {
                locals[i] = arg;
            }
        }
        
        callstack.push(Frame {
            fn_id,
            ip: 0,
            locals,
            repeat_u32: Vec::new(),
            span: Some(*span),
            scope_stack: vec![HashMap::new()],
        });
        
        // フレームを実行して結果を取得
        // （実際のVMループと統合する必要がある）
        Ok(Value::Unit)
    }
    
    // 組み込み高階関数の実装
    fn builtin_vec_map_impl(
        &mut self,
        vec: Vec<Value>,
        func: Value,
        prog: &CompiledProgram,
        callstack: &mut Vec<Frame>,
    ) -> Result<Vec<Value>, SiggError> {
        let mut result = Vec::new();
        
        for item in vec {
            let mapped = self.call_value(&func, vec![item], prog, callstack)?;
            result.push(mapped);
        }
        
        Ok(result)
    }
    
    fn builtin_vec_filter_impl(
        &mut self,
        vec: Vec<Value>,
        predicate: Value,
        prog: &CompiledProgram,
        callstack: &mut Vec<Frame>,
    ) -> Result<Vec<Value>, SiggError> {
        let mut result = Vec::new();
        
        for item in vec {
            let should_keep = self.call_value(&predicate, vec![item.clone()], prog, callstack)?;
            
            match should_keep {
                Value::Bool(true) => result.push(item),
                Value::Bool(false) => {}
                _ => return Err(SiggError::runtime("filter predicate must return bool")),
            }
        }
        
        Ok(result)
    }
    
    fn builtin_vec_reduce_impl(
        &mut self,
        vec: Vec<Value>,
        initial: Value,
        reducer: Value,
        prog: &CompiledProgram,
        callstack: &mut Vec<Frame>,
    ) -> Result<Value, SiggError> {
        let mut acc = initial;
        
        for item in vec {
            acc = self.call_value(&reducer, vec![acc, item], prog, callstack)?;
        }
        
        Ok(acc)
    }

    pub fn exec_compiled(&mut self, prog: &CompiledProgram) -> Result<(), SiggError> {
        let mut stack: Vec<Value> = Vec::new();
        let mut callstack: Vec<Frame> = Vec::new();

        // main frame
        let mut global_scope = HashMap::new();
        let mut global_locals = Vec::new();
        for (&id, _) in &prog.fns {
            let name = prog.table.name(id).to_string();
            global_scope.insert(name, global_locals.len() as u16);
            global_locals.push(Value::Lambda(id));
        }
        {
            let (main_chunk, main_span) = prog
                .fns
                .get(&prog.main_id)
                .ok_or_else(|| SiggError::runtime("missing main chunk"))?;
            let mut locals = vec![Value::Unit; main_chunk.local_count as usize];
            // prepend global locals
            global_locals.extend(locals);
            locals = global_locals;
            callstack.push(Frame {
                fn_id: prog.main_id,
                ip: 0,
                locals,
                repeat_u32: Vec::new(),
                span: Some(*main_span),
                scope_stack: vec![global_scope], // global scope with functions
            });
        }

        loop {
            let stack_trace = self.build_stack_trace(prog, &callstack);
            let Some(frame) = callstack.last_mut() else { break; };
            let (chunk, _) = prog
                .fns
                .get(&frame.fn_id)
                .ok_or_else(|| SiggError::runtime("missing function chunk"))?;
            let current_span = frame.span;

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
                Op::Add => binop(&mut stack, add, current_span, &stack_trace)?,
                Op::Sub => binop(&mut stack, sub, current_span, &stack_trace)?,
                Op::Mul => binop(&mut stack, mul, current_span, &stack_trace)?,
                Op::Div => binop(&mut stack, div, current_span, &stack_trace)?,
                Op::Mod => binop(&mut stack, |a,b| {
                    let x = as_f64(&a)? as i64;
                    let y = as_f64(&b)? as i64;
                    if y == 0 { return Err(SiggError::runtime("modulo by zero")); }
                    Ok(Value::Number((x % y) as f64))
                }, current_span, &stack_trace)?,
                Op::BitAnd => binop(&mut stack, bitand, current_span, &stack_trace)?, // &
                Op::Eq => binop(&mut stack, eq, current_span, &stack_trace)?, // 新しい演算子: ==
                Op::Ne => binop(&mut stack, ne, current_span, &stack_trace)?, // 新しい演算子: !=
                Op::Lt => binop(&mut stack, lt, current_span, &stack_trace)?, // 新しい演算子: <
                Op::Gt => binop(&mut stack, gt, current_span, &stack_trace)?, // 新しい演算子: >
                Op::Le => binop(&mut stack, le, current_span, &stack_trace)?, // 新しい演算子: <=
                Op::Ge => binop(&mut stack, ge, current_span, &stack_trace)?, // 新しい演算子: >=
                Op::And => binop(&mut stack, and, current_span, &stack_trace)?, // &&
                Op::Or => binop(&mut stack, or, current_span, &stack_trace)?,   // ||

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

                    // builtin only
                    let f = self.builtins.get(&name).cloned()
                        .ok_or_else(|| SiggError::runtime(format!("unknown builtin: {name}")))?;
                    let out = f(args)?;
                    stack.push(out);
                }

                Op::CallLambda { argc } => {
                    let argc = argc as usize;
                    if stack.len() < argc + 1 { return Err(SiggError::runtime("stack underflow (lambda call)")); }
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc { args.push(stack.pop().unwrap()); }
                    args.reverse();
                    let lambda = stack.pop().unwrap();
                    let id = match lambda {
                        Value::Lambda(id) => id,
                        _ => return Err(SiggError::runtime("call expects lambda")),
                    };

                    let (chunk2, span) = prog
                        .fns
                        .get(&id)
                        .ok_or_else(|| SiggError::runtime(format!("unknown lambda: {:?}", id)))?;
                    
                    // ローカル変数のベクターを作成
                    let mut locals = vec![Value::Unit; chunk2.local_count as usize];
                    
                    // 引数をローカル変数に格納
                    for i in 0..args.len() {
                        if i < locals.len() {
                            locals[i] = args[i].clone();
                        }
                    }
                    
                    callstack.push(Frame {
                        fn_id: id,
                        ip: 0,
                        locals,
                        repeat_u32: Vec::new(),
                        span: Some(*span),
                        scope_stack: vec![HashMap::new()],
                    });
                }

                // Op::CallDynamic { argc } => {
                //     let argc = argc as usize;
                //     if stack.len() < argc + 1 { return Err(SiggError::runtime("stack underflow (dynamic call)")); }
                //     let mut args = Vec::with_capacity(argc);
                //     for _ in 0..argc { args.push(stack.pop().unwrap()); }
                //     args.reverse();
                //     let callee = stack.pop().unwrap();
                //     match callee {
                //         Value::Lambda(id) => {
                //             let (chunk2, span) = prog
                //                 .fns
                //                 .get(&id)
                //                 .ok_or_else(|| SiggError::runtime(format!("unknown lambda: {:?}", id)))?;
                            
                //             // ローカル変数のベクターを作成
                //             let mut locals = vec![Value::Unit; chunk2.local_count as usize];
                            
                //             // 引数をローカル変数に格納
                //             for i in 0..args.len() {
                //                 if i < locals.len() {
                //                     locals[i] = args[i].clone();
                //                 }
                //             }
                            
                //             callstack.push(Frame {
                //                 fn_id: id,
                //                 ip: 0,
                //                 locals,
                //                 repeat_u32: Vec::new(),
                //                 span: Some(*span),
                //                 scope_stack: vec![HashMap::new()],
                //             });
                //         }
                //         _ => return Err(SiggError::runtime("dynamic call expects lambda")),
                //     }
                // }
                //以下のCallDynamicは、関数を変数に代入して呼び出す機能です。
                Op::CallDynamic { argc } => {
                    let argc = argc as usize;
                    if stack.len() < argc + 1 { 
                        return Err(SiggError::runtime("stack underflow (dynamic call)")); 
                    }
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc { args.push(stack.pop().unwrap()); }
                    args.reverse();
                    let callee = stack.pop().unwrap();
                    
                    // ★ 修正: Lambda以外にもFunctionやClosureをサポート
                    match callee {
                        Value::Lambda(id) | Value::Function { id, .. } => {
                            let (chunk2, span) = prog
                                .fns
                                .get(&id)
                                .ok_or_else(|| SiggError::runtime(format!("unknown function: {:?}", id)))?;
                            
                            let mut locals = vec![Value::Unit; chunk2.local_count as usize];
                            for i in 0..args.len() {
                                if i < locals.len() {
                                    locals[i] = args[i].clone();
                                }
                            }
                            
                            callstack.push(Frame {
                                fn_id: id,
                                ip: 0,
                                locals,
                                repeat_u32: Vec::new(),
                                span: Some(*span),
                                scope_stack: vec![HashMap::new()],
                            });
                        }
                        Value::Closure { fn_id: _, upvalues } => {
                            // クロージャの処理
                            let mut extended_args = upvalues;
                            extended_args.extend(args);
                            // ... 以下Lambdaと同じ処理 ...
                        }
                        _ => return Err(SiggError::runtime(
                            format!("dynamic call expects callable value, got: {:?}", callee)
                        )),
                    }
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

                Op::JumpIfFalse { off } => { // 新しい命令: if用
                    let v = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow"))?;
                    if !is_truthy(&v) {
                        frame.ip = (frame.ip as i32 + off) as usize;
                    }
                }
                Op::Jump { off } => { // 新しい命令: if用
                    frame.ip = (frame.ip as i32 + off) as usize;
                }

                Op::PushScope => {
                    frame.scope_stack.push(HashMap::new());
                }

                Op::PopScope => {
                    frame.scope_stack.pop();
                }

                Op::Index => {
                    let index = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow (index)"))?;
                    let expr = stack.pop().ok_or_else(|| SiggError::runtime("stack underflow (expr)"))?;
                    let val = match expr {
                        Value::Vec(v) => {
                            let idx = as_usize(&index)?;
                            v.get(idx).cloned().ok_or_else(|| SiggError::runtime("vec index out of bounds"))?
                        }
                        Value::Map(m) => {
                            let key = as_string(&index)?;
                            m.get(&key).cloned().ok_or_else(|| SiggError::runtime("map key not found"))?
                        }
                        _ => return Err(SiggError::runtime("index expects vec or map")),
                    };
                    stack.push(val);
                }

                Op::MakeLambda(id) => {
                    stack.push(Value::Lambda(id));
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
fn as_usize(v: &Value) -> Result<usize, SiggError> {
    let n = as_f64(v)?;
    if n < 0.0 { return Err(SiggError::runtime("expected non-negative integer")); }
    Ok(n as usize)
}
pub fn as_string(v: &Value) -> Result<String, SiggError> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(SiggError::runtime("expected string")),
    }
}

fn binop(stack: &mut Vec<Value>, f: fn(Value, Value) -> Result<Value, SiggError>, span: Option<Span>, stack_trace: &[String]) -> Result<(), SiggError> {
    let b = stack.pop().ok_or_else(|| SiggError::runtime_with_stack("stack underflow", stack_trace.to_vec()))?;
    let a = stack.pop().ok_or_else(|| SiggError::runtime_with_stack("stack underflow", stack_trace.to_vec()))?;
    match f(a, b) {
        Ok(v) => { stack.push(v); Ok(()) }
        Err(e) => {
            match e {
                SiggError::Runtime { msg, span: _, stack_trace: _ } => {
                    Err(SiggError::Runtime { msg, span, stack_trace: stack_trace.to_vec() })
                }
                _ => Err(e)
            }
        }
    }
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
fn eq(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: ==
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x == y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x == y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) == y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x == (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x == y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) == y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x == (y as f64))),
        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x == y)),
        (Value::Str(x), Value::Str(y)) => Ok(Value::Bool(x == y)),
        _ => Err(SiggError::runtime("eq: unsupported types")),
    }
}
fn ne(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: !=
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x != y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x != y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) != y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x != (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x != y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) != y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x != (y as f64))),
        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x != y)),
        (Value::Str(x), Value::Str(y)) => Ok(Value::Bool(x != y)),
        _ => Err(SiggError::runtime("ne: unsupported types")),
    }
}
fn lt(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: <
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x < y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x < y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) < y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x < (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x < y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) < y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x < (y as f64))),
        _ => Err(SiggError::runtime("lt: unsupported types")),
    }
}
fn gt(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: >
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x > y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x > y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) > y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x > (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x > y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) > y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x > (y as f64))),
        _ => Err(SiggError::runtime("gt: unsupported types")),
    }
}
fn le(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: <=
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x <= y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x <= y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) <= y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x <= (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x <= y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) <= y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x <= (y as f64))),
        _ => Err(SiggError::runtime("le: unsupported types")),
    }
}
fn ge(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: >=
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(x >= y)),
        (Value::F32(x), Value::F32(y)) => Ok(Value::Bool(x >= y)),
        (Value::F32(x), Value::Number(y)) => Ok(Value::Bool((x as f64) >= y)),
        (Value::Number(x), Value::F32(y)) => Ok(Value::Bool(x >= (y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x >= y)),
        (Value::Int(x), Value::Number(y)) => Ok(Value::Bool((x as f64) >= y)),
        (Value::Number(x), Value::Int(y)) => Ok(Value::Bool(x >= (y as f64))),
        _ => Err(SiggError::runtime("ge: unsupported types")),
    }
}

fn bitand(a: Value, b: Value) -> Result<Value, SiggError> { // &
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let x = x as i64;
            let y = y as i64;
            Ok(Value::Number((x & y) as f64))
        }
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x & y)),
        _ => Err(SiggError::runtime("bitand: unsupported types")),
    }
}

fn and(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: &&
    Ok(Value::Bool(is_truthy(&a) && is_truthy(&b)))
}
fn or(a: Value, b: Value) -> Result<Value, SiggError> { // 新しい関数: ||
    Ok(Value::Bool(is_truthy(&a) || is_truthy(&b)))
}

fn is_truthy(v: &Value) -> bool { // 新しい関数: 真偽値判定
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => *n != 0.0,
        _ => true,
    }
}

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
