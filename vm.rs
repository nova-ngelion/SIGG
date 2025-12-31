use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use crate::state::PocketState;
use crate::lexer::Lexer;
use crate::token::Tok;
use crate::builtins;
use crate::pocket::tensor::{Tensor, Complex};
use crate::bytecode::{CompiledProgram, FnId, Op};
use crate::error::SiggError;
use crate::value::{Grid, GridRef, Value};
use std::net::TcpStream;
use std::io::Write;

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

/// インタプリタが扱う値の型
#[derive(Clone)]
pub enum RuntimeValue {
    Tensor(Arc<RwLock<Tensor>>),
    Number(f64),
}

/// 変数の有効範囲（スコープ）を管理
pub struct Environment {
    pub parent: Option<Arc<RwLock<Environment>>>,
    pub values: HashMap<String, RuntimeValue>,
    // server.rs の PocketState への参照（空間への直接アクセス用）
    pub pocket_state: Arc<Mutex<PocketState>>,
    pub variables: HashMap<String, Value>,
    pub current_stream: Option<TcpStream>,
}

pub struct SiggInterpreter {
    pub pocket_state: Arc<Mutex<PocketState>>,
    pub variables: HashMap<String, Value>,
    pub current_stream: Option<TcpStream>, // これを確実に入れる
}
impl SiggInterpreter {
    pub fn new(pocket_state: Arc<Mutex<PocketState>>) -> Self {
        Self {
            pocket_state,
            variables: HashMap::new(),
            current_stream: None,
        }
    }

    /// 受信したソースコードを実行し、server.rs の空間を操作する
    pub fn run(&mut self, source: &str) -> Result<(), String> {
        let mut lexer = Lexer::new(source);
        // 本来はここで Parser を通して AST を作りますが、
        // 接続の具体化としてトークンを直接評価する流れを示します
        
        while let Ok(Some(token)) = lexer.next_token() {
            match token.kind {
                Tok::Let => self.handle_let(&mut lexer)?,
                Tok::Repeat => self.handle_repeat(&mut lexer)?,
                // server.rs の AI_TICK 等と連動させる処理をここに挟む
                _ => {}
            }
        }
        Ok(())
    }

    fn handle_let(&mut self, lexer: &mut Lexer) -> Result<(), String> {
        // 例: let w = tensor([1]);
        // 1. 変数名 (w) を取得
        // 2. tensor 関数呼び出しを解析
        // 3. Tensor::zeros(shape) を呼び出し、variables に登録
        // 4. 同時に、server.rs の PocketState に「新しい構造が生まれた」ことを報告
        let mut st = self.pocket_state.lock().unwrap();
        // st.next_handle を更新するなど、server.rs の流儀に従う
        Ok(())
    }

    fn handle_repeat(&mut self, lexer: &mut Lexer) -> Result<(), String> {
        // repeat (n) { ... }
        // server.rs の TAG_RUN_ONCE に相当するループを実行
        // ループごとに Tensor::backward() や discrete_laplacian() を呼び出す
        Ok(())
    }
}

pub struct VM {
    pub builtins: HashMap<String, fn(Vec<Value>) -> Result<Value, SiggError>>,
    pub globals: HashMap<String, Value>,
    pub seed: u64,
    pub current_stream: Option<TcpStream>,
}

impl VM {
    // src/vm.rs の VM::new() 内
    pub fn new() -> Self {
        let mut m: HashMap<String, BuiltinFn> = HashMap::new();
        let mut g: HashMap<String, Value> = HashMap::new();

        for b in builtins::builtins() {
            m.insert(b.name.to_string(), b.f);
            g.insert(b.name.to_string(), Value::NativeFunction(b.f));
        }

        // --- ここを追加 ---
        // visualize という名前だけ globals に登録しておく（コンパイラに見つけさせるため）
        // 値は何でも良いですが、識別しやすいように None 以外を入れます
        g.insert("visualize".to_string(), Value::Number(0.0)); 
        // ------------------

        Self { 
            builtins: m, 
            globals: g,
            seed: 0,
            current_stream: None, 
        }
    }
    // ストリームを外からセットするためのメソッドを追加
    pub fn set_stream(&mut self, stream: TcpStream) {
        self.current_stream = Some(stream);
    }
    // 2Dラプラシアンの実装
    pub fn compute_laplacian_2d(&mut self, tensor_val: Value) -> Result<Value, SiggError> {
        if let Value::Tensor(arc_tensor) = tensor_val {
            let tensor = arc_tensor.read().unwrap();
            let w = tensor.shape[0];
            let h = if tensor.shape.len() > 1 { tensor.shape[1] } else { 1 };
            
            let mut new_data = vec![Complex::new(0.0, 0.0); w * h];
            for y in 1..h-1 {
                for x in 1..w-1 {
                    let idx = y * w + x;
                    // 実部(re)を取り出して計算
                    let center = tensor.data[idx].re;
                    let neighbors = tensor.data[idx+1].re + tensor.data[idx-1].re + 
                                   tensor.data[idx+w].re + tensor.data[idx-w].re;
                    new_data[idx] = Complex::new(neighbors - 4.0 * center, 0.0);
                }
            }
            let mut out = Tensor::zeros(vec![w, h]);
            out.data = new_data;
            Ok(Value::Tensor(Arc::new(RwLock::new(out))))
        } else {
            Err(SiggError::runtime("Expected tensor for laplacian_2d"))
        }
    }
    // 可視化データの送信
    pub fn send_visualize_data(&mut self, val: Value) -> Result<Value, SiggError> {
        println!("DEBUG: send_visualize_data called!"); // ←これを追加
        if let Value::Tensor(t_lock) = val {
            let t = t_lock.read().unwrap();
            // ここでタグ 0x50 とデータを準備
            let mut buffer = vec![0x50];
            for c in &t.data {
                buffer.extend_from_slice(&(c.re as f32).to_le_bytes());
            }
    
            // フィールド名を self.current_stream に修正
            if let Some(ref mut stream) = self.current_stream {
                use std::io::Write;
                stream.write_all(&buffer).map_err(|e| SiggError::runtime(e.to_string()))?;
                stream.flush().map_err(|e| SiggError::runtime(e.to_string()))?;
                //println!("DEBUG: Sent {} bytes to Python", buffer.len());
            } else {
                //println!("DEBUG: No active stream to send data!"); // ←これが出たら接続が切れています
            }
        }
        // 戻り値を Result<(), ...> ではなく Result<Value, ...> に合わせる
        Ok(Value::Number(0.0)) 
    }
    pub fn native_t_set_2d(&mut self, args: Vec<Value>) -> Result<Value, SiggError> {
        if args.len() < 4 {
            return Err(SiggError::runtime("t_set_2d requires 4 arguments"));
        }

        if let Value::Tensor(arc_tensor) = &args[0] {
            let mut tensor = arc_tensor.write().unwrap();
            
            // args[1] (x) と args[2] (y) を数値として取得
            let x = match args[1] {
                Value::Number(n) => n as usize,
                Value::F32(f) => f as usize,
                _ => return Err(SiggError::runtime("x must be a number")),
            };
            let y = match args[2] {
                Value::Number(n) => n as usize,
                Value::F32(f) => f as usize,
                _ => return Err(SiggError::runtime("y must be a number")),
            };

            // args[3] は [re, im] の配列(List/Array)か
            // あなたの定義に合わせて修正（Value::List の場合が多いです）
            let (re, im) = match &args[3] {
                Value::List(list) => {
                    let r = match list[0] { Value::Number(n) => n as f32, Value::F32(f) => f, _ => 0.0 };
                    let i = if list.len() > 1 {
                        match list[1] { Value::Number(n) => n as f32, Value::F32(f) => f, _ => 0.0 }
                    } else { 0.0 };
                    (r, i)
                },
                _ => return Err(SiggError::runtime("Value must be a list [re, im]")),
            };

            let w = tensor.shape[0];
            let h = if tensor.shape.len() > 1 { tensor.shape[1] } else { 1 };

            if x < w && y < h {
                let idx = y * w + x;
                tensor.data[idx] = Complex::new(re as f64, im as f64);
                Ok(Value::Number(0.0))
            } else {
                Err(SiggError::runtime("Index out of bounds"))
            }
        } else {
            Err(SiggError::runtime("First arg must be tensor"))
        }
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
        //println!("DEBUG: VM is trying to call: {}", func); // これで何が呼ばれているか全表示
        match func {
            Value::Lambda(id) | Value::Function { id, .. } => {
                self.call_lambda(*id, args, prog, callstack)
            }
            Value::NativeFunction(f) => {
                f(args)
            }
            Value::Closure { fn_id, upvalues } => {
                // クロージャの場合、upvaluesをローカル変数として設定
                let mut extended_args = upvalues.clone();
                extended_args.extend(args);
                self.call_lambda(*fn_id, extended_args, prog, callstack)
            }
            Value::HostFunction { func, name } => {
                // 1. まず、VM自体が持つ特殊命令（2D拡張）かどうかをチェック
                match name.as_str() {
                    "t_set_2d" => {
                        return self.native_t_set_2d(args);      // return を付ける
                    }
                    "laplacian_2d" => {
                        let arg = args[0].clone();
                        return self.compute_laplacian_2d(arg);
                    }
                    "visualize" => {
                        //println!("DEBUG: VM is calling visualize function!!!"); // これを追加
                        let arg = args[0].clone();
                        return self.send_visualize_data(arg);
                    }                  
                    _ => {} // 次へ進む
                }
            
                // 2. 特殊命令でなければ、従来のHostFunctionとして実行
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
        //println!("DEBUG: exec_compiled started"); // ★ログ
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
        for (name, func) in &self.builtins {
            // 重複チェック（ユーザー定義関数と同名ならユーザー優先、または上書き）
            if !global_scope.contains_key(name) {
                global_scope.insert(name.clone(), global_locals.len() as u16);
                global_locals.push(Value::NativeFunction(*func));
            }
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
            //println!("DEBUG: Executing OP: {:?}", op); // ★ これを追加
            frame.ip += 1;
            //println!("DEBUG: Executing Op {:?}", op);

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
                Op::Add => {
                    let rhs = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let lhs = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    
                    match (lhs, rhs) {
                        (Value::Number(a), Value::Number(b)) => stack.push(Value::Number(a + b)),
                        (Value::Tensor(t1_lock), Value::Tensor(t2_lock)) => {
                            let mut t1 = t1_lock.write().unwrap(); // 書き込みロック
                            let t2 = t2_lock.read().unwrap();      // 読み取りロック
                            
                            if t1.data.len() == t2.data.len() {
                                for (v1, v2) in t1.data.iter_mut().zip(t2.data.iter()) {
                                    v1.re += v2.re;
                                    v1.im += v2.im;
                                }
                            }
                            drop(t1); // ロック解除
                            drop(t2);
                            stack.push(Value::Tensor(t1_lock));
                        }
                        _ => return Err(SiggError::runtime("Invalid types for add")),
                    }
                }
                // --- 引き算 (-) ---
                Op::Sub => {
                    let b = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let a = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    
                    let res = crate::builtins::builtin_sub(vec![a, b])?;
                    stack.push(res);
                }
                // --- 掛け算 (*) ---
                Op::Mul => {
                    let rhs = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let lhs = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    
                    match (lhs, rhs) {
                        (Value::Number(a), Value::Number(b)) => stack.push(Value::Number(a * b)),
                        (Value::Tensor(t_lock), Value::Number(n)) | (Value::Number(n), Value::Tensor(t_lock)) => {
                            let mut t = t_lock.write().unwrap();
                            
                            // n_f32 ではなく n_f64 として、f64 にキャストする
                            let n_f64 = n as f64; 
                            
                            for val in t.data.iter_mut() {
                                // これで f64 *= f64 の計算になり、エラーが消えます
                                val.re *= n_f64;
                                val.im *= n_f64;
                            }
                            drop(t);
                            stack.push(Value::Tensor(t_lock));
                        }
                        _ => return Err(SiggError::runtime("Invalid types for mul")),
                    }
                }
                
                // --- 割り算 (/) ---
                Op::Div => {
                    let b = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let a = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let res = crate::builtins::builtin_div(vec![a, b])?;
                    stack.push(res);
                }
                // --- 剰余 (%) ---
                Op::Mod => {
                    let b = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let a = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let res = crate::builtins::builtin_mod(vec![a, b])?;
                    stack.push(res);
                }
                // --- 単項マイナス (Neg) ---
                Op::Neg => {
                    let a = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    let res = crate::builtins::builtin_neg(vec![a])?;
                    stack.push(res);
                }
                // src/vm.rs の Op::Visualize 処理
                // match op { ... } の中
                Op::Visualize => {
                    let arg = stack.pop().ok_or(SiggError::runtime("Stack underflow"))?;
                    // self.send_visualize_data を呼ぶ
                    self.send_visualize_data(arg)?; 
                    stack.push(Value::Number(0.0)); // スタックの整合性を保つ
                    // 描画の安定のため少し待機
                    std::thread::sleep(std::time::Duration::from_millis(16)); 
                }
                Op::StoreElement | Op::StoreIndex => {
                    let val = stack.pop().ok_or(SiggError::runtime("Stack underflow: value"))?;
                    let idx_val = stack.pop().ok_or(SiggError::runtime("Stack underflow: index"))?;
                    let mut target = stack.pop().ok_or(SiggError::runtime("Stack underflow: target"))?;
                    
                    //println!("DEBUG: StoreIndex target={:?}, idx={:?}, val={:?}", target, idx_val, val);
                
                    match target {
                        Value::Tensor(ref t_arc) => {
                            let indices = match idx_val {
                                Value::Number(n) => vec![n as usize],
                                Value::Int(i) => vec![i as usize],
                                Value::List(ref l) | Value::Vec(ref l) => {
                                    l.iter().map(|v| match v {
                                        Value::Number(n) => *n as usize,
                                        Value::Int(i) => *i as usize,
                                        _ => 0
                                    }).collect()
                                },
                                _ => return Err(SiggError::runtime("Invalid index type")),
                            };
                            let c_val = match val {
                                Value::Number(n) => Complex { re: n, im: 0.0 },
                                Value::Int(i) => Complex { re: i as f64, im: 0.0 },
                                _ => return Err(SiggError::runtime("Only numbers can be stored in tensor")),
                            };
                            
                            // 書き換え実行
                            let mut t = t_arc.write().unwrap();
                            t.set(&indices, c_val);
                
                            // ★重要修正: テンソルもスタックに戻す必要があります！
                            // 次の命令(StoreLocal)がこれをPopして変数に紐付けるためです。
                            drop(t); // ロックを明示的に解放（念のため）
                            stack.push(target); 
                            Ok(())
                        }
                        
                        Value::List(ref mut list) | Value::Vec(ref mut list) => {
                            let idx = match idx_val {
                                Value::Number(n) => n as usize,
                                Value::Int(i) => i as usize,
                                _ => return Err(SiggError::runtime("List index must be a number")),
                            };
                
                            if idx < list.len() {
                                list[idx] = val;
                                //println!("DEBUG: List updated. New list: {:?}", list);
                            } else {
                                return Err(SiggError::runtime("List index out of bounds"));
                            }
                            
                            // 書き換え終わった target をスタックに戻す
                            stack.push(target); 
                            Ok(())
                        }
                
                        _ => Err(SiggError::runtime(format!("Target is not indexable: {:?}", target)))
                    }?; // matchの結果（Result）をチェック
                }
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
                    
                    // 1. まず引数をスタックから取り出す（共通処理）
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc { args.push(stack.pop().unwrap()); }
                    args.reverse();
                    
                    // 2. 関数名を取得
                    let name = prog.table.name(id).to_string();
                
                    // 3. 【重要】visualize の横取り処理（ここだけでOK）
                    if name == "visualize" {
                        //println!("DEBUG: INTERCEPTED 'visualize' call!");
                        
                        // 第1引数（tensor）を取り出して送信
                        if let Some(arg) = args.get(0) {
                            self.send_visualize_data(arg.clone())?;
                        }
                        
                        // 戻り値を積んで、次の命令へ進む
                        stack.push(Value::Number(0.0)); 
                        continue; 
                    }
                
                    // 4. それ以外の通常の関数呼び出し処理
                    let name_for_err = name.clone();
                    let func_val = if let Some(f) = self.builtins.get(&name) {
                        Value::HostFunction { 
                            func: Arc::new(*f), 
                            name: name.clone() 
                        }
                    } else if let Some(val) = self.globals.get(&name) {
                        val.clone()
                    } else {
                        return Err(SiggError::runtime(format!("unknown function: {name_for_err}")));
                    };
                
                    let out = self.call_value(&func_val, args, prog, &mut callstack)?;
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

                    // 1. まずユーザー定義関数 (prog.fns) にバイトコードがあるか探す
                    if let Some((chunk2, span)) = prog.fns.get(&id) {
                        // --- A. バイトコードがある場合（通常の関数実行） ---
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
                    } else {
                        // --- B. バイトコードがない場合（組み込み関数へのフォールバック） ---
                        let name = prog.table.name(id);
                        
                        if name == "visualize" {
                            // 1. visualize の横取り処理
                            //println!("DEBUG: INTERCEPTED 'visualize' via CallLambda!");
                            if let Some(arg) = args.get(0) {
                                self.send_visualize_data(arg.clone())?;
                            }
                            stack.push(Value::Number(0.0));
                            // ここでこの関数の処理は終わりなので、これ以上下には行かない
                        } else if let Some(func) = self.builtins.get(name) {
                            // 2. その他の組み込み関数の実行
                            let out = func(args)?;
                            stack.push(out);
                        } else {
                            // 3. 本当に見つからない場合
                            return Err(SiggError::runtime(format!("unknown lambda: {} (ID: {:?})", name, id)));
                        }
                    }
                }
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
                        Value::NativeFunction(f) => {
                            let out = f(args)?;
                            stack.push(out);
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
                    
                    //println!("DEBUG: Op::Index calling for target: {:?}, index: {:?}", expr, index);
                    
                    let val = match expr {
                        // ★ List と Vec の両方を許可する
                        Value::List(v) | Value::Vec(v) => {
                            let idx = as_usize(&index)?;
                            v.get(idx).cloned().ok_or_else(|| SiggError::runtime("index out of bounds"))?
                        }
                        Value::Map(m) => {
                            let key = as_string(&index)?;
                            m.get(&key).cloned().ok_or_else(|| SiggError::runtime("map key not found"))?
                        }
                        _ => return Err(SiggError::runtime(format!("index expects vec or map, but got {:?}", expr))),
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
