use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};
use crate::error::SiggError;
use crate::value::{Boundary, Grid, GridRef, Value};
use crate::vm::as_string;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use crate::pocket;
use crate::pocket::types::{WorldKey, ChunkKey, Hit};
use crate::state::PocketState;
use crate::pocket::PocketWorld;
use crate::pocket::compute::ComputeSpace;
use crate::ai_runtime;
use std::collections::HashMap;
use std::time::Instant;
use std::sync::{Arc, RwLock};
use crate::pocket::tensor::{Tensor, Complex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::io::{self, Write};
use std::fs;
use std::path::Path;
use serde_json::json;
use std::process::Command;

#[allow(dead_code)]
const PROG_MAX: i32 = 64;   // 命令領域は 0..63
#[allow(dead_code)]
const IO_BASE:  i32 = 100;  // io_in/io_out はここ以降へ


type EvalResult = Result<Value, SiggError>;

fn runtime_err(msg: &str) -> SiggError {
    SiggError::runtime(msg)
}

pub struct Builtin {
    pub name: &'static str,
    pub f: fn(Vec<Value>) -> Result<Value, SiggError>,
}

static LAST_DIGEST: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)]
static AI_TICK: AtomicU32 = AtomicU32::new(0);//new
static RAND_SEED: AtomicU64 = AtomicU64::new(12345);

pub fn last_digest_u32() -> u32 {
    LAST_DIGEST.load(Ordering::Relaxed) as u32
}

fn mix_hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

fn digest_grid_u32(g: &crate::value::Grid) -> u32 {
    // そこそこ安定して軽い：全要素を f32 bits で混ぜる
    let mut h: u32 = 0x811C9DC5;
    for &v in &g.data {
        let b = v.to_bits();
        h ^= b;
        h = mix_hash_u32(h);
    }
    h
}
fn builtin_pocket_open(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_open(u,v,w, chunk_size, z_dim, delta_path) -> handle
    need_n(&args, 6, "pocket_open")?;
    let world = as_worldkey(&args, 0)?;
    let chunk_size = as_usize(&args[3])?;
    let z_dim = as_usize(&args[4])?;
    let delta_path = match &args[5] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("pocket_open: delta_path must be string")),
    };

    let w = pocket::PocketWorld::open(world, chunk_size, z_dim, delta_path)
        .map_err(|e| SiggError::runtime(format!("pocket_open failed: {e}")))?;

    let mut reg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let id = reg.len() as u32;
    reg.push(w);
    Ok(Value::Handle(id))
}

fn builtin_atlas_new(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "atlas_new")?;
    let z_dim = as_usize(&args[0])?;
    let a = pocket::Atlas::new(z_dim);
    let mut reg = atlases().lock().map_err(|_| SiggError::runtime("atlas registry poisoned"))?;
    let id = reg.len() as u32;
    reg.push(a);
    Ok(Value::Handle(id))
}

fn builtin_pocket_read(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_read(world_h, x,y,z,lane) -> f32
    need_n(&args, 5, "pocket_read")?;
    let wh = as_u32_handle(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let lane = as_usize(&args[4])?;

    let mut reg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize).ok_or_else(|| SiggError::runtime("invalid world handle"))?;
    Ok(Value::F32(w.cell_read_f32(x,y,z,lane)))
}

fn builtin_pocket_write(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_write(world_h, x,y,z,lane,val) -> ()
    need_n(&args, 6, "pocket_write")?;
    let wh = as_u32_handle(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let lane = as_usize(&args[4])?;
    let val = as_f32(&args[5])?;

    let mut reg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize).ok_or_else(|| SiggError::runtime("invalid world handle"))?;
    w.cell_write_f32(x,y,z,lane,val);
    Ok(Value::Unit)
}

fn builtin_pocket_persist(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_persist(world_h)
    need_n(&args, 1, "pocket_persist")?;
    let wh = as_u32_handle(&args[0])?;
    let mut reg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize).ok_or_else(|| SiggError::runtime("invalid world handle"))?;
    w.persist().map_err(|e| SiggError::runtime(format!("persist failed: {e}")))?;
    Ok(Value::Unit)
}

fn builtin_atlas_query_topk(args: Vec<Value>) -> Result<Value, SiggError> {
    // atlas_query_topk(atlas_h, u,v,w, query_vec, topk) -> chunks ((cx,cy,cz),...)
    need_n(&args, 6, "atlas_query_topk")?;
    let ah = as_u32_handle(&args[0])?;
    let world = as_worldkey(&args, 1)?;
    let query = as_query_vec(&args[4])?;
    let topk = as_usize(&args[5])?;

    let reg = atlases().lock().map_err(|_| SiggError::runtime("atlas registry poisoned"))?;
    let a = reg.get(ah as usize).ok_or_else(|| SiggError::runtime("invalid atlas handle"))?;
    let chunks = a.query_topk(world, &query, topk);

    let out = chunks.into_iter().map(|(cx,cy,cz)| {
        Value::Tuple(vec![Value::Number(cx as f64), Value::Number(cy as f64), Value::Number(cz as f64)])
    }).collect();
    Ok(Value::Tuple(out))
}

fn builtin_pocket_trigger_extract_auto(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_trigger_extract_auto(world_h, chunks, query, steps, diffusion, threshold, topn)
    // -> (hits, used_steps, used_diffusion, used_threshold)
    need_n(&args, 7, "pocket_trigger_extract_auto")?;
    let wh = as_u32_handle(&args[0])?;
    let chunks = as_chunks(&args[1])?;
    let query = as_query_vec(&args[2])?;
    let steps = as_f64(&args[3])? as u32;
    let diffusion = as_f32(&args[4])?;
    let threshold = as_f32(&args[5])?;
    let topn = as_usize(&args[6])?;

    let mut reg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize).ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let (hits, used_steps, used_diff, used_thr) =
        w.compute_trigger_extract_auto(&chunks, &query, steps, diffusion, threshold, topn);

    Ok(Value::Tuple(vec![
        hits_to_value(hits),
        Value::Number(used_steps as f64),
        Value::F32(used_diff),
        Value::F32(used_thr),
    ]))
}

fn builtin_pocket_atlas_update_from_hits(args: Vec<Value>) -> Result<Value, SiggError> {
    // pocket_atlas_update(world_h, atlas_h, hits, beta, top_per_chunk)
    need_n(&args, 5, "pocket_atlas_update")?;
    let wh = as_u32_handle(&args[0])?;
    let ah = as_u32_handle(&args[1])?;
    let hits = value_to_hits(&args[2])?;
    let beta = as_f32(&args[3])?;
    let top_per_chunk = as_usize(&args[4])?;

    let mut wreg = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = wreg.get_mut(wh as usize).ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let mut areg = atlases().lock().map_err(|_| SiggError::runtime("atlas registry poisoned"))?;
    let a = areg.get_mut(ah as usize).ok_or_else(|| SiggError::runtime("invalid atlas handle"))?;

    w.atlas_update_from_hits(a, &hits, beta, top_per_chunk);
    Ok(Value::Unit)
}


pub fn builtin_add(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("add expects 2 arguments"));
    }
    match (&args[0], &args[1]) {
        // 数値 + 数値
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)), // Number型がある場合
        
        // Tensor + Tensor (Autograd対応)
        (Value::Tensor(a), Value::Tensor(b)) => {
            let res = Tensor::add_graph(a.clone(), b.clone());
            Ok(Value::Tensor(res))
        },

        // 型不一致などの場合
        _ => Err(SiggError::runtime("Invalid types for add (supports Number or Tensor)")),
    }
}

// 引き算 (-)
pub fn builtin_sub(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("sub expects 2 arguments"));
    }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
        
        (Value::Tensor(a), Value::Tensor(b)) => {
            let res = Tensor::sub_graph(a.clone(), b.clone());
            Ok(Value::Tensor(res))
        },
        _ => Err(SiggError::runtime("Invalid types for sub")),
    }
}

// 掛け算 (*)
pub fn builtin_mul(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("mul expects 2 arguments"));
    }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),

        (Value::Tensor(a), Value::Tensor(b)) => {
            let res = Tensor::mul_graph(a.clone(), b.clone());
            Ok(Value::Tensor(res))
        },
        _ => Err(SiggError::runtime("Invalid types for mul")),
    }
}
pub fn builtin_div(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("div expects 2 arguments"));
    }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 { return Err(SiggError::runtime("division by zero")); }
            Ok(Value::Int(a / b))
        },
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a / b)),

        // Tensor / Tensor
        (Value::Tensor(a), Value::Tensor(b)) => {
            let res = Tensor::div_graph(a.clone(), b.clone());
            Ok(Value::Tensor(res))
        },
        _ => Err(SiggError::runtime("Invalid types for div")),
    }
}
// 単項マイナス (Neg)
pub fn builtin_neg(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("neg expects 1 argument"));
    }
    match &args[0] {
        Value::Int(a) => Ok(Value::Int(-a)),
        Value::Float(a) => Ok(Value::Float(-a)),
        Value::Number(a) => Ok(Value::Number(-a)),
        
        // ★修正: 0-t ではなく、専用の neg_graph を呼ぶ
        Value::Tensor(t) => {
            let res = Tensor::neg_graph(t.clone());
            Ok(Value::Tensor(res))
        }
        _ => Err(SiggError::runtime("Invalid type for neg")),
    }
}
// 剰余 (%)
pub fn builtin_mod(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("mod expects 2 arguments"));
    }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 { return Err(SiggError::runtime("modulo by zero")); }
            Ok(Value::Int(a % b))
        },
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a % b)),
        
        // ★追加: TensorのMod対応 (高速化のため微分は切断)
        (Value::Tensor(a), Value::Tensor(b)) => {
            let res = Tensor::rem_graph(a.clone(), b.clone());
            Ok(Value::Tensor(res))
        },
        
        _ => Err(SiggError::runtime("Invalid types for mod")),
    }
}
// 1. builtin_exp 関数を追加
pub fn builtin_exp(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("exp expects 1 argument"));
    }
    match &args[0] {
        Value::Int(a) => Ok(Value::Float((*a as f64).exp())),
        Value::Float(a) => Ok(Value::Float(a.exp())),
        Value::Number(a) => Ok(Value::Float(a.exp())), // Number(f64)の場合
        Value::Tensor(t) => {
            let res = Tensor::exp_graph(t.clone());
            Ok(Value::Tensor(res))
        },
        _ => Err(SiggError::runtime("Invalid type for exp")),
    }
}
pub fn builtin_zero_grad(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("zero_grad expects 1 argument"));
    }
    match &args[0] {
        Value::Tensor(t) => {
            let mut tensor = t.write().unwrap();
            // 勾配を None に戻す（リセット）
            tensor.grad = None; 
            Ok(Value::Int(0)) // ダミーの戻り値
        },
        _ => Err(SiggError::runtime("zero_grad expects a tensor")),
    }
}


//new　↓

// ==============================
// Pocket registries (stateful builtins)
// ==============================
static WORLDS: OnceLock<Mutex<Vec<pocket::PocketWorld>>> = OnceLock::new();
static ATLASES: OnceLock<Mutex<Vec<pocket::Atlas>>> = OnceLock::new();
// ==============================
// AI runtime state (PocketState singleton)
// ==============================
static AI_STATE: OnceLock<Mutex<PocketState>> = OnceLock::new();

fn ai_state() -> &'static Mutex<PocketState> {
    AI_STATE.get_or_init(|| {
        // server.rs と同じデフォルト
        let data_dir: String = std::env::var("SIGG_DATA_DIR").unwrap_or_else(|_| "sigg_data".to_string());
        let z_dim_default: usize = std::env::var("SIGG_ZDIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);
        Mutex::new(PocketState::new(data_dir, z_dim_default))
    })
}
fn as_u32(v: &Value) -> Result<u32, SiggError> {
    let n = as_f64(v)?;
    if n < 0.0 { return Err(SiggError::runtime("expected u32")); }
    Ok(n as u32)
}
fn as_i32(v: &Value) -> Result<i32, SiggError> { Ok(as_f64(v)? as i32) }
#[allow(dead_code)]
fn as_worldkey3(args: &Vec<Value>, i: usize) -> Result<WorldKey, SiggError> {
    Ok((as_u32(&args[i])?, as_u32(&args[i+1])?, as_u32(&args[i+2])?))
}
fn worlds() -> &'static Mutex<Vec<pocket::PocketWorld>> {
    WORLDS.get_or_init(|| Mutex::new(Vec::new()))
}
fn atlases() -> &'static Mutex<Vec<pocket::Atlas>> {
    ATLASES.get_or_init(|| Mutex::new(Vec::new()))
}
#[allow(dead_code)]
fn world_get_mut(h: u32) -> Result<std::sync::MutexGuard<'static, Vec<pocket::PocketWorld>>, SiggError> {
    let g = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    if h as usize >= g.len() { return Err(SiggError::runtime("invalid world handle")); }
    Ok(g)
}
#[allow(dead_code)]
fn atlas_get_mut(h: u32) -> Result<std::sync::MutexGuard<'static, Vec<pocket::Atlas>>, SiggError> {
    let g = atlases().lock().map_err(|_| SiggError::runtime("atlas registry poisoned"))?;
    if h as usize >= g.len() { return Err(SiggError::runtime("invalid atlas handle")); }
    Ok(g)
}
fn as_u32_handle(v: &Value) -> Result<u32, SiggError> {
    match v {
        Value::Handle(h) => Ok(*h),
        Value::Number(n) if *n >= 0.0 => Ok(*n as u32),
        _ => Err(SiggError::runtime("expected handle")),
    }
}
fn as_worldkey(args: &[Value], i: usize) -> Result<WorldKey, SiggError> {
    let u = as_f64(&args[i])? as u32;
    let v = as_f64(&args[i+1])? as u32;
    let w = as_f64(&args[i+2])? as u32;
    Ok((u,v,w))
}
// query: either Grid([z_dim]) or Tuple(numbers...)
fn as_query_vec(v: &Value) -> Result<Vec<f32>, SiggError> {
    match v {
        Value::Grid(g) => Ok(g.data.clone()),
        Value::Tuple(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for it in xs { out.push(as_f32(it)?); }
            Ok(out)
        }
        _ => Err(SiggError::runtime("expected query vector (grid or tuple)")),
    }
}
fn as_chunks(v: &Value) -> Result<Vec<ChunkKey>, SiggError> {
    match v {
        Value::Tuple(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for ck in xs {
                match ck {
                    Value::Tuple(t) if t.len()==3 => {
                        out.push((as_i32(&t[0])?, as_i32(&t[1])?, as_i32(&t[2])?));
                    }
                    _ => return Err(SiggError::runtime("chunks must be tuple of (cx,cy,cz)")),
                }
            }
            Ok(out)
        }
        _ => Err(SiggError::runtime("expected chunks tuple")),
    }
}
fn hits_to_value(hits: Vec<Hit>) -> Value {
    // hit = (wu,wv,ww, x,y,z, score)
    let mut out: Vec<Value> = Vec::with_capacity(hits.len());
    for h in hits {
        out.push(Value::Tuple(vec![
            Value::Number(h.world.0 as f64),
            Value::Number(h.world.1 as f64),
            Value::Number(h.world.2 as f64),
            Value::Number(h.cell.0 as f64),
            Value::Number(h.cell.1 as f64),
            Value::Number(h.cell.2 as f64),
            Value::F32(h.score),
        ]));
    }
    Value::Tuple(out)
}
#[allow(dead_code)]
fn guard_io_xyz(mut p: (i32,i32,i32)) -> (i32,i32,i32) {
    // 命令領域(0..PROG_MAX-1)への衝突を防ぐ
    if p.0 >= 0 && p.0 < PROG_MAX {
        p.0 = IO_BASE;
    }
    p
}
#[allow(dead_code)]
fn guard_io_pair(io_in: (i32,i32,i32), io_out: (i32,i32,i32)) -> ((i32,i32,i32),(i32,i32,i32)) {
    let a = guard_io_xyz(io_in);
    let mut b = guard_io_xyz(io_out);

    // io_in と io_out が同じxにならないよう最低限ずらす
    if a.0 == b.0 {
        b.0 += 1;
    }

    // policy/modeセルが io_out+1 / io_out+2 を使う想定なら
    // それらも命令領域に落ちないように（念のため）
    if b.0 + 2 >= 0 && b.0 + 2 < PROG_MAX {
        b.0 = IO_BASE + 1;
    }
    (a,b)
}
fn value_to_hits(v: &Value) -> Result<Vec<Hit>, SiggError> {
    match v {
        Value::Tuple(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for it in xs {
                match it {
                    Value::Tuple(t) if t.len()==7 => {
                        let world = (
                            as_f64(&t[0])? as u32,
                            as_f64(&t[1])? as u32,
                            as_f64(&t[2])? as u32,
                        );
                        let cell = (
                            as_f64(&t[3])? as i32,
                            as_f64(&t[4])? as i32,
                            as_f64(&t[5])? as i32,
                        );
                        let score = as_f32(&t[6])?;
                        out.push(Hit { world, cell, score });
                    }
                    _ => return Err(SiggError::runtime("hits must be tuple of (wu,wv,ww,x,y,z,score)")),
                }
            }
            Ok(out)
        }
        _ => Err(SiggError::runtime("expected hits tuple")),
    }
}
#[allow(dead_code)]
fn load_program_into_space(space: &mut ComputeSpace, prog: &[u32]) {
    for (i, &inst) in prog.iter().enumerate() {
        space.write_cell_bits(i as i32, 0, 0, 0, inst);
    }
}
// ---------- basic builtins ----------
// 値をきれいな文字列に変換するヘルパー関数
fn value_to_string(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        // Value::F32(f) => f.to_string(), // 必要ならコメントアウト解除
        Value::Str(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Vec(v) | Value::List(v) => format!("{:?}", v), // リストはデバッグ表示
        _ => format!("{:?}", v), // その他はデバッグ表示
    }
}
pub fn builtin_print(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.is_empty() {
        println!();
        return Ok(Value::Int(0));
    }

    // 第1引数が文字列で、かつ "{}" を含んでいるかチェック
    let first_arg = &args[0];
    if let Value::Str(fmt_str) = first_arg {
        if fmt_str.contains("{}") {
            // --- フォーマット出力モード ---
            let mut output = fmt_str.clone();
            
            // args[1] 以降の引数を順番に埋め込む
            for i in 1..args.len() {
                let val_str = value_to_string(&args[i]);
                // 文字列中の最初の "{}" を値に置き換える
                output = output.replacen("{}", &val_str, 1);
            }
            
            println!("{}", output);
            return Ok(Value::Int(0));
        }
    }

    // --- 従来モード (引数をスペース区切りで表示) ---
    // 例: print(a, b) -> "10 20"
    let mut output = String::new();
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            output.push(' ');
        }
        output.push_str(&value_to_string(arg));
    }
    println!("{}", output);

    Ok(Value::Int(0))
}
fn format_value(v: &Value) -> String {
    match v {
        Value::Unit => "()".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(n) => n.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Str(s) => s.clone(), 

        // ★修正箇所: .read() を削除して直接 items.iter() を使います
        Value::List(items) => {
            let parts: Vec<String> = items.iter().map(|v| format_value(v)).collect();
            format!("[{}]", parts.join(", "))
        }

        Value::Tensor(_) => "<Tensor>".to_string(),//"".to_string(),
        
        // 関数やNativeFunctionなど
        _ => "<Value>".to_string(),//"".to_string(),
    }
}

fn builtin_vec_get(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "vec_get")?;
    match &args[0] {
        Value::Vec(v) => {
            let idx = as_usize(&args[1])?;
            v.get(idx).cloned().ok_or_else(|| SiggError::runtime("vec index out of bounds"))
        }
        _ => Err(SiggError::runtime("vec_get expects vec and index")),
    }
}

fn builtin_vec_set(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "vec_set")?;
    match &args[0] {
        Value::Vec(v) => {
            let idx = as_usize(&args[1])?;
            if idx >= v.len() {
                return Err(SiggError::runtime("vec index out of bounds"));
            }
            // Note: Vec is immutable in current design, return new vec
            let mut new_v = v.clone();
            new_v[idx] = args[2].clone();
            Ok(Value::Vec(new_v))
        }
        _ => Err(SiggError::runtime("vec_set expects vec, index, value")),
    }
}

fn builtin_map_get(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "map_get")?;
    match &args[0] {
        Value::Map(m) => {
            let key = as_string(&args[1])?;
            m.get(&key).cloned().ok_or_else(|| SiggError::runtime("map key not found"))
        }
        _ => Err(SiggError::runtime("map_get expects map and key")),
    }
}

fn builtin_map_set(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "map_set")?;
    match &args[0] {
        Value::Map(m) => {
            let key = as_string(&args[1])?;
            let mut new_m = m.clone();
            new_m.insert(key, args[2].clone());
            Ok(Value::Map(new_m))
        }
        _ => Err(SiggError::runtime("map_set expects map, key, value")),
    }
}

fn builtin_vec_new(_args: Vec<Value>) -> Result<Value, SiggError> {
    Ok(Value::Vec(Vec::new()))
}

fn builtin_map_new(_args: Vec<Value>) -> Result<Value, SiggError> {
    Ok(Value::Map(HashMap::new()))
}

fn builtin_vec_push(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "vec_push")?;
    match &args[0] {
        Value::Vec(v) => {
            let mut new_v = v.clone();
            new_v.push(args[1].clone());
            Ok(Value::Vec(new_v))
        }
        _ => Err(SiggError::runtime("vec_push expects vec and value")),
    }
}

fn builtin_vec_len(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "vec_len")?;
    match &args[0] {
        Value::Vec(v) => Ok(Value::F32(v.len() as f32)),
        _ => Err(SiggError::runtime("vec_len expects vec")),
    }
}

fn builtin_map_len(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "map_len")?;
    match &args[0] {
        Value::Map(m) => Ok(Value::F32(m.len() as f32)),
        _ => Err(SiggError::runtime("map_len expects map")),
    }
}

// noise2(w,h,seed) -> grid 0..1 (hash-based, coordinate deterministic)
fn builtin_noise2(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "noise2")?;
    let w = as_usize(&args[0])?;
    let h = as_usize(&args[1])?;
    let seed = as_i64(&args[2])? as u32;

    let mut out = Grid::new(vec![w, h], 0.0, Boundary::Wrap);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            out.data[idx] = noise01_from_xy_seed(x as u32, y as u32, seed);
        }
    }
    Ok(Value::Grid(Arc::new(out)))
}
// rand2(w,h,seed) -> grid 0..1 (LCG-based, sequential)
fn builtin_rand2(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "rand2")?;
    let w = as_usize(&args[0])?;
    let h = as_usize(&args[1])?;
    let seed = as_i64(&args[2])? as u64;

    let mut g = Grid::new(vec![w, h], 0.0, Boundary::Wrap);
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    for v in &mut g.data {
        *v = u32_to_01(lcg_next(&mut s));
    }
    Ok(Value::Grid(Arc::new(g)))
}
// mix(a,b,t): num/num/num or num/num/grid(t)
fn builtin_mix(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "mix")?;
    let a = args[0].clone();
    let b = args[1].clone();
    let t = args[2].clone();

    match (a, b, t) {
        (Value::Number(a), Value::Number(b), Value::Number(t)) => Ok(Value::Number(a * (1.0 - t) + b * t)),
        (Value::Number(a), Value::Number(b), Value::Grid(gt)) => {
            let tt = gt.as_ref();
            let mut out = Grid::new(tt.dims.clone(), 0.0, tt.boundary.clone());
            for i in 0..tt.data.len() {
                let ti = tt.data[i] as f64;
                out.data[i] = (a * (1.0 - ti) + b * ti) as f32;
            }
            Ok(Value::Grid(Arc::new(out)))
        }
        _ => Err(SiggError::runtime("mix: supported forms mix(num,num,num) or mix(num,num,grid)")),
    }
}
// clamp(x, lo, hi): x can be number/f32/grid
fn builtin_clamp(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "clamp")?;
    let lo = as_f32(&args[1])?;
    let hi = as_f32(&args[2])?;
    match &args[0] {
        Value::Number(n) => Ok(Value::Number((*n as f32).clamp(lo, hi) as f64)),
        Value::F32(x) => Ok(Value::F32((*x).clamp(lo, hi))),
        Value::Grid(g) => {
            let gg = g.as_ref();
            let mut out = Grid::new(gg.dims.clone(), 0.0, gg.boundary.clone());
            for i in 0..gg.data.len() { out.data[i] = gg.data[i].clamp(lo, hi); }
            Ok(Value::Grid(Arc::new(out)))
        }
        _ => Err(SiggError::runtime("clamp: expected number or grid")),
    }
}
// project(grid, constraint_id): id=0 => clamp 0..1
fn builtin_project(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "project")?;
    let id = as_usize(&args[1])?;
    let g = as_grid(&args[0])?;
    let gg = g.as_ref();
    match id {
        0 => {
            let mut out = Grid::new(gg.dims.clone(), 0.0, gg.boundary.clone());
            for i in 0..gg.data.len() { out.data[i] = gg.data[i].clamp(0.0, 1.0); }
            Ok(Value::Grid(Arc::new(out)))
        }
        _ => Err(SiggError::runtime("unknown constraint_id")),
    }
}
// reaction_gs(u,v,du,dv,f,k,dt) -> (u2,v2)  (scalar, single-thread baseline)
fn builtin_reaction_gs(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 7, "reaction_gs")?;
    let u = as_grid(&args[0])?;
    let v = as_grid(&args[1])?;
    let du = as_f32(&args[2])?;
    let dv = as_f32(&args[3])?;
    let ff = as_f32(&args[4])?;
    let kk = as_f32(&args[5])?;
    let dt = as_f32(&args[6])?;

    let uu = u.as_ref();
    let vv = v.as_ref();
    if uu.rank() != 2 || vv.rank() != 2 { return Err(SiggError::runtime("reaction_gs expects 2D grids")); }
    if uu.dims != vv.dims { return Err(SiggError::runtime("reaction_gs shape mismatch")); }
    let (w, h) = (uu.dims[0], uu.dims[1]);

    let mut out_u = Grid::new(vec![w, h], 0.0, uu.boundary.clone());
    let mut out_v = Grid::new(vec![w, h], 0.0, uu.boundary.clone());

    for y in 0..h {
        let yis = y as isize;
        for x in 0..w {
            let xis = x as isize;

            let u0 = uu.get2(xis, yis);
            let v0 = vv.get2(xis, yis);

            let lap_u =
                uu.get2(xis - 1, yis) + uu.get2(xis + 1, yis) + uu.get2(xis, yis - 1) + uu.get2(xis, yis + 1)
                - 4.0 * u0;
            let lap_v =
                vv.get2(xis - 1, yis) + vv.get2(xis + 1, yis) + vv.get2(xis, yis - 1) + vv.get2(xis, yis + 1)
                - 4.0 * v0;

            let uvv = u0 * v0 * v0;

            let du_dt = du * lap_u - uvv + ff * (1.0 - u0);
            let dv_dt = dv * lap_v + uvv - (ff + kk) * v0;

            let idx = y * w + x;
            out_u.data[idx] = (u0 + dt * du_dt).clamp(0.0, 1.0);
            out_v.data[idx] = (v0 + dt * dv_dt).clamp(0.0, 1.0);
        }
    }

    Ok(Value::Tuple(vec![Value::Grid(Arc::new(out_u)), Value::Grid(Arc::new(out_v))]))
}
fn builtin_ai_pocket_open(args: Vec<Value>) -> Result<Value, SiggError> {
    // ai_pocket_open(u,v,w, chunk_size, z_dim, delta_path) -> pocket_handle(u32)
    need_n(&args, 6, "ai_pocket_open")?;
    let world: WorldKey = (as_u32(&args[0])?, as_u32(&args[1])?, as_u32(&args[2])?);
    let chunk_size = as_usize(&args[3])?;
    let z_dim = as_usize(&args[4])?;
    let delta_path = match &args[5] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("ai_pocket_open: delta_path must be string")),
    };

    let pw = PocketWorld::open(world, chunk_size, z_dim, delta_path)
        .map_err(|e| SiggError::runtime(format!("ai_pocket_open failed: {e}")))?;

    let mut st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let h = st.next_handle;
    st.next_handle += 1;
    st.pockets.insert(h, pw);
    Ok(Value::Number(h as f64))
}
fn builtin_ai_pocket_write_f32(args: Vec<Value>) -> Result<Value, SiggError> {
    // ai_pocket_write_f32(pocket_h, x,y,z, lane, val)
    need_n(&args, 6, "ai_pocket_write_f32")?;
    let h = as_u32(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let lane = as_usize(&args[4])?;
    let val = as_f32(&args[5])?;

    let mut st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let p = st.pockets.get_mut(&h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    p.cell_write_f32(x,y,z,lane,val);
    Ok(Value::Unit)
}
fn builtin_ai_pocket_read_f32(args: Vec<Value>) -> Result<Value, SiggError> {
    // ai_pocket_read_f32(pocket_h, x,y,z, lane) -> f32
    need_n(&args, 5, "ai_pocket_read_f32")?;
    let h = as_u32(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let lane = as_usize(&args[4])?;

    let mut st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let p = st.pockets.get_mut(&h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    Ok(Value::F32(p.cell_read_f32(x,y,z,lane)))
}
fn builtin_ai_create(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "ai_create")?;
    let pocket_h = as_f64(&args[0])? as u32;

    let mut st = ai_state()
        .lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;

    let agent_id = crate::ai_runtime::ai_create_core(
        &mut st,
        (0, 0, 0),
        pocket_h,
        (100, 0, 0),
        (101, 0, 0),
        0.20,
        1024,
        1,
    )?;

    Ok(Value::Number(agent_id as f64))
}

fn builtin_ai_tick(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "ai_tick")?;
    let agent_id = as_f64(&args[0])? as u64;
    let budget   = as_f64(&args[1])? as u32;

    let mut st = ai_state()
        .lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;

    let r = crate::ai_runtime::ai_tick_core(&mut st, agent_id, budget)?;

    // デバッグ表示（好きなら維持）
    // println!("mu=\n{}\nmode=\n{}", r.mu, r.mode);
    // println!("tick=\n{}\nenv=\n{}\nin=\n{}\nout=\n{}\nexp=\n{}\nok=\n{}",
    //     r.tick, r.env_bit, r.in_bits, r.out_bits, r.expect, if r.ok {1} else {0}
    // );

    // ★ SIGG 側が期待している 8 要素
    Ok(Value::Tuple(vec![
        Value::Number(r.tick as f64),
        Value::Number(r.env_bit as f64),
        Value::Number(r.policy_bits as f64),
        Value::Number(r.in_bits as f64),
        Value::Number(r.out_bits as f64),
        Value::Number(r.expect as f64),
        Value::Number(if r.ok { 1.0 } else { 0.0 }),
        Value::Number(r.mu as f64),
        Value::Number(r.mode as f64),
    ]))
}
// ai_now_tick(agent) -> Number (u64相当)
fn builtin_ai_now_tick(args: Vec<Value>) -> Result<Value, SiggError> {
    // agent は将来拡張用。現状はグローバルtickを返すだけ。
    if args.len() != 1 {
        return Err(SiggError::runtime("ai_now_tick(agent): expected 1 arg"));
    }
    let _agent = &args[0];
    let t: u64 = ai_runtime::now_tick(); // ← ai_runtime 側に関数を足す（下に差分）
    Ok(Value::Number(t as f64))
}

fn builtin_ai_get_score(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "ai_get_score")?;
    let agent_id = as_f64(&args[0])? as u64;
    let st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let ag = st.agents.get(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
    Ok(Value::F32(f32::from_bits(ag.score_mu_bits)))
}
fn builtin_ai_get_mode(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "ai_get_mode")?;
    let agent_id = as_f64(&args[0])? as u64;
    let st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let ag = st.agents.get(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
    Ok(Value::Number(ag.mode as f64))
}

fn builtin_ai_get_env_period(_args: Vec<Value>) -> Result<Value, SiggError> {
    Ok(Value::Int(ai_runtime::get_env_period() as i64))
}

fn builtin_ai_set_env_period(args: Vec<Value>) -> Result<Value, SiggError> {
    let period = as_u32(&args[0])?;
    ai_runtime::set_env_period(period);
    Ok(Value::Unit)
}
// ai_get_mu(agent) -> F32 (mu)
fn builtin_ai_get_mu(args: Vec<Value>) -> Result<Value, SiggError> {
    // 実体は ai_get_score と同じ（mu の別名）
    builtin_ai_get_score(args)
}
fn need_n(args: &Vec<Value>, n: usize, name: &str) -> Result<(), SiggError> {
    if args.len() != n {
        return Err(SiggError::runtime(format!("{name} expects {n} args")));
    }
    Ok(())
}

fn builtin_vec_map(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("vec_map expects 2 args: vec, fn"));
    }
    
    let vec = match &args[0] {
        Value::Vec(v) => v.clone(),
        _ => return Err(SiggError::runtime("vec_map: first arg must be vec")),
    };
    
    let func = &args[1];
    
    let mut result = Vec::new();
    for item in vec {
        // ここではシンプルなホスト関数呼び出しを想定
        // 実際のVM統合では、Lambda/Closureの呼び出しを実装する必要がある
        match func {
            Value::HostFunction { func, .. } => {
                result.push(func(vec![item])?);
            }
            Value::Lambda(_) | Value::Closure { .. } => {
                // VM統合が必要 - ここではプレースホルダー
                return Err(SiggError::runtime("vec_map: lambda execution requires VM context"));
            }
            _ => return Err(SiggError::runtime("vec_map: second arg must be function")),
        }
    }
    
    Ok(Value::Vec(result))
}

// filter 関数: vec_filter(vec, predicate) -> vec
fn builtin_vec_filter(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("vec_filter expects 2 args: vec, predicate"));
    }
    
    let vec = match &args[0] {
        Value::Vec(v) => v.clone(),
        _ => return Err(SiggError::runtime("vec_filter: first arg must be vec")),
    };
    
    let func = &args[1];
    
    let mut result = Vec::new();
    for item in vec {
        let keep = match func {
            Value::HostFunction { func, .. } => {
                match func(vec![item.clone()])? {
                    Value::Bool(b) => b,
                    _ => return Err(SiggError::runtime("vec_filter: predicate must return bool")),
                }
            }
            _ => return Err(SiggError::runtime("vec_filter: second arg must be function")),
        };
        
        if keep {
            result.push(item);
        }
    }
    
    Ok(Value::Vec(result))
}

// reduce 関数: vec_reduce(vec, initial, reducer) -> value
fn builtin_vec_reduce(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 3 {
        return Err(SiggError::runtime("vec_reduce expects 3 args: vec, initial, reducer"));
    }
    
    let vec = match &args[0] {
        Value::Vec(v) => v.clone(),
        _ => return Err(SiggError::runtime("vec_reduce: first arg must be vec")),
    };
    
    let mut acc = args[1].clone();
    let func = &args[2];
    
    for item in vec {
        acc = match func {
            Value::HostFunction { func, .. } => {
                func(vec![acc, item])?
            }
            _ => return Err(SiggError::runtime("vec_reduce: third arg must be function")),
        };
    }
    
    Ok(acc)
}

// スナップショット作成
fn builtin_snapshot_create(args: Vec<Value>) -> Result<Value, SiggError> {
    let seed = if args.is_empty() {
        use std::time::SystemTime;
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
            .unwrap().as_secs()
    } else {
        match &args[0] {
            Value::Number(n) => *n as u64,
            Value::Int(n) => *n as u64,
            _ => return Err(SiggError::runtime("snapshot_create: seed must be number")),
        }
    };
    
    Ok(Value::Snapshot {
        timestamp: Instant::now(),
        seed,
        values: args,
        metadata: HashMap::new(),
    })
}

// ログ出力（構造化）
fn builtin_log(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.is_empty() {
        return Ok(Value::Unit);
    }
    
    let level = match &args[0] {
        Value::Str(s) if matches!(s.as_str(), "debug" | "info" | "warn" | "error") => s.as_str(),
        _ => "info",
    };
    
    let message = if args.len() > 1 {
        format!("{}", args[1])
    } else {
        format!("{}", args[0])
    };
    
    // 構造化ログとしてJSON形式で出力
    let log_entry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "level": level,
        "message": message,
    });
    
    println!("{}", log_entry);
    Ok(Value::Unit)
}

// 名前空間作成
fn builtin_namespace_create(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.is_empty() {
        return Err(SiggError::runtime("namespace_create requires name"));
    }
    
    let name = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("namespace_create: name must be string")),
    };
    
    Ok(Value::Namespace {
        name,
        members: HashMap::new(),
    })
}

//new　↑



// ---------- small helpers ----------


fn as_f64(v: &Value) -> Result<f64, SiggError> {
    match v {
        Value::Int(n) => Ok(*n as f64),
        Value::Float(f) => Ok(*f),
        Value::Number(n) => Ok(*n),
        Value::F32(x) => Ok(*x as f64),
        _ => Err(SiggError::runtime("expected number")),
    }
}
fn as_f32(v: &Value) -> Result<f32, SiggError> { Ok(as_f64(v)? as f32) }
fn as_usize(v: &Value) -> Result<usize, SiggError> {
    let n = as_f64(v)?;
    if n < 0.0 { return Err(SiggError::runtime("expected non-negative integer")); }
    Ok(n as usize)
}
fn as_i64(v: &Value) -> Result<i64, SiggError> { Ok(as_f64(v)? as i64) }

fn as_grid(v: &Value) -> Result<GridRef, SiggError> {
    match v {
        Value::Grid(g) => Ok(g.clone()),
        _ => Err(SiggError::runtime("expected grid")),
    }
}

// ---------- RNG helpers ----------
fn lcg_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

// hash-based (coordinate noise)
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}
fn u32_to_01(x: u32) -> f32 { (x as f32) / (u32::MAX as f32) }

#[inline]
fn noise01_from_xy_seed(x: u32, y: u32, seed: u32) -> f32 {
    let h = hash_u32(
        seed
            ^ x.wrapping_mul(0x9E37_79B1)
            ^ y.wrapping_mul(0x85EB_CA6B),
    );
    u32_to_01(h)
}
// world_step_gs(u,v,du,dv,f,k,dt,eps,seed,constraint_id) -> (u2,v2)
// - wrap neighborhood
// - hash-based noise injected on-the-fly
// - clamp/project included
// - rayon parallel rows
fn b_world_step_gs(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 10, "world_step_gs")?;

    let u = as_grid(&args[0])?;
    let v = as_grid(&args[1])?;
    let du = as_f32(&args[2])?;
    let dv = as_f32(&args[3])?;
    let f = as_f32(&args[4])?;
    let k = as_f32(&args[5])?;
    let dt = as_f32(&args[6])?;
    let eps = as_f32(&args[7])?;
    let seed = as_i64(&args[8])? as u32;
    let cid = as_i64(&args[9])?;

    if u.dims != v.dims { return Err(SiggError::runtime("world_step_gs: dim mismatch")); }
    if u.rank() != 2 { return Err(SiggError::runtime("world_step_gs: expects 2D grids")); }

    if cid != 0 {
        return Err(SiggError::runtime("world_step_gs: unknown constraint_id (only 0 supported)"));
    }

    let w = u.dims[0];
    let h = u.dims[1];

    if !matches!(u.boundary, Boundary::Wrap) || !matches!(v.boundary, Boundary::Wrap) {
        return Err(SiggError::runtime("world_step_gs: only Boundary::Wrap supported"));
    }

    let mut u2 = Grid::new(vec![w, h], 0.0, Boundary::Wrap);
    let mut v2 = Grid::new(vec![w, h], 0.0, Boundary::Wrap);

    u2.data
        .par_chunks_mut(w)
        .zip(v2.data.par_chunks_mut(w))
        .enumerate()
        .for_each(|(yy, (u_row, v_row))| {
            let y = yy;
            let ym = if y == 0 { h - 1 } else { y - 1 };
            let yp = if y + 1 == h { 0 } else { y + 1 };

            let y_u = y * w;
            let ym_u = ym * w;
            let yp_u = yp * w;

            for x in 0..w {
                let xm = if x == 0 { w - 1 } else { x - 1 };
                let xp = if x + 1 == w { 0 } else { x + 1 };

                let i = y_u + x;

                let c_u = u.data[i];
                let c_v = v.data[i];

                let lap_u =
                    u.data[y_u + xm] + u.data[y_u + xp] +
                    u.data[ym_u + x ] + u.data[yp_u + x ] -
                    4.0 * c_u;

                let lap_v =
                    v.data[y_u + xm] + v.data[y_u + xp] +
                    v.data[ym_u + x ] + v.data[yp_u + x ] -
                    4.0 * c_v;

                let uvv = c_u * c_v * c_v;
                let du_dt = du * lap_u - uvv + f * (1.0 - c_u);
                let dv_dt = dv * lap_v + uvv - (f + k) * c_v;

                let mut nu = c_u + dt * du_dt;
                let mut nv = c_v + dt * dv_dt;

                // bench_world.sigg の seed+777 に合わせる
                let rn = noise01_from_xy_seed(x as u32, yy as u32, seed.wrapping_add(777));
                nv = nv + (rn - 0.5) * eps;

                // project(id=0) == clamp 0..1
                nu = nu.clamp(0.0, 1.0);
                nv = nv.clamp(0.0, 1.0);

                u_row[x] = nu;
                v_row[x] = nv;
            }
        });

    Ok(Value::Tuple(vec![Value::Grid(Arc::new(u2)), Value::Grid(Arc::new(v2))]))
}

// ============================================================================
// Grid操作ヘルパー関数
// ============================================================================
/// 境界処理を考慮したGrid値の取得
fn get_with_boundary(g: &Grid, x: isize, y: isize, w: usize, h: usize) -> f32 {
    match g.boundary {
        Boundary::Wrap => {
            let xx = wrap_coord(x, w);
            let yy = wrap_coord(y, h);
            g.data[yy * w + xx]
        }
        Boundary::Clamp => {
            let xx = x.clamp(0, w as isize - 1) as usize;
            let yy = y.clamp(0, h as isize - 1) as usize;
            g.data[yy * w + xx]
        }
        Boundary::Zero => {
            if x < 0 || x >= w as isize || y < 0 || y >= h as isize {
                0.0
            } else {
                g.data[y as usize * w + x as usize]
            }
        }
        Boundary::Mirror => {
            let xx = mirror_coord(x, w);
            let yy = mirror_coord(y, h);
            g.data[yy * w + xx]
        }
    }
}
fn wrap_coord(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut r = i % n;
    if r < 0 { r += n; }
    r as usize
}
fn mirror_coord(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut r = i % (2 * n);
    if r < 0 { r += 2 * n; }
    if r >= n { r = 2 * n - r - 1; }
    r as usize
}
/// カーネルをVec<Vec<Float>>から変換
fn as_kernel(v: &Value) -> Result<Vec<Vec<f32>>, SiggError> {
    match v {
        Value::Vec(rows) => {
            let mut kernel = Vec::new();
            for row in rows {
                match row {
                    Value::Vec(cols) => {
                        let mut row_data = Vec::new();
                        for val in cols {
                            row_data.push(as_f32(val)?);
                        }
                        kernel.push(row_data);
                    }
                    _ => return Err(SiggError::runtime("kernel must be Vec<Vec<Float>>")),
                }
            }
            if kernel.is_empty() {
                return Err(SiggError::runtime("kernel cannot be empty"));
            }
            let width = kernel[0].len();
            for row in &kernel {
                if row.len() != width {
                    return Err(SiggError::runtime("kernel rows must have same length"));
                }
            }
            Ok(kernel)
        }
        _ => Err(SiggError::runtime("kernel must be Vec<Vec<Float>>")),
    }
}

// ============================================================================
// 1. grid_shift - グリッド平行移動
// ============================================================================
/// Grid を dx, dy だけ平行移動する
pub fn grid_shift_impl(g: &Grid, dx: i32, dy: i32) -> Grid {
    if g.rank() != 2 {
        panic!("grid_shift requires 2D grid");
    }
    
    let w = g.dims[0];
    let h = g.dims[1];
    let mut out = Grid::new(vec![w, h], 0.0, g.boundary);
    
    // 並列処理: 各行を並列に処理
    out.data.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let src_x = (x as i32 - dx) as isize;
                let src_y = (y as i32 - dy) as isize;
                
                // 境界処理を考慮して値を取得
                row[x] = get_with_boundary(g, src_x, src_y, w, h);
            }
        });
    
    out
}
fn builtin_grid_shift(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "grid_shift")?;
    let g = as_grid(&args[0])?;
    let dx = as_i32(&args[1])?;
    let dy = as_i32(&args[2])?;
    
    let result = grid_shift_impl(g.as_ref(), dx, dy);
    Ok(Value::Grid(Arc::new(result)))
}

// ============================================================================
// 2. grid_convolve - 畳み込み
// ============================================================================
/// 畳み込み演算（完全並列化版）
pub fn grid_convolve_impl(g: &Grid, kernel: &[Vec<f32>]) -> Grid {
    if g.rank() != 2 {
        panic!("grid_convolve requires 2D grid");
    }
    
    let kh = kernel.len();
    let kw = kernel[0].len();
    let ky_offset = kh as i32 / 2;
    let kx_offset = kw as i32 / 2;
    
    let w = g.dims[0];
    let h = g.dims[1];
    let mut out = Grid::new(vec![w, h], 0.0, g.boundary);
    
    // 並列処理: 各行を独立に処理
    out.data.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let mut sum = 0.0;
                
                // カーネルを適用
                for ky in 0..kh {
                    for kx in 0..kw {
                        let src_x = x as i32 + kx as i32 - kx_offset;
                        let src_y = y as i32 + ky as i32 - ky_offset;
                        
                        let val = get_with_boundary(
                            g, 
                            src_x as isize, 
                            src_y as isize, 
                            w, 
                            h
                        );
                        
                        sum += val * kernel[ky][kx];
                    }
                }
                
                row[x] = sum;
            }
        });
    
    out
}
fn builtin_grid_convolve(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "grid_convolve")?;
    let g = as_grid(&args[0])?;
    let kernel = as_kernel(&args[1])?;
    
    let result = grid_convolve_impl(g.as_ref(), &kernel);
    Ok(Value::Grid(Arc::new(result)))
}

// ============================================================================
// 3. grid_laplace - Laplacian オペレータ
// ============================================================================
/// Laplacian オペレータ（最適化版）
pub fn grid_laplace_impl(g: &Grid) -> Grid {
    if g.rank() != 2 {
        panic!("grid_laplace requires 2D grid");
    }
    
    let w = g.dims[0];
    let h = g.dims[1];
    let mut out = Grid::new(vec![w, h], 0.0, g.boundary);
    
    out.data.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let yi = y as isize;
            for x in 0..w {
                let xi = x as isize;
                
                let center = g.get2(xi, yi);
                let up = g.get2(xi, yi - 1);
                let down = g.get2(xi, yi + 1);
                let left = g.get2(xi - 1, yi);
                let right = g.get2(xi + 1, yi);
                
                // Laplacian: ∇²f = up + down + left + right - 4*center
                row[x] = up + down + left + right - 4.0 * center;
            }
        });
    
    out
}
fn builtin_grid_laplace(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "grid_laplace")?;
    let g = as_grid(&args[0])?;
    
    let result = grid_laplace_impl(g.as_ref());
    Ok(Value::Grid(Arc::new(result)))
}

// ============================================================================
// 4. grid_sobel - Sobel エッジ検出
// ============================================================================
/// Sobel X方向エッジ検出
pub fn grid_sobel_x_impl(g: &Grid) -> Grid {
    // Sobel X kernel:
    // [-1  0  1]
    // [-2  0  2]
    // [-1  0  1]
    let kernel = vec![
        vec![-1.0, 0.0, 1.0],
        vec![-2.0, 0.0, 2.0],
        vec![-1.0, 0.0, 1.0],
    ];
    grid_convolve_impl(g, &kernel)
}
fn builtin_grid_sobel_x(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "grid_sobel_x")?;
    let g = as_grid(&args[0])?;
    
    let result = grid_sobel_x_impl(g.as_ref());
    Ok(Value::Grid(Arc::new(result)))
}
/// Sobel Y方向エッジ検出
pub fn grid_sobel_y_impl(g: &Grid) -> Grid {
    // Sobel Y kernel:
    // [-1 -2 -1]
    // [ 0  0  0]
    // [ 1  2  1]
    let kernel = vec![
        vec![-1.0, -2.0, -1.0],
        vec![ 0.0,  0.0,  0.0],
        vec![ 1.0,  2.0,  1.0],
    ];
    grid_convolve_impl(g, &kernel)
}
fn builtin_grid_sobel_y(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "grid_sobel_y")?;
    let g = as_grid(&args[0])?;
    
    let result = grid_sobel_y_impl(g.as_ref());
    Ok(Value::Grid(Arc::new(result)))
}

// ============================================================================
// 5. grid_get / grid_set - 要素アクセス
// ============================================================================
fn builtin_grid_get(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "grid_get")?;
    let g = as_grid(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    
    if g.rank() != 2 {
        return Err(SiggError::runtime("grid_get requires 2D grid"));
    }
    
    let w = g.dims[0];
    let h = g.dims[1];
    let val = get_with_boundary(g.as_ref(), x as isize, y as isize, w, h);
    
    Ok(Value::F32(val))
}
fn builtin_grid_set(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 4, "grid_set")?;
    let g = as_grid(&args[0])?;
    let x = as_usize(&args[1])?;
    let y = as_usize(&args[2])?;
    let value = as_f32(&args[3])?;
    
    if g.rank() != 2 {
        return Err(SiggError::runtime("grid_set requires 2D grid"));
    }
    
    let w = g.dims[0];
    let h = g.dims[1];
    
    if x >= w || y >= h {
        return Err(SiggError::runtime("grid_set: index out of bounds"));
    }
    
    // Gridは不変なので、新しいGridを作成
    let mut new_grid = (*g).clone();
    new_grid.set2(x, y, value);
    
    Ok(Value::Grid(Arc::new(new_grid)))
}

// ============================================================================
// 6. grid_map - 高階関数（マップ）
// ============================================================================
fn builtin_grid_map(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("grid_map expects 2 args: grid, fn"));
    }
    
    let g = as_grid(&args[0])?;
    let func = &args[1];
    
    let mut new_grid = (*g).clone();
    
    // 現状はホスト関数のみサポート（将来的にVM統合が必要）
    match func {
        Value::HostFunction { func, .. } => {
            for i in 0..new_grid.data.len() {
                let val = Value::F32(new_grid.data[i]);
                let result = func(vec![val])?;
                new_grid.data[i] = as_f32(&result)?;
            }
        }
        _ => return Err(SiggError::runtime("grid_map: second arg must be function (VM integration needed for lambdas)")),
    }
    
    Ok(Value::Grid(Arc::new(new_grid)))
}

// ============================================================================
// 7. grid_reduce - 高階関数（リデュース）
// ============================================================================
fn builtin_grid_reduce(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 3 {
        return Err(SiggError::runtime("grid_reduce expects 3 args: grid, init, fn"));
    }
    
    let g = as_grid(&args[0])?;
    let mut acc = args[1].clone();
    let func = &args[2];
    
    match func {
        Value::HostFunction { func, .. } => {
            for &val in &g.data {
                let result = func(vec![acc, Value::F32(val)])?;
                acc = result;
            }
        }
        _ => return Err(SiggError::runtime("grid_reduce: third arg must be function")),
    }
    
    Ok(acc)
}

// ============================================================================
// 8. grid_filter - 高階関数（フィルタ）
// ============================================================================
fn builtin_grid_filter(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("grid_filter expects 2 args: grid, predicate"));
    }
    
    let g = as_grid(&args[0])?;
    let func = &args[1];
    
    let mut new_grid = (*g).clone();
    
    match func {
        Value::HostFunction { func, .. } => {
            for i in 0..new_grid.data.len() {
                let val = Value::F32(new_grid.data[i]);
                let result = func(vec![val])?;
                match result {
                    Value::Bool(false) => new_grid.data[i] = 0.0, // フィルタ: false -> 0
                    Value::Bool(true) => {}, // 保持
                    _ => return Err(SiggError::runtime("grid_filter: predicate must return bool")),
                }
            }
        }
        _ => return Err(SiggError::runtime("grid_filter: second arg must be function")),
    }
    
    Ok(Value::Grid(Arc::new(new_grid)))
}

// ============================================================================
// 9. grid_stats - 統計情報
// ============================================================================
/// グリッドの統計情報を並列計算
pub fn grid_stats_impl(g: &Grid) -> (f32, f32, f32, f32) {
    let n = g.data.len() as f32;
    
    // 並列reduce
    let sum: f32 = g.data.par_iter().sum();
    let min: f32 = g.data.par_iter().copied()
        .reduce(|| f32::INFINITY, f32::min);
    let max: f32 = g.data.par_iter().copied()
        .reduce(|| f32::NEG_INFINITY, f32::max);
    
    let mean = sum / n;
    
    // 標準偏差
    let variance: f32 = g.data.par_iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>() / n;
    
    let std = variance.sqrt();
    
    (min, max, mean, std)
}


// ============================================================================
// 1. pocket_read_chunk - チャンク単位の読み取り
// ============================================================================
/// チャンク全体をGridとして読み取る
/// pocket_read_chunk(world_h, cx, cy, cz) -> Grid
fn builtin_pocket_read_chunk(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 4, "pocket_read_chunk")?;
    let wh = as_u32_handle(&args[0])?;
    let cx = as_i32(&args[1])?;
    let cy = as_i32(&args[2])?;
    let cz = as_i32(&args[3])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    // チャンクのサイズとz次元を取得
    let chunk_size = w.chunk_size;
    let z_dim = w.z_dim;
    
    // チャンクのベース座標を計算
    let base_x = cx * chunk_size as i32;
    let base_y = cy * chunk_size as i32;
    let base_z = cz * chunk_size as i32;
    
    // チャンク全体を読み取る（3D -> 2D Grid変換）
    // dims = [chunk_size * chunk_size, z_dim]
    let total_cells = chunk_size * chunk_size;
    let mut data = vec![0.0f32; total_cells * z_dim];
    
    for dy in 0..chunk_size {
        for dx in 0..chunk_size {
            let x = base_x + dx as i32;
            let y = base_y + dy as i32;
            let z = base_z;
            
            // z_dim分のレーンを読み取る
            for lane in 0..z_dim {
                let cell_value = w.cell_read_f32(x, y, z, lane);
                let idx = (dy * chunk_size + dx) * z_dim + lane;
                data[idx] = cell_value;
            }
        }
    }
    
    let grid = Grid {
        dims: vec![total_cells, z_dim],
        data,
        boundary: Boundary::Zero, // チャンクは境界外は0
    };
    
    Ok(Value::Grid(Arc::new(grid)))
}

// ============================================================================
// 2. pocket_write_chunk - チャンク単位の書き込み
// ============================================================================
/// Gridをチャンク全体に書き込む
/// pocket_write_chunk(world_h, cx, cy, cz, grid) -> Unit
fn builtin_pocket_write_chunk(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 5, "pocket_write_chunk")?;
    let wh = as_u32_handle(&args[0])?;
    let cx = as_i32(&args[1])?;
    let cy = as_i32(&args[2])?;
    let cz = as_i32(&args[3])?;
    let grid = as_grid(&args[4])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let chunk_size = w.chunk_size;
    let z_dim = w.z_dim;
    
    // Gridのサイズを検証
    if grid.rank() != 2 {
        return Err(SiggError::runtime("pocket_write_chunk: grid must be 2D"));
    }
    
    let total_cells = chunk_size * chunk_size;
    if grid.dims[0] != total_cells || grid.dims[1] != z_dim {
        return Err(SiggError::runtime(
            format!("pocket_write_chunk: grid size must be [{}, {}]", total_cells, z_dim)
        ));
    }
    
    // チャンクのベース座標を計算
    let base_x = cx * chunk_size as i32;
    let base_y = cy * chunk_size as i32;
    let base_z = cz * chunk_size as i32;
    
    // Gridをチャンクに書き込む
    for dy in 0..chunk_size {
        for dx in 0..chunk_size {
            let x = base_x + dx as i32;
            let y = base_y + dy as i32;
            let z = base_z;
            
            for lane in 0..z_dim {
                let idx = (dy * chunk_size + dx) * z_dim + lane;
                let value = grid.data[idx];
                w.cell_write_f32(x, y, z, lane, value);
            }
        }
    }
    
    Ok(Value::Unit)
}

// ============================================================================
// 3. pocket_slice_xy - XY平面スライス取得
// ============================================================================
/// XY平面（z固定）のスライスをGridとして取得
/// pocket_slice_xy(world_h, z, x_min, x_max, y_min, y_max, lane) -> Grid
fn builtin_pocket_slice_xy(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 7, "pocket_slice_xy")?;
    let wh = as_u32_handle(&args[0])?;
    let z = as_i32(&args[1])?;
    let x_min = as_i32(&args[2])?;
    let x_max = as_i32(&args[3])?;
    let y_min = as_i32(&args[4])?;
    let y_max = as_i32(&args[5])?;
    let lane = as_usize(&args[6])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;
    
    let mut data = Vec::with_capacity(width * height);
    
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let value = w.cell_read_f32(x, y, z, lane);
            data.push(value);
        }
    }
    
    let grid = Grid {
        dims: vec![width, height],
        data,
        boundary: Boundary::Wrap,
    };
    
    Ok(Value::Grid(Arc::new(grid)))
}

// ============================================================================
// 4. pocket_slice_xz - XZ平面スライス取得
// ============================================================================
/// XZ平面（y固定）のスライスをGridとして取得
/// pocket_slice_xz(world_h, y, x_min, x_max, z_min, z_max, lane) -> Grid
fn builtin_pocket_slice_xz(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 7, "pocket_slice_xz")?;
    let wh = as_u32_handle(&args[0])?;
    let y = as_i32(&args[1])?;
    let x_min = as_i32(&args[2])?;
    let x_max = as_i32(&args[3])?;
    let z_min = as_i32(&args[4])?;
    let z_max = as_i32(&args[5])?;
    let lane = as_usize(&args[6])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let width = (x_max - x_min + 1) as usize;
    let depth = (z_max - z_min + 1) as usize;
    
    let mut data = Vec::with_capacity(width * depth);
    
    for z in z_min..=z_max {
        for x in x_min..=x_max {
            let value = w.cell_read_f32(x, y, z, lane);
            data.push(value);
        }
    }
    
    let grid = Grid {
        dims: vec![width, depth],
        data,
        boundary: Boundary::Wrap,
    };
    
    Ok(Value::Grid(Arc::new(grid)))
}

// ============================================================================
// 5. pocket_slice_yz - YZ平面スライス取得
// ============================================================================
/// YZ平面（x固定）のスライスをGridとして取得
/// pocket_slice_yz(world_h, x, y_min, y_max, z_min, z_max, lane) -> Grid
fn builtin_pocket_slice_yz(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 7, "pocket_slice_yz")?;
    let wh = as_u32_handle(&args[0])?;
    let x = as_i32(&args[1])?;
    let y_min = as_i32(&args[2])?;
    let y_max = as_i32(&args[3])?;
    let z_min = as_i32(&args[4])?;
    let z_max = as_i32(&args[5])?;
    let lane = as_usize(&args[6])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    let height = (y_max - y_min + 1) as usize;
    let depth = (z_max - z_min + 1) as usize;
    
    let mut data = Vec::with_capacity(height * depth);
    
    for z in z_min..=z_max {
        for y in y_min..=y_max {
            let value = w.cell_read_f32(x, y, z, lane);
            data.push(value);
        }
    }
    
    let grid = Grid {
        dims: vec![height, depth],
        data,
        boundary: Boundary::Wrap,
    };
    
    Ok(Value::Grid(Arc::new(grid)))
}

// ============================================================================
// 6. pocket_fill_region - 領域の一括書き込み
// ============================================================================
/// 指定領域を特定の値で埋める
/// pocket_fill_region(world_h, x_min, x_max, y_min, y_max, z_min, z_max, lane, value) -> Unit
fn builtin_pocket_fill_region(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 9, "pocket_fill_region")?;
    let wh = as_u32_handle(&args[0])?;
    let x_min = as_i32(&args[1])?;
    let x_max = as_i32(&args[2])?;
    let y_min = as_i32(&args[3])?;
    let y_max = as_i32(&args[4])?;
    let z_min = as_i32(&args[5])?;
    let z_max = as_i32(&args[6])?;
    let lane = as_usize(&args[7])?;
    let value = as_f32(&args[8])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    for z in z_min..=z_max {
        for y in y_min..=y_max {
            for x in x_min..=x_max {
                w.cell_write_f32(x, y, z, lane, value);
            }
        }
    }
    
    Ok(Value::Unit)
}

// ============================================================================
// 7. pocket_copy_region - 領域のコピー
// ============================================================================
/// 領域を別の場所にコピー
/// pocket_copy_region(world_h, src_x, src_y, src_z, dst_x, dst_y, dst_z, 
///                    width, height, depth, lane) -> Unit
fn builtin_pocket_copy_region(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 11, "pocket_copy_region")?;
    let wh = as_u32_handle(&args[0])?;
    let src_x = as_i32(&args[1])?;
    let src_y = as_i32(&args[2])?;
    let src_z = as_i32(&args[3])?;
    let dst_x = as_i32(&args[4])?;
    let dst_y = as_i32(&args[5])?;
    let dst_z = as_i32(&args[6])?;
    let width = as_i32(&args[7])?;
    let height = as_i32(&args[8])?;
    let depth = as_i32(&args[9])?;
    let lane = as_usize(&args[10])?;

    let mut reg = worlds().lock()
        .map_err(|_| SiggError::runtime("world registry poisoned"))?;
    let w = reg.get_mut(wh as usize)
        .ok_or_else(|| SiggError::runtime("invalid world handle"))?;

    // バッファに一時的にコピー
    let mut buffer = Vec::with_capacity((width * height * depth) as usize);
    
    for dz in 0..depth {
        for dy in 0..height {
            for dx in 0..width {
                let value = w.cell_read_f32(
                    src_x + dx, 
                    src_y + dy, 
                    src_z + dz, 
                    lane
                );
                buffer.push(value);
            }
        }
    }
    
    // バッファから書き込み
    let mut idx = 0;
    for dz in 0..depth {
        for dy in 0..height {
            for dx in 0..width {
                w.cell_write_f32(
                    dst_x + dx, 
                    dst_y + dy, 
                    dst_z + dz, 
                    lane,
                    buffer[idx]
                );
                idx += 1;
            }
        }
    }
    
    Ok(Value::Unit)
}


// ============================================================================
// AI拡張API
// ============================================================================
// ----------------------------------------------------------------------------
// 1. ai_pocket_write_vec - ベクトル書き込み
// ----------------------------------------------------------------------------
/// AIエージェントのPocketにベクトルデータを書き込む
/// ai_pocket_write_vec(pocket_h, x, y, z, lane, vec) -> Unit
fn builtin_ai_pocket_write_vec(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 6, "ai_pocket_write_vec")?;
    let h = as_u32(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let base_lane = as_usize(&args[4])?;
    let vec_data = as_float_vec(&args[5])?;

    let mut st = ai_state().lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let p = st.pockets.get_mut(&h)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    
    // ベクトルの各要素を連続したレーンに書き込む
    for (i, &value) in vec_data.iter().enumerate() {
        p.cell_write_f32(x, y, z, base_lane + i, value);
    }
    
    Ok(Value::Unit)
}

// ----------------------------------------------------------------------------
// 2. ai_pocket_read_vec - ベクトル読み取り
// ----------------------------------------------------------------------------
/// AIエージェントのPocketからベクトルデータを読み取る
/// ai_pocket_read_vec(pocket_h, x, y, z, lane, length) -> Vec<Float>
fn builtin_ai_pocket_read_vec(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 6, "ai_pocket_read_vec")?;
    let h = as_u32(&args[0])?;
    let x = as_i32(&args[1])?;
    let y = as_i32(&args[2])?;
    let z = as_i32(&args[3])?;
    let base_lane = as_usize(&args[4])?;
    let length = as_usize(&args[5])?;

    let mut st = ai_state().lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    let p = st.pockets.get_mut(&h)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    
    // 連続したレーンからベクトルを読み取る
    let mut vec_data = Vec::with_capacity(length);
    for i in 0..length {
        let value = p.cell_read_f32(x, y, z, base_lane + i);
        vec_data.push(Value::F32(value));
    }
    
    Ok(Value::Vec(vec_data))
}

// ----------------------------------------------------------------------------
// 3. ai_tick_batch - バッチtick
// ----------------------------------------------------------------------------
/// 複数エージェントを一括でtick
/// ai_tick_batch(agents: Vec<Int>, budget: Int) -> Vec<Tuple>
fn builtin_ai_tick_batch(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "ai_tick_batch")?;
    
    let agent_ids = match &args[0] {
        Value::Vec(v) => {
            v.iter()
                .map(|val| match val {
                    Value::Number(n) => Ok(*n as u64),
                    Value::Int(i) => Ok(*i as u64),
                    _ => Err(SiggError::runtime("ai_tick_batch: agents must be numbers")),
                })
                .collect::<Result<Vec<u64>, _>>()?
        }
        _ => return Err(SiggError::runtime("ai_tick_batch: first arg must be Vec")),
    };
    
    let budget = as_f64(&args[1])? as u32;
    
    // シーケンシャルに実行（MutexGuardはSendでないため）
    let mut results = Vec::with_capacity(agent_ids.len());
    
    for &agent_id in &agent_ids {
        let mut st = ai_state().lock()
            .map_err(|_| SiggError::runtime("ai_state poisoned"))?;
        
        match ai_runtime::ai_tick_core(&mut st, agent_id, budget) {
            Ok(r) => {
                results.push(Value::Tuple(vec![
                    Value::Number(r.tick as f64),
                    Value::Number(r.env_bit as f64),
                    Value::Number(r.policy_bits as f64),
                    Value::Number(r.in_bits as f64),
                    Value::Number(r.out_bits as f64),
                    Value::Number(r.expect as f64),
                    Value::Number(if r.ok { 1.0 } else { 0.0 }),
                    Value::Number(r.mu as f64),
                    Value::Number(r.mode as f64),
                ]));
            }
            Err(e) => {
                eprintln!("ai_tick_batch error for agent {}: {}", agent_id, e);
                results.push(Value::Unit);
            }
        }
    }
    
    Ok(Value::Vec(results))
}

// ----------------------------------------------------------------------------
// 4. ai_set_memory - エージェントメモリの設定
// ----------------------------------------------------------------------------
/// エージェントのメモリ領域に値を設定
/// ai_set_memory(agent_id: Int, address: Int, value: Float) -> Unit
fn builtin_ai_set_memory(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "ai_set_memory")?;
    let agent_id = as_f64(&args[0])? as u64;
    let address = as_i32(&args[1])?;
    let value = as_f32(&args[2])?;
    
    let mut st = ai_state().lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    
    // エージェント情報を先に取得
    let (io_out, pocket_handle) = {
        let agent = st.agents.get(&agent_id)
            .ok_or_else(|| SiggError::runtime("bad agent id"))?;
        (agent.io_out, agent.pocket_handle)
    };
    
    // Pocketへのアクセスは別のスコープで
    let pocket = st.pockets.get_mut(&pocket_handle)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    
    // メモリ領域（io_out位置を使用）
    let (x, y, z) = io_out;
    pocket.cell_write_f32(x + address, y, z, 0, value);
    
    Ok(Value::Unit)
}

// ----------------------------------------------------------------------------
// 5. ai_get_memory - エージェントメモリの取得
// ----------------------------------------------------------------------------
/// エージェントのメモリ領域から値を取得
/// ai_get_memory(agent_id: Int, address: Int) -> Float
fn builtin_ai_get_memory(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "ai_get_memory")?;
    let agent_id = as_f64(&args[0])? as u64;
    let address = as_i32(&args[1])?;
    
    let mut st = ai_state().lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;
    
    // エージェント情報を先に取得
    let (io_out, pocket_handle) = {
        let agent = st.agents.get(&agent_id)
            .ok_or_else(|| SiggError::runtime("bad agent id"))?;
        (agent.io_out, agent.pocket_handle)
    };
    
    // Pocketへのアクセスは別のスコープで
    let pocket = st.pockets.get_mut(&pocket_handle)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    
    let (x, y, z) = io_out;
    let value = pocket.cell_read_f32(x + address, y, z, 0);
    
    Ok(Value::F32(value))
}


// ============================================================================
// 文字列操作API
// ============================================================================

// ----------------------------------------------------------------------------
// 1. str_split - 文字列分割
// ----------------------------------------------------------------------------
/// 文字列を区切り文字で分割
/// str_split(s: String, sep: String) -> Vec<String>
fn builtin_str_split(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "str_split")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_split: first arg must be string")),
    };
    
    let sep = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_split: second arg must be string")),
    };
    
    let parts: Vec<Value> = s.split(&sep)
        .map(|part| Value::Str(part.to_string()))
        .collect();
    
    Ok(Value::Vec(parts))
}

// ----------------------------------------------------------------------------
// 2. str_join - 文字列結合
// ----------------------------------------------------------------------------
/// 文字列のベクトルを区切り文字で結合
/// str_join(v: Vec<String>, sep: String) -> String
fn builtin_str_join(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "str_join")?;
    
    let parts = match &args[0] {
        Value::Vec(v) => {
            v.iter()
                .map(|val| match val {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(SiggError::runtime("str_join: vector must contain strings")),
                })
                .collect::<Result<Vec<String>, _>>()?
        }
        _ => return Err(SiggError::runtime("str_join: first arg must be Vec")),
    };
    
    let sep = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_join: second arg must be string")),
    };
    
    let result = parts.join(&sep);
    Ok(Value::Str(result))
}

// ----------------------------------------------------------------------------
// 3. str_replace - 文字列置換
// ----------------------------------------------------------------------------
/// 文字列内の部分文字列を置換
/// str_replace(s: String, from: String, to: String) -> String
fn builtin_str_replace(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 3, "str_replace")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_replace: first arg must be string")),
    };
    
    let from = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_replace: second arg must be string")),
    };
    
    let to = match &args[2] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_replace: third arg must be string")),
    };
    
    let result = s.replace(&from, &to);
    Ok(Value::Str(result))
}

// ----------------------------------------------------------------------------
// 4. str_contains - 部分文字列チェック
// ----------------------------------------------------------------------------
/// 文字列が部分文字列を含むかチェック
/// str_contains(s: String, substr: String) -> Bool
fn builtin_str_contains(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "str_contains")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_contains: first arg must be string")),
    };
    
    let substr = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_contains: second arg must be string")),
    };
    
    Ok(Value::Bool(s.contains(&substr)))
}

// ----------------------------------------------------------------------------
// 5. str_starts_with / str_ends_with - 前後チェック
// ----------------------------------------------------------------------------
fn builtin_str_starts_with(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "str_starts_with")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_starts_with: first arg must be string")),
    };
    
    let prefix = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_starts_with: second arg must be string")),
    };
    
    Ok(Value::Bool(s.starts_with(&prefix)))
}
fn builtin_str_ends_with(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 2, "str_ends_with")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_ends_with: first arg must be string")),
    };
    
    let suffix = match &args[1] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_ends_with: second arg must be string")),
    };
    
    Ok(Value::Bool(s.ends_with(&suffix)))
}

// ----------------------------------------------------------------------------
// 6. str_trim - 前後空白削除
// ----------------------------------------------------------------------------
fn builtin_str_trim(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "str_trim")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_trim: arg must be string")),
    };
    
    Ok(Value::Str(s.trim().to_string()))
}

// ----------------------------------------------------------------------------
// 7. str_to_upper / str_to_lower - 大文字小文字変換
// ----------------------------------------------------------------------------
fn builtin_str_to_upper(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "str_to_upper")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_to_upper: arg must be string")),
    };
    
    Ok(Value::Str(s.to_uppercase()))
}
fn builtin_str_to_lower(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "str_to_lower")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_to_lower: arg must be string")),
    };
    
    Ok(Value::Str(s.to_lowercase()))
}

// ----------------------------------------------------------------------------
// 8. str_len - 文字列長
// ----------------------------------------------------------------------------
fn builtin_str_len(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "str_len")?;
    
    let s = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err(SiggError::runtime("str_len: arg must be string")),
    };
    
    Ok(Value::Number(s.len() as f64))
}

// ============================================================================
// ヘルパー関数
// ============================================================================
/// Vec<Float>への変換
fn as_float_vec(v: &Value) -> Result<Vec<f32>, SiggError> {
    match v {
        Value::Vec(vec) => {
            vec.iter()
                .map(|val| as_f32(val))
                .collect()
        }
        Value::Tuple(vec) => {
            vec.iter()
                .map(|val| as_f32(val))
                .collect()
        }
        _ => Err(SiggError::runtime("expected Vec or Tuple of numbers")),
    }
}



// src/builtins.rs の builtin_tensor_new 関数全体を書き換え

// src/builtins.rs の builtin_tensor_new 関数全体を書き換え

fn builtin_tensor_new(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("tensor expects 1 argument (shape list)"));
    }

    let shape_val = &args[0];
    let mut shape = Vec::new();

    // 1. リストからシェイプを取り出す
    match shape_val {
        Value::List(items) => {
            // ★修正1: .read().unwrap() を削除。items は &Vec<Value> なのでそのまま回せます。
            for item in items.iter() {
                // println!("DEBUG: tensor item check: {:?}", item);
                match item {
                    // ★修正2: Value::Int(n) の n は &i64 なので *n で実体化します
                    Value::Int(n) => shape.push(*n as usize),
                    // Floatの場合も許容して usize に変換
                    Value::Float(f) => shape.push(*f as usize),
                    Value::Number(n) => shape.push(*n as usize),
                    _ => return Err(SiggError::runtime("Shape must be a list of integers")),
                }
            }
        }
        _ => return Err(SiggError::runtime("tensor argument must be a list")),
    }

    // 2. テンソルを作成して返す
    // ★修正3: ここで Value::Tensor を作成して返す必要があります
    // (Tensor構造体の場所は crate::pocket::tensor::Tensor だと仮定しています)
    use crate::pocket::tensor::Tensor;
    use std::sync::{Arc, RwLock};

    let t = Tensor::zeros(shape);
    Ok(Value::Tensor(Arc::new(RwLock::new(t))))
}
pub fn builtin_tensor_laplacian(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("laplacian expects 1 argument"));
    }
    match &args[0] {
        Value::Tensor(t) => {
            // ★修正: 計算グラフ対応版を呼ぶ
            let res = Tensor::laplacian_graph(t.clone());
            Ok(Value::Tensor(res))
        },
        _ => Err(SiggError::runtime("laplacian expects a tensor")),
    }
}
pub fn builtin_tensor_get(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("t_get expects 2 arguments"));
    }
    
    // Tensor取得
    let tensor_arc = match &args[0] {
        Value::Tensor(t) => t.clone(),
        _ => return Err(SiggError::runtime("First argument must be a tensor")),
    };
    let tensor = tensor_arc.read().unwrap();

    // インデックス取得
    let indices: Vec<usize> = match &args[1] {
        Value::Vec(v) | Value::List(v) => { // ListもVecも許可
            let mut idxs = Vec::new();
            for val in v.iter() {
                match val {
                    Value::Int(i) => idxs.push(*i as usize), 
                    Value::Number(n) => idxs.push(*n as usize),
                    Value::Float(f) => idxs.push(*f as usize),
                    _ => return Err(SiggError::runtime("Indices must be numbers")),
                }
            }
            idxs
        },
        Value::Tensor(t_idx) => {
            let t = t_idx.read().unwrap();
            t.data.iter().map(|c| c.re as usize).collect()
        },
        Value::Int(i) => vec![*i as usize],
        Value::Number(n) => vec![*n as usize],
        Value::Float(f) => vec![*f as usize],
        _ => return Err(SiggError::runtime("Invalid index type")),
    };

    let c = tensor.get(&indices);
    // 数値として返す
    Ok(Value::Number(c.re))
}
pub fn builtin_tensor_set(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 3 {
        return Err(SiggError::runtime("t_set expects 3 arguments: (tensor, indices, value)"));
    }

    // 1. Tensor取得
    let tensor_arc = match &args[0] {
        Value::Tensor(t) => t.clone(),
        _ => return Err(SiggError::runtime("First argument must be a tensor")),
    };

    // 2. インデックス取得
    let indices: Vec<usize> = match &args[1] {
        // リスト系
        Value::Vec(v) | Value::List(v) => {
            let mut idxs = Vec::new();
            for val in v.iter() {
                match val {
                    Value::Int(i) => idxs.push(*i as usize), 
                    Value::Number(n) => idxs.push(*n as usize),
                    Value::Float(f) => idxs.push(*f as usize),
                    _ => return Err(SiggError::runtime("Indices must be numbers")),
                }
            }
            idxs
        },
        // 数値単体
        Value::Int(i) => vec![*i as usize],
        Value::Number(n) => vec![*n as usize],
        Value::Float(f) => vec![*f as usize],
        
        // Tensor型
        Value::Tensor(t_idx) => {
            let t = t_idx.read().unwrap();
            t.data.iter().map(|c| c.re as usize).collect()
        },
        
        _ => return Err(SiggError::runtime("Second argument must be a list of indices or a number")),
    };

    // 3. 値の変換 (修正ポイント: Complex を直接使用)
    let new_val = match &args[2] {
        // [re, im]
        Value::Vec(v) | Value::List(v) => {
            if v.len() != 2 {
                return Err(SiggError::runtime("Value must be [re, im]"));
            }
            let re = match v[0] { Value::Int(i)=>i as f64, Value::Number(n)=>n, Value::Float(f)=>f, _=>0.0 };
            let im = match v[1] { Value::Int(i)=>i as f64, Value::Number(n)=>n, Value::Float(f)=>f, _=>0.0 };
            
            // ★修正: crate::tensor::Complex ではなく Complex だけでOK
            Complex { re, im }
        },
        // 数値単体
        Value::Int(i) => Complex { re: *i as f64, im: 0.0 },
        Value::Number(n) => Complex { re: *n, im: 0.0 },
        Value::Float(f) => Complex { re: *f, im: 0.0 },
        
        _ => return Err(SiggError::runtime("Third argument must be [re, im] or a number")),
    };

    // 4. 書き込み
    {
        let mut t = tensor_arc.write().unwrap();
        t.set(&indices, new_val);
    }

    Ok(Value::Int(0))
}

pub fn builtin_list(args: Vec<Value>) -> EvalResult {
    // 引数をそのままListとして返すだけ
    Ok(Value::List(args))
}

fn builtin_grid_stats(args: Vec<Value>) -> Result<Value, SiggError> {
    need_n(&args, 1, "grid_stats")?;
    let g = as_grid(&args[0])?;
    
    let (min, max, mean, std) = grid_stats_impl(g.as_ref());
    
    // Tuple (min, max, mean, std) を返す
    Ok(Value::Tuple(vec![
        Value::F32(min),
        Value::F32(max),
        Value::F32(mean),
        Value::F32(std),
    ]))
}

// 勾配計算を開始する
fn builtin_backward(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("backward expects 1 argument"));
    }
    match &args[0] {
        Value::Tensor(t) => {
            let mut tensor = t.write().map_err(|e| SiggError::runtime(e.to_string()))?;
            tensor.backward();
            Ok(Value::Unit)
        }
        _ => Err(SiggError::runtime("backward expects a tensor")),
    }
}
fn builtin_grad(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("grad expects 1 argument"));
    }
    match &args[0] {
        Value::Tensor(t) => {
            let tensor = t.read().map_err(|e| SiggError::runtime(e.to_string()))?;
            if let Some(g) = &tensor.grad {
                let grad_tensor = Tensor {
                    data: g.clone(),
                    shape: tensor.shape.clone(),
                    grad: None,
                    requires_grad: false,
                    op: crate::pocket::tensor::OpType::Leaf,
                    parents: vec![],
                };
                Ok(Value::Tensor(Arc::new(RwLock::new(grad_tensor))))
            } else {
                let zero = Tensor::zeros(tensor.shape.clone());
                Ok(Value::Tensor(Arc::new(RwLock::new(zero))))
            }
        }
        _ => Err(SiggError::runtime("grad expects a tensor")),
    }
}
fn builtin_enable_grad(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("enable_grad expects 1 argument"));
    }
    match &args[0] {
        Value::Tensor(t) => {
            let mut tensor = t.write().map_err(|e| SiggError::runtime(e.to_string()))?;
            tensor.requires_grad = true;
            Ok(args[0].clone())
        }
        _ => Err(SiggError::runtime("enable_grad expects a tensor")),
    }
}
/// SIGG呼び出し: t_set_2d(world, x, y, [real, imag])
pub fn t_set_2d(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() < 4 { return Err(SiggError::runtime("4 args required")); }

    let arc_tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => return Err(SiggError::runtime("Arg 0 must be a Tensor")),
    };

    // ヘルパー関数を使用して数値を取得
    let x = as_f64(&args[1])? as usize;
    let y = as_f64(&args[2])? as usize;

    let (re, im) = match &args[3] {
        Value::List(l) => {
            let r = as_f64(&l[0])?;
            let i = if l.len() > 1 { as_f64(&l[1])? } else { 0.0 };
            (r, i)
        },
        _ => return Err(SiggError::runtime("Arg 3 must be a list")),
    };

    // ...以下、テンソルへの書き込み処理（前回と同じ）...
    let mut tensor = arc_tensor.write().unwrap();
    let idx = y * tensor.shape[0] + x;
    tensor.data[idx] = Complex::new(re, im);
    Ok(Value::Number(0.0))
}
/// 2次元ラプラシアン（拡散）を計算する
/// SIGG呼び出し: let diff = laplacian_2d(world);
pub fn laplacian_2d(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.is_empty() {
        return Err(SiggError::runtime("laplacian_2d requires a tensor"));
    }

    let arc_tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => return Err(SiggError::runtime("Arg must be a Tensor")),
    };

    let tensor = arc_tensor.read().unwrap();
    let w = tensor.shape[0];
    let h = if tensor.shape.len() > 1 { tensor.shape[1] } else { 1 };

    if w < 3 || h < 3 {
        return Err(SiggError::runtime("Tensor too small for 2D Laplacian"));
    }

    let mut new_data = vec![Complex::new(0.0, 0.0); w * h];

    // 5点差分法によるラプラス演算
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let center = tensor.data[idx];
            let left   = tensor.data[idx - 1];
            let right  = tensor.data[idx + 1];
            let up     = tensor.data[idx - w];
            let down   = tensor.data[idx + w];

            // Δf = f(x+1) + f(x-1) + f(y+1) + f(y-1) - 4f(x,y)
            let lap = Complex::new(
                right.re + left.re + up.re + down.re - 4.0 * center.re,
                right.im + left.im + up.im + down.im - 4.0 * center.im,
            );
            new_data[idx] = lap;
        }
    }

    let mut out_tensor = Tensor::zeros(vec![w, h]);
    out_tensor.data = new_data;
    Ok(Value::Tensor(Arc::new(RwLock::new(out_tensor))))
}
pub fn visualize_placeholder(_args: Vec<Value>) -> Result<Value, SiggError> {
    // 中身は空でOK。実行時は vm.rs 側で上書き（フック）されます。
    Ok(Value::Unit)
}

/// 数学関数のヘルパー（引数チェックと型変換を共通化）
fn math_unary_op<F>(args: Vec<Value>, name: &str, op: F) -> Result<Value, SiggError>
where
    F: Fn(f64) -> f64,
{
    if args.len() != 1 {
        return Err(SiggError::runtime(format!("{} expects 1 argument", name)));
    }

    match args[0] {
        Value::Number(n) => Ok(Value::Number(op(n))),
        Value::Int(i) => Ok(Value::Number(op(i as f64))),
        _ => Err(SiggError::runtime(format!("{} expects a number", name))),
    }
}
pub fn std_sin(args: Vec<Value>) -> Result<Value, SiggError> {
    math_unary_op(args, "sin", |n| n.sin())
}
pub fn std_cos(args: Vec<Value>) -> Result<Value, SiggError> {
    math_unary_op(args, "cos", |n| n.cos())
}
pub fn std_tan(args: Vec<Value>) -> Result<Value, SiggError> {
    math_unary_op(args, "tan", |n| n.tan())
}
pub fn std_sqrt(args: Vec<Value>) -> Result<Value, SiggError> {
    math_unary_op(args, "sqrt", |n| n.sqrt())
}
/// 指定した桁数でフォーマットする関数: fmt(value, precision)
pub fn std_fmt(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("fmt expects 2 arguments: (value, precision)"));
    }

    // args[0]: 値, args[1]: 桁数
    let target = &args[0];
    let prec_val = &args[1];

    let precision = match prec_val {
        Value::Number(n) => *n as usize,
        Value::Int(i) => *i as usize,
        _ => return Err(SiggError::runtime("Precision must be a number")),
    };

    let s = match target {
        Value::Number(n) => format!("{:.1$}", n, precision),
        Value::Int(i) => format!("{:.1$}", *i as f64, precision),
        Value::Str(s) => s.clone(), 
        // 以前の変更でValue::Stringを追加していればOK。
        // まだなら format!("{:?}", target) にしてください。
        
        _ => format!("{}", target),
    };

    Ok(Value::Str(s))
}
pub fn std_abs(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("abs expects 1 argument"));
    }

    match args[0] {
        // 浮動小数点の場合: 3.14 -> 3.14, -3.14 -> 3.14
        Value::Number(n) => Ok(Value::Number(n.abs())),
        
        // 整数の場合: 10 -> 10, -10 -> 10
        Value::Int(i) => Ok(Value::Int(i.abs())),
        
        _ => Err(SiggError::runtime("abs expects a number")),
    }
}

pub fn std_rand(args: Vec<Value>) -> Result<Value, SiggError> {
    // 引数は不要ですが、あっても無視するかエラーにするか選べます
    // ここでは引数なしを想定
    
    // 現在のシードを取得
    let mut seed = RAND_SEED.load(Ordering::Relaxed);

    // 初回実行時（または特定の値の時）に現在時刻でシードを初期化
    if seed == 12345 {
        let start = SystemTime::now();
        let since = start.duration_since(UNIX_EPOCH).unwrap_or_default();
        seed = since.as_nanos() as u64;
    }

    // 線形合同法 (LCG) による更新 (Knuthの定数などを使用)
    // seed = seed * 6364136223846793005 + 1442695040888963407
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    
    // 新しいシードを保存
    RAND_SEED.store(seed, Ordering::Relaxed);

    // 0.0 〜 1.0 の範囲の浮動小数点に変換
    // u64の最大値で割る
    let ret = (seed as f64) / (u64::MAX as f64);

    Ok(Value::Number(ret))
}
pub fn std_int(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("int() expects 1 argument"));
    }

    match &args[0] {
        // 小数 -> 整数 (小数点以下切り捨て)
        Value::Number(n) => Ok(Value::Int(*n as i64)),
        
        // 整数 -> 整数 (そのまま)
        Value::Int(i) => Ok(Value::Int(*i)),
        
        // 文字列 -> 整数 ("123" -> 123)
        Value::Str(s) => {
            match s.parse::<i64>() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => Err(SiggError::runtime(format!("int(): invalid number string '{}'", s))),
            }
        },

        // 真偽値 -> 整数 (true->1, false->0)
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),

        _ => Err(SiggError::runtime("int() expects a number or string")),
    }
}
pub fn std_floor(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("floor expects 1 argument"));
    }

    match args[0] {
        // 浮動小数点: 3.9 -> 3.0, -3.1 -> -4.0
        Value::Number(n) => Ok(Value::Number(n.floor())),
        
        // 整数: そのまま (5 -> 5)
        Value::Int(i) => Ok(Value::Int(i)),
        
        _ => Err(SiggError::runtime("floor expects a number")),
    }
}


/// ユーザーからの入力を受け取る: let text = input("Prompt> ");
pub fn std_input(args: Vec<Value>) -> Result<Value, SiggError> {
    // プロンプトメッセージがあれば表示
    if let Some(msg) = args.get(0) {
        print!("{}", msg);
        io::stdout().flush().map_err(|e| SiggError::runtime(e.to_string()))?;
    }

    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .map_err(|e| SiggError::runtime(e.to_string()))?;

    // 末尾の改行を削除
    let trimmed = buffer.trim_end().to_string();
    Ok(Value::Str(trimmed))
}
/// ファイルを読み込む: let content = read_file("memory.txt");
pub fn std_read_file(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("read_file expects 1 argument (filename)"));
    }

    let filename = match &args[0] {
        Value::Str(s) => s,
        _ => return Err(SiggError::runtime("filename must be a string")),
    };

    let content = fs::read_to_string(filename)
        .map_err(|e| SiggError::runtime(format!("Failed to read file: {}", e)))?;

    Ok(Value::Str(content))
}
/// ファイルに書き込む: write_file("memory.txt", "learned data");
pub fn std_write_file(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("write_file expects 2 arguments (filename, content)"));
    }

    let filename = match &args[0] {
        Value::Str(s) => s,
        _ => return Err(SiggError::runtime("filename must be a string")),
    };

    let content = match &args[1] {
        Value::Str(s) => s,
        _ => return Err(SiggError::runtime("content must be a string")),
    };

    fs::write(filename, content)
        .map_err(|e| SiggError::runtime(format!("Failed to write file: {}", e)))?;

    Ok(Value::Unit)
}

/// 文字列置換: let new_s = replace("hello world", "world", "AI");
pub fn std_replace(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 3 {
        return Err(SiggError::runtime("replace expects 3 arguments: (source, from, to)"));
    }

    let source = match &args[0] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 1 must be string")) };
    let from = match &args[1] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 2 must be string")) };
    let to = match &args[2] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 3 must be string")) };

    let result = source.replace(from, to);
    Ok(Value::Str(result))
}
/// 文字列分割: let parts = split("a,b,c", ",");
pub fn std_split(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("split expects 2 arguments: (source, delimiter)"));
    }

    let source = match &args[0] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 1 must be string")) };
    let delimiter = match &args[1] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 2 must be string")) };

    // 分割して Value::String のリストに変換
    let parts: Vec<Value> = source
        .split(delimiter)
        .map(|s| Value::Str(s.to_string()))
        .collect();

    // Value::List が定義されている前提 (なければ Value::Vec など適宜変更)
    Ok(Value::List(parts))
}

pub fn std_delete_file(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 { return Err(SiggError::runtime("delete_file expects 1 argument")); }
    let filename = match &args[0] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg must be string")) };
    
    fs::remove_file(filename).map_err(|e| SiggError::runtime(format!("Delete failed: {}", e)))?;
    Ok(Value::Bool(true))
}
/// ディレクトリ一覧取得: list_dir(".")
pub fn std_list_dir(args: Vec<Value>) -> Result<Value, SiggError> {
    let path_str = if args.is_empty() { ".".to_string() } else {
        match &args[0] { Value::Str(s) => s.clone(), _ => ".".to_string() }
    };

    let paths = fs::read_dir(path_str).map_err(|e| SiggError::runtime(format!("List dir failed: {}", e)))?;
    let mut list = Vec::new();

    for path in paths {
        if let Ok(entry) = path {
            if let Ok(name) = entry.file_name().into_string() {
                list.push(Value::Str(name));
            }
        }
    }
    Ok(Value::List(list))
}
pub fn std_try_eval(_args: Vec<Value>) -> Result<Value, SiggError> {
    // 中身は空でOK。実行時は vm.rs 側で上書き（フック）されます。
    Ok(Value::Unit)
}
/// LLMに問い合わせる: let response = ask_llm("system prompt", "user prompt");
pub fn std_ask_llm(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("ask_llm expects 2 arguments: (system_prompt, user_msg)"));
    }

    let system_prompt = match &args[0] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 1 must be string")) };
    let user_msg = match &args[1] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg 2 must be string")) };

    // Ollamaへのリクエストボディ
    let client = reqwest::blocking::Client::new();
    let body = json!({
        "model": "llama3.1",
        "stream": false, // 一括で返してもらう
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_msg }
        ],
        "options": {
            "temperature": 0.0 // コマンド解析なのでランダム性をなくす
        }
    });

    // リクエスト送信 (localhost:11434 はOllamaのデフォルトポート)
    let res = client.post("http://localhost:11434/api/chat")
        .json(&body)
        .send()
        .map_err(|e| SiggError::runtime(format!("LLM Request Failed: {}", e)))?;

    // レスポンスの解析
    let json_resp: serde_json::Value = res.json()
        .map_err(|e| SiggError::runtime(format!("Invalid JSON: {}", e)))?;

    let content = json_resp["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(Value::Str(content))
}

/// OSのシェルコマンドを実行: shell("cargo build --release")
pub fn std_shell(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 { return Err(SiggError::runtime("shell expects 1 argument")); }
    let cmd_str = match &args[0] { Value::Str(s) => s, _ => return Err(SiggError::runtime("arg must be string")) };

    println!("[System] Executing: {}", cmd_str);

    // Windowsなら cmd /C, Linux/Macなら sh -c
    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", cmd_str])
            .output()
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(cmd_str)
            .output()
    };

    match output {
        Ok(o) => {
            // 標準出力を取得
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            // 成功かどうかに関わらず、ログとして返す（あるいは終了コードを見る）
            Ok(Value::Str(format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr)))
        }
        Err(e) => Err(SiggError::runtime(format!("Shell command failed: {}", e))),
    }
}
/// try_unwrap(result_list)
/// try_evalの戻り値 [bool, val] を受け取る。
/// trueなら val を返し、falseなら val (エラーメッセージ) でランタイムエラーを起こす。
pub fn std_try_unwrap(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 1 {
        return Err(SiggError::runtime("try_unwrap expects 1 argument (the result list)"));
    }

    // 引数がリストかチェック
    match &args[0] {
        Value::List(items) => {
            if items.len() < 2 {
                return Err(SiggError::runtime("try_unwrap expects a list of length 2: [success, value]"));
            }
            
            // 0番目がBoolかチェック
            let success = match items[0] {
                Value::Bool(b) => b,
                _ => return Err(SiggError::runtime("First element of result must be a boolean")),
            };

            // 1番目が値
            let content = items[1].clone();

            if success {
                Ok(content)
            } else {
                // 失敗時はエラーとして停止させる（メッセージを見やすく整形）
                let msg = match content {
                    Value::Str(s) => s,
                    _ => format!("{:?}", content),
                };
                Err(SiggError::runtime(format!("Unwrap Panic: {}", msg)))
            }
        }
        _ => Err(SiggError::runtime("try_unwrap expects a List")),
    }
}
/// format("Hello {}, your score is {}", "World", 100)
/// 文字列内の "{}" を引数で順番に置き換える簡易フォーマッター
pub fn std_format(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() < 1 {
        return Err(SiggError::runtime("format expects at least 1 argument (template string)"));
    }

    let template = match &args[0] {
        Value::Str(s) => s,
        _ => return Err(SiggError::runtime("First argument to format must be a string")),
    };

    let mut result = String::new();
    let parts: Vec<&str> = template.split("{}").collect();
    let values = &args[1..]; // 埋め込む値たち

    // テンプレートの各部分と値を交互に結合
    for (i, part) in parts.iter().enumerate() {
        result.push_str(part);
        
        // 最後のパーツの後ろには値を入れない（splitの仕様上）
        // かつ、埋め込む値が残っている場合のみ追加
        if i < parts.len() - 1 {
            if i < values.len() {
                // Value型の文字列表現を取得（Display実装やデバッグ表示に依存）
                // ※ Value型に to_string_value() のようなメソッドがあればそれを使うのがベスト
                // ここでは簡易的に Debug表示などを使うか、型ごとに処理
                let val_str = match &values[i] {
                    Value::Str(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    v => format!("{:?}", v), // ListなどはDebug表示
                };
                result.push_str(&val_str);
            } else {
                // 値が足りない場合は {} をそのまま残すか、エラーにするか
                // ここでは安全に "(undefined)" とする
                result.push_str("(undefined)");
            }
        }
    }

    Ok(Value::Str(result))
}
/// pow(base, exponent) -> base^exponent
pub fn std_pow(args: Vec<Value>) -> Result<Value, SiggError> {
    if args.len() != 2 {
        return Err(SiggError::runtime("pow expects 2 arguments: (base, exponent)"));
    }

    let base = match args[0] {
        Value::Number(n) => n,
        _ => return Err(SiggError::runtime("Base must be a number")),
    };

    let exp = match args[1] {
        Value::Number(n) => n,
        _ => return Err(SiggError::runtime("Exponent must be a number")),
    };

    // f64のpowfを使用
    let res = base.powf(exp);
    Ok(Value::Number(res))
}


pub fn builtins() -> Vec<Builtin> {
    //println!("DEBUG: Loading builtins..."); // ★ これを追加
    vec![
        // Builtin { name: "", f: },
        Builtin { name: "exp", f: builtin_exp },
        Builtin { name: "zero_grad", f: builtin_zero_grad },
        Builtin { name: "print", f: builtin_print },
        Builtin { name: "vec_get", f: builtin_vec_get },
        Builtin { name: "vec_set", f: builtin_vec_set },
        Builtin { name: "vec_new", f: builtin_vec_new },
        Builtin { name: "vec_push", f: builtin_vec_push },
        Builtin { name: "vec_len", f: builtin_vec_len },
        Builtin { name: "map_get", f: builtin_map_get },
        Builtin { name: "map_set", f: builtin_map_set },
        Builtin { name: "map_new", f: builtin_map_new },
        Builtin { name: "map_len", f: builtin_map_len },
        Builtin { name: "noise2", f: builtin_noise2 },
        Builtin { name: "rand2", f: builtin_rand2 },
        Builtin { name: "mix", f: builtin_mix },
        Builtin { name: "clamp", f: builtin_clamp },
        Builtin { name: "project", f: builtin_project },
        Builtin { name: "reaction_gs", f: builtin_reaction_gs },
        Builtin { name: "world_step_gs", f: b_world_step_gs },
        Builtin { name: "pocket_open", f: builtin_pocket_open },
        Builtin { name: "atlas_new", f: builtin_atlas_new },
        Builtin { name: "pocket_read", f: builtin_pocket_read },
        Builtin { name: "pocket_write", f: builtin_pocket_write },
        Builtin { name: "pocket_persist", f: builtin_pocket_persist },
        Builtin { name: "atlas_query_topk", f: builtin_atlas_query_topk },
        Builtin { name: "pocket_trigger_extract_auto", f: builtin_pocket_trigger_extract_auto },
        Builtin { name: "pocket_atlas_update", f: builtin_pocket_atlas_update_from_hits },
        Builtin { name: "ai_pocket_open", f: builtin_ai_pocket_open },
        Builtin { name: "ai_pocket_write_f32", f: builtin_ai_pocket_write_f32 },
        Builtin { name: "ai_pocket_read_f32", f: builtin_ai_pocket_read_f32 },

        Builtin { name: "ai_create", f: builtin_ai_create },
        Builtin { name: "ai_tick", f: builtin_ai_tick },
        Builtin { name: "ai_get_score", f: builtin_ai_get_score },
        Builtin { name: "ai_get_mode", f: builtin_ai_get_mode },
        Builtin { name: "ai_get_env_period", f: builtin_ai_get_env_period },
        Builtin { name: "ai_set_env_period", f: builtin_ai_set_env_period },
        Builtin { name: "ai_get_mu",    f: builtin_ai_get_mu },
        Builtin { name: "ai_now_tick", f: builtin_ai_now_tick },
        Builtin { name: "vec_map", f: builtin_vec_map },
        Builtin { name: "vec_filter", f: builtin_vec_filter },
        Builtin { name: "vec_reduce", f: builtin_vec_reduce },
        Builtin { name: "snapshot_create", f: builtin_snapshot_create },
        Builtin { name: "log", f: builtin_log },
        Builtin { name: "namespace_create", f: builtin_namespace_create },
    
        // Grid拡張API
        Builtin { name: "grid_shift", f: builtin_grid_shift },
        Builtin { name: "grid_convolve", f: builtin_grid_convolve },
        Builtin { name: "grid_laplace", f: builtin_grid_laplace },
        Builtin { name: "grid_sobel_x", f: builtin_grid_sobel_x },
        Builtin { name: "grid_sobel_y", f: builtin_grid_sobel_y },
        Builtin { name: "grid_get", f: builtin_grid_get },
        Builtin { name: "grid_set", f: builtin_grid_set },
        Builtin { name: "grid_map", f: builtin_grid_map },
        Builtin { name: "grid_reduce", f: builtin_grid_reduce },
        Builtin { name: "grid_filter", f: builtin_grid_filter },
        Builtin { name: "grid_stats", f: builtin_grid_stats },

        // Pocket拡張API
        Builtin { name: "pocket_read_chunk", f: builtin_pocket_read_chunk },
        Builtin { name: "pocket_write_chunk", f: builtin_pocket_write_chunk },
        Builtin { name: "pocket_slice_xy", f: builtin_pocket_slice_xy },
        Builtin { name: "pocket_slice_xz", f: builtin_pocket_slice_xz },
        Builtin { name: "pocket_slice_yz", f: builtin_pocket_slice_yz },
        Builtin { name: "pocket_fill_region", f: builtin_pocket_fill_region },
        Builtin { name: "pocket_copy_region", f: builtin_pocket_copy_region },

        // AI拡張API
        Builtin { name: "ai_pocket_write_vec", f: builtin_ai_pocket_write_vec },
        Builtin { name: "ai_pocket_read_vec", f: builtin_ai_pocket_read_vec },
        Builtin { name: "ai_tick_batch", f: builtin_ai_tick_batch },
        Builtin { name: "ai_set_memory", f: builtin_ai_set_memory },
        Builtin { name: "ai_get_memory", f: builtin_ai_get_memory },
        
        // 文字列操作API
        Builtin { name: "str_split", f: builtin_str_split },
        Builtin { name: "str_join", f: builtin_str_join },
        Builtin { name: "str_replace", f: builtin_str_replace },
        Builtin { name: "str_contains", f: builtin_str_contains },
        Builtin { name: "str_starts_with", f: builtin_str_starts_with },
        Builtin { name: "str_ends_with", f: builtin_str_ends_with },
        Builtin { name: "str_trim", f: builtin_str_trim },
        Builtin { name: "str_to_upper", f: builtin_str_to_upper },
        Builtin { name: "str_to_lower", f: builtin_str_to_lower },
        Builtin { name: "str_len", f: builtin_str_len },

        Builtin { name: "tensor", f: builtin_tensor_new },
        Builtin { name: "laplacian", f: builtin_tensor_laplacian },
        Builtin { name: "t_get", f: builtin_tensor_get },
        Builtin { name: "t_set", f: builtin_tensor_set },
        Builtin { name: "list", f: builtin_list },
        Builtin { name: "backward", f: builtin_backward },
        Builtin { name: "grad", f: builtin_grad },
        Builtin { name: "enable_grad", f: builtin_enable_grad },
        Builtin { name: "t_set_2d", f: t_set_2d },
        Builtin { name: "laplacian_2d", f: laplacian_2d },
        Builtin { name: "visualize", f: visualize_placeholder },

        Builtin { name: "sin", f: std_sin },
        Builtin { name: "cos", f: std_cos },
        Builtin { name: "tan", f: std_tan },
        Builtin { name: "sqrt", f: std_sqrt },
        Builtin { name: "fmt", f: std_fmt },
        Builtin { name: "abs", f: std_abs },
        Builtin { name: "rand", f: std_rand },
        Builtin { name: "int", f: std_int },
        Builtin { name: "floor", f: std_floor },
        Builtin { name: "input", f: std_input },
        Builtin { name: "read_file", f: std_read_file },
        Builtin { name: "write_file", f: std_write_file },
        Builtin { name: "replace", f: std_replace },
        Builtin { name: "split", f: std_split },
        Builtin { name: "delete_file", f: std_delete_file },
        Builtin { name: "list_dir", f: std_list_dir },
        Builtin { name: "try_eval", f: std_try_eval },
        Builtin { name: "ask_llm", f: std_ask_llm },
        Builtin { name: "shell", f: std_shell },
        Builtin { name: "try_unwrap", f: std_try_unwrap },
        Builtin { name: "format", f: std_format },
        Builtin { name: "pow", f: std_pow },        
        //Builtin { name: "", f: },
    ]
}
