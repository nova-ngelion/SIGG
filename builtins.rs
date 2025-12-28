use std::sync::Arc;

use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};
use crate::error::SiggError;
use crate::value::{Boundary, Grid, GridRef, Value};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use crate::pocket;
use crate::pocket::types::{WorldKey, ChunkKey, Hit};
use crate::state::PocketState;
use crate::pocket::PocketWorld;
use crate::pocket::compute::ComputeSpace;
use crate::pocket::cpu::{CpuState, cpu_run_mem};
use crate::state::SiggAgent;
use crate::ai_runtime;

const PROG_MAX: i32 = 64;   // 命令領域は 0..63
const IO_BASE:  i32 = 100;  // io_in/io_out はここ以降へ
const ENV_PERIOD: u32 = 30; // 30tickごとに正解が変わる（好みで調整）




pub struct Builtin {
    pub name: &'static str,
    pub f: fn(Vec<Value>) -> Result<Value, SiggError>,
}

static LAST_DIGEST: AtomicU64 = AtomicU64::new(0);
static AI_TICK: AtomicU32 = AtomicU32::new(0);//new

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
fn as_worldkey3(args: &Vec<Value>, i: usize) -> Result<WorldKey, SiggError> {
    Ok((as_u32(&args[i])?, as_u32(&args[i+1])?, as_u32(&args[i+2])?))
}
fn worlds() -> &'static Mutex<Vec<pocket::PocketWorld>> {
    WORLDS.get_or_init(|| Mutex::new(Vec::new()))
}
fn atlases() -> &'static Mutex<Vec<pocket::Atlas>> {
    ATLASES.get_or_init(|| Mutex::new(Vec::new()))
}
fn world_get_mut(h: u32) -> Result<std::sync::MutexGuard<'static, Vec<pocket::PocketWorld>>, SiggError> {
    let g = worlds().lock().map_err(|_| SiggError::runtime("world registry poisoned"))?;
    if h as usize >= g.len() { return Err(SiggError::runtime("invalid world handle")); }
    Ok(g)
}
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
fn guard_io_xyz(mut p: (i32,i32,i32)) -> (i32,i32,i32) {
    // 命令領域(0..PROG_MAX-1)への衝突を防ぐ
    if p.0 >= 0 && p.0 < PROG_MAX {
        p.0 = IO_BASE;
    }
    p
}
fn guard_io_pair(io_in: (i32,i32,i32), io_out: (i32,i32,i32)) -> ((i32,i32,i32),(i32,i32,i32)) {
    let mut a = guard_io_xyz(io_in);
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
fn load_program_into_space(space: &mut ComputeSpace, prog: &[u32]) {
    for (i, &inst) in prog.iter().enumerate() {
        space.write_cell_bits(i as i32, 0, 0, 0, inst);
    }
}
// ---------- basic builtins ----------
fn builtin_print(args: Vec<Value>) -> Result<Value, SiggError> {
    for v in args {
        println!("{v}");
        if let Value::Grid(g) = &v {
            let d = digest_grid_u32(g.as_ref());
            LAST_DIGEST.store(d as u64, Ordering::Relaxed);
        }
    }
    Ok(Value::Unit)
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
// fn builtin_ai_create(args: Vec<Value>) -> Result<Value, SiggError> {
//     // ai_create(pocket_h, u,v,w, space_size, lanes) -> (agent_id, space_id, cpu_h)
//     need_n(&args, 6, "ai_create")?;
//     let pocket_h = as_u32(&args[0])?;
//     let world = as_worldkey3(&args, 1)?;
//     let space_size = as_usize(&args[4])?;
//     let lanes = as_usize(&args[5])?;

//     let orig_in = (0,0,0);
//     let orig_out = (1,0,0);
//     let (io_in, io_out) = guard_io_pair(orig_in, orig_out);
//     if io_in != orig_in || io_out != orig_out {
//         eprintln!("[AI] io relocated: in={orig_in:?}->{io_in:?}, out={orig_out:?}->{io_out:?}");
//     }
//     let pocket_in_addr    = (0,0,0);
//     let pocket_out_addr   = (1,0,0);
//     let pocket_score_addr = (2,0,0);
//     let pocket_policy_addr= (3,0,0);

//     let mut st = ai_state().lock().map_err(|_| SiggError::runtime("ai_state poisoned"))?;

//     // pocket存在確認 + world一致確認
//     {
//         let p = st.pockets.get(&pocket_h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
//         if p.world != world {
//             return Err(SiggError::runtime("ai_create: world mismatch"));
//         }
//     }

//     // ComputeSpace 作成
//     let space_id = st.next_space_id;
//     st.next_space_id += 1;
//     let mut space = ComputeSpace::new_blank(space_size as i32, lanes);

//     // server.rs と同じ命令ロード（あなたが server で成功させたのと同一にする）
//     let src = r#"
//     loop:
//         // mode が 1 なら policy を反転（LSBだけ使う）
//         LOAD  r5, r0, 103       // r5 = mode
//         BRZ   r5, cont          // mode==0ならスキップ
//         LOAD  r2, r0, 102       // r2 = policy_bits
//         MOVI  r3, 1
//         XOR   r2, r2, r3        // r2 ^= 1
//         STORE r2, r0, 102       // policy更新

//     cont:
//         // out = in + (policy&1) + 1
//         LOAD  r1, r0, 100       // in
//         LOAD  r2, r0, 102       // policy
//         MOVI  r3, 1
//         AND   r2, r2, r3        // policy_id = policy&1
//         MOVI  r4, 1
//         ADD   r2, r2, r4        // policy_id+1
//         ADD   r1, r1, r2        // in + (policy_id+1)
//         STORE r1, r0, 101       // out
//         JMP   loop
//     "#;

//     let prog = crate::pocket::asm::assemble(src)
//         .map_err(|e| SiggError::runtime(format!("assemble failed: {e}")))?;
//     load_program_into_space(&mut space, &prog);

//     st.compute_spaces.insert(space_id, space);


//     // CPU 作成
//     let cpu_h = st.next_handle;
//     st.next_handle += 1;
//     st.cpu_states.insert(cpu_h, CpuState::new(world));

//     // Agent 作成
//     let agent_id = st.next_agent_id;
//     st.next_agent_id += 1;

//     st.agents.insert(agent_id, SiggAgent {
//         id: agent_id,
//         world,
//         pocket_handle: pocket_h,
//         space_id,
//         cpu_handle: cpu_h,
//         sensors_flags: 0,
//         sensors_coords: vec![],
//         io_in,
//         io_out,
//         pocket_in_addr,
//         pocket_out_addr,
//         pocket_score_addr,
//         pocket_policy_addr,
//         score_mu_bits: 0.0f32.to_bits(),
//         score_beta: 0.20,
//         mode: 0,
//     });

//     Ok(Value::Tuple(vec![
//         Value::Number(agent_id as f64),
//         Value::Number(space_id as f64),
//         Value::Number(cpu_h as f64),
//     ]))
// }
fn builtin_ai_create(args: Vec<Value>) -> Result<Value, SiggError> {
    // 以前のSIGGプログラム互換：ai_create(pocket) の1引数
    need_n(&args, 1, "ai_create")?;
    let pocket_h = as_f64(&args[0])? as u32;

    // 命令領域(0..63)と衝突しないI/O（以前成功した100番台）
    let io_in:  (i32,i32,i32) = (100, 0, 0);
    let io_out: (i32,i32,i32) = (101, 0, 0);
    let score_beta: f32 = 0.20;

    // デフォルト（必要ならSIGG側で別builtin作って可変にしてOK）
    let space_size: i32 = 1024;
    let lanes: usize = 1;

    let mut st = ai_state()
        .lock()
        .map_err(|_| SiggError::runtime("ai_state poisoned"))?;

    // pocket から world を取得して整合させる
    let p = st.pockets
        .get(&pocket_h)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    let world = p.world; // (u,v,w)

    let agent_id: u64 = crate::ai_runtime::ai_create_core(
        &mut st,
        world,
        pocket_h,
        io_in,
        io_out,
        score_beta,
        space_size,
        lanes,
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

    // ログ（SIGG側printと二重になるなら消してOK）
    println!("mu=\n{}\nmode=\n{}", r.mu, r.mode);
    println!(
        "tick=\n{}\nenv=\n{}\nin=\n{}\nout=\n{}\nexp=\n{}\nok=\n{}",
        r.tick,
        r.env_bit,
        r.in_bits,
        r.out_bits,
        r.expect,
        if r.ok { 1 } else { 0 }
    );

    // SIGGプログラムの destructuring と一致させる（8要素）
    Ok(Value::Tuple(vec![
        Value::Number(r.tick as f64),
        Value::Number(r.env_bit as f64),
        Value::Number(r.in_bits as f64),
        Value::Number(r.out_bits as f64),
        Value::Number(r.expect as f64),
        Value::Number(if r.ok { 1.0 } else { 0.0 }),
        Value::Number(r.mu as f64),
        Value::Number(r.mode as f64),
    ]))
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
fn need_n(args: &Vec<Value>, n: usize, name: &str) -> Result<(), SiggError> {
    if args.len() != n {
        return Err(SiggError::runtime(format!("{name} expects {n} args")));
    }
    Ok(())
}

//new　↑



// ---------- small helpers ----------


fn as_f64(v: &Value) -> Result<f64, SiggError> {
    match v {
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

pub fn builtins() -> Vec<Builtin> {
    vec![
        Builtin { name: "print", f: builtin_print },
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

    
    ]
}
