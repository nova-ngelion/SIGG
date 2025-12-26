use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use crate::state::PocketState;
use crate::pocket::types::{WorldKey, hash64};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
    Arc, Condvar, Mutex,
};

use std::thread;
use crate::error::SiggError;
use crate::value::{Boundary, Grid};
use crate::bytecode;
use crate::pocket::cpu::{CpuState, cpu_run};
use crate::state::SiggAgent;
use crate::pocket::compute::ComputeSpace;
// use rayon::prelude::*;

type Compiled = bytecode::CompiledProgram;

lazy_static::lazy_static! {
    static ref CACHE: Mutex<HashMap<u64, Arc<Compiled>>> = Mutex::new(HashMap::new());
}

// ============================
// V4 protocol (fixed-binary API)
// ============================

const TAG_RUN_ONCE: u32 = 1;
const TAG_SUBMIT_BATCH: u32 = 0x20;
const TAG_POLL: u32 = 0x21;
const TAG_FETCH: u32 = 0x22;
const TAG_CANCEL: u32 = 0x23;

const TAG_POCKET_SEARCH_AND_TRIGGER: u32 = 0x24;
const TAG_ATLAS_QUERY: u32 = 0x25;
const TAG_POCKET_TRIGGER: u32 = 0x26;
const TAG_POCKET_OPEN: u32 = 0x27;

const OK: u32 = 0;
const ERR: u32 = 1;

// flags (bit)
const FLAG_EARLY_STOP: u32 = 1 << 0;
const SENSOR_FLAG_HAS_COORDS: u32 = 1 << 0;

// ---- ComputeSpace protocol ----
const TAG_COMPUTE_OPEN: u32  = 0x28; // open compute space in a world
const TAG_COMPUTE_WRITE: u32 = 0x29; // write cells
const TAG_COMPUTE_RUN: u32   = 0x30; // run rule110 steps
const TAG_COMPUTE_READ: u32  = 0x31; // read cells

const TAG_CPU_CREATE: u32     = 0x32;
const TAG_CPU_LOAD_PROG: u32  = 0x33;
const TAG_CPU_SET_BASE: u32   = 0x34;
const TAG_CPU_SET_REG: u32    = 0x35;
const TAG_CPU_GET_REGS: u32   = 0x36;
const TAG_CPU_RUN: u32        = 0x37;
const TAG_CPU_STATUS: u32     = 0x38;
const TAG_AI_CREATE: u32  = 0x39;
const TAG_AI_TICK: u32    = 0x40;
const TAG_AI_OBSERVE: u32 = 0x41;
const TAG_AI_SET_SENSORS: u32 = 0x42;
const TAG_CPU_ASM_LOAD: u32 = 0x43;
const TAG_AI_SET_IO: u32 = 0x44;
const TAG_POCKET_WRITE_F32: u32 = 0x45;
const TAG_POCKET_READ_F32: u32 = 0x46;






#[derive(Clone, Debug)]
struct RunSpec {
    w: u32,
    h: u32,
    steps: u32,
    seed: u32,
    du: f32,
    dv: f32,
    f: f32,
    k: f32,
    dt: f32,
    eps: f32,
    reaction_id: u32,
    constraint_id: u32,
    noise_id: u32,
}

#[derive(Clone, Debug)]
struct ResultEntry {
    score: f32,
    digest: u32,
    steps_done: u32,
    spec: RunSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobState {
    Queued,
    Running,
    Done,
    Canceled,
    Error,
}

#[derive(Debug)]
struct Job {
    id: u64,
    state: JobState,
    done: u32,
    total: u32,
    topk: usize,
    flags: u32,
    // top-k results (max-heap by score)
    results: Vec<ResultEntry>,
    // if error:
    err_msg: Option<String>,
    cancel: bool,
    // queue payload
    batch: Vec<RunSpec>,
}

#[derive(Clone)]
struct ServerCtx {
    pocket: Arc<Mutex<PocketState>>,

    next_job_id: Arc<AtomicU64>,
    jobs: Arc<Mutex<HashMap<u64, Job>>>,

    queue: Arc<(Mutex<VecDeque<u64>>, Condvar)>,

    shutdown: Arc<AtomicBool>,

}

#[derive(Clone)]
struct JobSpec {
    id: u64,
    topk: u32,
    flags: u32,
    runs: Vec<RunSpec>,
}

#[derive(Clone)]
struct JobResult {
    // top-k heap (min-heap by score)
    heap: BinaryHeap<Scored>,
    // book-keeping
    done: u32,
    total: u32,
    cancelled: bool,
}

#[derive(Clone)]
struct Scored {
    score: f32,
    digest: u32,
    steps_done: u32,
    seed: u32,
    w: u32,
    h: u32,
    du: f32,
    dv: f32,
    f: f32,
    k: f32,
    dt: f32,
    eps: f32,
    reaction_id: u32,
    constraint_id: u32,
    noise_id: u32,
}

// ============================
// Public entry
// ============================

pub fn serve(addr: &str) -> Result<(), SiggError> {
    let listener = TcpListener::bind(addr).map_err(|e| SiggError::io(e.to_string()))?;

    // ✅ 追加：環境変数から読み、無ければデフォルト
    let data_dir: String = std::env::var("SIGG_DATA_DIR").unwrap_or_else(|_| "sigg_data".to_string());
    let z_dim_default: usize = std::env::var("SIGG_ZDIM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64);

    let pocket_state = Arc::new(Mutex::new(PocketState::new(data_dir, z_dim_default)));

    let ctx = ServerCtx {
        pocket: pocket_state.clone(),
        next_job_id: Arc::new(AtomicU64::new(1)),
        jobs: Arc::new(Mutex::new(HashMap::new())),
        queue: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
        shutdown: Arc::new(AtomicBool::new(false)), // ✅ shutting_down → shutdown
    };

    // worker thread
    {
        let ctxw = ctx.clone();
        thread::spawn(move || worker_loop(ctxw));
    }

    loop {
        let (stream, _addr) = listener.accept().map_err(|e| SiggError::io(e.to_string()))?;
        let ctx2 = ctx.clone();
        thread::spawn(move || {
            if let Err(e) = handle_client(stream, ctx2) {
                eprintln!("[server] client error: {e}");
            }
        });
    }
}



// ============================
// Networking helpers
// ============================

// ---------- IO helpers ----------
fn read_exact(stream: &mut TcpStream, n: usize) -> Result<Vec<u8>, SiggError> {
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(buf)
}

fn read_u32(stream: &mut TcpStream) -> Result<u32, SiggError> {
    let b = read_exact(stream, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i32(stream: &mut TcpStream) -> Result<i32, SiggError> {
    let b = read_exact(stream, 4)?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(stream: &mut TcpStream) -> Result<u64, SiggError> {
    let b = read_exact(stream, 8)?;
    Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

fn read_f32(stream: &mut TcpStream) -> Result<f32, SiggError> {
    let b = read_exact(stream, 4)?;
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_f32_vec(stream: &mut TcpStream, n: usize) -> Result<Vec<f32>, SiggError> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n { v.push(read_f32(stream)?); }
    Ok(v)
}

fn read_u32_vec(stream: &mut TcpStream, n: usize) -> Result<Vec<u32>, SiggError> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_u32(stream)?);
    }
    Ok(v)
}


fn write_ok(stream: &mut TcpStream) -> Result<(), SiggError> {
    write_u32(stream, OK)?;
    Ok(())
}

fn write_err(stream: &mut TcpStream, msg: &str) -> Result<(), SiggError> {
    // status=1 + utf8 len + bytes
    write_u32(stream, 1)?;
    write_u32(stream, msg.len() as u32)?;
    stream.write_all(msg.as_bytes()).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(())
}

fn write_u32(stream: &mut TcpStream, x: u32) -> Result<(), SiggError> {
    stream.write_all(&x.to_le_bytes()).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(())
}

fn write_i32(stream: &mut TcpStream, x: i32) -> Result<(), SiggError> {
    stream.write_all(&x.to_le_bytes()).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(())
}

fn write_u64(stream: &mut TcpStream, x: u64) -> Result<(), SiggError> {
    stream.write_all(&x.to_le_bytes()).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(())
}

fn write_f32(stream: &mut TcpStream, x: f32) -> Result<(), SiggError> {
    stream.write_all(&x.to_le_bytes()).map_err(|e| SiggError::io(e.to_string()))?;
    Ok(())
}

// ---------- original world-gen spec read ----------
fn read_runspec(stream: &mut TcpStream) -> Result<RunSpec, SiggError> {
    Ok(RunSpec {
        w: read_u32(stream)?,
        h: read_u32(stream)?,
        steps: read_u32(stream)?,
        seed: read_u32(stream)?,
        du: read_f32(stream)?,
        dv: read_f32(stream)?,
        f: read_f32(stream)?,
        k: read_f32(stream)?,
        dt: read_f32(stream)?,
        eps: read_f32(stream)?,
        reaction_id: read_u32(stream)?,
        constraint_id: read_u32(stream)?,
        noise_id: read_u32(stream)?,
    })
}

fn compute_open(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let world: WorldKey = (u, v, w);

    let mut st = pocket_state.lock().unwrap();
    let space_id = st.compute_next_id;
    st.compute_next_id = 1;
    // とりあえずデフォルト（後でプロトコルに size/lanes を足してもOK）
    let space_size = read_u32(stream)? as i32;
    let lanes = read_u32(stream)? as usize;
    let cs = crate::pocket::ComputeSpace::new(world, space_id, space_size, lanes);


    write_ok(stream)?;
    write_u32(stream, space_id)?;
    Ok(())
}

fn compute_write(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let space_id = read_u32(stream)?;
    let n = read_u32(stream)? as usize;

    // payload: (x:i32, y:i32, z:i32, lane:u32, bits:u32) * n
    let mut cells: Vec<(i32, i32, i32, usize, u32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = read_i32(stream)?;
        let y = read_i32(stream)?;
        let z = read_i32(stream)?;
        let lane = read_u32(stream)? as usize;
        let bits = read_u32(stream)?;
        cells.push((x, y, z, lane, bits));
    }

    let mut st = pocket_state.lock().unwrap();
    let cs = st.compute_spaces.get_mut(&space_id).ok_or_else(|| SiggError::runtime("unknown compute space_id"))?;

    for (x, y, z, lane, bits) in cells {
        cs.write_cell_bits(x, y, z, lane, bits);
    }

    write_ok(stream)?;
    Ok(())
}


fn compute_run(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let space_id = read_u32(stream)?;
    let steps = read_u32(stream)?;

    let mut st = pocket_state.lock().unwrap();
    let cs = match st.compute_spaces.get_mut(&space_id) {
        Some(v) => v,
        None => return write_err(stream, "unknown compute space_id"),
    };
    // 1D実験: lane=0 を回す
    cs.step_rule110_1d(steps, 0);
    write_ok(stream)?;
    Ok(())
}

fn compute_read(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let space_id = read_u32(stream)?;
    let n = read_u32(stream)? as usize;

    let mut xs: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..n {
        xs.push(read_i32(stream)?);
    }

    let st = pocket_state.lock().unwrap();
    let cs = match st.compute_spaces.get(&space_id) {
        Some(v) => v,
        None => return write_err(stream, "unknown compute space_id"),
    };

    // xs: Vec<i32> -> coords4: Vec<(x,0,0,lane)>
    let coords4: Vec<(i32, i32, i32, usize)> =
        xs.iter().map(|&x| (x, 0, 0, 0usize)).collect();

    let vals = cs.read_cells_bits(&coords4);

    write_ok(stream)?;
    write_u32(stream, vals.len() as u32)?;
    for v in vals {
        write_u32(stream, v)?;
    }
    Ok(())
}


// ---------- Pocket handlers (tag-based, same stream) ----------
fn pocket_open(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let chunk_size = read_u32(stream)? as usize;

    let z_dim_req = read_u32(stream)? as usize; // clientが要求するz_dim
    let space_id = read_u32(stream)?;

    let mut st = pocket_state.lock().unwrap();

    // --- z_dim を Atlas と同期（Atlasが空なら作り直し、空でないなら拒否）
    let atlas_z = st.atlas.z_dim();
    if atlas_z != z_dim_req {
        if st.atlas.is_empty() {
            st.atlas = crate::pocket::atlas::Atlas::new(z_dim_req);
        } else {
            return write_err(
                stream,
                &format!(
                    "atlas z_dim mismatch: server={} client={} (restart server or use z_dim={})",
                    atlas_z, z_dim_req, atlas_z
                ),
            );
        }
    }

    // ★ここで確定した z_dim を使う
    let z_dim = st.atlas.z_dim();

    let world: WorldKey = (u, v, w);
    let handle = st.next_handle;
    st.next_handle += 1;

    let delta_path = format!("{}/pocket_deltas_{}_{}_{}_s{}.bin", st.data_dir, u, v, w, space_id);

    // --- ここが質問の部分：z_dim は「確定値」を渡す
    let pocket = crate::pocket::PocketWorld::open(world, chunk_size, z_dim, delta_path)
        .map_err(|e| SiggError::io(e.to_string()))?;

    st.pockets.insert(handle, pocket);

    write_ok(stream)?;
    write_u32(stream, handle)?;
    Ok(())
}


fn pocket_write(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let handle = read_u32(stream)?;
    let x = read_i32(stream)?;
    let y = read_i32(stream)?;
    let z = read_i32(stream)?;
    let version = read_u32(stream)?;
    let confidence = read_f32(stream)?;

    let mut st = pocket_state.lock().unwrap();
    let pocket = match st.pockets.get_mut(&handle) {
        Some(p) => p,
        None => return write_err(stream, "unknown pocket handle"),
    };

    let zvec = read_f32_vec(stream, pocket.z_dim)?;
    let parent_hash = hash64(&(handle, x, y, z, version));
    if let Err(e) = pocket.write_cell(x,y,z, zvec, version, parent_hash, confidence) {
        return write_err(stream, &format!("write_cell failed: {e}"));
    }

    write_ok(stream)?;
    write_u32(stream, 1)?;
    Ok(())
}

fn pocket_write_f32(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let pocket_h = read_u32(stream)?;
    let x = read_i32(stream)?;
    let y = read_i32(stream)?;
    let z = read_i32(stream)?;
    let lane = read_u32(stream)? as usize;
    let bits = read_u32(stream)?;
    let v = f32::from_bits(bits);

    let mut st = pocket_state.lock().unwrap();
    let pocket = st.pockets.get_mut(&pocket_h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;

    // PocketWorld が f32 API を持っている前提
    pocket.cell_write_f32(x, y, z, lane, v);

    write_ok(stream)?;
    Ok(())
}

fn pocket_read_f32(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let pocket_h = read_u32(stream)?;
    let x = read_i32(stream)?;
    let y = read_i32(stream)?;
    let z = read_i32(stream)?;
    let lane = read_u32(stream)? as usize;

    let mut st = pocket_state.lock().unwrap();
    let pocket = st.pockets.get_mut(&pocket_h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;

    let bits: u32 = pocket.cell_read_f32(x, y, z, lane).to_bits();

    write_ok(stream)?;
    write_u32(stream, bits)?;
    Ok(())
}



fn atlas_upsert(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let cx = read_i32(stream)?;
    let cy = read_i32(stream)?;
    let cz = read_i32(stream)?;

    let mut st = pocket_state.lock().unwrap();
    let z_dim = st.atlas.z_dim();
    let proto = read_f32_vec(stream, z_dim)?;

    if let Err(e) = st.atlas.upsert((u,v,w), (cx,cy,cz), proto) {
        return write_err(stream, &format!("atlas upsert failed: {e}"));
    }

    write_ok(stream)?;
    write_u32(stream, 1)?;
    Ok(())
}

fn atlas_query(
    stream: &mut TcpStream,
    pocket_state: &Arc<Mutex<PocketState>>,
) -> Result<(), SiggError> {
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let topk = read_u32(stream)? as usize;

    // query_vec を読むための z_dim は Atlas 側の設定を使う
    let z_dim = {
        let st = pocket_state.lock().unwrap();
        st.atlas.z_dim()
    };

    let query = read_f32_vec(stream, z_dim)?;

    let chunks = {
        let st = pocket_state.lock().unwrap();
        st.atlas.query_topk((u, v, w), &query, topk)
    };

    write_ok(stream)?;
    write_u32(stream, chunks.len() as u32)?;
    for (cx, cy, cz) in &chunks {
        write_i32(stream, *cx)?;
        write_i32(stream, *cy)?;
        write_i32(stream, *cz)?;
    }
    Ok(())
}


fn pocket_search_and_trigger(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    eprintln!("[pocket_search_and_trigger] enter");

    let handle = read_u32(stream)?;
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;

    let topk_chunks = read_u32(stream)? as usize;

    let steps0 = read_u32(stream)?;
    let diffusion0 = read_f32(stream)?;
    let threshold0 = read_f32(stream)?;
    let topn = read_u32(stream)? as usize;

    // z_dim を先に取得（query_vec読み取りのため）
    let z_dim = {
        let st = pocket_state.lock().unwrap();
        let p = st.pockets.get(&handle)
            .ok_or_else(|| SiggError::runtime("unknown pocket handle"))?;
        if p.world != (u, v, w) {
            return write_err(stream, "world mismatch: pocket.world != requested world");
        }
        p.z_dim
    };

    eprintln!("[pocket_search_and_trigger] reading query z_dim={}", z_dim);
    


    let query = read_f32_vec(stream, z_dim)?;
    eprintln!("[pocket_search_and_trigger] read query ok");

    let mut st = pocket_state.lock().unwrap();

    // pocket を一時的に map から外す（borrow 衝突回避）
    let mut pocket = match st.pockets.remove(&handle) {
        Some(p) => p,
        None => return write_err(stream, "unknown pocket handle"),
    };

    // seed（Atlasが空のとき立ち上げ用）
    for cx in -1..=1 {
        for cy in -1..=1 {
            for cz in -1..=1 {
                pocket.ensure_atlas_proto(&mut st.atlas, (cx, cy, cz));
            }
        }
    }

    let mut chunks = st.atlas.query_topk((u, v, w), &query, topk_chunks);
    if chunks.is_empty() { chunks.push((0, 0, 0)); }

    for &ck in &chunks {
        pocket.ensure_atlas_proto(&mut st.atlas, ck);
    }
    // --- SIGG constraint-based auto tuning (safe) ---
    let world: WorldKey = (u, v, w);

    // proto数（成熟度）
    let proto_count_f = st.atlas.count_protos_in_world((u, v, w)) as f32;

    // world_params を「最初に」確定（ここが重要：run 前）
    let mut wp_snapshot = st
        .world_params
        .get(&world)
        .cloned()
        .unwrap_or_else(|| crate::state::WorldParams {
            steps: steps0.clamp(2, 64),
            diffusion: diffusion0.clamp(0.02, 1.50),
            threshold: threshold0.clamp(0.02, 0.95),
        });

    // 未学習なら少し探索寄り（穏やか補正）
    let maturity = (proto_count_f / 256.0).clamp(0.0, 1.0);
    wp_snapshot.diffusion = (wp_snapshot.diffusion * (1.0 + (1.0 - maturity) * 0.20)).clamp(0.02, 1.50);
    wp_snapshot.steps = ((wp_snapshot.steps as f32) * (1.0 + (1.0 - maturity) * 0.10))
        .round()
        .clamp(2.0, 64.0) as u32;
    wp_snapshot.threshold = wp_snapshot.threshold.clamp(0.02, 0.95);

    // run に使うパラメータ（クライアントに返すのもこれ）
    let steps_run = wp_snapshot.steps;
    let diff_run  = wp_snapshot.diffusion;
    let thr_run   = wp_snapshot.threshold;

    // 実行（※ここで steps_run/diff_run/thr_run を使う）
    let (hits, _used_steps, _used_diff, _used_thr) =
        pocket.trigger_extract_auto(&chunks, &query, steps_run, diff_run, thr_run, topn);

    let hit_mu = st.atlas.hit_mu();
    let beta_now = st.atlas.beta();

    // ---- SIGG constraint tuner (world persistent) ----
    let hit_mean = if hits.is_empty() { 0.0 }
    else { hits.iter().map(|h| h.score).sum::<f32>() / hits.len() as f32 };

    let hit_count = hits.len() as f32;

    // 目標（SIGG制約）
    let target_mean = 0.28_f32;
    let target_hits = (topn as f32 * 0.60).max(4.0);
    let tol_hits    = (topn as f32 * 0.20).max(2.0);
    let topn_f      = topn as f32;

    // tuner は wp_snapshot をベースに更新
    let mut steps = wp_snapshot.steps;
    let mut diffusion = wp_snapshot.diffusion;
    let mut threshold = wp_snapshot.threshold;

    if hit_count < target_hits - tol_hits {
        threshold *= 0.95;
        diffusion *= 1.06;
        steps = (steps + 1).min(64);
    } else if hit_count > target_hits + tol_hits {
        threshold *= 1.06;
        diffusion *= 0.94;
        steps = steps.saturating_sub(1).max(2);
    }

    // 満タン近いなら強く絞る
    if hit_count >= topn_f * 0.95 {
        threshold *= 1.10;
        diffusion *= 0.90;
        steps = steps.saturating_sub(1).max(2);
    }

    // mean 高いなら絞る（thr↑, diff↓）
    if hit_mean > target_mean + 0.01 {
        threshold *= 1.03;
        diffusion *= 0.97;
    } else if hit_mean < target_mean - 0.01 {
        diffusion *= 1.03;
        threshold *= 0.99;
    }

    // clamp
    threshold = threshold.clamp(0.02, 0.95);
    diffusion = diffusion.clamp(0.02, 1.50);
    steps = steps.clamp(2, 64);

    // 保存（次回に効く）
    wp_snapshot.steps = steps;
    wp_snapshot.diffusion = diffusion;
    wp_snapshot.threshold = threshold;
    st.world_params.insert(world, wp_snapshot.clone());

    // atlas の観測＆更新
    st.atlas.observe_hit_mean(hit_mean);
    let beta = st.atlas.beta();
    pocket.atlas_update_from_hits(&mut st.atlas, &hits, beta, 16);

    let proto_count = st.atlas.count_protos_in_world((u, v, w));

    // ログ
    eprintln!(
        "[tune] hits={} mean={:.4} -> steps={} diff={:.3} thr={:.3}",
        hit_count as u32, hit_mean, wp_snapshot.steps, wp_snapshot.diffusion, wp_snapshot.threshold
    );
    eprintln!("[atlas] protos_in_world={}", proto_count);


    // worldの現状パラメータ（なければデフォルト）
    let mut wp_snapshot = st
        .world_params
        .get(&world)
        .cloned()
        .unwrap_or_else(|| crate::state::WorldParams {
            steps: steps0.clamp(2, 64),
            diffusion: diffusion0.clamp(0.02, 1.50),
            threshold: threshold0.clamp(0.02, 0.95),
        });


    let mut steps = wp_snapshot.steps;
    let mut diffusion = wp_snapshot.diffusion;
    let mut threshold = wp_snapshot.threshold;

    let topn_f = topn as f32;

    // --- 調整1：hits数（少なすぎ→広げる、多すぎ→絞る） ---
    if hit_count < target_hits - tol_hits {
        threshold *= 0.95;
        diffusion *= 1.06;
        steps = (steps + 1).min(64);
    } else if hit_count > target_hits + tol_hits {
        threshold *= 1.06;
        diffusion *= 0.94;
        steps = steps.saturating_sub(1).max(2);
    }

    // --- 強制制約：hitsが満タン近いなら強く絞る ---
    if (hit_count as f32) >= topn_f * 0.95 {
        threshold *= 1.10;
        diffusion *= 0.90;
        steps = steps.saturating_sub(1).max(2);
    }

    // --- 調整2：平均スコア（低い→探索寄り、高い→収束寄り） ---
    if hit_mean < target_mean - 0.01 {
        diffusion *= 1.03;
        threshold *= 0.99;
        steps = (steps + 1).min(64);
    } else if hit_mean > target_mean + 0.01 {
        threshold *= 1.03;
        diffusion *= 0.97;
        if hit_count > target_hits + tol_hits {
            steps = steps.saturating_sub(1).max(2);
        }
    }

    // --- clamp ---
    threshold = threshold.clamp(0.02, 0.95);
    diffusion = diffusion.clamp(0.02, 1.50);
    steps = steps.clamp(2, 64);

    // 更新結果をローカルWPに反映
    wp_snapshot.steps = steps;
    wp_snapshot.diffusion = diffusion;
    wp_snapshot.threshold = threshold;

    // ここで world_params に戻す（stへの借用を最短に）
    st.world_params.insert(world, wp_snapshot.clone());

    // ==============================
    // atlas側の観測＆更新（wp借用なし）
    // ==============================
    st.atlas.observe_hit_mean(hit_mean);
    let beta = st.atlas.beta();
    pocket.atlas_update_from_hits(&mut st.atlas, &hits, beta, 16);

    // proto_count（この時点で st.atlas は触れる）
    let proto_count = st.atlas.count_protos_in_world((u, v, w));      

    // 戻す
    st.pockets.insert(handle, pocket);

    // response
    // response (fixed layout)
    write_ok(stream)?;

    // used params (authoritative)
    write_u32(stream, steps_run)?;
    write_f32(stream, diff_run)?;
    write_f32(stream, thr_run)?;

    // atlas stats
    write_u32(stream, proto_count)?;
    write_f32(stream, hit_mu)?;
    write_f32(stream, beta_now)?;

    // chunks
    write_u32(stream, chunks.len() as u32)?;
    for (cx, cy, cz) in &chunks {
        write_i32(stream, *cx)?;
        write_i32(stream, *cy)?;
        write_i32(stream, *cz)?;
    }

    // stabilize: apply threshold + cap
    let mut hits2 = hits;
    hits2.retain(|h| h.score >= thr_run);
    hits2.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits2.truncate(topn);
    let hits = hits2;


    // hits
    write_u32(stream, hits.len() as u32)?;
    for h in &hits {
        write_i32(stream, h.cell.0)?;
        write_i32(stream, h.cell.1)?;
        write_i32(stream, h.cell.2)?;
        write_f32(stream, h.score)?;
    }

    Ok(())

}




fn pocket_trigger(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    // req:
    // handle u32
    // world u,v,w (u32*3)  ※必要ならチェック
    // n_chunks u32 + chunks (i32,i32,i32)*
    // steps u32, diffusion f32, threshold f32, topn u32
    // query_vec f32[z_dim]
    let handle = read_u32(stream)?;
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;

    let n_chunks = read_u32(stream)? as usize;
    let mut chunks: Vec<(i32, i32, i32)> = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        let cx = read_i32(stream)?;
        let cy = read_i32(stream)?;
        let cz = read_i32(stream)?;
        chunks.push((cx, cy, cz));
    }

    let steps0 = read_u32(stream)?;
    let diffusion0 = read_f32(stream)?;
    let threshold0 = read_f32(stream)?;
    let topn = read_u32(stream)? as usize;

    // z_dim を先に取得して query_vec 読み
    let z_dim = {
        let st = pocket_state.lock().unwrap();
        let p = st.pockets.get(&handle)
            .ok_or_else(|| SiggError::runtime("unknown pocket handle"))?;
        if p.world != (u, v, w) {
            return write_err(stream, "world mismatch: pocket.world != requested world");
        }
        p.z_dim
    };

    let query = read_f32_vec(stream, z_dim)?;

    // ここから本処理
    let mut st = pocket_state.lock().unwrap();

    // pocket を一旦取り出す（borrow衝突回避）
    let mut pocket = match st.pockets.remove(&handle) {
        Some(p) => p,
        None => return write_err(stream, "unknown pocket handle"),
    };

    // ★ chunk proto を A(base_chunk) で必要分だけ自動登録
    for &ck in &chunks {
        pocket.ensure_atlas_proto(&mut st.atlas, ck);
    }

    let world: WorldKey = (u, v, w);
    let (steps_in, diff_in, thr_in) = {
        let wp = st.world_params.get(&world).copied().unwrap_or(crate::state::WorldParams::default());
        (wp.steps, wp.diffusion, wp.threshold)
    };

    let (hits, used_steps, used_diff, used_thr) =
        pocket.trigger_extract_auto(&chunks, &query, steps_in, diff_in, thr_in, topn);


    // ★ hit平均で atlas 更新（地形は固定、地図が育つ）
    pocket.atlas_update_from_hits(&mut st.atlas, &hits, 0.10, 16);

    // pocket を戻す
    st.pockets.insert(handle, pocket);

    // response
    write_ok(stream)?;
    write_u32(stream, used_steps)?;
    write_f32(stream, used_diff)?;
    write_f32(stream, used_thr)?;

    write_u32(stream, hits.len() as u32)?;
    for h in &hits {
        write_i32(stream, h.cell.0)?;
        write_i32(stream, h.cell.1)?;
        write_i32(stream, h.cell.2)?;
        write_f32(stream, h.score)?;
    }
    write_err(stream, "pocket_trigger not implemented in this snippet")?;
    Ok(())
}

fn hash_query_f32(query: &[f32]) -> u64 {
    // f32 の bits をそのまま畳む（安定・高速）
    let mut acc: u64 = 1469598103934665603u64; // FNV offset
    for &v in query {
        acc ^= v.to_bits() as u64;
        acc = acc.wrapping_mul(1099511628211u64);
    }
    acc
}


fn pocket_persist(
    stream: &mut TcpStream,
    pocket_state: &Arc<Mutex<PocketState>>,
) -> Result<(), SiggError> {
    let handle = read_u32(stream)?;

    let mut st = pocket_state.lock().unwrap();
    let pocket = st
        .pockets
        .get_mut(&handle)
        .ok_or_else(|| SiggError::runtime("unknown pocket handle"))?;

    pocket
        .persist()
        .map_err(|e| SiggError::io(e.to_string()))?;
    write_ok(stream)?;
    write_u32(stream, 1)?;
    Ok(())
}


fn handle_client(mut stream: TcpStream, ctx: ServerCtx) -> Result<(), SiggError> {
    loop {
        let tag = match read_u32(&mut stream) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        match tag {
            TAG_POCKET_OPEN => { pocket_open(&mut stream, &ctx.pocket)?; }
            TAG_ATLAS_QUERY => { atlas_query(&mut stream, &ctx.pocket)?; }
            TAG_POCKET_TRIGGER => { pocket_trigger(&mut stream, &ctx.pocket)?; }
            TAG_POCKET_SEARCH_AND_TRIGGER => { pocket_search_and_trigger(&mut stream, &ctx.pocket)?; }
            TAG_COMPUTE_OPEN => { compute_open(&mut stream, &ctx.pocket)?; }
            TAG_COMPUTE_WRITE => { compute_write(&mut stream, &ctx.pocket)?; }
            TAG_COMPUTE_RUN => { compute_run(&mut stream, &ctx.pocket)?; }
            TAG_COMPUTE_READ => { compute_read(&mut stream, &ctx.pocket)?; }
            TAG_CPU_CREATE     => { cpu_create(&mut stream, &ctx.pocket)?; }
            TAG_CPU_LOAD_PROG  => { cpu_load_prog(&mut stream, &ctx.pocket)?; }
            TAG_CPU_SET_BASE   => { cpu_set_base(&mut stream, &ctx.pocket)?; }
            TAG_CPU_SET_REG    => { cpu_set_reg(&mut stream, &ctx.pocket)?; }
            TAG_CPU_GET_REGS   => { cpu_get_regs(&mut stream, &ctx.pocket)?; }
            TAG_CPU_RUN        => { cpu_run_req(&mut stream, &ctx.pocket)?; }
            TAG_CPU_STATUS     => { cpu_status(&mut stream, &ctx.pocket)?; }
            TAG_AI_CREATE  => { ai_create(&mut stream, &ctx.pocket)?; }
            TAG_AI_TICK    => { ai_tick(&mut stream, &ctx.pocket)?; }
            TAG_AI_OBSERVE => { ai_observe(&mut stream, &ctx.pocket)?; }
            TAG_AI_SET_SENSORS => { ai_set_sensors(&mut stream, &ctx.pocket)?; }
            TAG_CPU_ASM_LOAD => { cpu_asm_load(&mut stream, &ctx.pocket)?; }
            TAG_AI_SET_IO => { ai_set_io(&mut stream, &ctx.pocket)?; }
            TAG_POCKET_WRITE_F32 => { pocket_write_f32(&mut stream, &ctx.pocket)?; }
            TAG_POCKET_READ_F32 => { pocket_read_f32(&mut stream, &ctx.pocket)?; }



            TAG_RUN_ONCE => {
                let spec = read_runspec(&mut stream)?;
                let flags = 0u32;
                let early = (flags & FLAG_EARLY_STOP) != 0;
                let (digest, score, steps_done) = run_world(spec, early)?;
                write_ok(&mut stream)?;
                write_u32(&mut stream, digest)?;
                write_f32(&mut stream, score)?;
                write_u32(&mut stream, steps_done)?;
            }

            TAG_SUBMIT_BATCH => {
                let n = read_u32(&mut stream)? as usize;
                let topk = read_u32(&mut stream)? as usize;
                let flags = read_u32(&mut stream)?;
            
                let job_id = ctx.next_job_id.fetch_add(1, AtomicOrdering::Relaxed);
            
                // ✅ 先に RunSpec を全部受け取って job.batch に入れる
                let mut batch: Vec<RunSpec> = Vec::with_capacity(n);
                for _ in 0..n {
                    batch.push(read_runspec(&mut stream)?);
                }
            
                {
                    let mut jobs = ctx.jobs.lock().unwrap();
                    jobs.insert(job_id, Job {
                        id: job_id,
                        state: JobState::Queued,
                        done: 0,
                        total: n as u32,
                        topk,
                        flags,
                        results: Vec::new(),
                        err_msg: None,
                        cancel: false,
                        batch,
                    });
                }
            
                // ✅ queue には job_id だけ積む（worker_loop の設計と一致）
                {
                    let (q_mu, q_cv) = &*ctx.queue;
                    let mut q = q_mu.lock().unwrap();
                    q.push_back(job_id);
                    q_cv.notify_one();
                }
            
                write_ok(&mut stream)?;
                write_u64(&mut stream, job_id)?;
            }
            

            TAG_POLL => {
                let job_id = read_u64(&mut stream)?;
                let jobs = ctx.jobs.lock().unwrap();
                let Some(job) = jobs.get(&job_id) else {
                    write_err(&mut stream, "unknown job")?;
                    continue;
                };

                write_ok(&mut stream)?;
                write_u32(&mut stream, job.state as u32)?;
                write_u32(&mut stream, job.done)?;
                write_u32(&mut stream, job.total)?;
            }

            TAG_FETCH => {
                let job_id = read_u64(&mut stream)?;
                let jobs = ctx.jobs.lock().unwrap();
                let Some(job) = jobs.get(&job_id) else {
                    write_err(&mut stream, "unknown job")?;
                    continue;
                };

                write_ok(&mut stream)?;
                write_u32(&mut stream, job.results.len() as u32)?;
                for e in &job.results {
                    write_f32(&mut stream, e.score)?;
                    write_u32(&mut stream, e.digest)?;
                    write_u32(&mut stream, e.steps_done)?;
                    write_u32(&mut stream, e.spec.seed)?;
                }
            }

            TAG_CANCEL => {
                let job_id = read_u64(&mut stream)?;
                let mut jobs = ctx.jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.cancel = true;
                    job.state = JobState::Canceled;
                }
                write_ok(&mut stream)?;
            }
            

            _ => {
                write_err(&mut stream, "unknown tag")?;
            }
        }
    }
}

// ============================================================
// ★ CPU handlers
// ============================================================
//
// TAG_CPU_CREATE:
//   u,v,w (u32,u32,u32)
//   returns: ok + cpu_handle(u32)
//
fn cpu_create(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let world = (u, v, w);

    let mut st = pocket_state.lock().unwrap();
    let handle = st.next_handle;
    st.next_handle += 1;
    st.cpu_states.insert(handle, CpuState::new(world));

    write_ok(stream)?;
    write_u32(stream, handle)?;
    Ok(())
}

// TAG_CPU_LOAD_PROG:
//   cpu_handle(u32) + n(u32) + program[n](u32...)
fn cpu_load_prog(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let space_id = read_u32(stream)?; // どの ComputeSpace に書くか
    let n = read_u32(stream)? as usize;
    let prog = read_u32_vec(stream, n)?;

    let mut st = pocket_state.lock().unwrap();

    // CPU を取得して初期化 + base座標を抜き出す
    let (base_x, base_y, base_z) = {
        let cpu = st
            .cpu_states
            .get_mut(&h)
            .ok_or_else(|| SiggError::runtime("unknown cpu handle"))?;
        cpu.pc = 0;
        cpu.halted = false;
        (cpu.base_x, cpu.base_y, cpu.base_z)
    };

    // ComputeSpace を取得してプログラムを書き込む（lane=0固定）
    let cs = st
        .compute_spaces
        .get_mut(&space_id)
        .ok_or_else(|| SiggError::runtime("bad compute space"))?;

    for (i, &w) in prog.iter().enumerate() {
        let x = base_x + i as i32;
        cs.write_cell_bits(x, base_y, base_z, 0usize, w);
    }

    write_ok(stream)?;
    Ok(())
}

fn cpu_asm_load(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let space_id = read_u32(stream)?;
    let text_len = read_u32(stream)? as usize;
    let text_bytes = read_exact(stream, text_len)?;
    let text = String::from_utf8_lossy(&text_bytes);

    let prog = crate::pocket::asm::assemble(&text)
        .map_err(|e| SiggError::runtime(&format!("asm error: {e}")))?;

    // CPU を取得して初期化 + base座標を抜き出す
    let mut st = pocket_state.lock().unwrap();

    let (base_x, base_y, base_z) = {
        let cpu = st.cpu_states.get_mut(&h).ok_or_else(|| SiggError::runtime("unknown cpu handle"))?;
        cpu.pc = 0;
        cpu.halted = false;
        (cpu.base_x, cpu.base_y, cpu.base_z)
    };

    // ComputeSpace に書く（lane0）
    let cs = st.compute_spaces.get_mut(&space_id).ok_or_else(|| SiggError::runtime("bad compute space"))?;
    for (i, &w) in prog.iter().enumerate() {
        let x = base_x + i as i32;
        cs.write_cell_bits(x, base_y, base_z, 0usize, w);
    }

    write_ok(stream)?;
    Ok(())
}


// TAG_CPU_SET_BASE:
//   cpu_handle(u32) + base_x(i32) base_y(i32) base_z(i32) + lane(u32)
fn cpu_set_base(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let bx = read_i32(stream)?;
    let by = read_i32(stream)?;
    let bz = read_i32(stream)?;
    let lane = read_u32(stream)?;
    let mut st = pocket_state.lock().unwrap();
    let cpu = match st.cpu_states.get_mut(&h) {
        Some(c) => c,
        None => return write_err(stream, "unknown cpu handle"),
    };
    cpu.base_x = bx;
    cpu.base_y = by;
    cpu.base_z = bz;
    cpu.lane = lane;
    write_ok(stream)?;
    Ok(())
}

// TAG_CPU_SET_REG:
//   cpu_handle(u32) + reg_id(u32 0..15) + value(u32)
fn cpu_set_reg(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let rid = read_u32(stream)? as usize;
    let val = read_u32(stream)?;
    if rid >= 16 { return write_err(stream, "reg id out of range"); }
    // ※ 下への変更を入れると、クライアントから r8〜r15 をセットしたい場合に ERR になります。
    // （本来は CPU 側を16本に揃えるのが筋）
    // if rid >= cpu.regs.len() { return write_err(stream, "reg id out of range"); }
    let mut st = pocket_state.lock().unwrap();
    let cpu = match st.cpu_states.get_mut(&h) {
        Some(c) => c,
        None => return write_err(stream, "unknown cpu handle"),
    };
    cpu.regs[rid] = val;
    write_ok(stream)?;
    Ok(())
}

// TAG_CPU_GET_REGS:
//   cpu_handle(u32) -> ok + pc(u32) halted(u32) + regs[16](u32...)
fn cpu_get_regs(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let st = pocket_state.lock().unwrap();
    let cpu = match st.cpu_states.get(&h) {
        Some(c) => c,
        None => return write_err(stream, "unknown cpu handle"),
    };
    write_ok(stream)?;
    write_u32(stream, cpu.pc)?;
    write_u32(stream, if cpu.halted { 1 } else { 0 })?;
    // クライアント仕様は regs[16] を返す前提。
    // CpuState 側が 8本でも落ちないように 0 埋めで返す。
    for i in 0..16 {
        let v = *cpu.regs.get(i).unwrap_or(&0u32);
        write_u32(stream, v)?;
    }
    Ok(())
}

// TAG_CPU_RUN:
//   cpu_handle(u32) + pocket_handle(u32) + budget(u32)
//   -> ok + ran(u32) + pc(u32) + halted(u32)
//
// “どのPocketをメモリにするか”を明示するため pocket_handle を渡す
fn cpu_run_req(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let cpu_h = read_u32(stream)?;
    let _pocket_h = read_u32(stream)?; // 分岐なしなら不要。受信だけして捨てる
    let budget = read_u32(stream)?;
    let space_id = read_u32(stream)?;

    let (mut space, mut cpu) = {
        let mut st = pocket_state.lock().unwrap();
        let space = st.compute_spaces.remove(&space_id).ok_or_else(|| SiggError::runtime("bad compute space"))?;
        let cpu = st.cpu_states.remove(&cpu_h).ok_or_else(|| SiggError::runtime("bad cpu"))?;
        (space, cpu)
    };

    let ran = {
        let mut mem = space.as_pocket_adapter_mut();
        crate::pocket::cpu::cpu_run_mem(&mut mem, &mut cpu, budget)?
    };

    let pc_out = cpu.pc;
    let halted_out: u32 = if cpu.halted { 1 } else { 0 };
    {
        let mut st = pocket_state.lock().unwrap();
        st.compute_spaces.insert(space_id, space);
        st.cpu_states.insert(cpu_h, cpu);
    }

    write_ok(stream)?;
    write_u32(stream, ran)?;
    write_u32(stream, pc_out)?;
    write_u32(stream, halted_out)?;
    Ok(())
}



// TAG_CPU_STATUS:
//   cpu_handle(u32) -> ok + steps(u64) + pc(u32) + halted(u32)
fn cpu_status(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let h = read_u32(stream)?;
    let st = pocket_state.lock().unwrap();
    let cpu = match st.cpu_states.get(&h) {
        Some(c) => c,
        None => return write_err(stream, "unknown cpu handle"),
    };
    write_ok(stream)?;
    write_u64(stream, cpu.steps)?;
    write_u32(stream, cpu.pc)?;
    write_u32(stream, if cpu.halted { 1 } else { 0 })?;
    Ok(())
}

// ============================
// Worker loop (async batch execution)
// ============================

fn worker_loop(ctx: ServerCtx) {
    loop {
        if ctx.shutdown.load(AtomicOrdering::Relaxed) {
            return;
        }

        let job_id = {
            let (q_mu, q_cv) = &*ctx.queue;
            let mut q = q_mu.lock().unwrap();
            while q.is_empty() {
                q = q_cv.wait(q).unwrap();
                if ctx.shutdown.load(AtomicOrdering::Relaxed) {
                    return;
                }
            }
            q.pop_front().unwrap()
        };

        // mark running + take batch
        let (batch, topk, flags) = {
            let mut jobs = ctx.jobs.lock().unwrap();
            let job = match jobs.get_mut(&job_id) {
                Some(j) => j,
                None => continue,
            };
            if job.cancel {
                job.state = JobState::Canceled;
                continue;
            }
            job.state = JobState::Running;
            (job.batch.clone(), job.topk, job.flags)
        };

        // run
        let mut heap: BinaryHeap<MinByScore> = BinaryHeap::new();
        let mut done: u32 = 0;

        for spec in batch {
            // cancel check
            {
                let jobs = ctx.jobs.lock().unwrap();
                if let Some(j) = jobs.get(&job_id) {
                    if j.cancel {
                        drop(jobs);
                        let mut jobs2 = ctx.jobs.lock().unwrap();
                        if let Some(j2) = jobs2.get_mut(&job_id) {
                            j2.state = JobState::Canceled;
                        }
                        return;
                    }
                }
            }

            let early = (flags & FLAG_EARLY_STOP) != 0;
            let out = run_world(spec.clone(), early);

            match out {
                Ok((digest, score, steps_done)) => {
                    done += 1;

                    // keep topk largest score
                    let entry = ResultEntry {
                        score,
                        digest,
                        steps_done,
                        spec,
                    };

                    if heap.len() < topk {
                        heap.push(MinByScore(entry));
                    } else if let Some(mut worst) = heap.peek_mut() {
                        // worst is min (because wrapper reverses)
                        if entry.score > worst.0.score {
                            *worst = MinByScore(entry);
                        }
                    }
                }
                Err(e) => {
                    let mut jobs = ctx.jobs.lock().unwrap();
                    if let Some(j) = jobs.get_mut(&job_id) {
                        j.state = JobState::Error;
                        j.err_msg = Some(format!("{e}"));
                    }
                    continue;
                }
            }

            // update progress occasionally
            if done % 8 == 0 {
                let mut jobs = ctx.jobs.lock().unwrap();
                if let Some(j) = jobs.get_mut(&job_id) {
                    j.done = done;
                }
            }
        }

        // finalize results sorted desc
        let mut results: Vec<ResultEntry> = heap.into_iter().map(|x| x.0).collect();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut jobs = ctx.jobs.lock().unwrap();
        if let Some(j) = jobs.get_mut(&job_id) {
            j.done = done;
            j.results = results;
            if j.state != JobState::Canceled && j.state != JobState::Error {
                j.state = JobState::Done;
            }
        }
    }
}

// wrapper so BinaryHeap acts like "min-heap by score"
#[derive(Clone, Debug)]
struct MinByScore(ResultEntry);

impl PartialEq for MinByScore {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.to_bits() == other.0.score.to_bits()
    }
}
impl Eq for MinByScore {}

impl PartialOrd for MinByScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinByScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 「最小スコアが先頭（worst）」になるように逆順
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}


// ============================
// World core (reaction-diffusion + noise + projection)
// ============================

#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[inline]
fn u32_to_01(x: u32) -> f32 {
    (x as f32) / (u32::MAX as f32)
}

#[inline]
fn noise01_from_xy_seed(x: u32, y: u32, seed: u32) -> f32 {
    let h = hash_u32(seed ^ x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA6B));
    u32_to_01(h)
}

#[inline]
fn wrap_dec(i: usize, n: usize) -> usize {
    if i == 0 { n - 1 } else { i - 1 }
}
#[inline]
fn wrap_inc(i: usize, n: usize) -> usize {
    let j = i + 1;
    if j == n { 0 } else { j }
}

fn run_world(spec: RunSpec, early_stop: bool) -> Result<(u32, f32, u32), SiggError> {
    // Only support 2D wrap for max speed (your bench assumption)
    if spec.w == 0 || spec.h == 0 {
        return Err(SiggError::runtime("invalid dims"));
    }
    if spec.reaction_id != 0 {
        return Err(SiggError::runtime("reaction_id unsupported (only 0=GS)"));
    }
    if spec.constraint_id != 0 {
        return Err(SiggError::runtime("constraint_id unsupported (only 0=clamp01)"));
    }
    if spec.noise_id != 0 {
        return Err(SiggError::runtime("noise_id unsupported (only 0=hash-noise)"));
    }

    let w = spec.w as usize;
    let h = spec.h as usize;
    let n = w * h;

    // init: u ~ 1.0, v ~ 0.0 with noise
    // (match your "その場生成ノイズ" 公平方式)
    let mut u = Grid::new(vec![w, h], 0.0, Boundary::Wrap);
    let mut v = Grid::new(vec![w, h], 0.0, Boundary::Wrap);

    for yy in 0..h {
        for xx in 0..w {
            let rn = noise01_from_xy_seed(xx as u32, yy as u32, spec.seed);
            let uu = (1.0 * (1.0 - rn) + 0.5 * rn).clamp(0.0, 1.0);
            let vv = (0.0 * (1.0 - rn) + 0.25 * rn).clamp(0.0, 1.0);
            let i = yy * w + xx;
            u.data[i] = uu;
            v.data[i] = vv;
        }
    }

    // ping-pong buffers (avoid alloc per step)
    let mut u2 = vec![0.0f32; n];
    let mut v2 = vec![0.0f32; n];

    // early-stop thresholds (tunable)
    let var_min = 1e-8f32;

    let mut steps_done: u32 = 0;
    for t in 0..spec.steps {
        steps_done = t + 1;

        // compute next
        for yy in 0..h {
            let ym = wrap_dec(yy, h);
            let yp = wrap_inc(yy, h);
            let y0 = yy * w;
            let y_m = ym * w;
            let y_p = yp * w;

            for xx in 0..w {
                let xm = wrap_dec(xx, w);
                let xp = wrap_inc(xx, w);

                let i = y0 + xx;

                let cu = u.data[i];
                let cv = v.data[i];

                let lap_u =
                    u.data[y0 + xm] + u.data[y0 + xp] +
                    u.data[y_m + xx] + u.data[y_p + xx] -
                    4.0 * cu;

                let lap_v =
                    v.data[y0 + xm] + v.data[y0 + xp] +
                    v.data[y_m + xx] + v.data[y_p + xx] -
                    4.0 * cv;

                // Gray-Scott
                let uvv = cu * cv * cv;
                let du_dt = spec.du * lap_u - uvv + spec.f * (1.0 - cu);
                let dv_dt = spec.dv * lap_v + uvv - (spec.f + spec.k) * cv;

                let mut nu = cu + spec.dt * du_dt;
                let mut nv = cv + spec.dt * dv_dt;

                // exploration noise each step: seed + 777 (match your bench)
                let rn = noise01_from_xy_seed(xx as u32, yy as u32, spec.seed.wrapping_add(777));
                nv = nv + (rn - 0.5) * spec.eps;

                // hard constraint (clamp01)
                nu = nu.clamp(0.0, 1.0);
                nv = nv.clamp(0.0, 1.0);

                u2[i] = nu;
                v2[i] = nv;
            }
        }

        // swap
        u.data.copy_from_slice(&u2);
        v.data.copy_from_slice(&v2);

        // early-stop checks
        if early_stop && (t % 8 == 7) {
            let (score, flags_bad) = evaluate_simple(&v.data);
            if flags_bad {
                break;
            }
            if score < var_min {
                break;
            }
        }
    }

    // final evaluate + digest
    let (score, _bad) = evaluate_simple(&v.data);
    let digest = digest_u32(&u.data, &v.data, w, h);
    Ok((digest, score, steps_done))
}

// evaluate: variance of v (simple but effective for exploration score)
fn evaluate_simple(v: &[f32]) -> (f32, bool) {
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    for &x in v {
        if !x.is_finite() {
            return (0.0, true);
        }
        sum += x;
        sum2 += x * x;
    }
    let n = v.len() as f32;
    let mean = sum / n;
    let var = (sum2 / n) - mean * mean;
    (var.max(0.0), false)
}

// deterministic digest from a few samples (cheap)
fn digest_u32(u: &[f32], v: &[f32], w: usize, h: usize) -> u32 {
    let mut x = 0x1234_5678u32;
    let picks = [
        0usize,
        (w / 2) + (h / 2) * w,
        (w - 1) + 0 * w,
        0 + (h - 1) * w,
        (w - 1) + (h - 1) * w,
    ];
    for &i in &picks {
        let a = u[i].to_bits();
        let b = v[i].to_bits();
        x = hash_u32(x ^ a);
        x = hash_u32(x ^ b);
    }
    x
}
pub fn main_server_bin() -> Result<(), SiggError> {
    let args: Vec<String> = std::env::args().collect();
    let mut addr = "127.0.0.1:39999".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--addr" && i + 1 < args.len() {
            addr = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }
    serve(&addr)
}

// =========================================================
// SIGG-AI API
// =========================================================
//
// AI_CREATE:
//   u32 handle
//   u32 world_u, world_v, world_w
//   u32 space_size (cells per axis, e.g. 128)
//   u32 lane_count (register lanes, e.g. 8 or 16)
//
// returns:
//   ok
//   u64 agent_id
//   u32 space_id
//
fn ai_create(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let pocket_h = read_u32(stream)?;
    let u = read_u32(stream)?;
    let v = read_u32(stream)?;
    let w = read_u32(stream)?;
    let space_size = read_u32(stream)? as usize;
    let lanes = read_u32(stream)? as usize;

    // ★追加：I/O座標
    let io_in  = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?);
    let io_out = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?);
    let pocket_in_addr  = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?);
    let pocket_out_addr = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?); // ★追加
    let pocket_score_addr = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?);
    // ★追加：policy座標
    let pocket_policy_addr = (read_i32(stream)?, read_i32(stream)?, read_i32(stream)?);



    let mut st = pocket_state.lock().unwrap();

    // pocket存在確認
    let p = st.pockets.get(&pocket_h)
        .ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
    if p.world != (u, v, w) {
        return write_err(stream, "world mismatch");
    }

    // ComputeSpace 作成
    let space_id = st.next_space_id;
    st.next_space_id += 1;
    st.compute_spaces.insert(
        space_id,
        ComputeSpace::new_blank(space_size as i32, lanes),
    );

    // CPU 作成（handle 管理）
    let cpu_handle = st.next_handle;
    st.next_handle += 1;
    st.cpu_states.insert(cpu_handle, CpuState::new((u, v, w)));

    // Agent 作成（参照のみ）
    let agent_id = st.next_agent_id;
    st.next_agent_id += 1;

    st.agents.insert(agent_id, SiggAgent {
        id: agent_id,
        world: (u, v, w),
        pocket_handle: pocket_h,
        space_id,
        cpu_handle,
        sensors_flags: 0,
        sensors_coords: vec![],
        io_in,
        io_out,
        pocket_in_addr,
        pocket_out_addr,
        pocket_score_addr,
        pocket_policy_addr,
        score_mu_bits: 0.0f32.to_bits(),
        score_beta: 0.20,
        mode: 0,
        
    });
    

    write_ok(stream)?;
    write_u64(stream, agent_id)?;
    write_u32(stream, space_id)?;
    write_u32(stream, cpu_handle)?;
    Ok(())
}



// AI_TICK:
//   u64 agent_id
//   u32 budget (cpu steps)
// returns:
//   ok
//   u32 ran
//   u32 pc
//   u32 halted(0/1)
//
// --- agentから必要値を抜く（値コピーだけ。中間ロックを増やさない）
fn ai_tick(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let agent_id = read_u64(stream)?;
    let budget   = read_u32(stream)?;

    // --- 0) agent から必要値を値として抜く（ロック短く）
    let (cpu_h, space_id, pocket_h, io_in, io_out, pin, pout, pscore, ppolicy, beta, mu_bits, sensors3) = {
        let st = pocket_state.lock().unwrap();
        let ag = st.agents.get(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
        (
            ag.cpu_handle,
            ag.space_id,
            ag.pocket_handle,
            ag.io_in,
            ag.io_out,
            ag.pocket_in_addr,
            ag.pocket_out_addr,
            ag.pocket_score_addr,
            ag.pocket_policy_addr,
            ag.score_beta,
            ag.score_mu_bits,
            ag.sensors_coords.clone(),
        )
    };

    // --- 1) cpu/space/pocket を remove して同時に触る（借用衝突回避）
    let (mut cpu, mut space, mut pocket) = {
        let mut st = pocket_state.lock().unwrap();
        let cpu    = st.cpu_states.remove(&cpu_h).ok_or_else(|| SiggError::runtime("bad cpu"))?;
        let space  = st.compute_spaces.remove(&space_id).ok_or_else(|| SiggError::runtime("bad space"))?;
        let pocket = st.pockets.remove(&pocket_h).ok_or_else(|| SiggError::runtime("bad pocket handle"))?;
        (cpu, space, pocket)
    };

    // --- 2) Pocket -> in_bits
    let in_bits: u32 = pocket.cell_read_f32(pin.0, pin.1, pin.2, 0).to_bits();

    // --- 3) Pocket(policy) -> ComputeSpace(policy_cell)  (CPUが読み書きできるように)
    let policy_cell = (io_out.0 + 1, io_out.1, io_out.2);
    let mode_cell   = (io_out.0 + 2, io_out.1, io_out.2);

    let policy_bits_in: u32 = pocket
        .cell_read_f32(ppolicy.0, ppolicy.1, ppolicy.2, 0)
        .to_bits();
    space.write_cell_bits(policy_cell.0, policy_cell.1, policy_cell.2, 0, policy_bits_in);

    // --- 4) ComputeSpace(io_in) <- in_bits
    space.write_cell_bits(io_in.0, io_in.1, io_in.2, 0, in_bits);

    // --- 5) CPU実行
    let ran = {
        let mut mem = space.as_pocket_adapter_mut();
        crate::pocket::cpu::cpu_run_mem(&mut mem, &mut cpu, budget)?
    };

    // --- 6) out_bits を読む
    let mut out_bits: u32 = space.read_cell_bits(io_out.0, io_out.1, io_out.2, 0);

    // --- 7) policy を CPU から読み戻し → Pocketへ保存
    let policy_bits_after: u32 = space.read_cell_bits(policy_cell.0, policy_cell.1, policy_cell.2, 0);
    pocket.cell_write_f32(ppolicy.0, ppolicy.1, ppolicy.2, 0, f32::from_bits(policy_bits_after));

    // --- 8) 成否判定（+1）
    // policy_id: 0 => +1, 1 => +2（最小: 1bit）
    let policy_id: u32 = policy_bits_after & 1;
    let expect: u32 = in_bits.wrapping_add(policy_id + 1);
    let ok_policy: bool = out_bits == expect;

    // --- 9) 失敗なら fallback = policy（かつ ComputeSpace上の出力も揃える）
    if !ok_policy {
        // fallback は「policyに従った期待値」を採用
        out_bits = expect;

        space.write_cell_bits(io_out.0, io_out.1, io_out.2, 0, out_bits);
    }

    // --- 10) Pocketへ out を書き戻し
    pocket.cell_write_f32(pout.0, pout.1, pout.2, 0, f32::from_bits(out_bits));
    // 自己フィードバック：Pocket(in) も out で上書き
    pocket.cell_write_f32(pin.0, pin.1, pin.2, 0, f32::from_bits(out_bits));

    // --- 11) score(EWMA) : policyに従って成功したら 1.0
    let score_now: f32 = if ok_policy { 1.0 } else { 0.0 };
    let mu_prev = f32::from_bits(mu_bits);
    let mu_new  = (1.0 - beta) * mu_prev + beta * score_now;
    let mu_new_bits = mu_new.to_bits();

    pocket.cell_write_f32(pscore.0, pscore.1, pscore.2, 0, mu_new);

    // mode を ComputeSpaceに出す（policyセルと衝突させない）
    let new_mode_bits: u32 = if mu_new < 0.5 { 1 } else { 0 };
    space.write_cell_bits(mode_cell.0, mode_cell.1, mode_cell.2, 0, new_mode_bits);

    // Agent側にも保存（次tickに持ち越す）
    {
        let mut st = pocket_state.lock().unwrap();
        let ag = st.agents.get_mut(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
        ag.score_mu_bits = mu_new_bits;
        ag.mode = new_mode_bits;
    }

    // --- 12) センサー観測
    let mut observations = Vec::with_capacity(sensors3.len());
    for (x, y, z) in sensors3 {
        observations.push(space.read_cell_bits(x, y, z, 0));
    }

    let pc_out = cpu.pc;
    let halted_out: u32 = if cpu.halted { 1 } else { 0 };

    // --- 13) 戻す
    {
        let mut st = pocket_state.lock().unwrap();
        st.cpu_states.insert(cpu_h, cpu);
        st.compute_spaces.insert(space_id, space);
        st.pockets.insert(pocket_h, pocket);
    }

    // --- 14) 応答
    write_ok(stream)?;
    write_u32(stream, ran)?;
    write_u32(stream, pc_out)?;
    write_u32(stream, halted_out)?;
    write_u32(stream, in_bits)?;
    write_u32(stream, out_bits)?;
    write_u32(stream, observations.len() as u32)?;
    for v in observations { write_u32(stream, v)?; }
    Ok(())
}




// AI_OBSERVE:
//   u64 agent_id
//   u32 n
//   repeated: i32 x, i32 y, i32 z
// returns:
//   ok
//   u32 n
//   repeated: u32 bits (f32 bits)
//
fn ai_observe(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let agent_id = read_u64(stream)?;
    let n = read_u32(stream)? as usize;

    let mut coords3: Vec<(i32, i32, i32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = read_i32(stream)?;
        let y = read_i32(stream)?;
        let z = read_i32(stream)?;
        coords3.push((x, y, z));
    }

    let space_id = {
        let st = pocket_state.lock().unwrap();
        let ag = st.agents.get(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
        ag.space_id
    };

    let vals: Vec<u32> = {
        let st = pocket_state.lock().unwrap();
        let cs = st.compute_spaces.get(&space_id).ok_or_else(|| SiggError::runtime("bad space"))?;

        // (x,y,z) -> (x,y,z,lane=0)
        let coords4: Vec<(i32, i32, i32, usize)> =
            coords3.iter().map(|&(x, y, z)| (x, y, z, 0usize)).collect();

        cs.read_cells_bits(&coords4)
    };

    write_ok(stream)?;
    write_u32(stream, vals.len() as u32)?;
    for b in vals {
        write_u32(stream, b)?;
    }
    Ok(())
}

fn ai_set_sensors(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let agent_id = read_u64(stream)?;
    let n = read_u32(stream)? as usize;

    let mut coords: Vec<(i32,i32,i32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = read_i32(stream)?;
        let y = read_i32(stream)?;
        let z = read_i32(stream)?;
        coords.push((x,y,z));
    }

    let mut st = pocket_state.lock().unwrap();
    let ag = st.agents.get_mut(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;

    // ✅ センサー座標を差し替え
    ag.sensors_coords = coords;

    // ✅ フラグも立てる（任意だけど運用が安定する）
    if ag.sensors_coords.is_empty() {
        ag.sensors_flags &= !SENSOR_FLAG_HAS_COORDS;
    } else {
        ag.sensors_flags |= SENSOR_FLAG_HAS_COORDS;
    }

    write_ok(stream)?;
    Ok(())
}

fn ai_set_io(stream: &mut TcpStream, pocket_state: &Arc<Mutex<PocketState>>) -> Result<(), SiggError> {
    let agent_id = read_u64(stream)?;

    let in_x = read_i32(stream)?; let in_y = read_i32(stream)?; let in_z = read_i32(stream)?;
    let out_x = read_i32(stream)?; let out_y = read_i32(stream)?; let out_z = read_i32(stream)?;

    let mem_x = read_i32(stream)?; let mem_y = read_i32(stream)?; let mem_z = read_i32(stream)?; // pocket入力アドレス
    let nen_x = read_i32(stream)?; let nen_y = read_i32(stream)?; let nen_z = read_i32(stream)?;
    let ses_x = read_i32(stream)?; let ses_y = read_i32(stream)?; let ses_z = read_i32(stream)?;

    let mut st = pocket_state.lock().unwrap();
    let ag = st.agents.get_mut(&agent_id).ok_or_else(|| SiggError::runtime("bad agent"))?;
    ag.io_in = (in_x,in_y,in_z);
    ag.io_out = (out_x,out_y,out_z);
    ag.pocket_in_addr = (mem_x,mem_y,mem_z);
    ag.pocket_out_addr = (nen_x,nen_y,nen_z);
    ag.pocket_score_addr = (ses_x,ses_y,ses_z);

    write_ok(stream)?;
    Ok(())
}





