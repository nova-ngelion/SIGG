use std::sync::Arc;

use rayon::prelude::*;

use crate::error::SiggError;
use crate::value::{Boundary, Grid, GridRef, Value};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Builtin {
    pub name: &'static str,
    pub f: fn(Vec<Value>) -> Result<Value, SiggError>,
}

static LAST_DIGEST: AtomicU64 = AtomicU64::new(0);

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

// ---------- small helpers ----------
fn need_n(args: &Vec<Value>, n: usize, name: &str) -> Result<(), SiggError> {
    if args.len() != n {
        return Err(SiggError::runtime(format!("{name} expects {n} args")));
    }
    Ok(())
}

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
    ]
}
