use super::types::{WorldKey, ChunkKey, Hit};

/// 6近傍拡散＋閾値＋topN抽出
pub fn react_and_extract(
    world: WorldKey,
    _chunk: ChunkKey,
    chunk_origin: (i32,i32,i32),
    s: usize,
    z_dim: usize,
    z: &[f32],              // len = s*s*s*z_dim
    query_vec: &[f32],       // len = z_dim
    steps: u32,
    diffusion: f32,
    threshold: f32,
    topn: usize,
) -> Vec<Hit> {
    if query_vec.len() != z_dim { return vec![]; }
    if z.len() != s*s*s*z_dim { return vec![]; }

    let q = l2normed(query_vec);

    // score field
    let mut a = vec![0.0f32; s*s*s];
    for idx in 0..(s*s*s) {
        let base = idx*z_dim;
        let cell = &z[base..base+z_dim];
        a[idx] = cos_sim_l2normed(cell, &q);
    }

    // diffuse
    let mut b = vec![0.0f32; a.len()];
    let d = diffusion.clamp(0.0, 1.0);
    let inv = 1.0 - d;

    for _ in 0..steps {
        for x in 0..s {
            for y in 0..s {
                for zc in 0..s {
                    let i = lin(x,y,zc,s);
                    let nb = (
                        a[lin((x+1)%s,y,zc,s)] +
                        a[lin((x+s-1)%s,y,zc,s)] +
                        a[lin(x,(y+1)%s,zc,s)] +
                        a[lin(x,(y+s-1)%s,zc,s)] +
                        a[lin(x,y,(zc+1)%s,s)] +
                        a[lin(x,y,(zc+s-1)%s,s)]
                    ) / 6.0;
                    b[i] = inv*a[i] + d*nb;
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
    }

    // filter + topn
    let mut hits: Vec<(f32, (usize,usize,usize))> = Vec::new();
    for x in 0..s {
        for y in 0..s {
            for zc in 0..s {
                let i = lin(x,y,zc,s);
                let sc = a[i];
                if sc >= threshold {
                    hits.push((sc, (x,y,zc)));
                }
            }
        }
    }
    hits.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(topn);

    let (ox,oy,oz) = chunk_origin;
    hits.into_iter().map(|(score,(lx,ly,lz))| {
        Hit {
            world,
            cell: (ox + lx as i32, oy + ly as i32, oz + lz as i32),
            score,
        }
    }).collect()
}

fn lin(x: usize, y: usize, z: usize, s: usize) -> usize {
    x*s*s + y*s + z
}
fn l2normed(v: &[f32]) -> Vec<f32> {
    let n = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-8);
    v.iter().map(|x| x/n).collect()
}
fn cos_sim_l2normed(cell: &[f32], q_l2: &[f32]) -> f32 {
    let n = (cell.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-8);
    let mut s = 0.0f32;
    for (i, &q) in q_l2.iter().enumerate() {
        s += (cell[i]/n) * q;
    }
    s
}
