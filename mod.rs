pub mod types;
pub mod store;
pub mod atlas;
pub mod react;
pub mod lru;
pub mod compute;
pub mod cpu;
pub mod asm;


use std::path::Path;

use crate::pocket::lru::LruCache;

use std::collections::HashMap;
use types::{WorldKey, ChunkKey, CellDelta, Hit};
use store::DeltaStore;

pub use atlas::Atlas;
pub use types::{CellKey, PocketHandle};
pub use compute::ComputeSpace;

pub struct PocketWorld {
    pub world: WorldKey,
    pub chunk_size: usize,
    pub z_dim: usize,
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim_bytes: usize,

    pub cache: LruCache<ChunkKey, Vec<u8>>,
    pub store: DeltaStore,
}


impl PocketWorld {

    pub fn cell_read_f32(&mut self, x: i32, y: i32, z: i32, lane: usize) -> f32 {
        let cs = self.chunk_size as i32;
        let cx = div_floor_i32(x, cs);
        let cy = div_floor_i32(y, cs);
        let cz = div_floor_i32(z, cs);
        let lx = (x - cx * cs) as usize;
        let ly = (y - cy * cs) as usize;
        let lz = (z - cz * cs) as usize;
    
        let chunk = (cx, cy, cz);
        let data = self.get_chunk(chunk);
        let idx = (lx * self.chunk_size * self.chunk_size + ly * self.chunk_size + lz) * self.z_dim + lane;
        if idx < data.len() { data[idx] } else { 0.0 }
    }
    
    pub fn cell_write_f32(&mut self, x: i32, y: i32, z: i32, lane: usize, val: f32) {
        let cs = self.chunk_size as i32;
        let cx = div_floor_i32(x, cs);
        let cy = div_floor_i32(y, cs);
        let cz = div_floor_i32(z, cs);
        let lx = (x - cx * cs) as usize;
        let ly = (y - cy * cs) as usize;
        let lz = (z - cz * cs) as usize;
    
        let chunk = (cx, cy, cz);
        let mut data = self.get_chunk(chunk);
        let idx = (lx * self.chunk_size * self.chunk_size + ly * self.chunk_size + lz) * self.z_dim + lane;
        if idx < data.len() {
            data[idx] = val;
            // cacheへ戻す（bytes化して保存）
            self.cache.insert(chunk, Self::f32s_to_bytes(&data));
        }
    }
    
    fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }
    
    fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(b.len() / 4);
        let mut i = 0;
        while i + 4 <= b.len() {
            out.push(f32::from_le_bytes([b[i], b[i+1], b[i+2], b[i+3]]));
            i += 4;
        }
        out
    }
    
    pub fn cpu_load_u32(&mut self, x: i32, y: i32, z: i32, lane: usize) -> u32 {
        // TODO: あなたのPocketWorld内部セル読み出しへ接続
        let f = self.cell_read_f32(x,y,z,lane); f.to_bits()
    }

    pub fn cpu_store_u32(&mut self, x: i32, y: i32, z: i32, lane: usize, v: u32) {
        // TODO: あなたのPocketWorld内部セル書き込みへ接続
        self.cell_write_f32(x,y,z,lane, f32::from_bits(v));
        let _ = (x, y, z, lane, v);
    }

    pub fn open(
        world: WorldKey,
        chunk_size: usize,
        z_dim: usize,
        delta_path: String,
    ) -> Result<Self, std::io::Error> {
        let x_dim = chunk_size;
        let y_dim = chunk_size;
        let z_dim_bytes = z_dim * std::mem::size_of::<f32>();

        let store = DeltaStore::open(delta_path);

        Ok(Self {
            world,
            chunk_size,
            z_dim,
            x_dim,
            y_dim,
            z_dim_bytes,
            cache: LruCache::new(256),
            store,
        })
    }

    pub fn persist(&mut self) -> Result<(), crate::error::SiggError> {
        self.store.persist()
    }

    pub fn cell_to_chunk(&self, x: i32, y: i32, z: i32) -> (ChunkKey, (usize, usize, usize)) {
        let cs = self.chunk_size as i32;
        let cx = div_floor_i32(x, cs);
        let cy = div_floor_i32(y, cs);
        let cz = div_floor_i32(z, cs);

        let lx = (x - cx * cs) as usize;
        let ly = (y - cy * cs) as usize;
        let lz = (z - cz * cs) as usize;

        ((cx, cy, cz), (lx, ly, lz))
    }

    pub fn chunk_origin(&self, chunk: ChunkKey) -> (i32,i32,i32) {
        let s = self.chunk_size as i32;
        (chunk.0*s, chunk.1*s, chunk.2*s)
    }

    pub const COMPUTE_CHUNK_OFFSET: i32 = 1 << 15; // 32768 chunks 離す（十分遠い）

    #[inline]
    pub fn compute_chunk_key(&self, ck: ChunkKey) -> ChunkKey {
        (ck.0 + Self::COMPUTE_CHUNK_OFFSET, ck.1, ck.2)
    }

    #[inline]
    pub fn memory_chunk_key(&self, ck_compute: ChunkKey) -> ChunkKey {
        (ck_compute.0 - Self::COMPUTE_CHUNK_OFFSET, ck_compute.1, ck_compute.2)
    }

    #[inline]
    pub fn compute_cell(&self, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
        // cell 座標を chunk オフセットに合わせて平行移動
        (x + Self::COMPUTE_CHUNK_OFFSET * self.chunk_size as i32, y, z)
    }

    #[inline]
    pub fn memory_cell(&self, x_compute: i32, y: i32, z: i32) -> (i32, i32, i32) {
        (x_compute - Self::COMPUTE_CHUNK_OFFSET * self.chunk_size as i32, y, z)
    }

    pub fn write_cell(
        &mut self,
        x: i32, y: i32, z: i32,
        zvec: Vec<f32>,
        version: u32,
        parent_hash: u64,
        confidence: f32,
    ) -> Result<(), String> {
        if zvec.len() != self.z_dim {
            return Err(format!("zvec len {} != z_dim {}", zvec.len(), self.z_dim));
        }
        let (chunk, (lx, ly, lz)) = self.cell_to_chunk(x, y, z);
        let local: (u16,u16,u16) = (lx as u16, ly as u16, lz as u16);


        self.store.set_cell(
            self.world,
            chunk,
            local,
            CellDelta { z: zvec, version, parent_hash, confidence }
        );

        if self.cache.contains_key(&chunk) {
            // いまのcacheを作り直して入れ替える（更新頻度が高いなら最適化する）
            self.cache.remove(&chunk);
        
            // chunk を再構築して入れ直す
            let rebuilt = self.get_chunk(chunk);
            self.cache.insert(chunk, Self::f32s_to_bytes(&rebuilt));
        }
        

        Ok(())
    }

    pub fn get_chunk(&mut self, chunk: ChunkKey) -> Vec<f32> {
        if let Some(vb) = self.cache.get(&chunk) {
            return Self::bytes_to_f32s(vb);
        }
        let base = self.base_chunk(chunk);
        let mut buf = Self::f32s_to_bytes(&base);
        self.store.apply_to_chunk(self.world, chunk, self.chunk_size, self.z_dim, &mut buf);
        let mut base = Self::bytes_to_f32s(&buf);
    
        if let Some(cells) = self.store.get_chunk_map(self.world, chunk) {
            for ((lx,ly,lz), delta) in cells {
                let idx = self.cell_index(*lx as usize, *ly as usize, *lz as usize);
                let off = idx*self.z_dim;
                base[off..off+self.z_dim].copy_from_slice(&delta.z);
            }
        }
        self.cache.insert(chunk, Self::f32s_to_bytes(&base));
        base
    }
    

    pub fn trigger_extract(
        &mut self,
        chunks: &[ChunkKey],
        query: &[f32],
        steps: u32,
        diffusion: f32,
        threshold: f32,
        topn: usize,
    ) -> Vec<Hit> {
        let mut all: Vec<Hit> = Vec::new();
        for &ck in chunks {
            let z = self.get_chunk(ck);
            let origin = self.chunk_origin(ck);
            let mut hits = react::react_and_extract(
                self.world,
                ck,
                origin,
                self.chunk_size,
                self.z_dim,
                &z,
                query,
                steps,
                diffusion,
                threshold,
                topn,
            );
            all.append(&mut hits);
        }
        all.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(topn);
        all
    }
    

    fn cell_index(&self, lx: usize, ly: usize, lz: usize) -> usize {
        (ly * self.x_dim + lx) * self.z_dim + lz
    }

    fn base_chunk(&self, chunk: ChunkKey) -> Vec<f32> {
        // 依存ゼロの deterministic base（低ランク+小ノイズ）
        let s = self.chunk_size;
        let z_dim = self.z_dim;
        let n_cells = s*s*s;
        let mut out = vec![0.0f32; n_cells*z_dim];

        let seed = types::hash64(&(self.world, chunk, s as u32, z_dim as u32));
        let mut rng = SplitMix64::new(seed);

        let rank = 8usize.min(z_dim);
        let mut a = vec![0.0f32; rank*z_dim];
        for i in 0..(rank*z_dim) {
            a[i] = rng.f32n();
        }

        for x in 0..s {
            let gx = (x as f32)/(s as f32 - 1.0) * 2.0 - 1.0;
            for y in 0..s {
                let gy = (y as f32)/(s as f32 - 1.0) * 2.0 - 1.0;
                for z in 0..s {
                    let gz = (z as f32)/(s as f32 - 1.0) * 2.0 - 1.0;
                    let feats = [
                        gx, gy, gz,
                        gx*gy, gy*gz, gz*gx,
                        (std::f32::consts::PI*gx).sin(),
                        (std::f32::consts::PI*gy).cos()
                    ];
                    let idx = (x*s*s + y*s + z) * z_dim;
                    for k in 0..z_dim {
                        let mut v = 0.0f32;
                        for r in 0..rank {
                            v += feats[r] * a[r*z_dim + k];
                        }
                        v += rng.f32n()*0.05;
                        out[idx + k] = v;
                    }
                }
            }
        }
        out
    }

    pub fn atlas_update_from_hits(
        &mut self,
        atlas: &mut Atlas,
        hits: &[Hit],
        beta: f32,          // 0.1 推奨
        top_per_chunk: usize,
    ) {
        if hits.is_empty() { return; }

        // chunk -> Vec<(score, cell_xyz)>
        let mut per: HashMap<(i32,i32,i32), Vec<(f32,(i32,i32,i32))>> = HashMap::new();

        for h in hits {
            let (ck, _) = self.cell_to_chunk(h.cell.0, h.cell.1, h.cell.2);
            per.entry(ck).or_default().push((h.score, h.cell));
        }

        for (ck, mut list) in per {
            // スコア順に上位だけ使う（平均が安定する）
            list.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            if list.len() > top_per_chunk { list.truncate(top_per_chunk); }

            // chunk の z を取得（LRU経由）
            let z = self.get_chunk(ck);
            let origin = self.chunk_origin(ck);

            let mut avg = vec![0.0f32; self.z_dim];
            let mut n = 0.0f32;

            for (_sc, cell) in list {
                let lx = (cell.0 - origin.0) as usize;
                let ly = (cell.1 - origin.1) as usize;
                let lz = (cell.2 - origin.2) as usize;
                if lx >= self.chunk_size || ly >= self.chunk_size || lz >= self.chunk_size { continue; }

                let idx = lx*self.chunk_size*self.chunk_size + ly*self.chunk_size + lz;
                let off = idx*self.z_dim;
                for i in 0..self.z_dim {
                    avg[i] += z[off+i];
                }
                n += 1.0;
            }

            if n <= 0.0 { continue; }
            for i in 0..self.z_dim { avg[i] /= n; }

            // 既存protoがあれば EMA、なければそのまま
            // atlas は map を直接公開してないので「upsertで上書き」するため、
            // query_topk用のプロトタイプを上書きする形にする。
            //
            // ここでは Atlas 側に get_proto を追加するのが綺麗なので、
            // 次の 2-2 を入れてください。
            if let Some(old) = atlas.get_proto(self.world, ck) {
                let mut p = old.clone();
                let b = beta.clamp(0.0, 1.0);
                for i in 0..self.z_dim {
                    p[i] = (1.0 - b)*p[i] + b*avg[i];
                }
                let _ = atlas.upsert(self.world, ck, p);
            } else {
                let _ = atlas.upsert(self.world, ck, avg);
            }
        }
    }
    pub fn trigger_extract_auto(
        &mut self,
        chunks: &[ChunkKey],
        query: &[f32],
        mut steps: u32,
        mut diffusion: f32,
        mut threshold: f32,
        topn: usize,
    ) -> (Vec<Hit>, u32, f32, f32) {
        let mut best_hits = self.trigger_extract(chunks, query, steps, diffusion, threshold, topn);

        // 1回目で十分なら終わり
        if is_good(&best_hits, threshold, topn) {
            return (best_hits, steps, diffusion, threshold);
        }

        // SIGG制約に基づく簡易調整
        let (hc, mx) = hit_stats(&best_hits);

        // 目標：topn の 30%〜150% 程度ヒットしてて、maxがthresholdを少し上回る
        if hc == 0 || mx < threshold + 0.02 {
            steps = (steps + 2).min(64);
            diffusion = (diffusion + 0.12).min(1.0);
            threshold = (threshold - 0.03).max(-1.0);
        } else if hc < (topn.max(1) as f32 * 0.3) as usize {
            steps = (steps + 1).min(64);
            diffusion = (diffusion + 0.06).min(1.0);
        } else if hc > (topn.max(1) as f32 * 1.5) as usize {
            steps = steps.saturating_sub(1).max(1);
            diffusion = (diffusion - 0.06).max(0.0);
            threshold = (threshold + 0.02).min(1.0);
        }

        let hits2 = self.trigger_extract(chunks, query, steps, diffusion, threshold, topn);

        // 2回目が改善してなければ1回目を返す（保守的）
        if score_sum(&hits2) >= score_sum(&best_hits) {
            best_hits = hits2;
        }
        (best_hits, steps, diffusion, threshold)
    }
    
    pub fn ensure_compute_atlas_proto(&mut self, atlas: &mut Atlas, ck_mem: ChunkKey) {
        let ck = self.compute_chunk_key(ck_mem);
        self.ensure_atlas_proto(atlas, ck);
    }

    /// Compute-space で trigger_extract_auto を実行して hits を返す。
    /// hits の cell 座標は "メモリ座標系" へ戻して返す（上位ロジックが扱いやすい）
    pub fn compute_trigger_extract_auto(
        &mut self,
        chunks_mem: &[ChunkKey],
        query: &[f32],
        steps: u32,
        diffusion: f32,
        threshold: f32,
        topn: usize,
    ) -> (Vec<Hit>, u32, f32, f32) {
        let chunks_compute: Vec<ChunkKey> = chunks_mem
            .iter()
            .copied()
            .map(|ck| self.compute_chunk_key(ck))
            .collect();

        let (mut hits, used_steps, used_diff, used_thr) =
            self.trigger_extract_auto(&chunks_compute, query, steps, diffusion, threshold, topn);

        // cell をメモリ座標系に戻す
        for h in &mut hits {
            let (mx, my, mz) = self.memory_cell(h.cell.0, h.cell.1, h.cell.2);
            h.cell = (mx, my, mz);
        }

        (hits, used_steps, used_diff, used_thr)
    }

    pub fn chunk_proto_from_base(&self, chunk: ChunkKey) -> Vec<f32> {
        // base_chunk を生成して、セル方向に平均プーリングして proto 化
        let z = self.base_chunk(chunk);
        let n_cells = self.chunk_size * self.chunk_size * self.chunk_size;
        let mut proto = vec![0.0f32; self.z_dim];
    
        for c in 0..n_cells {
            let off = c * self.z_dim;
            for k in 0..self.z_dim {
                proto[k] += z[off + k];
            }
        }
    
        let inv = 1.0 / (n_cells as f32);
        for k in 0..self.z_dim {
            proto[k] *= inv;
        }
        proto
    }
    
    pub fn ensure_atlas_proto(&self, atlas: &mut crate::pocket::atlas::Atlas, chunk: ChunkKey) {
        if !atlas.has_proto(self.world, chunk) {
            let proto = self.chunk_proto_from_base(chunk);
            let _ = atlas.upsert(self.world, chunk, proto);
        }
    }


    pub fn read_cell_vec(&mut self, x: i32, y: i32, z: i32) -> Vec<f32> {
        let (chunk, local) = self.cell_to_chunk(x, y, z);
        let buf = self.get_chunk(chunk);
        let (lx, ly, lz) = local;
        let s = self.chunk_size;
        let idx = (lx as usize) * s * s + (ly as usize) * s + (lz as usize);
        let off = idx * self.z_dim;
        buf[off..off + self.z_dim].to_vec()
    }
    
}

fn hit_stats(hits: &[Hit]) -> (usize, f32) {
    let mut mx = -1e9f32;
    for h in hits { if h.score > mx { mx = h.score; } }
    (hits.len(), mx)
}

fn score_sum(hits: &[Hit]) -> f32 {
    hits.iter().map(|h| h.score).sum()
}

fn is_good(hits: &[Hit], threshold: f32, topn: usize) -> bool {
    if hits.is_empty() { return false; }
    let (hc, mx) = hit_stats(hits);
    if mx < threshold + 0.01 { return false; }
    let lo = (topn.max(1) as f32 * 0.2) as usize;
    let hi = (topn.max(1) as f32 * 2.0) as usize;
    hc >= lo && hc <= hi
}

// floor division for negative coords
fn div_floor(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if (r != 0) && ((r > 0) != (b > 0)) { q -= 1; }
    q
}

fn div_floor_i32(a: i32, b: i32) -> i32 { div_floor(a, b) }

/// 依存ゼロ RNG
struct SplitMix64 { x: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { x: seed } }
    fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f32n(&mut self) -> f32 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-12);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        let r = (-2.0*u1.ln()).sqrt();
        let theta = 2.0*std::f64::consts::PI*u2;
        (r*theta.cos()) as f32
    }
}
