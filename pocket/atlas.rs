use std::collections::HashMap;

use super::types::{WorldKey, ChunkKey};

pub struct Atlas {
    z_dim: usize,
    // world -> chunk -> prototype embedding
    protos: HashMap<WorldKey, HashMap<ChunkKey, Vec<f32>>>,

    hit_mu: f32,     // EWMA mean
    hit_beta: f32,
}

impl Atlas {
    pub fn new(z_dim: usize) -> Self {
        Self {
            z_dim,
            protos: HashMap::new(),
            hit_mu: 0.25,
            hit_beta: 0.10,
        }
    }

    pub fn z_dim(&self) -> usize { self.z_dim }

    /// ✅ world内のproto数
    pub fn count_protos_in_world(&self, world: WorldKey) -> u32 {
        self.protos
            .get(&world)
            .map(|m| m.len() as u32)
            .unwrap_or(0)
    }

    pub fn observe_hit_mean(&mut self, mean: f32) {
        let mean = mean.clamp(0.0, 1.0);
        let alpha = 0.05;
        self.hit_mu = (1.0 - alpha) * self.hit_mu + alpha * mean;

        let target = 0.28;
        let mu = self.hit_mu.max(1e-4);
        let mut b = 0.10 * (target / mu);
        b = b.clamp(0.02, 0.25);
        self.hit_beta = b;
    }

    pub fn beta(&self) -> f32 { self.hit_beta }
    pub fn hit_mu(&self) -> f32 { self.hit_mu }

    pub fn upsert(&mut self, world: WorldKey, chunk: ChunkKey, proto: Vec<f32>) -> Result<(), String> {
        if proto.len() != self.z_dim {
            return Err(format!("proto len {} != z_dim {}", proto.len(), self.z_dim));
        }
        self.protos.entry(world).or_insert_with(HashMap::new).insert(chunk, proto);
        Ok(())
    }

    pub fn query_topk(&self, world: WorldKey, query: &[f32], topk: usize) -> Vec<ChunkKey> {
        if query.len() != self.z_dim { return vec![]; }
        let Some(map) = self.protos.get(&world) else { return vec![]; };

        let q = l2normed(query);

        let mut scored: Vec<(f32, ChunkKey)> = Vec::with_capacity(map.len());
        for (ck, proto) in map {
            let p = l2normed(proto);
            let sim = dot(&q, &p);
            scored.push((sim, *ck));
        }
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(topk).map(|(_,ck)| ck).collect()
    }

    pub fn get_proto(&self, world: WorldKey, chunk: ChunkKey) -> Option<&Vec<f32>> {
        self.protos.get(&world)?.get(&chunk)
    }

    pub fn has_proto(&self, world: WorldKey, chunk: ChunkKey) -> bool {
        self.protos.get(&world).map(|m| m.contains_key(&chunk)).unwrap_or(false)
    }

    pub fn is_empty(&self) -> bool {
        self.protos.is_empty()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x,y)| x*y).sum()
}

fn l2normed(v: &[f32]) -> Vec<f32> {
    let n = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-8);
    v.iter().map(|x| x/n).collect()
}
