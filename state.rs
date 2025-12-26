use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::pocket::ComputeSpace;
use crate::pocket::cpu::CpuState;
use crate::pocket::atlas::Atlas;
use crate::pocket::types::{WorldKey};
use crate::pocket::types::SpaceKey;

#[derive(Clone, Copy, Debug)]
pub struct WorldParams {
    pub steps: u32,
    pub diffusion: f32,
    pub threshold: f32,
}

pub struct PocketState {
    pub next_handle: u32,
    pub pockets: HashMap<u32, crate::pocket::PocketWorld>,
    pub atlas: crate::pocket::atlas::Atlas,
    pub data_dir: String,
    pub z_dim_default: usize, // ★追加
    pub world_params: HashMap<WorldKey, WorldParams>,
    pub atlases: HashMap<SpaceKey, Atlas>,
    pub space_params: HashMap<SpaceKey, WorldParams>,
    // ---- ComputeSpace (暗黙巨大空間内の別スペースで計算) ----
    pub compute_next_id: u32,
    pub compute_spaces: HashMap<u32, ComputeSpace>, // space_id -> ComputeSpace
    // ★ compute用: handle -> cpu状態
    pub cpu_states: HashMap<u32, CpuState>,
    // --- 追加：計算用スペース（Pocketとは別）
    pub next_space_id: u32,
    // --- 追加：AI（Agent）管理
    pub next_agent_id: u64,
    pub agents: HashMap<u64, SiggAgent>,

}

impl Default for WorldParams {
    fn default() -> Self {
        Self { steps: 6, diffusion: 0.25, threshold: 0.20 }
    }
}

impl PocketState {
    pub fn new(data_dir: String, z_dim_default: usize) -> Self {
        let atlas = crate::pocket::atlas::Atlas::new(z_dim_default as usize);
        Self {
            next_handle: 1,
            pockets: HashMap::new(),
            atlas,
            data_dir,
            z_dim_default,
            world_params: HashMap::new(),
            atlases: HashMap::new(),
            space_params: HashMap::new(),
            compute_next_id: 1,
            compute_spaces: HashMap::new(),
            cpu_states: HashMap::new(),
            next_space_id: 1,
            next_agent_id: 1,
            agents: HashMap::new(),

        }
    }
    pub fn atlas_for_mut(&mut self, sk: SpaceKey, z_dim: usize) -> &mut Atlas {
        self.atlases.entry(sk).or_insert_with(|| Atlas::new(z_dim))
    }
    pub fn atlas_for(&self, sk: SpaceKey) -> Option<&Atlas> {
        self.atlases.get(&sk)
    }
}
// ============================
// SIGG-AI（最小骨格）
// ============================

#[derive(Clone)]
pub struct SiggAgent {
    pub id: u64,
    pub world: WorldKey,      // “頭の中の世界”の属するworld
    pub pocket_handle: u32,   // 記憶（Pocket）
    pub space_id: u32,        // 計算（ComputeSpace）
    pub cpu_handle: u32,  // CPUハンドル（server管理用
    pub sensors_flags: u32,                 // ← これは 観測能力ビットフラグ bit flag 用
    pub sensors_coords: Vec<(i32, i32, i32)>, // ← これが観測点の本体
    // ★追加：ComputeSpace側の I/O（lane=0固定なら lane は持たない）
    pub io_in:  (i32, i32, i32),
    pub io_out: (i32, i32, i32),
    // ★追加：Pocket側の入力アドレス（最小は固定でOK）
    pub pocket_in_addr: (i32, i32, i32),
    pub pocket_out_addr: (i32, i32, i32),
    pub pocket_score_addr: (i32, i32, i32),
    pub pocket_policy_addr: (i32,i32,i32),
    // ★追加（最小の“自己状態”）
    pub score_mu_bits: u32, // f32 bits
    pub score_beta: f32,
    pub mode: u32, // 0=normal, 1=fallback
    
}
