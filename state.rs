use std::collections::HashMap;

use crate::pocket::{Atlas, ComputeSpace, PocketWorld};
use crate::pocket::cpu::CpuState;
use crate::pocket::types::{WorldKey, SpaceKey};

#[derive(Clone, Copy, Debug)]
pub struct WorldParams {
    pub steps: u32,
    pub diffusion: f32,
    pub threshold: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self { steps: 8, diffusion: 0.35, threshold: 0.10 }
    }
}

/// サーバ常駐の “世界＋AI管理” 状態
pub struct PocketState {
    pub next_handle: u32,
    pub pockets: HashMap<u32, PocketWorld>,

    /// デフォルト Atlas（簡易）
    pub atlas: Atlas,

    pub data_dir: String,
    pub z_dim_default: usize,

    pub world_params: HashMap<WorldKey, WorldParams>,
    pub atlases: HashMap<SpaceKey, Atlas>,
    pub space_params: HashMap<SpaceKey, WorldParams>,

    // ---- ComputeSpace (暗黙巨大空間内の別スペースで計算) ----
    pub compute_next_id: u32,
    pub compute_spaces: HashMap<u32, ComputeSpace>, // space_id -> ComputeSpace
    pub cpu_states: HashMap<u32, CpuState>,         // cpu_handle -> CpuState

    // ---- Agent ----
    pub next_space_id: u32,
    pub next_agent_id: u64,
    pub agents: HashMap<u64, SiggAgent>,
}

impl PocketState {
    pub fn new(data_dir: String, z_dim_default: usize) -> Self {
        Self {
            next_handle: 1,
            pockets: HashMap::new(),
            atlas: Atlas::new(z_dim_default),

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

    /// worldごとのパラメータ（無ければデフォルト）
    pub fn world_params(&self, w: WorldKey) -> WorldParams {
        self.world_params.get(&w).copied().unwrap_or_default()
    }

    pub fn set_world_params(&mut self, w: WorldKey, p: WorldParams) {
        self.world_params.insert(w, p);
    }

    /// 便利：agentのscore(EWMA)を読み出す
    pub fn agent_score(&self, agent_id: u64) -> Option<f32> {
        self.agents.get(&agent_id).map(|a| f32::from_bits(a.score_mu_bits))
    }
}

#[derive(Clone)]
pub struct SiggAgent {
    pub id: u64,
    pub world: WorldKey,    // “頭の中の世界”の属するworld
    pub pocket_handle: u32, // 記憶（PocketWorld）
    pub space_id: u32,      // 計算（ComputeSpace）
    pub cpu_handle: u32,    // CPUハンドル（server管理）

    pub sensors_flags: u32,
    pub sensors_coords: Vec<(i32, i32, i32)>,

    // ComputeSpace側 I/O
    pub io_in: (i32, i32, i32),
    pub io_out: (i32, i32, i32),

    // Pocket側 I/O（最小は固定でOK）
    pub pocket_in_addr: (i32, i32, i32),
    pub pocket_out_addr: (i32, i32, i32),
    pub pocket_score_addr: (i32, i32, i32),
    pub pocket_policy_addr: (i32, i32, i32),

    // 自己状態（EWMA）
    pub score_mu_bits: u32, // f32 bits
    pub score_beta: f32,
    pub mode: u32, // 0=normal, 1=fallback
}

impl SiggAgent {
    #[inline]
    pub fn score_mu(&self) -> f32 { f32::from_bits(self.score_mu_bits) }

    #[inline]
    pub fn set_score_mu(&mut self, v: f32) { self.score_mu_bits = v.to_bits(); }
}
