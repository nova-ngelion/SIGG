use std::hash::{Hash, Hasher};

pub type WorldKey = (u32, u32, u32); // (u,v,w)
pub type ChunkKey = (i32, i32, i32); // (cx,cy,cz)
pub type CellKey  = (i32, i32, i32); // (x,y,z)
pub type SpaceId = u32;              // 0 = memory, 1.. = compute
pub type SpaceKey = (WorldKey, SpaceId);

#[derive(Clone, Copy, Debug)]
pub struct PocketHandle(pub u32);

#[derive(Clone, Debug)]
pub struct CellDelta {
    pub z: Vec<f32>,      // len = z_dim
    pub version: u32,
    pub parent_hash: u64,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub world: WorldKey,
    pub cell: CellKey,
    pub score: f32,
}

// 小さく安定したハッシュ（依存ゼロ）
pub fn hash64<T: Hash>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

// src/pocket/types.rs

pub fn seed32(world: WorldKey, chunk: ChunkKey, query_hash: u64, salt: u64) -> u32 {
    // 安定 seed：world + chunk + query_hash + salt
    let h = hash64(&(world.0, world.1, world.2, chunk.0, chunk.1, chunk.2, query_hash, salt));
    // u32へ（上位/下位をxorして偏りを減らす）
    let lo = (h & 0xFFFF_FFFF) as u32;
    let hi = (h >> 32) as u32;
    lo ^ hi
}

#[derive(Clone, Copy, Debug)]
pub enum SpaceKind {
    Memory,
    Compute,
}
impl SpaceKind {
    pub fn id(self) -> SpaceId {
        match self { SpaceKind::Memory => 0, SpaceKind::Compute => 1 }
    }
}


