use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use super::types::{CellDelta, ChunkKey, WorldKey};

pub type LocalKey = (u16, u16, u16);

#[derive(Clone)]
struct IndexRec {
    world: WorldKey,
    chunk: ChunkKey,
    local: LocalKey,
    // ここは最低限。必要なら後で拡張
    version: u32,
    parent_hash: u64,
    confidence: f32,
    // z は別で保持（map側に入れる）
}

/// 差分ストア（最小実装）
/// - map: (world,chunk) -> (local -> CellDelta)
/// - index/file/path: “持っているだけ”でもコンパイルが通るように用意
pub struct DeltaStore {
    pub path: String,
    pub file: Option<File>,
    pub index: Vec<IndexRec>,
    pub map: HashMap<(WorldKey, ChunkKey), HashMap<LocalKey, CellDelta>>,
}

impl DeltaStore {
    pub fn open(path: String) -> Self {
        // なくても作る。読めるなら読む（今は空でOK）
        let file = OpenOptions::new().read(true).write(true).create(true).open(&path).ok();

        Self {
            path,
            file,
            index: Vec::new(),
            map: HashMap::new(),
        }
    }
    pub fn persist(&mut self) -> Result<(), crate::error::SiggError> {
        // いまは append 書き込みなので “何もしない” でもOK
        Ok(())
    }

    pub fn new(path: String) -> Self {
        Self::open(path)
    }

    pub fn set_cell(&mut self, world: WorldKey, chunk: ChunkKey, local: LocalKey, d: CellDelta) {
        self.map
            .entry((world, chunk))
            .or_insert_with(HashMap::new)
            .insert(local, d.clone());

        self.index.push(IndexRec {
            world,
            chunk,
            local,
            version: d.version,
            parent_hash: d.parent_hash,
            confidence: d.confidence,
        });

        // 永続化は後で本格化でOK。まずはクラッシュしないように軽く。
        if let Some(f) = self.file.as_mut() {
            let _ = Self::append_bin(f, world, chunk, local, &d);
        }
    }

    /// (world,chunk) に溜まってる差分を取得
    pub fn get_chunk_map(&self, world: WorldKey, chunk: ChunkKey) -> Option<&HashMap<LocalKey, CellDelta>> {
        self.map.get(&(world, chunk))
    }

    /// chunk_buf(u8) に差分を適用（f32をlittle-endian bytesで書く）
    pub fn apply_to_chunk(
        &self,
        world: WorldKey,
        chunk: ChunkKey,
        chunk_size: usize,
        z_dim: usize,
        chunk_buf: &mut [u8],
    ) {
        let Some(cells) = self.get_chunk_map(world, chunk) else { return; };

        for (&(lx, ly, lz), d) in cells.iter() {
            let lx = lx as usize;
            let ly = ly as usize;
            let lz = lz as usize;

            // セルの先頭lane位置（f32単位）
            let cell_index = (lx * chunk_size * chunk_size + ly * chunk_size + lz) * z_dim;

            // z vector 書き込み（足りない分は無視）
            let lanes = d.z.len().min(z_dim);
            for lane in 0..lanes {
                let f = d.z[lane];
                let off = (cell_index + lane) * 4;
                if off + 4 <= chunk_buf.len() {
                    chunk_buf[off..off + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    fn append_bin(
        f: &mut File,
        world: WorldKey,
        chunk: ChunkKey,
        local: LocalKey,
        d: &CellDelta,
    ) -> std::io::Result<()> {
        // 超簡易 append（互換性より “まず動く”）
        // フォーマットは後でちゃんと決めてからでOK
        f.seek(SeekFrom::End(0))?;
        f.write_all(&world.0.to_le_bytes())?;
        f.write_all(&world.1.to_le_bytes())?;
        f.write_all(&world.2.to_le_bytes())?;
        f.write_all(&chunk.0.to_le_bytes())?;
        f.write_all(&chunk.1.to_le_bytes())?;
        f.write_all(&chunk.2.to_le_bytes())?;
        f.write_all(&local.0.to_le_bytes())?;
        f.write_all(&local.1.to_le_bytes())?;
        f.write_all(&local.2.to_le_bytes())?;
        f.write_all(&d.version.to_le_bytes())?;
        f.write_all(&d.parent_hash.to_le_bytes())?;
        f.write_all(&d.confidence.to_le_bytes())?;
        let n = d.z.len() as u32;
        f.write_all(&n.to_le_bytes())?;
        for &v in &d.z {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}
