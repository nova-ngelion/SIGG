use std::collections::HashMap;

use crate::pocket::types::WorldKey;

/// ComputeSpace: 計算専用の内部空間メモリ
/// - (x,y,z,lane) -> u32(bits) を HashMap で保持
/// - CPU からは「u32ロード/ストア」できれば良い（PocketWorldとは別実装）
#[derive(Debug, Clone)]
pub struct ComputeSpace {
    pub world: WorldKey,
    pub space_id: u32,
    pub size: i32,
    pub lanes: usize,
    pub map: HashMap<(i32, i32, i32, usize), u32>,
}

impl ComputeSpace {
    pub fn new(world: WorldKey, space_id: u32, size: i32, lanes: usize) -> Self {
        Self {
            world,
            space_id,
            size,
            lanes: lanes.max(1),
            map: HashMap::new(),
        }
    }

    /// world/space_id 未確定の空間（作成直後に server 側で埋めてもOK）
    pub fn new_blank(size: i32, lanes: usize) -> Self {
        Self {
            world: (0, 0, 0),
            space_id: 0,
            size,
            lanes: lanes.max(1),
            map: HashMap::new(),
        }
    }

    #[inline]
    fn check_lane(&self, lane: usize) {
        debug_assert!(lane < self.lanes);
    }

    #[inline]
    pub fn read_cell_bits(&self, x: i32, y: i32, z: i32, lane: usize) -> u32 {
        self.check_lane(lane);
        *self.map.get(&(x, y, z, lane)).unwrap_or(&0u32)
    }

    #[inline]
    pub fn write_cell_bits(&mut self, x: i32, y: i32, z: i32, lane: usize, bits: u32) {
        self.check_lane(lane);
        if bits == 0 {
            self.map.remove(&(x, y, z, lane));
        } else {
            self.map.insert((x, y, z, lane), bits);
        }
    }

    pub fn read_cells_bits(&self, coords: &[(i32, i32, i32, usize)]) -> Vec<u32> {
        coords
            .iter()
            .map(|&(x, y, z, lane)| self.read_cell_bits(x, y, z, lane))
            .collect()
    }

    pub fn write_cells_bits(&mut self, cells: &[(i32, i32, i32, usize, u32)]) {
        for &(x, y, z, lane, bits) in cells {
            self.write_cell_bits(x, y, z, lane, bits);
        }
    }

    /// 1D Rule110 (lane 固定, y=0,z=0 を使う) を steps 回回す
    /// - CPU動作確認用の簡易ステップ。将来は任意ルールに拡張。
    pub fn step_rule110_1d(&mut self, steps: u32, lane: usize) {
        self.check_lane(lane);
        if steps == 0 {
            return;
        }
        // 対象の x 範囲を推定（0/1セルのみ）
        let mut min_x = 0i32;
        let mut max_x = 0i32;
        let mut first = true;
        for &(x, y, z, ln) in self.map.keys() {
            if y == 0 && z == 0 && ln == lane {
                if first {
                    min_x = x;
                    max_x = x;
                    first = false;
                } else {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
        }
        if first {
            // 何も無いなら終了
            return;
        }

        // 少し広げる
        min_x -= 2;
        max_x += 2;

        for _ in 0..steps {
            let mut next: HashMap<(i32, i32, i32, usize), u32> = HashMap::new();

            for x in min_x..=max_x {
                let l = (self.read_cell_bits(x - 1, 0, 0, lane) & 1) as u32;
                let c = (self.read_cell_bits(x, 0, 0, lane) & 1) as u32;
                let r = (self.read_cell_bits(x + 1, 0, 0, lane) & 1) as u32;
                let pat = (l << 2) | (c << 1) | r; // 0..7

                // Rule110: 111->0,110->1,101->1,100->0,011->1,010->1,001->1,000->0
                let out = match pat {
                    0b111 => 0,
                    0b110 => 1,
                    0b101 => 1,
                    0b100 => 0,
                    0b011 => 1,
                    0b010 => 1,
                    0b001 => 1,
                    0b000 => 0,
                    _ => 0,
                };

                if out != 0 {
                    next.insert((x, 0, 0, lane), out);
                }
            }

            // lane=他 など既存データは保持したいなら merge するが、
            // いまは 1D 実験用途なので lane の y=0,z=0 だけ置換する。
            // 同じ lane の y=0,z=0 のキーを削除
            self.map.retain(|&(x, y, z, ln), _| !(y == 0 && z == 0 && ln == lane && x >= min_x && x <= max_x));
            // next を反映
            for (k, v) in next {
                self.map.insert(k, v);
            }
        }
    }

    pub fn as_pocket_adapter_mut(&mut self) -> ComputePocketAdapter<'_> {
        ComputePocketAdapter { space: self }
    }
}

pub struct ComputePocketAdapter<'a> {
    pub space: &'a mut ComputeSpace,
}


