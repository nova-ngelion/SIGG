use crate::error::SiggError;

// ===== Memory abstraction used by CPU =====
pub trait CpuMem {
    fn cpu_load_u32(&mut self, x: i32, y: i32, z: i32, lane: usize) -> u32;
    fn cpu_store_u32(&mut self, x: i32, y: i32, z: i32, lane: usize, v: u32);
}

// PocketWorld 側が bits/u32 読み書きAPIを持っている前提（なければ cell_read_bits / cell_write_bits に合わせてください）
impl CpuMem for crate::pocket::PocketWorld {
    #[inline]
    fn cpu_load_u32(&mut self, x: i32, y: i32, z: i32, lane: usize) -> u32 {
        self.cell_read_f32(x, y, z, lane).to_bits()
    }
    #[inline]
    fn cpu_store_u32(&mut self, x: i32, y: i32, z: i32, lane: usize, v: u32) {
        self.cell_write_f32(x, y, z, lane, f32::from_bits(v))
    }
}

// ComputeSpace 用の “Pocket 互換アダプタ”
impl<'a> CpuMem for crate::pocket::compute::ComputePocketAdapter<'a> {
    #[inline]
    fn cpu_load_u32(&mut self, x: i32, y: i32, z: i32, lane: usize) -> u32 {
        self.space.read_cell_bits(x, y, z, lane)
    }
    #[inline]
    fn cpu_store_u32(&mut self, x: i32, y: i32, z: i32, lane: usize, v: u32) {
        self.space.write_cell_bits(x, y, z, lane, v)
    }
}

#[derive(Clone, Debug)]
pub struct CpuState {
    pub world: (u32, u32, u32),
    pub regs: [u32; 16],
    pub pc: u32,
    pub halted: bool,
    pub steps: u64,
    pub base_x: i32,
    pub base_y: i32,
    pub base_z: i32,
    pub lane: u32,
    pub program: Vec<u32>,
}

impl CpuState {
    pub fn new(world: (u32, u32, u32)) -> Self {
        Self {
            world,
            regs: [0; 16],
            pc: 0,
            halted: false,
            steps: 0,
            base_x: 0,
            base_y: 0,
            base_z: 0,
            lane: 0,
            program: Vec::new(),
        }
    }
}

// ===== ISA =====
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    NOP   = 0x00,
    HALT  = 0x01,
    MOVI  = 0x02, // rd = imm (sign-extend 12)
    ADD   = 0x03, // rd = ra + rb
    SUB   = 0x04, // rd = ra - rb
    MUL   = 0x05, // rd = ra * rb
    XOR   = 0x06,
    AND   = 0x07,
    OR    = 0x08,
    SHL   = 0x09,
    SHR   = 0x10,
    LOAD  = 0x11, // rd = MEM[ra+imm]
    STORE = 0x12, // MEM[ra+imm] = rd
    JMP   = 0x13,
    BRZ   = 0x14, // if rd==0 jump
    BRNZ  = 0x15, // if rd!=0 jump
}

#[inline]
fn sign12(inst: u32) -> i32 {
    let v = (inst & 0x0FFF) as i32;
    if (v & 0x0800) != 0 { v | !0x0FFF } else { v }
}

#[inline]
fn decode(inst: u32) -> (Op, usize, usize, usize, i32) {
    let op = ((inst >> 24) & 0xFF) as u8;
    let rd = ((inst >> 20) & 0x0F) as usize;
    let ra = ((inst >> 16) & 0x0F) as usize;
    let rb = ((inst >> 12) & 0x0F) as usize;
    let imm = sign12(inst);
    let op = match op {
        0x00 => Op::NOP,
        0x01 => Op::HALT,
        0x02 => Op::MOVI,
        0x03 => Op::ADD,
        0x04 => Op::SUB,
        0x05 => Op::MUL,
        0x06 => Op::XOR,
        0x07 => Op::AND,
        0x08 => Op::OR,
        0x09 => Op::SHL,
        0x10 => Op::SHR,
        0x11 => Op::LOAD,
        0x12 => Op::STORE,
        0x13 => Op::JMP,
        0x14 => Op::BRZ,
        0x15 => Op::BRNZ,
        _ => Op::NOP,
    };
    (op, rd, ra, rb, imm)
}

// addr(ワード) -> (x,y,z) : 最小実装として x 方向にだけ展開
#[inline]
fn addr_to_cell(base_x: i32, base_y: i32, base_z: i32, addr: i32) -> (i32, i32, i32) {
    (base_x + addr, base_y, base_z)
}

// ===== main loop =====

pub fn cpu_run_mem(mem: &mut impl CpuMem, cpu: &mut CpuState, budget: u32) -> Result<u32, SiggError> {
    let mut ran: u32 = 0;
    while ran < budget && !cpu.halted {
        cpu_step_mem(mem, cpu)?;
        ran += 1;
    }
    Ok(ran)
}

// PocketWorld を渡す旧APIも残したいなら、cpu_run は cpu_run_mem を呼ぶだけでOK
pub fn cpu_run(pocket: &mut crate::pocket::PocketWorld, cpu: &mut CpuState, budget: u32) -> Result<u32, SiggError> {
    cpu_run_mem(pocket, cpu, budget)
}

pub fn cpu_step_mem(mem: &mut impl CpuMem, cpu: &mut CpuState) -> Result<(), SiggError> {
    if cpu.halted {
        return Ok(());
    }

    // 命令はメモリ(ComputeSpace/PocketWorld)の lane0 に置く設計
    let pc = cpu.pc;
    let x = cpu.base_x.wrapping_add(pc as i32);
    let y = cpu.base_y;
    let z = cpu.base_z;
    let inst = mem.cpu_load_u32(x, y, z, 0);

    // if cfg!(debug_assertions) && CpuState {
    //     eprintln!(
    //         "[cpu_step] pc={} (x,y,z)=({},{},{}) inst=0x{:08x} decoded={:?}",
    //         pc, x, y, z, inst, decode(inst)
    //     );
    // }
    

    cpu.pc = cpu.pc.wrapping_add(1);
    cpu.steps = cpu.steps.wrapping_add(1);

    let (op, rd, ra, rb, imm) = decode(inst);
    let rd = rd as usize;
    let ra = ra as usize;
    let rb = rb as usize;

    match op {
        Op::NOP => {}
        Op::HALT => {cpu.halted = true;}
        // MOVI rd, imm
        Op::MOVI => {
            cpu.regs[rd] = imm as u32;
        }
        Op::ADD => {cpu.regs[rd] = cpu.regs[ra].wrapping_add(cpu.regs[rb]);}
        Op::SUB => {cpu.regs[rd] = cpu.regs[ra].wrapping_sub(cpu.regs[rb]);}
        Op::MUL => {cpu.regs[rd] = cpu.regs[ra].wrapping_mul(cpu.regs[rb]);}
        Op::XOR => {cpu.regs[rd] = cpu.regs[ra] ^ cpu.regs[rb];}
        Op::AND => {cpu.regs[rd] = cpu.regs[ra] & cpu.regs[rb];}
        Op::OR  => {cpu.regs[rd] = cpu.regs[ra] | cpu.regs[rb];}
        Op::SHL => {
            let sh = (cpu.regs[rb] & 31) as u32;
            cpu.regs[rd] = cpu.regs[ra] << sh;
        }
        Op::SHR => {
            let sh = (cpu.regs[rb] & 31) as u32;
            cpu.regs[rd] = cpu.regs[ra] >> sh;
        }
        Op::LOAD => {
            let addr = (cpu.regs[ra] as i32).wrapping_add(imm);
            let (x, y, z) = addr_to_cell(cpu.base_x, cpu.base_y, cpu.base_z, addr);
            cpu.regs[rd] = mem.cpu_load_u32(x, y, z, 0);
        }
        Op::STORE => {
            let addr = (cpu.regs[ra] as i32).wrapping_add(imm);
            let (x, y, z) = addr_to_cell(cpu.base_x, cpu.base_y, cpu.base_z, addr);
            mem.cpu_store_u32(x, y, z, 0, cpu.regs[rd]);
        }
        Op::JMP => {
            cpu.pc = (cpu.pc as i32).wrapping_add(imm) as u32;
        }
        Op::BRZ => {
            if cpu.regs[rd] == 0 {
                cpu.pc = (cpu.pc as i32).wrapping_add(imm) as u32;
            }
        }
        Op::BRNZ => {
            if cpu.regs[rd] != 0 {
                cpu.pc = (cpu.pc as i32).wrapping_add(imm) as u32;
            }
        }
    }

    Ok(())
}



