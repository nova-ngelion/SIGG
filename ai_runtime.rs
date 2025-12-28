// src/ai_runtime.rs
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::SiggError;
use crate::pocket::compute::ComputeSpace;
use crate::pocket::cpu::cpu_run_mem;
use crate::state::{PocketState, SiggAgent};

// =====================
// 共有の環境(②用)
// =====================
static AI_TICK: AtomicU32 = AtomicU32::new(0);
pub const ENV_PERIOD: u32 = 30; // 10〜50で好み調整

// =====================
// メモリマップ規約(①)
// =====================
pub const PROG_MAX: i32 = 64;   // 命令領域: x < 64 を予約
pub const IO_BASE: i32 = 100;   // I/O領域はここへ退避

#[derive(Debug, Clone)]
pub struct TickResult {
    pub tick: u32,
    pub env_bit: u32,
    pub in_bits: u32,
    pub out_bits: u32,
    pub expect: u32,
    pub ok: bool,
    pub mu: f32,
    pub mode: u32,
    pub policy_bits: u32,
}

// ---- I/O collision guard ----
pub fn guard_io_xyz(mut p: (i32, i32, i32)) -> (i32, i32, i32) {
    if p.0 >= 0 && p.0 < PROG_MAX {
        p.0 = IO_BASE;
    }
    p
}

pub fn guard_io_pair(
    io_in: (i32, i32, i32),
    io_out: (i32, i32, i32),
) -> ((i32, i32, i32), (i32, i32, i32)) {
    let mut a = guard_io_xyz(io_in);
    let mut b = guard_io_xyz(io_out);

    if a.0 == b.0 {
        b.0 += 1;
    }
    // policy/modeは io_out+1/+2 を使うので、そこも命令領域に落ちないように
    if (b.0 + 2) >= 0 && (b.0 + 2) < PROG_MAX {
        b.0 = IO_BASE + 1;
    }
    (a, b)
}

fn load_program_into_space(space: &mut ComputeSpace, prog: &[u32]) {
    // 命令は (x=0.., y=0, z=0, lane=0) に置く（cpu_step_memの設計通り）
    for (i, &inst) in prog.iter().enumerate() {
        space.write_cell_bits(i as i32, 0, 0, 0, inst);
    }
}

// =====================
// ②：共通 ai_tick_core
// =====================
pub fn ai_tick_core(
    st: &mut PocketState,
    agent_id: u64,
    budget: u32,
) -> Result<TickResult, SiggError> {
    // agent
    let ag: &mut SiggAgent = st
        .agents
        .get_mut(&agent_id)
        .ok_or_else(|| SiggError::runtime("agent not found".to_string()))?;

    // compute space
    let space: &mut ComputeSpace = st
        .compute_spaces
        .get_mut(&ag.space_id)
        .ok_or_else(|| SiggError::runtime("compute space not found".to_string()))?;

    // cpu state
    let cpu = st
        .cpu_states
        .get_mut(&ag.cpu_handle)
        .ok_or_else(|| SiggError::runtime("cpu state not found".to_string()))?;

    // I/O は固定（以前の成功配置）
    let io_in  = (100, 0, 0);
    let io_out = (101, 0, 0);
    let policy_cell = (102, 0, 0);

    // tick/env
    let tick = AI_TICK.fetch_add(1, Ordering::Relaxed);
    let env_bit: u32 = (tick / ENV_PERIOD) % 2;

    // === 入力 ===
    // まずはデバッグ優先で固定（self-feedback をやめる）
    let in_bits: u32 = 0;
    space.write_cell_bits(io_in.0, io_in.1, io_in.2, 0, in_bits);

    // === 重要：policy(=102) を必ず更新 ===
    // env_bit をそのまま policy_bits に入れる（0/1）
    let policy_bits: u32 = env_bit & 1;
    space.write_cell_bits(policy_cell.0, policy_cell.1, policy_cell.2, 0, policy_bits);

    // CPU実行
    cpu.pc = 0;
    cpu.halted = false;
    {
        let mut mem = space.as_pocket_adapter_mut();
        cpu_run_mem(&mut mem, cpu, budget)?;
    }

    // 出力回収
    let out_bits = space.read_cell_bits(io_out.0, io_out.1, io_out.2, 0);

    // === 期待値 ===
    // CPUの計算式に一致：out = in + (policy_bits&1) + 1
    let expect = in_bits.wrapping_add((policy_bits & 1).wrapping_add(1));
    let ok = out_bits == expect;

    // === EWMA 更新 ===
    let mut mu = f32::from_bits(ag.score_mu_bits);
    let score_now: f32 = if ok { 1.0 } else { 0.0 };
    let beta = ag.score_beta.max(0.0).min(1.0);
    mu = mu * (1.0 - beta) + score_now * beta;
    ag.score_mu_bits = mu.to_bits();

    // mode（とりあえず維持でもOK。必要なら以前の閾値制御）
    if !ok {
        ag.mode = 1;
    } else if mu > 0.6 {
        ag.mode = 0;
    }

    Ok(TickResult {
        tick,
        env_bit,
        in_bits,
        out_bits,
        expect,
        ok,
        mu,
        mode: ag.mode,
        policy_bits,
    })
}


// =====================
// 共通 ai_create_core（統合のため）
// ※いまの your builtins/server の create と置き換え可能
// =====================
pub fn ai_create_core(
    st: &mut PocketState,
    world: (u32, u32, u32),
    pocket_handle: u32,
    io_in: (i32, i32, i32),
    io_out: (i32, i32, i32),
    score_beta: f32,
    compute_size: i32,
    lanes: usize,
) -> Result<u64, SiggError> {
    // space
    let space_id = st.compute_next_id;
    st.compute_next_id = st.compute_next_id.wrapping_add(1);

    let mut space = ComputeSpace::new_blank(compute_size.max(64), lanes.max(1));
    space.world = world;
    space.space_id = space_id;
    st.compute_spaces.insert(space_id, space.clone());

    // --- CPUプログラムをアセンブルしてロード（旧ai_create互換：100/101/102固定）
    let src = r#"
    loop:
        LOAD  r1, r0, 100      // r1 = in_bits
        LOAD  r2, r0, 102      // r2 = policy_bits
        MOVI  r3, 1
        AND   r2, r2, r3       // r2 = policy_id (0 or 1)
        MOVI  r4, 1
        ADD   r2, r2, r4       // r2 = policy_id + 1
        ADD   r1, r1, r2       // r1 = in_bits + (policy_id+1)
        STORE r1, r0, 101      // out_bits
        JMP   loop
    "#;

    let prog = crate::pocket::asm::assemble(src)
        .map_err(|e| SiggError::runtime(format!("assemble failed: {e}")))?;

    // これが旧ai_createの「命令を空間へ書く」要
    load_program_into_space(&mut space, &prog);


    // cpu
    let cpu_handle = st.next_handle;
    st.next_handle = st.next_handle.wrapping_add(1);

    let mut cpu = crate::pocket::cpu::CpuState::new(world);
    cpu.base_x = 0;
    cpu.base_y = 0;
    cpu.base_z = 0;
    cpu.lane = 0;
    st.cpu_states.insert(cpu_handle, cpu);

    // agent
    let agent_id = st.next_agent_id;
    st.next_agent_id = st.next_agent_id.wrapping_add(1);

    let (io_in, io_out) = guard_io_pair(io_in, io_out);

    let ag = SiggAgent {
        id: agent_id,
        world,
        pocket_handle,
        space_id,
        cpu_handle,

        sensors_flags: 0,
        sensors_coords: vec![],

        io_in,
        io_out,

        // Pocket側は最小固定でもOK（現状に合わせて）
        pocket_in_addr: (0, 0, 0),
        pocket_out_addr: (1, 0, 0),
        pocket_score_addr: (2, 0, 0),
        pocket_policy_addr: (3, 0, 0),

        score_mu_bits: 0f32.to_bits(),
        score_beta,
        mode: 1,
    };

    st.agents.insert(agent_id, ag);
    Ok(agent_id)
}
