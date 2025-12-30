use std::collections::HashMap;

/// Minimal SIGG-CPU assembler (text -> Vec<u32>)
///
/// Supported:
/// - NOP, HALT
/// - MOVI rd, imm
/// - ADD/SUB/MUL/XOR/AND/OR/SHL/SHR rd, ra, rb
/// - LOAD rd, ra, imm
/// - STORE rd, ra, imm
/// - JMP label|imm
/// - BRZ rd, label|imm
/// - BRNZ rd, label|imm
///
/// Labels:
/// - `name:`
///
/// Notes:
/// - imm is 12-bit signed (stored as imm & 0x0FFF)
/// - branch/jump offset is relative to **next** instruction (pc is already incremented in cpu_step_mem)
pub fn assemble(src: &str) -> Result<Vec<u32>, String> {
    // 1) tokenize lines + collect labels (1st pass)
    let mut labels: HashMap<String, i32> = HashMap::new();
    let mut items: Vec<Vec<String>> = Vec::new(); // tokenized instructions only

    for raw in src.lines() {
        let line = raw
            .split("//").next().unwrap_or("")
            .split('#').next().unwrap_or("")
            .split(';').next().unwrap_or("")
            .trim();
        if line.is_empty() { continue; }

        // allow "label: instr ..."
        let mut rest = line;
        loop {
            if let Some(idx) = rest.find(':') {
                let (l, r) = rest.split_at(idx);
                let name = l.trim();
                if name.is_empty() { return Err("empty label".into()); }
                let pc = items.len() as i32;
                labels.insert(name.to_string(), pc);
                rest = r[1..].trim();
                if rest.is_empty() { break; }
                // continue parsing remaining "instr ..."
                // but ensure we don't treat something like "http:" etc; here minimal OK
                continue;
            }
            break;
        }

        if rest.is_empty() { continue; }

        let tokens = split_tokens(rest);
        if !tokens.is_empty() {
            items.push(tokens);
        }
    }

    // 2) encode (2nd pass)
    let mut out: Vec<u32> = Vec::with_capacity(items.len());
    for (pc, toks) in items.iter().enumerate() {
        let pc = pc as i32;
        let m = toks[0].to_uppercase();

        let inst = match m.as_str() {
            "NOP"  => enc(op("NOP"), 0, 0, 0, 0),
            "HALT" => enc(op("HALT"), 0, 0, 0, 0),

            "MOVI" => {
                let rd = parse_reg(req(toks, 1)?)?;
                let imm = parse_imm_or_label(req(toks, 2)?, pc, &labels, /*is_rel=*/false)?;
                enc(op("MOVI"), rd, 0, 0, imm)
            }

            "ADD" | "SUB" | "MUL" | "XOR" | "AND" | "OR" | "SHL" | "SHR" => {
                let rd = parse_reg(req(toks, 1)?)?;
                let ra = parse_reg(req(toks, 2)?)?;
                let rb = parse_reg(req(toks, 3)?)?;
                enc(op(&m), rd, ra, rb, 0)
            }

            "LOAD" => {
                let rd = parse_reg(req(toks, 1)?)?;
                let ra = parse_reg(req(toks, 2)?)?;
                let imm = parse_imm_or_label(req(toks, 3)?, pc, &labels, /*is_rel=*/false)?;
                enc(op("LOAD"), rd, ra, 0, imm)
            }
            "STORE" => {
                let rd = parse_reg(req(toks, 1)?)?;
                let ra = parse_reg(req(toks, 2)?)?;
                let imm = parse_imm_or_label(req(toks, 3)?, pc, &labels, /*is_rel=*/false)?;
                enc(op("STORE"), rd, ra, 0, imm)
            }

            "JMP" => {
                let imm = parse_imm_or_label(req(toks, 1)?, pc, &labels, /*is_rel=*/true)?;
                enc(op("JMP"), 0, 0, 0, imm)
            }
            "BRZ" | "BRNZ" => {
                let rd = parse_reg(req(toks, 1)?)?;
                let imm = parse_imm_or_label(req(toks, 2)?, pc, &labels, /*is_rel=*/true)?;
                enc(op(&m), rd, 0, 0, imm)
            }

            _ => return Err(format!("unknown mnemonic: {}", m)),
        };

        out.push(inst);
    }

    Ok(out)
}

fn split_tokens(s: &str) -> Vec<String> {
    // split by whitespace and commas
    s.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

fn req<'a>(toks: &'a [String], i: usize) -> Result<&'a str, String> {
    toks.get(i).map(|s| s.as_str()).ok_or_else(|| format!("missing operand {}", i))
}

fn parse_reg(s: &str) -> Result<usize, String> {
    let s = s.trim().to_lowercase();
    if !s.starts_with('r') { return Err(format!("bad reg: {}", s)); }
    let n: usize = s[1..].parse().map_err(|_| format!("bad reg: {}", s))?;
    if n > 15 { return Err(format!("reg out of range: {}", s)); }
    Ok(n)
}

fn parse_i32_auto(s: &str) -> Result<i32, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).map_err(|_| format!("bad hex imm: {}", s))
    } else {
        s.parse::<i32>().map_err(|_| format!("bad imm: {}", s))
    }
}

fn parse_imm_or_label(s: &str, pc: i32, labels: &HashMap<String, i32>, is_rel: bool) -> Result<i32, String> {
    if let Ok(v) = parse_i32_auto(s) {
        return Ok(v);
    }
    let target = labels.get(s).ok_or_else(|| format!("unknown label: {}", s))?;
    if is_rel {
        // relative to next instruction
        Ok(*target - (pc + 1))
    } else {
        Ok(*target)
    }
}

fn op(name: &str) -> u8 {
    match name {
        "NOP"   => 0x00,
        "HALT"  => 0x01,
        "MOVI"  => 0x02,
        "ADD"   => 0x03,
        "SUB"   => 0x04,
        "MUL"   => 0x05,
        "XOR"   => 0x06,
        "AND"   => 0x07,
        "OR"    => 0x08,
        "SHL"   => 0x09,
        "SHR"   => 0x10,
        "LOAD"  => 0x11,
        "STORE" => 0x12,
        "JMP"   => 0x13,
        "BRZ"   => 0x14,
        "BRNZ"  => 0x15,
        _ => 0x00,
    }
}

fn enc(op: u8, rd: usize, ra: usize, rb: usize, imm: i32) -> u32 {
    let imm12 = (imm as i32) & 0x0FFF;
    ((op as u32) << 24)
        | ((rd as u32) << 20)
        | ((ra as u32) << 16)
        | ((rb as u32) << 12)
        | (imm12 as u32)
}
