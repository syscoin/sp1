use std::{borrow::Borrow, collections::BTreeMap};
use std::io::{Seek, SeekFrom, Write};

use anyhow::{anyhow, Context, Result};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use sha2::{Digest, Sha256};
use sp1_stark::air::MachineAir;
use sp1_core_machine::operations::poseidon2::{
    air::{external_linear_layer_mut, internal_linear_layer_mut},
    permutation::{Poseidon2Cols, Poseidon2Degree3Cols, NUM_POSEIDON2_DEGREE3_COLS},
    NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH,
};
use sp1_primitives::RC_16_30_U32;

use crate::{components::CpuProverComponents, InnerSC, SP1Prover};
use sp1_stark::MachineProver;
use sp1_stark::StarkGenericConfig;

/// Abstraction over where tables come from (in-memory traces vs PVOR on disk).
pub trait RowProvider {
    fn dims(&self, table: &str) -> Option<(u32, u32)>;
    fn read_row_u32(&mut self, table: &str, row: u32) -> Result<Vec<u32>>;
}

#[derive(Clone, Debug)]
pub struct ShrinkTagParams<'a> {
    pub statement_hex: &'a str,
    pub armer_id: u32,
    pub tag_seed_hex: &'a str,
    pub shape_id: &'a str,
    pub alpha_hex: Option<&'a str>,
    pub alpha_seed_hex: Option<&'a str>,
    pub print_alpha: bool,
}

#[derive(Clone, Debug)]
pub struct ShrinkTagResult {
    pub residuals_emitted: u64,
    pub tag_mod_p: [u64; 4],
    pub alpha_mod_p: Option<[u64; 4]>,
    pub unlock: Option<bool>,
}

// -----------------------------
// Residual stream file (PVRS v0)
// -----------------------------

const PVRS_MAGIC: [u8; 4] = *b"PVRS";
const PVRS_VERSION_V0: u32 = 0;

#[derive(Clone, Debug)]
pub struct ShrinkResidualStreamParams<'a> {
    /// Statement bytes (hex, without 0x) used to derive statement_hash.
    pub statement_hex: &'a str,
    /// Domain separation / shape id string (bound into header as sha256(shape_id)).
    pub shape_id: &'a str,
    /// Block size (number of residual u32 values per block).
    pub slot_count: u32,
}

#[derive(Clone, Debug)]
pub struct ShrinkResidualStreamResult {
    pub slot_count: u32,
    pub block_count: u64,
    pub residuals_emitted: u64,
    pub statement_hash: [u8; 32],
    pub shape_id_hash: [u8; 32],
}

fn sha256_32(data: &[u8]) -> [u8; 32] {
    let d = Sha256::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

/// Writes a strictly ordered residual stream (same order used by `compute_shrink_tag`) as:
/// - fixed-size header (PVRS v0)
/// - followed by blocks of `slot_count` little-endian u32 residuals (padding last block with zeros).
///
/// Note: This is intended as the SP1 → PVUGC bridge for the PQ streaming evaluator.
pub fn write_shrink_residual_stream<P: RowProvider, W: Write + Seek>(
    p: &mut P,
    out: &mut W,
    params: ShrinkResidualStreamParams<'_>,
) -> Result<ShrinkResidualStreamResult> {
    anyhow::ensure!(params.slot_count > 0, "slot_count must be > 0");

    let statement_bytes = hex::decode(params.statement_hex.trim()).context("decode --statement-hex")?;
    let statement_hash = sha256_32(&statement_bytes);
    let shape_id_hash = sha256_32(params.shape_id.as_bytes());

    // Header (fixed 92 bytes):
    // magic[4], version(u32), slot_count(u32), block_count(u64), residuals_emitted(u64),
    // statement_hash[32], shape_id_hash[32]
    out.write_all(&PVRS_MAGIC)?;
    out.write_all(&PVRS_VERSION_V0.to_le_bytes())?;
    out.write_all(&params.slot_count.to_le_bytes())?;
    out.write_all(&0u64.to_le_bytes())?; // block_count placeholder
    out.write_all(&0u64.to_le_bytes())?; // residuals_emitted placeholder
    out.write_all(&statement_hash)?;
    out.write_all(&shape_id_hash)?;

    struct BlockWriter<'a, W: Write> {
        out: &'a mut W,
        slot_count: usize,
        buf: Vec<u32>,
        idx: usize,
        block_count: u64,
        residuals_emitted: u64,
        io_err: Option<anyhow::Error>,
    }

    impl<'a, W: Write> BlockWriter<'a, W> {
        fn new(out: &'a mut W, slot_count: usize) -> Self {
            Self {
                out,
                slot_count,
                buf: vec![0u32; slot_count],
                idx: 0,
                block_count: 0,
                residuals_emitted: 0,
                io_err: None,
            }
        }

        fn emit(&mut self, r: u32) {
            self.residuals_emitted += 1;
            if self.io_err.is_some() {
                return;
            }
            self.buf[self.idx] = r;
            self.idx += 1;
            if self.idx == self.slot_count {
                if let Err(e) = self.flush_block() {
                    self.io_err = Some(e);
                }
            }
        }

        fn flush_block(&mut self) -> Result<()> {
            for &x in &self.buf {
                self.out.write_all(&x.to_le_bytes())?;
            }
            self.block_count += 1;
            self.idx = 0;
            // Reset buffer to zeros for deterministic padding behavior on final partial block.
            self.buf.fill(0);
            Ok(())
        }

        fn finish(&mut self) {
            if self.io_err.is_some() {
                return;
            }
            if self.idx > 0 {
                if let Err(e) = self.flush_block() {
                    self.io_err = Some(e);
                }
            }
        }
    }

    let mut bw = BlockWriter::new(out, params.slot_count as usize);
    let mut emit = |r: u32| bw.emit(r);
    walk_shrink_residuals(p, &mut emit)?;
    bw.finish();
    if let Some(e) = bw.io_err.take() {
        return Err(e);
    }

    // Patch header counts.
    let block_count = bw.block_count;
    let residuals_emitted = bw.residuals_emitted;
    out.seek(SeekFrom::Start(4 + 4 + 4))?;
    out.write_all(&block_count.to_le_bytes())?;
    out.write_all(&residuals_emitted.to_le_bytes())?;
    out.seek(SeekFrom::End(0))?;

    Ok(ShrinkResidualStreamResult {
        slot_count: params.slot_count,
        block_count,
        residuals_emitted,
        statement_hash,
        shape_id_hash,
    })
}

// -----------------------------
// Tag computation
// -----------------------------

const BABYBEAR_P: u32 = 2_013_265_921;

#[inline]
fn bb_add(a: u32, b: u32) -> u32 {
    let s = a as u64 + b as u64;
    (s % (BABYBEAR_P as u64)) as u32
}

#[inline]
fn bb_sub(a: u32, b: u32) -> u32 {
    let p = BABYBEAR_P as u64;
    let a = a as u64;
    let b = b as u64;
    ((a + p - b) % p) as u32
}

#[inline]
fn bb_mul(a: u32, b: u32) -> u32 {
    let p = BABYBEAR_P as u64;
    ((a as u64 * b as u64) % p) as u32
}

#[inline]
fn bb_one() -> u32 {
    1
}

type BBExt = BinomialExtensionField<BabyBear, 4>;

#[inline]
fn bb_from_u32(x: u32) -> BabyBear {
    BabyBear::from_canonical_u32(x)
}

#[inline]
fn ext_from_block_u32(v: [u32; 4]) -> BBExt {
    let limbs = [bb_from_u32(v[0]), bb_from_u32(v[1]), bb_from_u32(v[2]), bb_from_u32(v[3])];
    BBExt::from_base_slice(&limbs)
}

#[inline]
fn ext_from_base_u32(v: u32) -> BBExt {
    let limbs = [bb_from_u32(v), bb_from_u32(0), bb_from_u32(0), bb_from_u32(0)];
    BBExt::from_base_slice(&limbs)
}

#[inline]
fn ext_eq(a: &BBExt, b: &BBExt) -> bool {
    <BBExt as AbstractExtensionField<BabyBear>>::as_base_slice(a)
        .iter()
        .zip(<BBExt as AbstractExtensionField<BabyBear>>::as_base_slice(b).iter())
        .all(|(x, y)| x.as_canonical_u32() == y.as_canonical_u32())
}

#[inline]
fn baby_eq_u32(a: BabyBear, b_u32: u32) -> bool {
    a.as_canonical_u32() == b_u32
}

const P64_CANDIDATES: [u64; 4] = [
    18446744069414584321,
    18446744073709551557,
    18446744073709551533,
    18446744073709551521,
];

fn mod_add(p: u64, a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % (p as u128)) as u64
}

fn mod_mul(p: u64, a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % (p as u128)) as u64
}

fn mod_mul_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % (m as u128)) as u64
}

fn mod_pow_u64(mut a: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = mod_mul_u64(r, a, m);
        }
        a = mod_mul_u64(a, a, m);
        e >>= 1;
    }
    r
}

fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    const SMALL: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in &SMALL {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d >>= 1;
        s += 1;
    }
    // Deterministic Miller-Rabin bases for 64-bit.
    // See: https://miller-rabin.appspot.com/
    const BASES: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];
    'outer: for &a in &BASES {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mod_mul_u64(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

fn parse_hex_32(name: &str, hex_str: &str) -> Result<[u8; 32]> {
    let b = hex::decode(hex_str.trim()).with_context(|| format!("failed to decode {name}"))?;
    anyhow::ensure!(b.len() == 32, "{name} must be 32 bytes (64 hex chars)");
    let mut out = [0u8; 32];
    out.copy_from_slice(&b);
    Ok(out)
}

fn alpha_from_seed(alpha_seed: &[u8; 32], shape_id: &str, sh: &[u8; 32], armer_id: u32) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"pvugc.alpha.v0");
    h.update(alpha_seed);
    h.update(shape_id.as_bytes());
    h.update(sh);
    h.update(armer_id.to_le_bytes());
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

fn coeff_for(seed: &[u8; 32], sh: &[u8; 32], armer_id: u32, idx: u64, limb: u32, p: u64) -> u64 {
    let mut h = Sha256::new();
    h.update(seed);
    h.update(sh);
    h.update(armer_id.to_le_bytes());
    h.update(idx.to_le_bytes());
    h.update(limb.to_le_bytes());
    let d = h.finalize();
    let mut w = [0u8; 8];
    w.copy_from_slice(&d[..8]);
    u64::from_le_bytes(w) % p
}

fn coeff_vec(seed: &[u8; 32], sh: &[u8; 32], armer_id: u32, idx: u64, primes: &[u64; 4]) -> [u64; 4] {
    [
        coeff_for(seed, sh, armer_id, idx, 0, primes[0]),
        coeff_for(seed, sh, armer_id, idx, 1, primes[1]),
        coeff_for(seed, sh, armer_id, idx, 2, primes[2]),
        coeff_for(seed, sh, armer_id, idx, 3, primes[3]),
    ]
}

#[inline]
fn row_u32(row: &[u32], col: u32) -> u32 {
    row[col as usize]
}

fn walk_shrink_residuals<P: RowProvider>(p: &mut P, emit: &mut dyn FnMut(u32)) -> Result<()> {
    // Debugging: if SP1_SHRINK_DEBUG_FIRST_NONZERO=1, abort on the first nonzero residual
    // and print its global index + section + row.
    let debug_first_nonzero = std::env::var("SP1_SHRINK_DEBUG_FIRST_NONZERO")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let dbg_idx = std::cell::Cell::new(0u64);
    let dbg_section = std::cell::Cell::new("init");
    let dbg_row = std::cell::Cell::new(0u32);

    let emit_inner: &mut dyn FnMut(u32) = emit;
    let mut emit = |v: u32| {
        if debug_first_nonzero {
            let idx = dbg_idx.get();
            if v != 0 {
                panic!(
                    "first nonzero residual: idx={idx} section={} row={} v={v}",
                    dbg_section.get(),
                    dbg_row.get()
                );
            }
            dbg_idx.set(idx + 1);
        }
        emit_inner(v);
    };

    // Select
    {
        dbg_section.set("Select");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) = p.dims("Select").ok_or_else(|| anyhow!("missing table 'Select'"))?;
        anyhow::ensure!(cols == 5, "Select.cols expected 5, got {cols}");
        let (prows, pcols) =
            p.dims("pre/Select").ok_or_else(|| anyhow!("missing table 'pre/Select'"))?;
        anyhow::ensure!(pcols == 8, "pre/Select.cols expected 8, got {pcols}");
        anyhow::ensure!(rows == prows, "Select.rows != pre/Select.rows");
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let r = p.read_row_u32("Select", row)?;
            let bit = row_u32(&r, 0);
            let out1 = row_u32(&r, 1);
            let out2 = row_u32(&r, 2);
            let in1 = row_u32(&r, 3);
            let in2 = row_u32(&r, 4);
            emit(if bb_mul(bit, bb_sub(bit, bb_one())) == 0 { 0 } else { 1 });
            let rhs1 = bb_add(bb_mul(bit, in2), bb_mul(bb_sub(bb_one(), bit), in1));
            emit(if bb_sub(out1, rhs1) == 0 { 0 } else { 1 });
            let rhs2 = bb_add(bb_mul(bit, in1), bb_mul(bb_sub(bb_one(), bit), in2));
            emit(if bb_sub(out2, rhs2) == 0 { 0 } else { 1 });
        }
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let r = p.read_row_u32("pre/Select", row)?;
            let is_real = row_u32(&r, 0);
            emit(if bb_mul(is_real, bb_sub(is_real, bb_one())) == 0 { 0 } else { 1 });
            if is_real == 0 {
                let mut ok = true;
                for c in 1..8u32 {
                    ok &= row_u32(&r, c) == 0;
                }
                emit(if ok { 0 } else { 1 });
            } else {
                emit(0);
            }
        }
    }

    // PublicValues
    {
        dbg_section.set("PublicValues");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) =
            p.dims("PublicValues").ok_or_else(|| anyhow!("missing table 'PublicValues'"))?;
        anyhow::ensure!(cols == 1, "PublicValues.cols expected 1, got {cols}");
        let (prows, pcols) = p
            .dims("pre/PublicValues")
            .ok_or_else(|| anyhow!("missing table 'pre/PublicValues'"))?;
        anyhow::ensure!(pcols == 10, "pre/PublicValues.cols expected 10, got {pcols}");
        anyhow::ensure!(rows == prows, "PublicValues.rows != pre/PublicValues.rows");
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("PublicValues", row)?;
            let pp = p.read_row_u32("pre/PublicValues", row)?;
            let pv_element = row_u32(&m, 0);
            let mut sum = 0u32;
            let mut ok = true;
            for i in 0..8u32 {
                let v = row_u32(&pp, i);
                ok &= bb_mul(v, bb_sub(v, bb_one())) == 0;
                sum = bb_add(sum, v);
            }
            ok &= bb_mul(sum, bb_sub(sum, bb_one())) == 0;
            emit(if ok { 0 } else { 1 });
            let addr = row_u32(&pp, 8);
            let mult = row_u32(&pp, 9);
            let ok2 = if sum == 0 { pv_element == 0 && addr == 0 && mult == 0 } else { true };
            emit(if ok2 { 0 } else { 1 });
        }
    }

    // BaseAlu
    {
        dbg_section.set("BaseAlu");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) = p.dims("BaseAlu").ok_or_else(|| anyhow!("missing table 'BaseAlu'"))?;
        anyhow::ensure!(cols == 12, "BaseAlu.cols expected 12, got {cols}");
        let (prows, pcols) =
            p.dims("pre/BaseAlu").ok_or_else(|| anyhow!("missing table 'pre/BaseAlu'"))?;
        anyhow::ensure!(pcols == 32, "pre/BaseAlu.cols expected 32, got {pcols}");
        anyhow::ensure!(rows == prows, "BaseAlu.rows != pre/BaseAlu.rows");
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("BaseAlu", row)?;
            let pp = p.read_row_u32("pre/BaseAlu", row)?;
            for lane in 0..4u32 {
                let v_off = lane * 3;
                let out = row_u32(&m, v_off + 0);
                let in1 = row_u32(&m, v_off + 1);
                let in2 = row_u32(&m, v_off + 2);
                let p_off = lane * 8;
                let addr_out = row_u32(&pp, p_off + 0);
                let addr_in1 = row_u32(&pp, p_off + 1);
                let addr_in2 = row_u32(&pp, p_off + 2);
                let is_add = row_u32(&pp, p_off + 3);
                let is_sub = row_u32(&pp, p_off + 4);
                let is_mul = row_u32(&pp, p_off + 5);
                let is_div = row_u32(&pp, p_off + 6);
                let mult = row_u32(&pp, p_off + 7);
                let mut ok_flags = true;
                for &b in &[is_add, is_sub, is_mul, is_div] {
                    ok_flags &= bb_mul(b, bb_sub(b, bb_one())) == 0;
                }
                let sum = bb_add(bb_add(is_add, is_sub), bb_add(is_mul, is_div));
                ok_flags &= bb_mul(sum, bb_sub(sum, bb_one())) == 0;
                emit(if ok_flags { 0 } else { 1 });
                let rhs_add = bb_add(in1, in2);
                let rhs_sub = bb_sub(in1, in2);
                let rhs_mul = bb_mul(in1, in2);
                let r_add = bb_mul(is_add, bb_sub(out, rhs_add));
                let r_sub = bb_mul(is_sub, bb_sub(out, rhs_sub));
                let r_mul = bb_mul(is_mul, bb_sub(out, rhs_mul));
                let r_div = bb_mul(is_div, bb_sub(in1, bb_mul(out, in2)));
                emit(if (r_add | r_sub | r_mul | r_div) == 0 { 0 } else { 1 });
                let ok_pad = if sum == 0 {
                    out == 0
                        && in1 == 0
                        && in2 == 0
                        && addr_out == 0
                        && addr_in1 == 0
                        && addr_in2 == 0
                        && mult == 0
                } else {
                    true
                };
                emit(if ok_pad { 0 } else { 1 });
            }
        }
    }

    // ExtAlu
    {
        dbg_section.set("ExtAlu");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) = p.dims("ExtAlu").ok_or_else(|| anyhow!("missing table 'ExtAlu'"))?;
        let (prows, pcols) =
            p.dims("pre/ExtAlu").ok_or_else(|| anyhow!("missing table 'pre/ExtAlu'"))?;
        // Current shrink-v1 layout:
        // - main ExtAlu: per lane stores (out, in1, in2) as 3 extension elems (3*4 base limbs = 12 cols)
        // - pre/ExtAlu: per lane stores 8 base values (addr_out, addr_in1, addr_in2, flags(4), mult)
        anyhow::ensure!(
            pcols % 8 == 0,
            "pre/ExtAlu.cols must be a multiple of 8 (per-lane metadata), got {pcols}"
        );
        let lanes = pcols / 8;
        anyhow::ensure!(
            cols == lanes * 12,
            "ExtAlu.cols expected {}, got {cols}",
            lanes * 12
        );
        anyhow::ensure!(rows == prows, "ExtAlu.rows != pre/ExtAlu.rows");
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("ExtAlu", row)?;
            let pp = p.read_row_u32("pre/ExtAlu", row)?;
            for lane in 0..lanes {
                let v_off = lane * 12;
                let out = ext_from_block_u32([
                    row_u32(&m, v_off + 0),
                    row_u32(&m, v_off + 1),
                    row_u32(&m, v_off + 2),
                    row_u32(&m, v_off + 3),
                ]);
                let in1 = ext_from_block_u32([
                    row_u32(&m, v_off + 4),
                    row_u32(&m, v_off + 5),
                    row_u32(&m, v_off + 6),
                    row_u32(&m, v_off + 7),
                ]);
                let in2 = ext_from_block_u32([
                    row_u32(&m, v_off + 8),
                    row_u32(&m, v_off + 9),
                    row_u32(&m, v_off + 10),
                    row_u32(&m, v_off + 11),
                ]);
                let p_off = lane * 8;
                let _addr_out = row_u32(&pp, p_off + 0);
                let _addr_in1 = row_u32(&pp, p_off + 1);
                let _addr_in2 = row_u32(&pp, p_off + 2);
                let is_add = row_u32(&pp, p_off + 3);
                let is_sub = row_u32(&pp, p_off + 4);
                let is_mul = row_u32(&pp, p_off + 5);
                let is_div = row_u32(&pp, p_off + 6);
                let _mult = row_u32(&pp, p_off + 7);
                let mut ok_flags = true;
                for &b in &[is_add, is_sub, is_mul, is_div] {
                    ok_flags &= bb_mul(b, bb_sub(b, bb_one())) == 0;
                }
                let sum = bb_add(bb_add(is_add, is_sub), bb_add(is_mul, is_div));
                ok_flags &= bb_mul(sum, bb_sub(sum, bb_one())) == 0;
                emit(if ok_flags { 0 } else { 1 });
                // IMPORTANT: ExtAlu values are extension field elements; checking only limb0 is
                // insufficient. We must enforce full extension equality.
                let ok_add = ext_eq(&out, &(in1 + in2));
                let ok_sub = ext_eq(&out, &(in1 - in2));
                let ok_mul = ext_eq(&out, &(in1 * in2));
                let ok_div = ext_eq(&in1, &(out * in2));
                let fail = (is_add != 0 && !ok_add)
                    || (is_sub != 0 && !ok_sub)
                    || (is_mul != 0 && !ok_mul)
                    || (is_div != 0 && !ok_div);
                emit(if fail { 1 } else { 0 });
                // Padding behavior for ExtAlu rows is not guaranteed to be fully zeroed, so we do
                // not emit a padding residual here (keeps valid proofs at residual=0).
                emit(0);
            }
        }
    }

    // BatchFRI
    //
    // Implements the local + transition constraints from `BatchFRIChip` (recursion-core),
    // omitting the memory interaction wiring (consistent with v0 residual stream scope).
    {
        dbg_section.set("BatchFRI");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) = p.dims("BatchFRI").ok_or_else(|| anyhow!("missing table 'BatchFRI'"))?;
        anyhow::ensure!(cols == 13, "BatchFRI.cols expected 13, got {cols}");
        let (prows, pcols) =
            p.dims("pre/BatchFRI").ok_or_else(|| anyhow!("missing table 'pre/BatchFRI'"))?;
        anyhow::ensure!(pcols == 6, "pre/BatchFRI.cols expected 6, got {pcols}");
        anyhow::ensure!(rows == prows, "BatchFRI.rows != pre/BatchFRI.rows");

        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("BatchFRI", row)?;
            let pp = p.read_row_u32("pre/BatchFRI", row)?;
            let is_real = row_u32(&pp, 0);
            let is_end = row_u32(&pp, 1);
            // is_real, is_end are booleans in the AIR.
            emit(if bb_mul(is_real, bb_sub(is_real, bb_one())) == 0 { 0 } else { 1 });
            emit(if bb_mul(is_end, bb_sub(is_end, bb_one())) == 0 { 0 } else { 1 });

            let acc = ext_from_block_u32([
                row_u32(&m, 0),
                row_u32(&m, 1),
                row_u32(&m, 2),
                row_u32(&m, 3),
            ]);
            let alpha_pow = ext_from_block_u32([
                row_u32(&m, 4),
                row_u32(&m, 5),
                row_u32(&m, 6),
                row_u32(&m, 7),
            ]);
            let p_at_z = ext_from_block_u32([
                row_u32(&m, 8),
                row_u32(&m, 9),
                row_u32(&m, 10),
                row_u32(&m, 11),
            ]);
            let p_at_x = row_u32(&m, 12);

            let expected = alpha_pow * (p_at_z - ext_from_base_u32(p_at_x));

            // First row constraint: acc = alpha_pow * (p_at_z - p_at_x)
            // (not explicitly gated by is_real in the AIR; padded rows are zero and satisfy).
            if row == 0 {
                emit(if ext_eq(&acc, &expected) { 0 } else { 1 });
            } else {
                emit(0);
            }

            if row + 1 < rows {
                let next_m = p.read_row_u32("BatchFRI", row + 1)?;
                let next_acc = ext_from_block_u32([
                    row_u32(&next_m, 0),
                    row_u32(&next_m, 1),
                    row_u32(&next_m, 2),
                    row_u32(&next_m, 3),
                ]);
                let next_alpha_pow = ext_from_block_u32([
                    row_u32(&next_m, 4),
                    row_u32(&next_m, 5),
                    row_u32(&next_m, 6),
                    row_u32(&next_m, 7),
                ]);
                let next_p_at_z = ext_from_block_u32([
                    row_u32(&next_m, 8),
                    row_u32(&next_m, 9),
                    row_u32(&next_m, 10),
                    row_u32(&next_m, 11),
                ]);
                let next_p_at_x = row_u32(&next_m, 12);
                let next_expected = next_alpha_pow * (next_p_at_z - ext_from_base_u32(next_p_at_x));

                // Transition:
                // - if is_end: next.acc = next_expected
                // - else: next.acc = acc + next_expected
                let want = if is_end == 1 { next_expected } else { acc + next_expected };
                emit(if ext_eq(&next_acc, &want) { 0 } else { 1 });
            } else {
                emit(0);
            }
        }
    }

    // MemoryVar
    //
    // Note: The current recursion AIR for MemoryVar is interaction-only (send_block) and does not
    // impose local constraints on (addr, value, mult) beyond what is enforced by the grand product.
    // Therefore, we emit no local residuals here (v0 residual stream currently does not model the
    // interaction argument).
    {
        let _ = p
            .dims("MemoryVar")
            .ok_or_else(|| anyhow!("missing table 'MemoryVar'"))?;
        let _ = p
            .dims("pre/MemoryVar")
            .ok_or_else(|| anyhow!("missing table 'pre/MemoryVar'"))?;
    }

    // MemoryConst
    //
    // Same as MemoryVar: interaction-only in the current AIR; emit no local residuals (v0).
    {
        let _ = p
            .dims("pre/MemoryConst")
            .ok_or_else(|| anyhow!("missing table 'pre/MemoryConst'"))?;
        let _ = p
            .dims("MemoryConst")
            .ok_or_else(|| anyhow!("missing table 'MemoryConst'"))?;
    }

    // ExpReverseBitsLen local+transition constraints (no global memory interactions)
    {
        dbg_section.set("ExpReverseBitsLen");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }
        let (rows, cols) = p
            .dims("ExpReverseBitsLen")
            .ok_or_else(|| anyhow!("missing table 'ExpReverseBitsLen'"))?;
        anyhow::ensure!(cols == 7, "ExpReverseBitsLen.cols expected 7, got {cols}");
        let (prows, pcols) = p
            .dims("pre/ExpReverseBitsLen")
            .ok_or_else(|| anyhow!("missing table 'pre/ExpReverseBitsLen'"))?;
        anyhow::ensure!(pcols == 10, "pre/ExpReverseBitsLen.cols expected 10, got {pcols}");
        anyhow::ensure!(rows == prows, "ExpReverseBitsLen.rows != pre/ExpReverseBitsLen.rows");
        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("ExpReverseBitsLen", row)?;
            let pp = p.read_row_u32("pre/ExpReverseBitsLen", row)?;
            let x = row_u32(&m, 0);
            let current_bit = row_u32(&m, 1);
            let prev_accum_sq = row_u32(&m, 2);
            let prev_accum_sq_mul = row_u32(&m, 3);
            let accum = row_u32(&m, 4);
            let accum_sq = row_u32(&m, 5);
            let multiplier = row_u32(&m, 6);
            let is_first = row_u32(&pp, 7);
            let is_last = row_u32(&pp, 8);
            let is_real = row_u32(&pp, 9);
            emit(if bb_mul(bb_mul(is_real, current_bit), bb_sub(multiplier, x)) == 0 { 0 } else { 1 });
            emit(if bb_mul(bb_mul(is_real, bb_sub(bb_one(), current_bit)), bb_sub(multiplier, bb_one())) == 0 { 0 } else { 1 });
            emit(if bb_mul(is_real, bb_sub(prev_accum_sq_mul, bb_mul(prev_accum_sq, multiplier))) == 0 { 0 } else { 1 });
            emit(if bb_mul(is_first, bb_sub(accum, multiplier)) == 0 { 0 } else { 1 });
            emit(if bb_mul(bb_mul(is_real, bb_sub(bb_one(), is_first)), bb_sub(accum, prev_accum_sq_mul)) == 0 { 0 } else { 1 });
            emit(if bb_mul(is_real, bb_sub(accum_sq, bb_mul(accum, accum))) == 0 { 0 } else { 1 });
            if row + 1 < rows {
                let next_pp = p.read_row_u32("pre/ExpReverseBitsLen", row + 1)?;
                let next_m = p.read_row_u32("ExpReverseBitsLen", row + 1)?;
                let next_is_real = row_u32(&next_pp, 9);
                let next_x = row_u32(&next_m, 0);
                let next_prev = row_u32(&next_m, 2);
                emit(if bb_mul(bb_mul(next_is_real, bb_sub(bb_one(), is_last)), bb_sub(x, next_x)) == 0 { 0 } else { 1 });
                emit(if bb_mul(bb_mul(next_is_real, bb_sub(bb_one(), is_last)), bb_sub(next_prev, accum_sq)) == 0 { 0 } else { 1 });
            } else {
                emit(0);
                emit(0);
            }
        }
    }

    // Poseidon2WideDeg3
    //
    // Mirrors `Poseidon2WideChip<3>::eval` (recursion-core) and the underlying poseidon2 AIR.
    // We include:
    // - external + internal round state transition constraints
    // - sbox columns constraints (degree-3 layout has explicit sbox state columns)
    //
    // Note: recursion-core also has memory interactions via `pre/Poseidon2WideDeg3`, but v0 PVRS
    // scope intentionally omits the memory interaction wiring (same as other chips here).
    {
        dbg_section.set("Poseidon2WideDeg3");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }

        let (rows, cols) = p
            .dims("Poseidon2WideDeg3")
            .ok_or_else(|| anyhow!("missing table 'Poseidon2WideDeg3'"))?;
        anyhow::ensure!(
            cols as usize == NUM_POSEIDON2_DEGREE3_COLS,
            "Poseidon2WideDeg3.cols expected {NUM_POSEIDON2_DEGREE3_COLS}, got {cols}"
        );
        let (prows, pcols) = p
            .dims("pre/Poseidon2WideDeg3")
            .ok_or_else(|| anyhow!("missing table 'pre/Poseidon2WideDeg3'"))?;
        anyhow::ensure!(pcols == 49, "pre/Poseidon2WideDeg3.cols expected 49, got {pcols}");
        anyhow::ensure!(rows == prows, "Poseidon2WideDeg3.rows != pre/Poseidon2WideDeg3.rows");

        for row in 0..rows {
            if debug_first_nonzero {
                dbg_row.set(row);
            }
            let m = p.read_row_u32("Poseidon2WideDeg3", row)?;

            let local: &Poseidon2Degree3Cols<u32> = m.as_slice().borrow();

            // External rounds
            for r in 0..NUM_EXTERNAL_ROUNDS {
                // local_state := external_rounds_state[r]
                let mut local_state: [BabyBear; WIDTH] = core::array::from_fn(|i| {
                    bb_from_u32(local.external_rounds_state()[r][i])
                });

                // For the first round, apply the linear layer.
                if r == 0 {
                    external_linear_layer_mut(&mut local_state);
                }

                // Add the round constants.
                let round = if r < NUM_EXTERNAL_ROUNDS / 2 { r } else { r + NUM_INTERNAL_ROUNDS };
                let add_rc: [BabyBear; WIDTH] = core::array::from_fn(|i| {
                    local_state[i] + BabyBear::from_wrapped_u32(RC_16_30_U32[round][i])
                });

                // Check the explicit sbox state columns (degree-3 layout has them).
                if let Some(external_sbox) = local.external_rounds_sbox() {
                    for i in 0..WIDTH {
                        let calc = add_rc[i] * add_rc[i] * add_rc[i];
                        emit(if baby_eq_u32(calc, external_sbox[r][i]) { 0 } else { 1 });
                    }
                }

                // Apply sbox (deg-7) and linear layer to compute the next state.
                let mut state: [BabyBear; WIDTH] = core::array::from_fn(|i| {
                    let s3 = add_rc[i] * add_rc[i] * add_rc[i];
                    // s7 = s3^2 * add_rc
                    (s3 * s3) * add_rc[i]
                });
                external_linear_layer_mut(&mut state);

                let next_state: [u32; WIDTH] = if r == (NUM_EXTERNAL_ROUNDS / 2) - 1 {
                    *local.internal_rounds_state()
                } else if r == NUM_EXTERNAL_ROUNDS - 1 {
                    *local.perm_output()
                } else {
                    local.external_rounds_state()[r + 1]
                };

                for i in 0..WIDTH {
                    emit(if baby_eq_u32(state[i], next_state[i]) { 0 } else { 1 });
                }
            }

            // Internal rounds
            {
                let mut state: [BabyBear; WIDTH] =
                    core::array::from_fn(|i| bb_from_u32(local.internal_rounds_state()[i]));
                let s0 = local.internal_rounds_s0();

                for r in 0..NUM_INTERNAL_ROUNDS {
                    let round = r + NUM_EXTERNAL_ROUNDS / 2;
                    let add_rc = if r == 0 {
                        state[0]
                    } else {
                        bb_from_u32(s0[r - 1])
                    } + BabyBear::from_wrapped_u32(RC_16_30_U32[round][0]);

                    let s3_calc = add_rc * add_rc * add_rc;
                    if let Some(internal_sbox) = local.internal_rounds_sbox() {
                        emit(if baby_eq_u32(s3_calc, internal_sbox[r]) { 0 } else { 1 });
                    }

                    let s7 = (s3_calc * s3_calc) * add_rc;
                    state[0] = s7;
                    internal_linear_layer_mut(&mut state);

                    if r < NUM_INTERNAL_ROUNDS - 1 {
                        emit(if baby_eq_u32(state[0], s0[r]) { 0 } else { 1 });
                    }
                }

                // Connect internal output to the middle external state.
                let external_mid = local.external_rounds_state()[NUM_EXTERNAL_ROUNDS / 2];
                for i in 0..WIDTH {
                    emit(if baby_eq_u32(state[i], external_mid[i]) { 0 } else { 1 });
                }
            }
        }
    }

    // ------------------------------------------------------------
    // LogUp / permutation (interaction) argument checks
    // ------------------------------------------------------------
    //
    // This makes the residual stream "complete" w.r.t. the SP1 shrink wrapper verifier relation:
    // we validate the *permutation traces* (the LogUp witness) using the same per-chip
    // interactions recorded from the AIR, and the same (alpha,beta) challenges used to generate
    // those permutation traces (exported in PVOR as `perm_challenges`).
    //
    // Concretely, for each included chip with local interactions:
    // - validate the exported `perm/<ChipName>` entries by enforcing the *same AIR constraints*
    //   as `sp1_stark::eval_permutation_constraints` (product*entry == numerator), which avoids
    //   any division semantics and is robust to denom==0 corner cases.
    // - verify the cumulative sum (last column) matches the running sum of row entries.
    {
        dbg_section.set("Permutation");
        if debug_first_nonzero {
            eprintln!("debug: section={} start_idx={}", dbg_section.get(), dbg_idx.get());
        }

        // `public_values` encodes the exact slice observed by the challenger before sampling
        // `perm_challenges` in the exporter.
        let (pv_rows, pv_cols) = p
            .dims("public_values")
            .ok_or_else(|| anyhow!("missing table 'public_values' (re-export PVOR with updated exporter)"))?;
        anyhow::ensure!(pv_rows == 1, "public_values.rows expected 1, got {pv_rows}");
        let pv_u32 = p.read_row_u32("public_values", 0)?;
        anyhow::ensure!(
            pv_u32.len() == pv_cols as usize,
            "public_values row len mismatch"
        );
        let public_values: Vec<BabyBear> = pv_u32.into_iter().map(bb_from_u32).collect();

        // Re-derive the permutation challenges (alpha,beta) from the transcript:
        // challenger.observe_slice(public_values[0..num_pv_elts]); challenger.observe(main_commit);
        //
        // This binds the LogUp/permutation/memory checks to the same Fiat–Shamir challenges SP1 uses.
        let prover: SP1Prover<CpuProverComponents> = SP1Prover::uninitialized();
        let machine = prover.shrink_prover.machine();
        let config = machine.config();
        let pcs: &<InnerSC as StarkGenericConfig>::Pcs = config.pcs();

        // Build the main-trace list in the same deterministic order as the exporter: sort by chip name.
        let mut trace_names: Vec<String> = machine
            .chips()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        trace_names.sort();

        let mut domains_and_traces: Vec<(sp1_stark::Dom<InnerSC>, RowMajorMatrix<sp1_stark::Val<InnerSC>>)> =
            Vec::with_capacity(trace_names.len());
        for name in trace_names.iter() {
            let (rows, cols) = p.dims(name).ok_or_else(|| anyhow!("missing table '{name}'"))?;
            let mut vals: Vec<sp1_stark::Val<InnerSC>> = Vec::with_capacity((rows as usize) * (cols as usize));
            for r in 0..rows {
                let row_u32 = p.read_row_u32(name, r)?;
                anyhow::ensure!(row_u32.len() == cols as usize, "row len mismatch for table '{name}'");
                for x in row_u32 {
                    vals.push(bb_from_u32(x));
                }
            }
            let mat = RowMajorMatrix::new(vals, cols as usize);
            let domain = <<InnerSC as StarkGenericConfig>::Pcs as Pcs<
                sp1_stark::Challenge<InnerSC>,
                sp1_stark::Challenger<InnerSC>,
            >>::natural_domain_for_degree(pcs, mat.height());
            domains_and_traces.push((domain, mat));
        }

        let (main_commit, _main_data): (sp1_stark::Com<InnerSC>, sp1_stark::PcsProverData<InnerSC>) =
            <<InnerSC as StarkGenericConfig>::Pcs as Pcs<
                sp1_stark::Challenge<InnerSC>,
                sp1_stark::Challenger<InnerSC>,
            >>::commit(pcs, domains_and_traces);

        let mut challenger = config.challenger();
        challenger.observe_slice(&public_values[0..machine.num_pv_elts()]);
        challenger.observe(main_commit);
        let perm_alpha: sp1_stark::Challenge<InnerSC> = challenger.sample_ext_element();
        let perm_beta: sp1_stark::Challenge<InnerSC> = challenger.sample_ext_element();

        // `perm_challenges` encodes [alpha (4 limbs), beta (4 limbs)] as BabyBear base limbs.
        let chal = p
            .dims("perm_challenges")
            .ok_or_else(|| anyhow!("missing table 'perm_challenges'"))?;
        anyhow::ensure!(chal.0 == 1, "perm_challenges.rows expected 1, got {}", chal.0);
        anyhow::ensure!(chal.1 == 8, "perm_challenges.cols expected 8, got {}", chal.1);
        let chal_row = p.read_row_u32("perm_challenges", 0)?;
        let alpha = ext_from_block_u32([chal_row[0], chal_row[1], chal_row[2], chal_row[3]]);
        let beta = ext_from_block_u32([chal_row[4], chal_row[5], chal_row[6], chal_row[7]]);
        // Enforce transcript binding: exported perm_challenges must match re-derived challenges.
        emit(if ext_eq(&alpha, &perm_alpha) { 0 } else { 1 });
        emit(if ext_eq(&beta, &perm_beta) { 0 } else { 1 });

        // Use the re-derived challenges for the LogUp checks.
        let random_elements = [perm_alpha, perm_beta];

        for chip in machine.chips() {
            let chip_name = chip.name();

            // Only check chips that are present in this PVOR.
            let Some((rows, _cols)) = p.dims(&chip_name) else {
                continue;
            };

            // Skip chips with no local interactions (permutation trace width == 0).
            if chip.permutation_width() == 0 {
                continue;
            }

            // This PVRS checker currently enforces the *local-scope* permutation constraints.
            // If a chip uses global-scope interactions, we must also enforce the 14 "global sum"
            // last-row constraints in `eval_permutation_constraints`.
            //
            // Fail loudly rather than silently under-checking.
            anyhow::ensure!(
                chip.commit_scope() == sp1_stark::air::InteractionScope::Local,
                "chip {chip_name} has commit_scope=Global; PVRS permutation checker must be extended to enforce global-scope constraints"
            );

            let perm_table = format!("perm/{chip_name}");
            let meta_table = format!("perm_meta/{chip_name}");
            let (perm_rows, perm_cols_u32) =
                p.dims(&perm_table).ok_or_else(|| anyhow!("missing table '{perm_table}'"))?;
            anyhow::ensure!(
                perm_rows == rows,
                "{perm_table}.rows ({perm_rows}) != {chip_name}.rows ({rows})"
            );
            anyhow::ensure!(
                perm_cols_u32 % 4 == 0,
                "{perm_table}.cols must be multiple of 4 (flattened extension elems), got {perm_cols_u32}"
            );

            let meta = p
                .dims(&meta_table)
                .ok_or_else(|| anyhow!("missing table '{meta_table}'"))?;
            anyhow::ensure!(meta.0 == 1, "{meta_table}.rows expected 1, got {}", meta.0);
            anyhow::ensure!(meta.1 == 2, "{meta_table}.cols expected 2, got {}", meta.1);
            let meta_row = p.read_row_u32(&meta_table, 0)?;
            let batch_size = meta_row[0] as usize;
            let _num_interactions = meta_row[1] as usize;
            anyhow::ensure!(
                batch_size == chip.logup_batch_size(),
                "{meta_table}.batch_size ({batch_size}) != chip.logup_batch_size() ({})",
                chip.logup_batch_size()
            );

            let perm_width = (perm_cols_u32 as usize) / 4; // in extension field elements
            anyhow::ensure!(perm_width == chip.permutation_width(), "{perm_table}.width ({perm_width}) != chip.permutation_width() ({})", chip.permutation_width());
            anyhow::ensure!(perm_width >= 1, "{perm_table}.width must be >= 1");
            let entry_cols = perm_width - 1;

            // Optional preprocessed row; if the pre table is missing, treat as empty slice.
            let pre_table = format!("pre/{chip_name}");
            let has_pre = p.dims(&pre_table).is_some();
            if let Some((prows, _)) = p.dims(&pre_table) {
                anyhow::ensure!(prows == rows, "{pre_table}.rows ({prows}) != {chip_name}.rows ({rows})");
            }

            // Only local-scope interactions contribute to the local permutation trace.
            let (scoped_sends, scoped_receives) = sp1_stark::scoped_interactions(chip.sends(), chip.receives());
            let local_sends = scoped_sends
                .get(&sp1_stark::air::InteractionScope::Local)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let local_receives = scoped_receives
                .get(&sp1_stark::air::InteractionScope::Local)
                .map(Vec::as_slice)
                .unwrap_or(&[]);

            let mut running_sum = BBExt::zero();

            for row in 0..rows {
                if debug_first_nonzero {
                    dbg_row.set(row);
                }

                let main_u32 = p.read_row_u32(&chip_name, row)?;
                let main_f: Vec<BabyBear> = main_u32.into_iter().map(bb_from_u32).collect();
                let pre_f: Vec<BabyBear> = if has_pre {
                    let pre_u32 = p.read_row_u32(&pre_table, row)?;
                    pre_u32.into_iter().map(bb_from_u32).collect()
                } else {
                    Vec::new()
                };

                let perm_u32 = p.read_row_u32(&perm_table, row)?;
                anyhow::ensure!(
                    perm_u32.len() == perm_width * 4,
                    "{perm_table} row len expected {}, got {}",
                    perm_width * 4,
                    perm_u32.len()
                );

                // Enforce AIR-style per-entry constraints:
                // entry * Π rlc_i == Σ m_i * Π_{j!=i} rlc_j (over a batch chunk).
                //
                // This matches `sp1_stark::eval_permutation_constraints` and avoids any inversion.
                let alpha = random_elements[0];
                let beta = random_elements[1];

                let total = local_sends.len() + local_receives.len();
                anyhow::ensure!(
                    entry_cols == total.div_ceil(batch_size),
                    "{perm_table}.entry_cols ({entry_cols}) != ceil(num_local_interactions ({total}) / batch_size ({batch_size}))"
                );

                // We'll also accumulate the per-row sum of entries for cumulative-sum checking.
                let mut row_sum = BBExt::zero();
                for entry_idx in 0..entry_cols {
                    // Slice the interactions for this batch.
                    let start = entry_idx * batch_size;
                    let end = core::cmp::min(start + batch_size, total);

                    // Compute rlcs and multiplicities (with send/receive sign).
                    let mut rlcs: Vec<BBExt> = Vec::with_capacity(end - start);
                    let mut mults: Vec<BabyBear> = Vec::with_capacity(end - start);

                    // Flatten sends then receives, matching `eval_permutation_constraints`.
                    for flat_i in start..end {
                        let (interaction, is_send) = if flat_i < local_sends.len() {
                            (&local_sends[flat_i], true)
                        } else {
                            (&local_receives[flat_i - local_sends.len()], false)
                        };

                        let mut rlc = alpha;
                        let mut betas = beta.powers();

                        // β^0 * argument_index
                        let beta0 = betas
                            .next()
                            .expect("beta.powers() must yield at least one element");
                        rlc += beta0 * BBExt::from_canonical_usize(interaction.argument_index());

                        // Σ β^j * value_j
                        for (columns, bj) in interaction.values.iter().zip(betas) {
                            let v: BabyBear = columns.apply::<BabyBear, BabyBear>(pre_f.as_slice(), main_f.as_slice());
                            rlc += bj * BBExt::from_base(v);
                        }
                        rlcs.push(rlc);

                        let mut m: BabyBear =
                            interaction.multiplicity.apply::<BabyBear, BabyBear>(pre_f.as_slice(), main_f.as_slice());
                        if !is_send {
                            m = -m;
                        }
                        mults.push(m);
                    }

                    // Compute product and numerator.
                    let mut product = BBExt::one();
                    for rlc in rlcs.iter() {
                        product *= *rlc;
                    }
                    let mut numerator = BBExt::zero();
                    for (i, &m) in mults.iter().enumerate() {
                        // Π_{j!=i} rlc_j
                        let mut all_but_i = BBExt::one();
                        for (j, rlc) in rlcs.iter().enumerate() {
                            if j != i {
                                all_but_i *= *rlc;
                            }
                        }
                        numerator += BBExt::from_base(m) * all_but_i;
                    }

                    // Load the exported entry.
                    let entry = ext_from_block_u32([
                        perm_u32[4 * entry_idx],
                        perm_u32[4 * entry_idx + 1],
                        perm_u32[4 * entry_idx + 2],
                        perm_u32[4 * entry_idx + 3],
                    ]);

                    let ok = ext_eq(&(product * entry), &numerator);
                    emit(if ok { 0 } else { 1 });
                    row_sum += entry;
                }

                // Check running sum column.
                running_sum = running_sum + row_sum;
                let got_cum = ext_from_block_u32([
                    perm_u32[4 * entry_cols],
                    perm_u32[4 * entry_cols + 1],
                    perm_u32[4 * entry_cols + 2],
                    perm_u32[4 * entry_cols + 3],
                ]);
                emit(if ext_eq(&got_cum, &running_sum) { 0 } else { 1 });
            }
        }
    }

    Ok(())
}

pub fn compute_shrink_tag<P: RowProvider>(p: &mut P, params: ShrinkTagParams<'_>) -> Result<ShrinkTagResult> {
    let statement_bytes = hex::decode(params.statement_hex.trim()).context("decode --statement-hex")?;
    let statement_hash = Sha256::digest(&statement_bytes);
    let mut sh = [0u8; 32];
    sh.copy_from_slice(&statement_hash);

    let seed = parse_hex_32("--tag-seed-hex", params.tag_seed_hex)?;

    for (i, &pp) in P64_CANDIDATES.iter().enumerate() {
        anyhow::ensure!(is_prime_u64(pp), "tag prime[{i}] is not prime: {pp}");
    }
    let primes = P64_CANDIDATES;
    let mut tag = [0u64; 4];
    let mut residuals_emitted: u64 = 0;

    let mut emit = |r: u32| {
        let coeff = coeff_vec(&seed, &sh, params.armer_id, residuals_emitted, &primes);
        let rr = r as u64;
        for i in 0..4 {
            let pp = primes[i];
            let term = mod_mul(pp, coeff[i], rr % pp);
            tag[i] = mod_add(pp, tag[i], term);
        }
        residuals_emitted += 1;
    };

    walk_shrink_residuals(p, &mut emit)?;
    // NOTE: The tag is defined over the residual stream emitted by `walk_shrink_residuals`,
    // including the permutation/interaction (LogUp) checks when present in the PVOR.

    let mut alpha_mod_p = None;
    let mut unlock = None;
    if params.alpha_hex.is_some() || params.alpha_seed_hex.is_some() {
        let alpha_bytes = if let Some(a) = params.alpha_hex {
            parse_hex_32("--alpha-hex", a)?
        } else {
            let s = parse_hex_32("--alpha-seed-hex", params.alpha_seed_hex.expect("checked above"))?;
            alpha_from_seed(&s, params.shape_id, &sh, params.armer_id)
        };
        if params.print_alpha {
            println!("tag.alpha_hex: {}", hex::encode(alpha_bytes));
        }
        let mut a_limb = [0u64; 4];
        for i in 0..4 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&alpha_bytes[i * 8..(i + 1) * 8]);
            a_limb[i] = u64::from_le_bytes(w);
        }
        let a_mod = [
            a_limb[0] % primes[0],
            a_limb[1] % primes[1],
            a_limb[2] % primes[2],
            a_limb[3] % primes[3],
        ];
        alpha_mod_p = Some(a_mod);
        unlock = Some(tag == a_mod);
    }

    Ok(ShrinkTagResult {
        residuals_emitted,
        tag_mod_p: tag,
        alpha_mod_p,
        unlock,
    })
}

/// Helper provider for in-memory traces (as produced by `generate_traces` / `setup`).
pub struct InMemoryRowProvider<'a> {
    by_name: BTreeMap<&'a str, &'a p3_matrix::dense::RowMajorMatrix<BabyBear>>,
}

impl<'a> InMemoryRowProvider<'a> {
    pub fn new(
        main: &'a [(&'a str, &'a p3_matrix::dense::RowMajorMatrix<BabyBear>)],
        pre: &'a [(&'a str, &'a p3_matrix::dense::RowMajorMatrix<BabyBear>)],
    ) -> Self {
        let mut by_name = BTreeMap::new();
        for (n, m) in main {
            by_name.insert(*n, *m);
        }
        for (n, m) in pre {
            by_name.insert(*n, *m);
        }
        Self { by_name }
    }
}

impl RowProvider for InMemoryRowProvider<'_> {
    fn dims(&self, table: &str) -> Option<(u32, u32)> {
        self.by_name.get(table).map(|m| (m.height() as u32, m.width() as u32))
    }

    fn read_row_u32(&mut self, table: &str, row: u32) -> Result<Vec<u32>> {
        let m = self
            .by_name
            .get(table)
            .ok_or_else(|| anyhow!("missing table '{table}'"))?;
        anyhow::ensure!(row < m.height() as u32, "row out of range for {table}");
        let w = m.width() as u32;
        let start = (row * w) as usize;
        let end = start + (w as usize);
        Ok(m.values[start..end].iter().map(|f| f.as_canonical_u32()).collect())
    }
}

impl RowProvider for crate::pvor::PvorReader {
    fn dims(&self, table: &str) -> Option<(u32, u32)> {
        self.table(table).map(|t| (t.rows, t.cols))
    }

    fn read_row_u32(&mut self, table: &str, row: u32) -> Result<Vec<u32>> {
        crate::pvor::PvorReader::read_row_u32(self, table, row)
    }
}


