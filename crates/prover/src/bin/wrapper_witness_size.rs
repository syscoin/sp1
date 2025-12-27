use std::collections::BTreeSet;

use sp1_core_machine::shape::CoreShapeConfig;
use sp1_prover::{
    components::CpuProverComponents, shapes::SP1ProofShape, SP1Prover,
};
use sp1_recursion_core::{machine::RecursionAirEventCount, shape::RecursionShapeConfig};
use sp1_recursion_core::{
    chips::{
        alu_base::{NUM_BASE_ALU_COLS, NUM_BASE_ALU_ENTRIES_PER_ROW},
        alu_ext::{NUM_EXT_ALU_COLS, NUM_EXT_ALU_ENTRIES_PER_ROW},
        batch_fri::NUM_BATCH_FRI_COLS,
        exp_reverse_bits::NUM_EXP_REVERSE_BITS_LEN_COLS,
        mem::{
            constant::{NUM_CONST_MEM_ENTRIES_PER_ROW, NUM_MEM_INIT_COLS as NUM_MEM_CONST_COLS},
            variable::{NUM_MEM_INIT_COLS as NUM_MEM_VAR_COLS, NUM_VAR_MEM_ENTRIES_PER_ROW},
        },
        public_values::NUM_PUBLIC_VALUES_COLS,
        select::SELECT_COLS,
    },
    runtime::D,
};
use sp1_core_machine::operations::poseidon2::permutation::NUM_POSEIDON2_DEGREE3_COLS;

fn usage() -> ! {
    eprintln!("usage: wrapper_witness_size [compress|shrink|deferred|recursion] [reduce_batch_size]");
    eprintln!();
    eprintln!("Prints shape-fixed sizing metrics for Track-A (A2) oracle choice:");
    eprintln!("- dummy input size for the wrapper circuit (bincode bytes)");
    eprintln!("- recursion program instruction count + total_memory");
    eprintln!("- event counts (proxy for trace height) for the recursion AIR chips");
    std::process::exit(2);
}

fn main() {
    let kind = std::env::args().nth(1).unwrap_or_else(|| "compress".to_string());
    let reduce_batch_size: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2".to_string())
        .parse()
        .unwrap_or(2);

    let kind = kind.to_lowercase();
    if kind.starts_with('-') {
        usage();
    }

    let core_shape_config = CoreShapeConfig::default();
    let recursion_shape_config = RecursionShapeConfig::default();

    // Generate the (default) proof shapes used by the prover pipeline, then derive the Merkle
    // height exactly as SP1 does when committing to the allowed VK set.
    let all_shapes: BTreeSet<SP1ProofShape> =
        SP1ProofShape::generate(&core_shape_config, &recursion_shape_config, reduce_batch_size)
            .collect();
    let merkle_height = all_shapes.len().next_power_of_two().ilog2() as usize;

    let selected: SP1ProofShape = match kind.as_str() {
        "compress" => all_shapes
            .iter()
            .filter_map(|s| matches!(s, SP1ProofShape::Compress(_)).then(|| s.clone()))
            .last()
            .unwrap_or_else(|| usage()),
        "shrink" => all_shapes
            .iter()
            .filter_map(|s| matches!(s, SP1ProofShape::Shrink(_)).then(|| s.clone()))
            .last()
            .unwrap_or_else(|| usage()),
        "deferred" => all_shapes
            .iter()
            .filter_map(|s| matches!(s, SP1ProofShape::Deferred(_)).then(|| s.clone()))
            .last()
            .unwrap_or_else(|| usage()),
        "recursion" => all_shapes
            .iter()
            .filter_map(|s| matches!(s, SP1ProofShape::Recursion(_)).then(|| s.clone()))
            .last()
            .unwrap_or_else(|| usage()),
        _ => usage(),
    };

    let program_shape = sp1_prover::shapes::SP1CompressProgramShape::from_proof_shape(
        selected.clone(),
        merkle_height,
    );

    // Build the recursion wrapper program for the selected shape.
    // Note: this is a compile-time fixed-shape program; the oracle (A2) size is driven by its
    // trace/witness footprint, not the application runtime.
    let prover: SP1Prover<CpuProverComponents> = SP1Prover::uninitialized();
    let program = prover.program_from_shape(program_shape.clone(), None);

    // Count instructions and approximate event counts (proxy for trace heights).
    let instr_count = program.inner.iter().count();
    let event_counts: RecursionAirEventCount =
        program.inner.iter().fold(RecursionAirEventCount::default(), |c, i| c + i);

    println!("== Track-A (A2) wrapper sizing ==");
    println!("kind: {kind}");
    println!("reduce_batch_size: {reduce_batch_size}");
    println!("num_shapes(default configs): {}", all_shapes.len());
    println!("vk_merkle_height(default configs): {merkle_height}");
    println!("program.total_memory: {}", program.total_memory);
    println!("program.instruction_count: {instr_count}");

    println!();
    println!("Event counts (proxy for wrapper trace size):");
    println!("  mem_const_events: {}", event_counts.mem_const_events);
    println!("  mem_var_events: {}", event_counts.mem_var_events);
    println!("  base_alu_events: {}", event_counts.base_alu_events);
    println!("  ext_alu_events: {}", event_counts.ext_alu_events);
    println!("  poseidon2_wide_events: {}", event_counts.poseidon2_wide_events);
    println!("  fri_fold_events: {}", event_counts.fri_fold_events);
    println!("  batch_fri_events: {}", event_counts.batch_fri_events);
    println!("  select_events: {}", event_counts.select_events);
    println!("  exp_reverse_bits_len_events: {}", event_counts.exp_reverse_bits_len_events);

    // -------------------------------------------------------------------------
    // Rough sizing estimates for an "oracleized trace" view (A2-style).
    //
    // These are not exact because:
    // - many chips pad to a power-of-two row count, and some shapes may fix row counts;
    // - we only estimate main trace tables (not preprocessed, not quotient/composition, etc.);
    // - we assume 4 bytes/field element (BabyBear).
    //
    // But this gives a good first-order sanity check against ~100MB targets.
    // -------------------------------------------------------------------------
    fn next_pow2(n: usize) -> usize {
        match n {
            0 | 1 => 1,
            _ => n.next_power_of_two(),
        }
    }
    fn div_ceil(a: usize, b: usize) -> usize {
        (a + (b - 1)) / b
    }

    let mem_const_rows_raw = div_ceil(event_counts.mem_const_events, NUM_CONST_MEM_ENTRIES_PER_ROW);
    let mem_var_rows_raw = div_ceil(event_counts.mem_var_events, NUM_VAR_MEM_ENTRIES_PER_ROW);
    let base_alu_rows_raw = div_ceil(event_counts.base_alu_events, NUM_BASE_ALU_ENTRIES_PER_ROW);
    let ext_alu_rows_raw = div_ceil(event_counts.ext_alu_events, NUM_EXT_ALU_ENTRIES_PER_ROW);
    let poseidon_rows_raw = event_counts.poseidon2_wide_events;
    let batch_fri_rows_raw = event_counts.batch_fri_events;
    let select_rows_raw = event_counts.select_events;
    let exp_rows_raw = event_counts.exp_reverse_bits_len_events;
    let public_values_rows_raw = 16; // PUB_VALUES_LOG_HEIGHT = 4

    // Pad to powers-of-two (typical for these AIRs). For shrink/compress, shapes often fix these,
    // but this is a good conservative approximation.
    let mem_const_rows = next_pow2(mem_const_rows_raw);
    let mem_var_rows = next_pow2(mem_var_rows_raw);
    let base_alu_rows = next_pow2(base_alu_rows_raw);
    let ext_alu_rows = next_pow2(ext_alu_rows_raw);
    let poseidon_rows = next_pow2(poseidon_rows_raw);
    let batch_fri_rows = next_pow2(batch_fri_rows_raw);
    let select_rows = next_pow2(select_rows_raw);
    let exp_rows = next_pow2(exp_rows_raw);
    let public_values_rows = public_values_rows_raw;

    // Column counts: these constants are defined as `size_of::<Cols<u8>>()` so they equal
    // the number of base-field columns. Memory columns are in units of Blocks, where Block has D limbs.
    const BYTES_PER_BABYBEAR: usize = 4;
    let poseidon_cols = NUM_POSEIDON2_DEGREE3_COLS; // compress/shrink use DEGREE=3 recursion for Poseidon2Wide

    let mem_const_bytes = mem_const_rows * NUM_MEM_CONST_COLS * BYTES_PER_BABYBEAR;
    let mem_var_bytes = mem_var_rows * NUM_MEM_VAR_COLS * BYTES_PER_BABYBEAR;
    let base_alu_bytes = base_alu_rows * NUM_BASE_ALU_COLS * BYTES_PER_BABYBEAR;
    let ext_alu_bytes = ext_alu_rows * NUM_EXT_ALU_COLS * BYTES_PER_BABYBEAR;
    let poseidon_bytes = poseidon_rows * poseidon_cols * BYTES_PER_BABYBEAR;
    let batch_fri_bytes = batch_fri_rows * NUM_BATCH_FRI_COLS * BYTES_PER_BABYBEAR;
    let select_bytes = select_rows * SELECT_COLS * BYTES_PER_BABYBEAR;
    let exp_bytes = exp_rows * NUM_EXP_REVERSE_BITS_LEN_COLS * BYTES_PER_BABYBEAR;
    let public_values_bytes = public_values_rows * NUM_PUBLIC_VALUES_COLS * BYTES_PER_BABYBEAR;

    let total_main_trace_bytes = mem_const_bytes
        + mem_var_bytes
        + base_alu_bytes
        + ext_alu_bytes
        + poseidon_bytes
        + batch_fri_bytes
        + select_bytes
        + exp_bytes
        + public_values_bytes;

    println!();
    println!("Approx padded row counts (main trace):");
    println!("  D (Block limbs): {D}");
    println!("  MemoryConst rows: {mem_const_rows} (raw {mem_const_rows_raw})");
    println!("  MemoryVar   rows: {mem_var_rows} (raw {mem_var_rows_raw})");
    println!("  BaseAlu     rows: {base_alu_rows} (raw {base_alu_rows_raw})");
    println!("  ExtAlu      rows: {ext_alu_rows} (raw {ext_alu_rows_raw})");
    println!("  Poseidon2Wide rows: {poseidon_rows} (raw {poseidon_rows_raw})");
    println!("  BatchFRI    rows: {batch_fri_rows} (raw {batch_fri_rows_raw})");
    println!("  Select      rows: {select_rows} (raw {select_rows_raw})");
    println!("  ExpReverseBitsLen rows: {exp_rows} (raw {exp_rows_raw})");
    println!("  PublicValues rows: {public_values_rows}");

    println!();
    println!("Approx main-trace size (@4B per BabyBear element):");
    println!(
        "  total_main_trace_bytes ≈ {} (~{:.2} MiB)",
        total_main_trace_bytes,
        (total_main_trace_bytes as f64) / (1024.0 * 1024.0)
    );
    println!("  breakdown (MiB):");
    println!("    MemoryConst: {:.2}", (mem_const_bytes as f64) / (1024.0 * 1024.0));
    println!("    MemoryVar  : {:.2}", (mem_var_bytes as f64) / (1024.0 * 1024.0));
    println!("    BaseAlu    : {:.2}", (base_alu_bytes as f64) / (1024.0 * 1024.0));
    println!("    ExtAlu     : {:.2}", (ext_alu_bytes as f64) / (1024.0 * 1024.0));
    println!("    Poseidon2Wide: {:.2}", (poseidon_bytes as f64) / (1024.0 * 1024.0));
    println!("    BatchFRI   : {:.2}", (batch_fri_bytes as f64) / (1024.0 * 1024.0));
    println!("    Select     : {:.2}", (select_bytes as f64) / (1024.0 * 1024.0));
    println!("    ExpRevBits : {:.2}", (exp_bytes as f64) / (1024.0 * 1024.0));
    println!("    PublicValues: {:.4}", (public_values_bytes as f64) / (1024.0 * 1024.0));

    println!();
    println!("Notes:");
    println!("  - These counts are for the recursion wrapper program itself (fixed-shape).");
    println!("  - Track-A π will need to expose enough oracle data for hidden spot-checks over this trace,");
    println!("    plus memory-consistency tables for the *app* trace (encoded into the wrapper inputs).");
    println!("  - If you later swap Merkle/FRI for tensor/PCP proximity in the wrapper, the event mix changes.");
}


