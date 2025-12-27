use std::{fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use sp1_sdk::{SP1ProofWithPublicValues, SP1Proof};

fn main() -> Result<()> {
    let arg1 = std::env::args().nth(1).unwrap_or_default();
    if arg1.is_empty() || arg1 == "-h" || arg1 == "--help" || arg1.starts_with('-') {
        eprintln!("usage: oracle_size <path-to-proof.bin>");
        eprintln!();
        eprintln!("Prints rough lower-bound sizing numbers for oracleizing an SP1 compressed proof.");
        return Ok(());
    }

    let path: PathBuf = arg1.into();

    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;

    // First try the SDK proof format (this is what we actually want).
    if let Ok(proof) = bincode::deserialize::<SP1ProofWithPublicValues>(&bytes) {
        println!("sp1_version: {}", proof.sp1_version);
        println!("proof_mode: {:?}", proof.proof.mode());

        match &proof.proof {
            SP1Proof::Compressed(reduce) => {
                println!("== Compressed (SP1ReduceProof) ==");
                println!("vk.chip_information.len = {}", reduce.vk.chip_information.len());

                let chips = &reduce.proof.opened_values.chips;
                println!("opened_values.chips.len = {}", chips.len());

                // Estimate: for each chip, base-domain size is 2^log_degree; width from vk info.
                // NOTE: this is a *lower bound* for an oracleized proof; full oracleization would also
                // include quotient/composition and (optionally) LDE + proximity encodings / FRI-like layers.
                let mut base_elems_main: u128 = 0;
                for (i, chip_opening) in chips.iter().enumerate() {
                    let log_degree = chip_opening.log_degree;
                    let n = 1u128 << log_degree;
                    let (name, _domain, dim) = &reduce.vk.chip_information[i];
                    let w = dim.width as u128;
                    base_elems_main += n * w;
                    println!(
                        "- chip[{i}] {name}: log_degree={log_degree} rows={n} width={}",
                        dim.width
                    );
                }

                // BabyBear is a 32-bit prime field in SP1; assume 4 bytes/element for rough sizing.
                // For extension fields, actual stored element size differs; this is just a baseline.
                let bytes_per_val: u128 = 4;
                let base_bytes_main = base_elems_main * bytes_per_val;

                println!();
                println!("Lower-bound main-trace table size (base domain only):");
                println!("  elems: {base_elems_main}");
                println!(
                    "  bytes (@4B/elem): {base_bytes_main} (~{:.2} MiB)",
                    (base_bytes_main as f64) / (1024.0 * 1024.0)
                );

                // Show how blowup affects an oracleized LDE export (if we export LDE codewords).
                // SP1's default/compressed FRI configs use log_blowup in {1,2,3}.
                for log_blowup in [1u32, 2, 3] {
                    let lde_bytes = base_bytes_main * (1u128 << log_blowup);
                    println!(
                        "  LDE blowup 2^{log_blowup}: main LDE bytes ~ {} (~{:.2} MiB)",
                        lde_bytes,
                        (lde_bytes as f64) / (1024.0 * 1024.0)
                    );
                }

                println!();
                println!("Rule of thumb for full oracleization (very rough):");
                println!("  total_oracle_bytes ≈ main_LDE_bytes * c");
                println!("  where c accounts for permutation/quotient/composition + proximity/LDT layers.");
                println!("  If exporting full FRI layers as codewords, c can be >> 2.");
                println!("  If using tensor/PCP proximity instead, c may be closer to ~2–6 depending on encoding.");
            }
            other => {
                println!("Unsupported for sizing in this tool: {other}");
                println!("Tip: pass a proof created with `.compressed()` so we can read vk + log_degree.");
            }
        }

        return Ok(());
    }

    // Second: detect a common confusion — `wrapped_proof.bin` in the repo is a wrap-template `ShardProof<OuterSC>`.
    if bincode::deserialize::<sp1_stark::ShardProof<sp1_prover::OuterSC>>(&bytes).is_ok() {
        println!("This file is a bincode'd `ShardProof<OuterSC>` wrap-template (PLONK/Groth16 circuit build),");
        println!("not an SDK `SP1ProofWithPublicValues`.");
        println!();
        println!("For `oracle_size`, pass the proof produced by sp1-sdk (it must contain `SP1Proof::Compressed`).");
        return Ok(());
    }

    Err(anyhow!(
        "unrecognized bincode payload: not an SDK `SP1ProofWithPublicValues` and not a wrap-template `ShardProof<OuterSC>`"
    ))?

    Ok(())
}

trait ProofModeExt {
    fn mode(&self) -> &'static str;
}

impl ProofModeExt for SP1Proof {
    fn mode(&self) -> &'static str {
        match self {
            SP1Proof::Core(_) => "Core",
            SP1Proof::Compressed(_) => "Compressed",
            SP1Proof::Plonk(_) => "Plonk",
            SP1Proof::Groth16(_) => "Groth16",
        }
    }
}


