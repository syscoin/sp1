use std::{fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use sp1_sdk::{SP1ProofWithPublicValues, SP1Proof};
use sp1_stark::SP1ReduceProof;
use sp1_prover::InnerSC;

fn usage() -> ! {
    eprintln!("usage: extract_reduce_proof <sdk-proof-with-pis.bin> <out-reduce-proof.bin>");
    eprintln!();
    eprintln!("Input must be bincode-serialized `SP1ProofWithPublicValues` whose `proof` is `SP1Proof::Compressed`.");
    eprintln!("Output is bincode-serialized `SP1ReduceProof<InnerSC>` (vk + single shard proof).");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let in_path: PathBuf = std::env::args().nth(1).map(Into::into).unwrap_or_else(|| usage());
    let out_path: PathBuf = std::env::args().nth(2).map(Into::into).unwrap_or_else(|| usage());

    let bytes = fs::read(&in_path).with_context(|| format!("read {}", in_path.display()))?;
    let proof: SP1ProofWithPublicValues =
        bincode::deserialize(&bytes).context("bincode deserialize SP1ProofWithPublicValues")?;

    let SP1Proof::Compressed(reduce) = proof.proof else {
        return Err(anyhow!("expected SP1Proof::Compressed, got {:?}", proof.proof.mode()));
    };

    let reduce: SP1ReduceProof<InnerSC> = (*reduce).clone();
    let out = bincode::serialize(&reduce).context("bincode serialize SP1ReduceProof<InnerSC>")?;
    fs::write(&out_path, out).with_context(|| format!("write {}", out_path.display()))?;

    println!("wrote {}", out_path.display());
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


