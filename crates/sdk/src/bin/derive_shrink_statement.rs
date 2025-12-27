use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use sp1_sdk::{SP1ProofWithPublicValues, SP1Proof};
use sp1_primitives::io::sha256_hash;
use sp1_stark::SP1ReduceProof;
use sp1_prover::InnerSC;

fn usage() -> ! {
    eprintln!("usage: derive_shrink_statement <sdk-proof-with-pis.bin> [shape_id]");
    eprintln!();
    eprintln!("Prints canonical Track-A shrink statement bytes used for tag binding:");
    eprintln!("- statement_hex (pass this to `export_shrink_oracle --statement-hex` or `shrink_tag --statement-hex`)");
    eprintln!("- statement_hash (sha256(statement_bytes))");
    eprintln!("- component hashes (vk_hash, public_values_hash)");
    std::process::exit(2);
}

fn push_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    let len: u32 = bytes.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn main() -> Result<()> {
    let proof_with_pis: PathBuf = std::env::args().nth(1).map(Into::into).unwrap_or_else(|| usage());
    let shape_id: String = std::env::args().nth(2).unwrap_or_else(|| "shrink_v1".to_string());

    // Use the SDK proof format loader (handles both SP1ProofWithPublicValues and ProofFromNetwork).
    let proof: SP1ProofWithPublicValues =
        SP1ProofWithPublicValues::load(&proof_with_pis).with_context(|| format!("load {}", proof_with_pis.display()))?;

    let SP1Proof::Compressed(reduce) = proof.proof else {
        return Err(anyhow!(
            "expected SP1Proof::Compressed, got {:?} (tip: generate with `.compressed()`)",
            proof.proof.mode()
        ));
    };
    let reduce: SP1ReduceProof<InnerSC> = (*reduce).clone();

    // Hash the VK and public values (keeps statement bytes small and stable).
    let vk_bytes = bincode::serialize(&reduce.vk).context("bincode serialize reduce.vk")?;
    let vk_hash_vec = sha256_hash(&vk_bytes);
    let pv_hash_vec = sha256_hash(proof.public_values.as_slice());
    anyhow::ensure!(vk_hash_vec.len() == 32, "sha256(vk_bytes) must be 32 bytes");
    anyhow::ensure!(pv_hash_vec.len() == 32, "sha256(public_values) must be 32 bytes");
    let mut vk_hash = [0u8; 32];
    let mut pv_hash = [0u8; 32];
    vk_hash.copy_from_slice(&vk_hash_vec);
    pv_hash.copy_from_slice(&pv_hash_vec);

    // Canonical statement encoding (v0):
    //
    // statement_bytes :=
    //   "sp1.tracka.shrink.statement.v0" ||
    //   len(shape_id) || shape_id ||
    //   len(sp1_version) || sp1_version ||
    //   vk_hash(32) ||
    //   pv_hash(32)
    //
    // NOTE: This is the byte string you should pass as `--statement-hex` to tag computation.
    let mut statement_bytes = Vec::new();
    statement_bytes.extend_from_slice(b"sp1.tracka.shrink.statement.v0");
    push_len_prefixed(&mut statement_bytes, shape_id.as_bytes());
    push_len_prefixed(&mut statement_bytes, proof.sp1_version.as_bytes());
    statement_bytes.extend_from_slice(&vk_hash);
    statement_bytes.extend_from_slice(&pv_hash);

    let statement_hash_vec = sha256_hash(&statement_bytes);
    anyhow::ensure!(statement_hash_vec.len() == 32, "sha256(statement_bytes) must be 32 bytes");
    let mut statement_hash = [0u8; 32];
    statement_hash.copy_from_slice(&statement_hash_vec);

    println!("shape_id: {}", shape_id);
    println!("sp1_version: {}", proof.sp1_version);
    println!("vk_hash: {}", hex::encode(vk_hash));
    println!("public_values_hash: {}", hex::encode(pv_hash));
    println!("statement_hash: {}", hex::encode(statement_hash));
    println!("statement_hex: {}", hex::encode(statement_bytes));

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


