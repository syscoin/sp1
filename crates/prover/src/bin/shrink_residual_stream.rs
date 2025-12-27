use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use sp1_prover::{pvor::PvorReader, shrink_tag};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input PVOR oracle file path (as produced by `export_shrink_oracle`).
    #[arg(long)]
    pvor: PathBuf,

    /// Statement bytes (hex, without 0x) used to derive statement_hash (binds the residual stream).
    #[arg(long)]
    statement_hex: String,

    /// Domain separation / shape id (bound as sha256(shape_id) in header).
    #[arg(long, default_value = "shrink_v1")]
    shape_id: String,

    /// Slot/block size B: number of residual u32 values per block.
    #[arg(long)]
    slot_count: u32,

    /// Output file path for PVRS stream.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut r = PvorReader::open(&args.pvor)?;
    let mut f = File::create(&args.out)?;

    let res = shrink_tag::write_shrink_residual_stream(
        &mut r,
        &mut f,
        shrink_tag::ShrinkResidualStreamParams {
            statement_hex: &args.statement_hex,
            shape_id: &args.shape_id,
            slot_count: args.slot_count,
        },
    )?;

    println!("pvrs.slot_count: {}", res.slot_count);
    println!("pvrs.block_count: {}", res.block_count);
    println!("pvrs.residuals_emitted: {}", res.residuals_emitted);
    println!("pvrs.statement_hash: {}", hex::encode(res.statement_hash));
    println!("pvrs.shape_id_hash: {}", hex::encode(res.shape_id_hash));
    println!("pvrs.out: {}", args.out.display());
    Ok(())
}

