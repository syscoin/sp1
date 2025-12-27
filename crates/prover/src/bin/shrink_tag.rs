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

    /// Secret seed (hex, 32 bytes) used to derive the coefficient stream for the tag.
    #[arg(long)]
    tag_seed_hex: String,

    /// Statement bytes (hex, without 0x) used to derive statement_hash (binds the tag).
    #[arg(long)]
    statement_hex: String,

    /// Armer id (domain separation for coefficient stream and alpha derivation).
    #[arg(long, default_value_t = 0)]
    armer_id: u32,

    /// Domain separation string used in alpha derivation.
    #[arg(long, default_value = "shrink_v1")]
    shape_id: String,

    /// Optional 256-bit alpha target (hex, 32 bytes). If provided, prints UNLOCK=YES iff tag==alpha.
    #[arg(long)]
    alpha_hex: Option<String>,

    /// Optional secret seed (hex, 32 bytes) to derive alpha per statement+armer. Alternative to --alpha-hex.
    #[arg(long)]
    alpha_seed_hex: Option<String>,

    /// If set, print derived alpha bytes as hex.
    #[arg(long, default_value_t = false)]
    print_alpha: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut r = PvorReader::open(&args.pvor)?;

    let res = shrink_tag::compute_shrink_tag(
        &mut r,
        shrink_tag::ShrinkTagParams {
            statement_hex: &args.statement_hex,
            armer_id: args.armer_id,
            tag_seed_hex: &args.tag_seed_hex,
            shape_id: &args.shape_id,
            alpha_hex: args.alpha_hex.as_deref(),
            alpha_seed_hex: args.alpha_seed_hex.as_deref(),
            print_alpha: args.print_alpha,
        },
    )?;

    println!("tag.residuals_emitted: {}", res.residuals_emitted);
    println!(
        "tag.tag_256 (4x64 mod primes): {:016x} {:016x} {:016x} {:016x}",
        res.tag_mod_p[0], res.tag_mod_p[1], res.tag_mod_p[2], res.tag_mod_p[3]
    );

    if let Some(a) = res.alpha_mod_p {
        println!(
            "tag.alpha_256 (4x64 mod primes): {:016x} {:016x} {:016x} {:016x}",
            a[0], a[1], a[2], a[3]
        );
    }
    if let Some(ok) = res.unlock {
        println!("tag.UNLOCK: {}", if ok { "YES" } else { "NO" });
    }

    Ok(())
}


