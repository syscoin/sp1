use std::{fs::File, io::Write, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs;
use p3_field::AbstractExtensionField;
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use sp1_core_machine::reduce::SP1ReduceProof;
use sp1_prover::{components::CpuProverComponents, pvor, shrink_tag, InnerSC, SP1Prover};
use sp1_recursion_compiler::config::InnerConfig;
use sp1_recursion_circuit::machine::SP1CompressWitnessValues;
use sp1_recursion_core::Runtime as RecursionRuntime;
use sp1_stark::{air::MachineAir, MachineProver, MachineRecord, StarkGenericConfig};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input: bincode-serialized `SP1ReduceProof<InnerSC>` (i.e. the sp1-sdk Compressed proof payload).
    #[arg(long)]
    compressed_reduce_proof: PathBuf,

    /// Output PVOR oracle file path.
    #[arg(long)]
    out: PathBuf,

    /// If provided, compute and print the full-scan shrink tag after exporting.
    /// Secret seed (hex, 32 bytes) used to derive the coefficient stream for the tag.
    #[arg(long)]
    tag_seed_hex: Option<String>,

    /// Statement bytes (hex, without 0x) used to derive statement_hash (binds the tag).
    #[arg(long)]
    statement_hex: Option<String>,

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

    /// If set, print derived alpha bytes as hex (useful to embed into an external lock artifact).
    #[arg(long, default_value_t = false)]
    print_alpha: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let proof_bytes = std::fs::read(&args.compressed_reduce_proof).with_context(|| {
        format!("read {}", args.compressed_reduce_proof.display())
    })?;
    let reduce_proof: SP1ReduceProof<InnerSC> =
        bincode::deserialize(&proof_bytes).context("bincode deserialize SP1ReduceProof<InnerSC>")?;

    let prover: SP1Prover<CpuProverComponents> = SP1Prover::uninitialized();

    // Build shrink input (same as `SP1Prover::shrink`, but we stop after runtime execution + trace gen).
    let SP1ReduceProof { vk: compressed_vk, proof: compressed_proof } = reduce_proof;
    let input = SP1CompressWitnessValues {
        vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
        is_complete: true,
    };
    let input_with_merkle = prover.make_merkle_proofs(input);

    let program = prover.shrink_program(
        sp1_prover::ShrinkAir::<BabyBear>::shrink_shape(),
        &input_with_merkle,
    );

    let mut runtime = RecursionRuntime::<sp1_stark::Val<InnerSC>, sp1_stark::Challenge<InnerSC>, _>::new(
        program.clone(),
        prover.shrink_prover.config().perm.clone(),
    );

    let mut witness_stream = Vec::new();
    sp1_recursion_circuit::witness::Witnessable::<InnerConfig>::write(
        &input_with_merkle,
        &mut witness_stream,
    );
    runtime.witness_stream = witness_stream.into();
    runtime.run().map_err(|e| anyhow!("runtime error: {e}"))?;

    // Preprocessed traces live in the proving key returned by `setup`.
    let (shrink_pk, _shrink_vk) = prover.shrink_prover.setup(&program);

    let mut named_traces = prover.shrink_prover.generate_traces(&runtime.record);
    named_traces.sort_by(|(a, _), (b, _)| a.cmp(b));

    // Map pk preprocessed traces back to chip names (order is via `chip_ordering` indices).
    let mut pre_named = shrink_pk
        .chip_ordering
        .iter()
        .map(|(name, idx)| (format!("pre/{name}"), *idx))
        .collect::<Vec<_>>();
    pre_named.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut out = File::create(&args.out).with_context(|| format!("create {}", args.out.display()))?;
    let pre_names_storage: Vec<String> = pre_named.iter().map(|(name, _)| name.clone()).collect();
    // We'll add extra PVOR tables for the permutation (interaction) argument:
    // - perm_challenges (alpha,beta) sampled from challenger after observing (public_values, main_commit)
    // - perm_meta/<chip> with (batch_size, num_interactions)
    // - perm/<chip> permutation trace (extension field elems flattened to u32 limbs)
    //
    // This enables PVRS generation to validate the same LogUp/permutation constraints SP1 uses.
    let mut metas = Vec::with_capacity(pre_named.len() + named_traces.len() + 256);

    // Build chip ordering consistent with our `named_traces` order.
    let chip_ordering = named_traces
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.to_owned(), i))
        .collect::<HashMap<_, _>>();

    // Compute main commitment (same as SP1 prover) and derive the local permutation challenges.
    let machine = prover.shrink_prover.machine();
    let config = machine.config();
    let pcs: &<InnerSC as StarkGenericConfig>::Pcs = config.pcs();
    let domains_and_traces: Vec<(sp1_stark::Dom<InnerSC>, p3_matrix::dense::RowMajorMatrix<sp1_stark::Val<InnerSC>>)> =
        named_traces
            .iter()
            .map(|(_, trace)| {
                let domain = < <InnerSC as StarkGenericConfig>::Pcs as Pcs<
                    sp1_stark::Challenge<InnerSC>,
                    sp1_stark::Challenger<InnerSC>,
                > >::natural_domain_for_degree(pcs, trace.height());
                (domain, trace.to_owned())
            })
            .collect();
    let (main_commit, _main_data): (sp1_stark::Com<InnerSC>, sp1_stark::PcsProverData<InnerSC>) =
        < <InnerSC as StarkGenericConfig>::Pcs as Pcs<
            sp1_stark::Challenge<InnerSC>,
            sp1_stark::Challenger<InnerSC>,
        > >::commit(pcs, domains_and_traces);

    let public_values = runtime.record.public_values::<sp1_stark::Val<InnerSC>>();
    let mut challenger = config.challenger();
    let pv_slice = &public_values[0..machine.num_pv_elts()];
    // Export the observed public values so downstream PVRS/tag code can re-derive the same
    // Fiat–Shamir challenges (perm_challenges) and bind the LogUp/memory checks to the transcript.
    metas.push(pvor::TableMeta {
        name: "public_values",
        rows: 1,
        cols: pv_slice.len() as u32,
        values_u32_le: Box::new(pv_slice.iter().map(|f| f.as_canonical_u32())),
    });
    challenger.observe_slice(pv_slice);
    challenger.observe(main_commit);

    let perm_alpha: sp1_stark::Challenge<InnerSC> = challenger.sample_ext_element();
    let perm_beta: sp1_stark::Challenge<InnerSC> = challenger.sample_ext_element();

    // Store alpha,beta as u32 limbs (8 u32 total for degree-4 binomial extension).
    let mut perm_chal_u32 = Vec::with_capacity(
        <sp1_stark::Challenge<InnerSC> as AbstractExtensionField<sp1_stark::Val<InnerSC>>>::D * 2,
    );
    for c in [perm_alpha, perm_beta] {
        for limb in <sp1_stark::Challenge<InnerSC> as AbstractExtensionField<sp1_stark::Val<InnerSC>>>::as_base_slice(&c)
        {
            perm_chal_u32.push(limb.as_canonical_u32());
        }
    }
    metas.push(pvor::TableMeta {
        name: "perm_challenges",
        rows: 1,
        cols: perm_chal_u32.len() as u32,
        values_u32_le: Box::new(perm_chal_u32.into_iter()),
    });

    // Generate and export permutation traces per chip (only for chips with local interactions).
    let chips = machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
    let mut perm_tables: Vec<(String, p3_matrix::dense::RowMajorMatrix<sp1_stark::Challenge<InnerSC>>, u32, u32)> =
        Vec::new();
    for chip in chips.iter() {
        let name = chip.name();
        // Only export if there are local interactions (permutation width > 0).
        if chip.permutation_width() == 0 {
            continue;
        }
        let &main_idx = chip_ordering
            .get(&name)
            .ok_or_else(|| anyhow!("missing main trace for chip {name}"))?;
        let main_trace = &named_traces[main_idx].1;
        let pre_trace = shrink_pk
            .chip_ordering
            .get(&name)
            .map(|&idx| &shrink_pk.traces[idx]);
        let (perm_trace, _local_sum) = chip.generate_permutation_trace::<sp1_stark::Challenge<InnerSC>>(
            pre_trace,
            main_trace,
            &[perm_alpha, perm_beta],
        );
        let batch_size = chip.logup_batch_size() as u32;
        let num_interactions = chip.num_interactions() as u32;
        perm_tables.push((name, perm_trace, batch_size, num_interactions));
    }

    // Preprocessed first (prefixed with `pre/`), then main traces.
    for (nm, (_name, idx)) in pre_names_storage
        .iter()
        .map(|s| s.as_str())
        .zip(pre_named.iter())
    {
        let mat = &shrink_pk.traces[*idx];
        metas.push(pvor::TableMeta {
            name: nm,
            rows: mat.height() as u32,
            cols: mat.width() as u32,
            values_u32_le: Box::new(mat.values.iter().map(|f| f.as_canonical_u32())),
        });
    }
    for (name, mat) in named_traces.iter() {
        metas.push(pvor::TableMeta {
            name: name.as_str(),
            rows: mat.height() as u32,
            cols: mat.width() as u32,
            values_u32_le: Box::new(mat.values.iter().map(|f| f.as_canonical_u32())),
        });
    }

    // Append permutation trace tables and per-chip meta.
    // We flatten each extension element to 4 u32 limbs (BabyBear base field).
    for (chip_name, trace, batch_size, num_interactions) in perm_tables.into_iter() {
        let perm_table_name: &'static str =
            Box::leak(format!("perm/{chip_name}").into_boxed_str());
        let meta_table_name: &'static str =
            Box::leak(format!("perm_meta/{chip_name}").into_boxed_str());

        let d = <sp1_stark::Challenge<InnerSC> as AbstractExtensionField<sp1_stark::Val<InnerSC>>>::D;
        let rows = trace.height() as u32;
        let cols = (trace.width() * d) as u32;
        let values = trace.values.into_iter().flat_map(move |ef| {
            <sp1_stark::Challenge<InnerSC> as AbstractExtensionField<sp1_stark::Val<InnerSC>>>::as_base_slice(&ef)
                .iter()
                .map(|x| x.as_canonical_u32())
                .collect::<Vec<_>>()
        });

        metas.push(pvor::TableMeta {
            name: perm_table_name,
            rows,
            cols,
            values_u32_le: Box::new(values),
        });

        metas.push(pvor::TableMeta {
            name: meta_table_name,
            rows: 1,
            cols: 2,
            values_u32_le: Box::new([batch_size, num_interactions].into_iter()),
        });
    }

    let tables_len = metas.len();
    pvor::write_u32_tables_streaming(&mut out, &mut metas)?;
    out.flush()?;
    println!("wrote shrink oracle to {}", args.out.display());
    println!("tables: {}", tables_len);

    // Optional tag computation (debug convenience).
    if let Some(tag_seed_hex) = args.tag_seed_hex.as_deref() {
        let statement_hex = args
            .statement_hex
            .as_deref()
            .ok_or_else(|| anyhow!("--tag-seed-hex requires --statement-hex"))?;

        // Build an in-memory row provider: main tables use their chip name; pre tables are "pre/<chip>".
        let mut main_refs = Vec::with_capacity(named_traces.len());
        for (name, mat) in named_traces.iter() {
            main_refs.push((name.as_str(), mat));
        }
        let mut pre_name_storage = Vec::with_capacity(shrink_pk.chip_ordering.len());
        let mut pre_idx_storage = Vec::with_capacity(shrink_pk.chip_ordering.len());
        for (name, idx) in shrink_pk.chip_ordering.iter() {
            pre_name_storage.push(format!("pre/{name}"));
            pre_idx_storage.push(*idx);
        }
        let mut pre_refs = Vec::with_capacity(pre_name_storage.len());
        for (i, idx) in pre_idx_storage.iter().enumerate() {
            pre_refs.push((pre_name_storage[i].as_str(), &shrink_pk.traces[*idx]));
        }

        let mut provider = shrink_tag::InMemoryRowProvider::new(&main_refs, &pre_refs);
        let res = shrink_tag::compute_shrink_tag(
            &mut provider,
            shrink_tag::ShrinkTagParams {
                statement_hex,
                armer_id: args.armer_id,
                tag_seed_hex,
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
    }

    Ok(())
}


