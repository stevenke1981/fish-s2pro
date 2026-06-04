use std::env;
use std::process::ExitCode;

use fish_s2_parity::{
    compare_fast_ar_frame_dump_files, compare_generated_codes_dump_files,
    compare_rvq_lookup_dump_files, compare_semantic_token_dump_files, compare_slow_ar_dump_files,
    compare_wav_files, metrics_from_wav_file, ParityError, ParityTolerance, Result,
    RvqLookupTolerance, SlowArTensorTolerance,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("metrics") => {
            let path = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let metrics = metrics_from_wav_file(path)?;
            println!("sample_rate={}", metrics.sample_rate);
            println!("channels={}", metrics.channels);
            println!("bits_per_sample={}", metrics.bits_per_sample);
            println!("duration_seconds={:.6}", metrics.duration_seconds);
            println!("rms={:.8}", metrics.rms);
            println!("peak={:.8}", metrics.peak);
            println!("envelope_frames={}", metrics.envelope_rms.len());
            Ok(())
        }
        Some("compare") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report = compare_wav_files(expected, actual, ParityTolerance::default())?;
            println!("passed={}", report.passed);
            println!(
                "duration_delta_seconds={:.6}",
                report.duration_delta_seconds
            );
            println!("rms_delta={:.8}", report.rms_delta);
            println!("envelope_mae={:.8}", report.envelope_mae);
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message("WAV parity failed".into()))
            }
        }
        Some("compare-semantic-tokens") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report = compare_semantic_token_dump_files(expected, actual)?;
            println!("passed={}", report.passed);
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message("semantic token parity failed".into()))
            }
        }
        Some("compare-fast-ar-frame") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report = compare_fast_ar_frame_dump_files(expected, actual)?;
            println!("passed={}", report.passed);
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message("Fast-AR frame parity failed".into()))
            }
        }
        Some("compare-generated-codes") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report = compare_generated_codes_dump_files(expected, actual)?;
            println!("passed={}", report.passed);
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message(
                    "generated codebook parity failed".into(),
                ))
            }
        }
        Some("compare-rvq-lookup") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report =
                compare_rvq_lookup_dump_files(expected, actual, RvqLookupTolerance::default())?;
            println!("passed={}", report.passed);
            println!("latent_l2_delta={:.8}", report.l2_delta);
            println!("latent_mean_abs_delta={:.8}", report.mean_abs_delta);
            println!("latent_max_abs_delta={:.8}", report.max_abs_delta);
            println!("latent_first8_mae={:.8}", report.first8_mae);
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message("RVQ lookup parity failed".into()))
            }
        }
        Some("compare-slow-ar") => {
            let expected = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let actual = args.next().ok_or_else(|| ParityError::Message(usage()))?;
            let report =
                compare_slow_ar_dump_files(expected, actual, SlowArTensorTolerance::default())?;
            println!("passed={}", report.passed);
            for delta in &report.tensor_deltas {
                println!(
                    "tensor={} l2_delta={:.8} mean_abs_delta={:.8} max_abs_delta={:.8} first8_mae={:.8}",
                    delta.name,
                    delta.l2_delta,
                    delta.mean_abs_delta,
                    delta.max_abs_delta,
                    delta.first8_mae
                );
            }
            for failure in &report.failures {
                println!("failure={failure}");
            }
            if report.passed {
                Ok(())
            } else {
                Err(ParityError::Message("Slow-AR tensor parity failed".into()))
            }
        }
        _ => Err(ParityError::Message(usage())),
    }
}

fn usage() -> String {
    "usage:\n  fish_s2_parity metrics <wav>\n  fish_s2_parity compare <golden.wav> <candidate.wav>\n  fish_s2_parity compare-slow-ar <expected.json> <actual.json>\n  fish_s2_parity compare-semantic-tokens <expected.json> <actual.json>\n  fish_s2_parity compare-fast-ar-frame <expected.json> <actual.json>\n  fish_s2_parity compare-generated-codes <expected.json> <actual.json>\n  fish_s2_parity compare-rvq-lookup <expected.json> <actual.json>"
        .to_string()
}
