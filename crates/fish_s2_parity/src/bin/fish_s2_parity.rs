use std::env;
use std::process::ExitCode;

use fish_s2_parity::{
    compare_wav_files, metrics_from_wav_file, ParityError, ParityTolerance, Result,
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
        _ => Err(ParityError::Message(usage())),
    }
}

fn usage() -> String {
    "usage:\n  fish_s2_parity metrics <wav>\n  fish_s2_parity compare <golden.wav> <candidate.wav>"
        .to_string()
}
