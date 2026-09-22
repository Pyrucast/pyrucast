//! The `spill` allocator, driven through the `solve_peak` harness in a child
//! process — the allocator is process-wide and reads its environment once, so
//! each configuration needs a process of its own.
#![cfg(all(unix, feature = "spill"))]

use std::process::{Command, Output};

/// Run `solve_peak <mode> <n>`, spilling every block of 4 KiB or more into
/// `dir` when one is given.
fn run(mode: &str, n: usize, dir: Option<&str>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_solve_peak"));
    cmd.args([mode, &n.to_string()]);
    match dir {
        Some(d) => cmd
            .env("PYRUCAST_SPILL_DIR", d)
            .env("PYRUCAST_SPILL_MIN", "4096"),
        None => cmd.env_remove("PYRUCAST_SPILL_DIR"),
    };
    cmd.output().unwrap()
}

fn stdout(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

/// The `FNV-1a = …` of the solution.
fn fingerprint(text: &str) -> String {
    text.lines()
        .find_map(|l| l.split_once("FNV-1a = ").map(|(_, h)| h.to_string()))
        .unwrap()
}

/// `(anonymous, file-backed)` peaks of the solve stage, in bytes.
fn peaks(text: &str) -> (usize, usize) {
    let line = text
        .lines()
        .find_map(|l| l.split_once("solve peaks (bytes): "))
        .unwrap()
        .1;
    let field = |key: &str| -> usize {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key))
            .unwrap()
            .parse()
            .unwrap()
    };
    (field("anon="), field("file="))
}

fn spill_dir() -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("spill");
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_str().unwrap().to_string()
}

/// A spilled solve hands faer the same bytes, so it must return the same bits —
/// on both direct paths.
#[test]
fn spilling_changes_no_bit_of_the_solution() {
    let dir = spill_dir();
    for mode in ["thermal-lu", "thermal-cholesky"] {
        let plain = stdout(&run(mode, 12, None));
        let spilled = stdout(&run(mode, 12, Some(&dir)));
        assert_eq!(fingerprint(&plain), fingerprint(&spilled), "{mode}");
    }
}

/// The factorization's memory leaves the anonymous set for the file-backed
/// one — measured, not assumed.
#[test]
fn the_factors_move_to_file_backed_memory() {
    let dir = spill_dir();
    let (anon_plain, file_plain) = peaks(&stdout(&run("thermal-cholesky", 25, None)));
    let (anon_spilled, file_spilled) = peaks(&stdout(&run("thermal-cholesky", 25, Some(&dir))));
    assert!(
        anon_spilled < anon_plain / 2,
        "anonymous peak: {anon_spilled} spilled vs {anon_plain} plain"
    );
    assert!(
        file_spilled > file_plain + (anon_plain - anon_spilled) / 2,
        "file-backed peak: {file_spilled} spilled vs {file_plain} plain"
    );
}

/// A spill directory that cannot be opened stops the run, loudly — rather than
/// running unspilled into the out-of-memory killer.
#[test]
fn an_unusable_directory_aborts_with_a_message() {
    let out = run("thermal-lu", 2, Some("/nonexistent/pyrucast-spill"));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("PYRUCAST_SPILL_DIR"));
}
