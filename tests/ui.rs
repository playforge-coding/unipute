//! Checks that a kernel the macro cannot accept fails with a message that
//! says what to do about it.
//!
//! Error messages are most of what a macro's users experience, so they are
//! worth pinning down. Run with `TRYBUILD=overwrite` to refresh the expected
//! output after changing a message on purpose.

#[test]
fn rejected_kernels_explain_themselves() {
    // The expected output includes rustc's own formatting, which changes
    // between toolchain versions. CI runs these once on a known toolchain and
    // sets this variable everywhere else, so a rustc update shows up as one
    // failure to look at rather than one per platform.
    if std::env::var_os("UNIPUTE_SKIP_UI_TESTS").is_some() {
        eprintln!("skipped because UNIPUTE_SKIP_UI_TESTS is set");
        return;
    }

    let harness = trybuild::TestCases::new();
    harness.compile_fail("tests/ui/*.rs");
}
