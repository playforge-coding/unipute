//! Errors produced while lowering or writing.

use core::fmt;

use unipute_ir::{Stage, Target};

/// Anything that can go wrong between a [`Kernel`](unipute_ir::Kernel) and
/// finished shader output.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The kernel uses a pipeline stage the naga back end does not handle yet.
    UnsupportedStage(Stage),
    /// A workgroup dimension was zero.
    EmptyWorkgroup,
    /// A type cannot be expressed in naga IR, or is not allowed where it was
    /// used.
    UnsupportedType(String),
    /// The kernel is structurally wrong, such as indexing something that is
    /// not an array.
    Invalid(String),
    /// Naga rejected the module we built. This means a bug in Unipute rather
    /// than in the user's kernel, so the message is passed through verbatim.
    Validation(String),
    /// A naga writer failed.
    Write { target: Target, message: String },
    /// The requested target exists but this build was compiled without it.
    TargetNotEnabled(Target),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStage(stage) => {
                write!(
                    f,
                    "the naga back end does not support the {} stage yet",
                    stage.name()
                )
            }
            Self::EmptyWorkgroup => f.write_str("every workgroup dimension must be at least 1"),
            Self::UnsupportedType(message) => write!(f, "unsupported type: {message}"),
            Self::Invalid(message) => write!(f, "invalid kernel: {message}"),
            Self::Validation(message) => {
                write!(f, "naga rejected the generated module: {message}")
            }
            Self::Write { target, message } => {
                write!(f, "could not write {target}: {message}")
            }
            Self::TargetNotEnabled(target) => write!(
                f,
                "target {target} is not available, enable the \"{target}\" feature"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
