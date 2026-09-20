//! The type system shared by every Unipute front end and back end.

use core::fmt;

/// A primitive numeric or boolean type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    Bool,
    I32,
    U32,
    F32,
}

impl Scalar {
    /// The size of a value of this type in bytes.
    pub const fn width(self) -> u8 {
        match self {
            // Booleans have no defined host layout, but shader languages still
            // treat them as 4 byte values internally.
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
        }
    }

    /// The Rust name that the `kernel` macro accepts for this type.
    pub const fn rust_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
        }
    }

    /// Parses a Rust type name such as `f32` into a scalar.
    pub fn from_rust_name(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "i32" => Some(Self::I32),
            "u32" => Some(Self::U32),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    /// Whether arithmetic on this type is floating point.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32)
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.rust_name())
    }
}

/// The number of components in a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VectorSize {
    Two = 2,
    Three = 3,
    Four = 4,
}

impl VectorSize {
    pub const fn count(self) -> u8 {
        self as u8
    }

    pub const fn from_count(count: u8) -> Option<Self> {
        match count {
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            _ => None,
        }
    }
}

/// A value type.
///
/// This is deliberately small. Matrices, atomics, textures and samplers are
/// reserved for later releases, and the back ends are written so that adding
/// them does not change the shape of this enum's existing variants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Vector {
        size: VectorSize,
        scalar: Scalar,
    },
    /// An array. A length of `None` means the array is sized at dispatch time,
    /// which is what a Rust slice parameter lowers to.
    Array {
        element: Box<Type>,
        len: Option<u32>,
    },
}

impl Type {
    /// Convenience constructor for a scalar type.
    pub const fn scalar(scalar: Scalar) -> Self {
        Self::Scalar(scalar)
    }

    /// Convenience constructor for a vector type.
    pub const fn vector(size: VectorSize, scalar: Scalar) -> Self {
        Self::Vector { size, scalar }
    }

    /// Convenience constructor for a runtime sized array.
    pub fn slice(element: Type) -> Self {
        Self::Array {
            element: Box::new(element),
            len: None,
        }
    }

    /// The scalar every component of this type is made of, if there is one.
    pub fn component_scalar(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(scalar) | Self::Vector { scalar, .. } => Some(*scalar),
            Self::Array { .. } => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Vector { size, scalar } => write!(f, "vec{}<{scalar}>", size.count()),
            Self::Array {
                element,
                len: Some(len),
            } => write!(f, "[{element}; {len}]"),
            Self::Array { element, len: None } => write!(f, "[{element}]"),
        }
    }
}
