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

/// One member of a [`StructType`], at the byte offset the GPU reads it from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructMember {
    pub name: String,
    pub ty: Type,
    /// Distance in bytes from the start of the struct.
    pub offset: u32,
}

/// A struct with an explicit byte layout.
///
/// A struct in a buffer is read by the host and by the shader, and the two
/// have to agree about every byte. So the offsets are written down here rather
/// than left for each back end to work out: [`StructType::new`] computes them
/// by the rules every shader language shares, and a back end checks them
/// against the same rules rather than trusting them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructType {
    pub name: String,
    /// In declaration order, with offsets that never decrease.
    pub members: Vec<StructMember>,
    /// The size of the whole struct in bytes, including the padding at the
    /// end that rounds it up to its alignment.
    pub size: u32,
}

impl StructType {
    /// Lays members out one after another the way a storage buffer expects.
    ///
    /// Each member starts at the next multiple of its alignment, and the
    /// struct is padded at the end to a multiple of its own alignment, which
    /// is the largest alignment among its members. That is the layout WGSL,
    /// SPIR-V, MSL and HLSL all agree on for a storage buffer, and for a
    /// uniform too as long as no member is itself a struct or an array.
    ///
    /// Returns `None` if a member has no fixed size, meaning a runtime sized
    /// array.
    pub fn new(
        name: impl Into<String>,
        members: impl IntoIterator<Item = (String, Type)>,
    ) -> Option<Self> {
        let mut laid_out = Vec::new();
        let mut cursor = 0;
        let mut alignment = 1;
        for (name, ty) in members {
            let size = ty.size()?;
            let align = ty.alignment();
            let offset = round_up(cursor, align);
            laid_out.push(StructMember { name, ty, offset });
            cursor = offset + size;
            alignment = alignment.max(align);
        }
        Some(Self {
            name: name.into(),
            members: laid_out,
            size: round_up(cursor, alignment),
        })
    }

    /// The alignment of the struct, which is the largest among its members.
    pub fn alignment(&self) -> u32 {
        self.members
            .iter()
            .map(|member| member.ty.alignment())
            .max()
            .unwrap_or(1)
    }

    /// Finds a member by name, with its position in [`StructType::members`].
    pub fn member(&self, name: &str) -> Option<(u32, &StructMember)> {
        self.members
            .iter()
            .enumerate()
            .find(|(_, member)| member.name == name)
            .map(|(index, member)| (index as u32, member))
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
    /// A struct, carrying its own layout. Two struct types are the same type
    /// when their name, members and offsets are all the same.
    Struct(StructType),
    /// A `u32` or `i32` that several invocations may update at once, through
    /// the atomic operations and nothing else. It has the size and alignment
    /// of its scalar, so the host sees a plain integer.
    ///
    /// Atomics live in storage buffers and workgroup memory. A local, a
    /// parameter or a uniform cannot be one, since no invocation could race
    /// another for it there.
    Atomic(Scalar),
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
    ///
    /// An atomic answers `None`. It holds a scalar, but it is not one, and a
    /// front end that let an untyped literal take its type from an atomic
    /// would be letting the atomic stand where a value goes.
    pub fn component_scalar(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(scalar) | Self::Vector { scalar, .. } => Some(*scalar),
            Self::Array { .. } | Self::Struct(_) | Self::Atomic(_) => None,
        }
    }

    /// The size of one value in bytes, or `None` for a runtime sized array.
    ///
    /// These are the rules every shader language shares for a storage
    /// buffer. A scalar is 4 bytes. A vector is its components side by side,
    /// so a `vec3` is 12 bytes, even though it aligns to 16. An array is its
    /// [stride](Type::stride) times its length, and a struct is whatever its
    /// layout says.
    pub fn size(&self) -> Option<u32> {
        Some(match self {
            Self::Scalar(scalar) | Self::Atomic(scalar) => u32::from(scalar.width()),
            Self::Vector { size, scalar } => u32::from(size.count()) * u32::from(scalar.width()),
            Self::Array {
                element,
                len: Some(len),
            } => element.stride()? * len,
            Self::Array { len: None, .. } => return None,
            Self::Struct(def) => def.size,
        })
    }

    /// The alignment a value of this type needs in a buffer, in bytes.
    ///
    /// A scalar aligns to 4, a `vec2` to 8, and a `vec3` or `vec4` to 16. An
    /// array aligns like its element and a struct like its most demanding
    /// member.
    pub fn alignment(&self) -> u32 {
        match self {
            Self::Scalar(scalar) | Self::Atomic(scalar) => u32::from(scalar.width()),
            Self::Vector { size, scalar } => {
                let width = u32::from(scalar.width());
                match size {
                    VectorSize::Two => 2 * width,
                    VectorSize::Three | VectorSize::Four => 4 * width,
                }
            }
            Self::Array { element, .. } => element.alignment(),
            Self::Struct(def) => def.alignment(),
        }
    }

    /// The distance between two values of this type in an array, which is the
    /// size rounded up to the alignment. `None` for a runtime sized array.
    pub fn stride(&self) -> Option<u32> {
        Some(round_up(self.size()?, self.alignment()))
    }
}

/// Rounds `value` up to the next multiple of `alignment`.
pub const fn round_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
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
            Self::Struct(def) => f.write_str(&def.name),
            Self::Atomic(Scalar::I32) => f.write_str("AtomicI32"),
            Self::Atomic(Scalar::U32) => f.write_str("AtomicU32"),
            Self::Atomic(scalar) => write!(f, "Atomic<{scalar}>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vec3_takes_twelve_bytes_but_aligns_to_sixteen() {
        let vec3 = Type::vector(VectorSize::Three, Scalar::F32);
        assert_eq!(vec3.size(), Some(12));
        assert_eq!(vec3.alignment(), 16);
        assert_eq!(vec3.stride(), Some(16));
    }

    #[test]
    fn a_struct_pads_each_member_to_its_alignment() {
        let def = StructType::new(
            "Body",
            [
                ("mass".to_owned(), Type::scalar(Scalar::F32)),
                (
                    "position".to_owned(),
                    Type::vector(VectorSize::Three, Scalar::F32),
                ),
                ("charge".to_owned(), Type::scalar(Scalar::F32)),
            ],
        )
        .unwrap();

        let offsets: Vec<u32> = def.members.iter().map(|member| member.offset).collect();
        assert_eq!(offsets, [0, 16, 28]);
        assert_eq!(def.size, 32);
        assert_eq!(def.alignment(), 16);
        assert_eq!(def.member("charge").map(|(index, _)| index), Some(2));
    }

    #[test]
    fn a_member_that_fits_after_a_vec3_is_not_pushed_out() {
        let def = StructType::new(
            "Particle",
            [
                (
                    "position".to_owned(),
                    Type::vector(VectorSize::Three, Scalar::F32),
                ),
                ("mass".to_owned(), Type::scalar(Scalar::F32)),
            ],
        )
        .unwrap();
        assert_eq!(def.members[1].offset, 12);
        assert_eq!(def.size, 16);
    }

    #[test]
    fn a_runtime_sized_member_has_no_layout() {
        let def = StructType::new(
            "Bad",
            [("tail".to_owned(), Type::slice(Type::scalar(Scalar::F32)))],
        );
        assert!(def.is_none());
    }
}
