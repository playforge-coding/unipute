//! Turning Rust expression syntax into Unipute IR expressions.
//!
//! The one interesting part is how untyped integer literals are handled. In
//! Rust `0` takes its type from context. Here, an unsuffixed literal is marked
//! [`Typed::weak`], and a binary operator that sees a weak side and a strong
//! side re-reads the weak one with the strong one's scalar type. That is
//! enough for `index < count` and `x * 2` to mean what they look like they
//! mean, without a real inference engine.

use syn::spanned::Spanned;
use unipute_ir as ir;

use super::function::Signature;
use super::types::value_type;
use super::{Binding, Scope};

/// An expression together with what is known about its type.
pub struct Typed {
    pub expr: ir::Expr,
    /// `None` when the type could not be worked out.
    pub ty: Option<ir::Type>,
    /// Whether this is a literal that has not committed to a type yet.
    pub weak: bool,
}

impl Typed {
    fn strong(expr: ir::Expr, ty: ir::Type) -> Self {
        Self {
            expr,
            ty: Some(ty),
            weak: false,
        }
    }

    fn scalar(expr: ir::Expr, scalar: ir::Scalar) -> Self {
        Self::strong(expr, ir::Type::scalar(scalar))
    }

    /// The scalar every component of this expression is made of.
    fn component(&self) -> Option<ir::Scalar> {
        self.ty.as_ref().and_then(ir::Type::component_scalar)
    }
}

impl Scope<'_> {
    /// Reads an expression, optionally steering untyped literals towards a
    /// scalar type.
    pub fn expr(&mut self, expr: &syn::Expr, expected: Option<ir::Scalar>) -> syn::Result<Typed> {
        match expr {
            syn::Expr::Lit(literal) => self.literal(literal, expected),
            syn::Expr::Path(path) => self.path(path),
            syn::Expr::Paren(paren) => self.expr(&paren.expr, expected),
            syn::Expr::Group(group) => self.expr(&group.expr, expected),
            syn::Expr::Unary(unary) => self.unary(unary, expected),
            syn::Expr::Binary(binary) => self.binary(binary),
            syn::Expr::Cast(cast) => self.cast(cast),
            syn::Expr::Index(index) => self.index(index),
            syn::Expr::Field(field) => self.field(field),
            syn::Expr::MethodCall(call) => self.method_call(call),
            syn::Expr::Call(call) => self.call(call, expected),
            other => Err(syn::Error::new_spanned(
                other,
                "this expression is not supported inside a kernel",
            )),
        }
    }

    fn literal(
        &mut self,
        literal: &syn::ExprLit,
        expected: Option<ir::Scalar>,
    ) -> syn::Result<Typed> {
        match &literal.lit {
            syn::Lit::Bool(value) => Ok(Typed::scalar(
                ir::Expr::Literal(ir::Literal::Bool(value.value)),
                ir::Scalar::Bool,
            )),
            syn::Lit::Int(value) => {
                let suffix = value.suffix();
                let scalar = if suffix.is_empty() {
                    // No suffix, so take the expected type if there is one.
                    // Integers standing alone default to `i32`, as in Rust.
                    expected
                        .filter(|scalar| !matches!(scalar, ir::Scalar::Bool))
                        .unwrap_or(ir::Scalar::I32)
                } else {
                    ir::Scalar::from_rust_name(suffix).ok_or_else(|| {
                        syn::Error::new_spanned(
                            value,
                            format!("`{suffix}` is not a scalar type Unipute supports"),
                        )
                    })?
                };
                let parsed = match scalar {
                    ir::Scalar::U32 => ir::Literal::U32(value.base10_parse()?),
                    ir::Scalar::I32 => ir::Literal::I32(value.base10_parse()?),
                    ir::Scalar::F32 => ir::Literal::F32(value.base10_parse()?),
                    ir::Scalar::Bool => {
                        return Err(syn::Error::new_spanned(
                            value,
                            "an integer literal cannot be a `bool`",
                        ));
                    }
                };
                Ok(Typed {
                    expr: ir::Expr::Literal(parsed),
                    ty: Some(ir::Type::scalar(scalar)),
                    weak: suffix.is_empty(),
                })
            }
            syn::Lit::Float(value) => {
                let suffix = value.suffix();
                if !suffix.is_empty() && suffix != "f32" {
                    return Err(syn::Error::new_spanned(
                        value,
                        "only `f32` floating point literals are supported",
                    ));
                }
                Ok(Typed::scalar(
                    ir::Expr::Literal(ir::Literal::F32(value.base10_parse()?)),
                    ir::Scalar::F32,
                ))
            }
            other => Err(syn::Error::new_spanned(
                other,
                "only integer, float and bool literals are supported",
            )),
        }
    }

    fn path(&mut self, path: &syn::ExprPath) -> syn::Result<Typed> {
        let name = path
            .path
            .get_ident()
            .ok_or_else(|| {
                syn::Error::new_spanned(path, "only plain variable names can be used here")
            })?
            .to_string();

        match self.lookup(&name) {
            Some(Binding::Local { id, ty }) => Ok(Typed::strong(ir::Expr::Local(id), ty)),
            Some(Binding::Param { id, ty }) => Ok(Typed::strong(ir::Expr::Param(id), ty)),
            // A buffer has no value of its own, but naming it is how you index
            // it, so the array type is passed along for `index` to use.
            Some(Binding::Resource { id, ty }) => Ok(Typed::strong(ir::Expr::Resource(id), ty)),
            Some(Binding::Shared { id, ty }) => Ok(Typed::strong(ir::Expr::Shared(id), ty)),
            None => Err(syn::Error::new_spanned(path, self.unknown_name(&name))),
        }
    }

    /// The message for a name that is not in scope. Inside a helper it says
    /// why the kernel's buffers are not among the names it could have been.
    fn unknown_name(&self, name: &str) -> String {
        match self.helper {
            Some(helper) => format!(
                "`{name}` is not a parameter or a local variable of `{}`, and a nested function \
                 cannot reach the kernel's buffers or workgroup memory, so pass it what it needs",
                helper.name
            ),
            None => format!("`{name}` is not a kernel parameter or a local variable"),
        }
    }

    fn unary(
        &mut self,
        unary: &syn::ExprUnary,
        expected: Option<ir::Scalar>,
    ) -> syn::Result<Typed> {
        let value = self.expr(&unary.expr, expected)?;
        let op = match unary.op {
            syn::UnOp::Neg(_) => ir::UnaryOp::Negate,
            syn::UnOp::Not(_) => ir::UnaryOp::Not,
            _ => {
                return Err(syn::Error::new_spanned(
                    unary,
                    "dereferencing is not supported inside a kernel",
                ));
            }
        };
        let ty = value.ty.clone();
        let weak = value.weak;
        Ok(Typed {
            expr: ir::Expr::Unary {
                op,
                value: Box::new(value.expr),
            },
            ty,
            weak,
        })
    }

    fn binary(&mut self, binary: &syn::ExprBinary) -> syn::Result<Typed> {
        let op = binary_op(binary)?;
        let (lhs, rhs) = self.balanced_operands(&binary.left, &binary.right)?;

        let ty = if op.is_comparison() {
            Some(ir::Type::scalar(ir::Scalar::Bool))
        } else {
            lhs.ty.clone().or_else(|| rhs.ty.clone())
        };
        Ok(Typed {
            expr: ir::Expr::Binary {
                op: op.ir,
                lhs: Box::new(lhs.expr),
                rhs: Box::new(rhs.expr),
            },
            ty,
            weak: false,
        })
    }

    /// Reads both sides of a binary operator, letting an untyped literal on
    /// one side pick up the scalar type of the other.
    pub fn balanced_operands(
        &mut self,
        left: &syn::Expr,
        right: &syn::Expr,
    ) -> syn::Result<(Typed, Typed)> {
        let lhs = self.expr(left, None)?;
        let rhs = self.expr(right, lhs.component())?;

        // The right hand side may have been the one carrying the type, in
        // which case the left hand side needs a second look.
        if lhs.weak
            && !rhs.weak
            && let Some(scalar) = rhs.component()
        {
            let lhs = self.expr(left, Some(scalar))?;
            return Ok((lhs, rhs));
        }
        Ok((lhs, rhs))
    }

    fn cast(&mut self, cast: &syn::ExprCast) -> syn::Result<Typed> {
        let target = value_type(&cast.ty)?;
        let ir::Type::Scalar(scalar) = target else {
            return Err(syn::Error::new_spanned(
                &cast.ty,
                "`as` can only cast to a scalar type",
            ));
        };
        let value = self.expr(&cast.expr, None)?;
        Ok(Typed::scalar(
            ir::Expr::Cast {
                value: Box::new(value.expr),
                to: scalar,
            },
            scalar,
        ))
    }

    fn index(&mut self, index: &syn::ExprIndex) -> syn::Result<Typed> {
        let base = self.expr(&index.expr, None)?;
        let element = match &base.ty {
            Some(ir::Type::Array { element, .. }) => Some((**element).clone()),
            Some(other) => {
                return Err(syn::Error::new_spanned(
                    &index.expr,
                    format!("`{other}` cannot be indexed"),
                ));
            }
            None => None,
        };
        // Shader index expressions are unsigned, so an untyped literal index
        // becomes a `u32` rather than the `i32` it would be in Rust.
        let position = self.expr(&index.index, Some(ir::Scalar::U32))?;
        Ok(Typed {
            expr: ir::Expr::Index {
                base: Box::new(base.expr),
                index: Box::new(position.expr),
            },
            ty: element,
            weak: false,
        })
    }

    fn field(&mut self, field: &syn::ExprField) -> syn::Result<Typed> {
        let syn::Member::Named(name) = &field.member else {
            return Err(syn::Error::new_spanned(
                &field.member,
                "use `.x`, `.y`, `.z` or `.w` to read a vector component",
            ));
        };
        let component = match name.to_string().as_str() {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            "w" => 3,
            other => {
                return Err(syn::Error::new_spanned(
                    name,
                    format!("`{other}` is not a vector component, use `x`, `y`, `z` or `w`"),
                ));
            }
        };
        let base = self.expr(&field.base, None)?;
        let ty = match &base.ty {
            Some(ir::Type::Vector { size, scalar }) => {
                if component >= size.count() {
                    return Err(syn::Error::new_spanned(
                        name,
                        format!("`{name}` is out of range for a vec{}", size.count()),
                    ));
                }
                Some(ir::Type::scalar(*scalar))
            }
            Some(other) => {
                return Err(syn::Error::new_spanned(
                    &field.base,
                    format!("`{other}` has no components"),
                ));
            }
            None => None,
        };
        Ok(Typed {
            expr: ir::Expr::Component {
                base: Box::new(base.expr),
                index: component,
            },
            ty,
            weak: false,
        })
    }

    fn method_call(&mut self, call: &syn::ExprMethodCall) -> syn::Result<Typed> {
        let method = call.method.to_string();
        if method != "len" {
            return Err(syn::Error::new_spanned(
                &call.method,
                format!("`{method}` is not a method Unipute supports, only `len` is"),
            ));
        }
        if !call.args.is_empty() {
            return Err(syn::Error::new_spanned(
                &call.args,
                "`len` takes no arguments",
            ));
        }
        let receiver = self.expr(&call.receiver, None)?;
        match (&receiver.expr, &receiver.ty) {
            (ir::Expr::Resource(resource), Some(ir::Type::Array { len: None, .. })) => Ok(
                Typed::scalar(ir::Expr::ArrayLength(*resource), ir::Scalar::U32),
            ),
            // Workgroup memory has its length written in the kernel, so there
            // is nothing to ask the driver: the length is a plain number.
            (ir::Expr::Shared(_), Some(ir::Type::Array { len: Some(len), .. })) => Ok(
                Typed::scalar(ir::Expr::Literal(ir::Literal::U32(*len)), ir::Scalar::U32),
            ),
            _ => Err(syn::Error::new_spanned(
                &call.receiver,
                "`len` only works on a slice parameter or on workgroup memory that is an array",
            )),
        }
    }

    fn call(&mut self, call: &syn::ExprCall, expected: Option<ir::Scalar>) -> syn::Result<Typed> {
        let syn::Expr::Path(path) = &*call.func else {
            return Err(syn::Error::new_spanned(
                &call.func,
                "only calls to built-in functions are supported",
            ));
        };
        let name = path
            .path
            .get_ident()
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    &call.func,
                    "only calls to built-in functions are supported",
                )
            })?
            .to_string();

        if let Some(built_in) = ir::BuiltIn::from_intrinsic_name(&name) {
            if let Some(helper) = self.helper {
                return Err(syn::Error::new_spanned(
                    &call.func,
                    format!(
                        "`{name}` is only available in the kernel body, so `{}` has to take what \
                         it needs as a parameter",
                        helper.name
                    ),
                ));
            }
            if !call.args.is_empty() {
                return Err(syn::Error::new_spanned(
                    &call.args,
                    format!("`{name}` takes no arguments"),
                ));
            }
            let ty = match built_in {
                ir::BuiltIn::LocalInvocationIndex => ir::Type::scalar(ir::Scalar::U32),
                _ => ir::Type::vector(ir::VectorSize::Three, ir::Scalar::U32),
            };
            return Ok(Typed::strong(ir::Expr::BuiltIn(built_in), ty));
        }

        if let Some(size) = vector_constructor(&name) {
            return self.compose(call, size, expected);
        }

        if let Some(function) = ir::MathFn::from_intrinsic_name(&name) {
            return self.math(call, function);
        }

        if let Some(signature) = self.function_named(&name) {
            return self.user_call(call, signature);
        }

        Err(syn::Error::new_spanned(
            &call.func,
            format!("`{name}` is not a built-in Unipute provides, or a `fn` in this kernel"),
        ))
    }

    /// A call to one of the kernel's own nested functions, used as a value.
    fn user_call(&mut self, call: &syn::ExprCall, signature: &Signature) -> syn::Result<Typed> {
        let args = self.call_args(call, signature)?;
        let Some(result) = signature.result.clone() else {
            return Err(syn::Error::new_spanned(
                call,
                format!(
                    "`{}` returns nothing, so its call has no value",
                    signature.name
                ),
            ));
        };
        Ok(Typed::strong(
            ir::Expr::Call {
                function: signature.id,
                args,
            },
            result,
        ))
    }

    /// Reads the arguments of a call to a nested function, checking them
    /// against its parameters.
    ///
    /// The types are checked here rather than left to the back end. A mismatch
    /// that got through would come back as a naga validation failure, which is
    /// how Unipute reports a bug in itself, not a mistake in a kernel.
    pub fn call_args(
        &mut self,
        call: &syn::ExprCall,
        signature: &Signature,
    ) -> syn::Result<Vec<ir::Expr>> {
        if call.args.len() != signature.params.len() {
            return Err(syn::Error::new_spanned(
                call,
                format!(
                    "`{}` takes {} arguments but got {}",
                    signature.name,
                    signature.params.len(),
                    call.args.len()
                ),
            ));
        }
        let mut args = Vec::with_capacity(call.args.len());
        for (argument, param) in call.args.iter().zip(&signature.params) {
            let value = self.expr(argument, param.ty.component_scalar())?;
            if let Some(ty) = &value.ty
                && ty != &param.ty
            {
                return Err(syn::Error::new_spanned(
                    argument,
                    format!(
                        "`{}` takes `{}` for `{}`, but this is `{ty}`",
                        signature.name, param.ty, param.name
                    ),
                ));
            }
            args.push(value.expr);
        }
        Ok(args)
    }

    fn compose(
        &mut self,
        call: &syn::ExprCall,
        size: ir::VectorSize,
        expected: Option<ir::Scalar>,
    ) -> syn::Result<Typed> {
        if call.args.len() != usize::from(size.count()) {
            return Err(syn::Error::new_spanned(
                &call.args,
                format!(
                    "`vec{}` takes {} components but got {}",
                    size.count(),
                    size.count(),
                    call.args.len()
                ),
            ));
        }
        let mut components = Vec::with_capacity(call.args.len());
        let mut scalar = expected;
        for arg in &call.args {
            let component = self.expr(arg, scalar)?;
            if scalar.is_none() && !component.weak {
                scalar = component.component();
            }
            components.push(component);
        }
        let scalar = scalar.ok_or_else(|| {
            syn::Error::new_spanned(
                call,
                "the component type of this vector is unclear, add a suffix such as `1.0f32`",
            )
        })?;

        // Once the component type is settled, re-read any untyped literal
        // components so they agree with it.
        let mut settled = Vec::with_capacity(components.len());
        for (component, arg) in components.into_iter().zip(&call.args) {
            if component.weak {
                settled.push(self.expr(arg, Some(scalar))?.expr);
            } else {
                settled.push(component.expr);
            }
        }

        Ok(Typed::strong(
            ir::Expr::Compose {
                size,
                scalar,
                components: settled,
            },
            ir::Type::vector(size, scalar),
        ))
    }

    fn math(&mut self, call: &syn::ExprCall, function: ir::MathFn) -> syn::Result<Typed> {
        if call.args.len() != function.arity() {
            return Err(syn::Error::new_spanned(
                &call.args,
                format!(
                    "`{}` takes {} arguments but got {}",
                    function.intrinsic_name(),
                    function.arity(),
                    call.args.len()
                ),
            ));
        }
        let mut args = Vec::with_capacity(call.args.len());
        let mut settled: Option<ir::Scalar> = None;
        for arg in &call.args {
            let value = self.expr(arg, settled)?;
            if settled.is_none() && !value.weak {
                settled = value.component();
            }
            args.push(value);
        }

        let first_ty = args.first().and_then(|arg| arg.ty.clone());
        let mut lowered = Vec::with_capacity(args.len());
        for (arg, syntax) in args.into_iter().zip(&call.args) {
            if arg.weak && settled.is_some() {
                lowered.push(self.expr(syntax, settled)?.expr);
            } else {
                lowered.push(arg.expr);
            }
        }

        let ty = math_result_type(function, first_ty);
        Ok(Typed {
            expr: ir::Expr::Math {
                function,
                args: lowered,
            },
            ty,
            weak: false,
        })
    }
}

/// The type a math function returns, given the type of its first argument.
fn math_result_type(function: ir::MathFn, argument: Option<ir::Type>) -> Option<ir::Type> {
    match function {
        // These collapse a vector down to one number.
        ir::MathFn::Dot | ir::MathFn::Length => argument
            .and_then(|ty| ty.component_scalar())
            .map(ir::Type::scalar),
        // Everything else keeps the shape of its first argument.
        _ => argument,
    }
}

fn vector_constructor(name: &str) -> Option<ir::VectorSize> {
    match name {
        "vec2" => Some(ir::VectorSize::Two),
        "vec3" => Some(ir::VectorSize::Three),
        "vec4" => Some(ir::VectorSize::Four),
        _ => None,
    }
}

/// A binary operator, with a note about whether it produces a `bool`.
pub struct BinaryOp {
    pub ir: ir::BinaryOp,
    comparison: bool,
}

impl BinaryOp {
    pub const fn is_comparison(&self) -> bool {
        self.comparison
    }
}

fn binary_op(binary: &syn::ExprBinary) -> syn::Result<BinaryOp> {
    use ir::BinaryOp as B;
    let (ir, comparison) = match binary.op {
        syn::BinOp::Add(_) => (B::Add, false),
        syn::BinOp::Sub(_) => (B::Subtract, false),
        syn::BinOp::Mul(_) => (B::Multiply, false),
        syn::BinOp::Div(_) => (B::Divide, false),
        syn::BinOp::Rem(_) => (B::Modulo, false),
        syn::BinOp::And(_) => (B::LogicalAnd, false),
        syn::BinOp::Or(_) => (B::LogicalOr, false),
        syn::BinOp::BitXor(_) => (B::Xor, false),
        syn::BinOp::BitAnd(_) => (B::And, false),
        syn::BinOp::BitOr(_) => (B::Or, false),
        syn::BinOp::Shl(_) => (B::ShiftLeft, false),
        syn::BinOp::Shr(_) => (B::ShiftRight, false),
        syn::BinOp::Eq(_) => (B::Equal, true),
        syn::BinOp::Lt(_) => (B::Less, true),
        syn::BinOp::Le(_) => (B::LessEqual, true),
        syn::BinOp::Ne(_) => (B::NotEqual, true),
        syn::BinOp::Ge(_) => (B::GreaterEqual, true),
        syn::BinOp::Gt(_) => (B::Greater, true),
        _ => {
            return Err(syn::Error::new(
                binary.op.span(),
                "this operator is not supported inside a kernel",
            ));
        }
    };
    Ok(BinaryOp { ir, comparison })
}

/// Maps a compound assignment such as `+=` onto its plain operator.
pub fn compound_op(op: syn::BinOp) -> Option<ir::BinaryOp> {
    use ir::BinaryOp as B;
    Some(match op {
        syn::BinOp::AddAssign(_) => B::Add,
        syn::BinOp::SubAssign(_) => B::Subtract,
        syn::BinOp::MulAssign(_) => B::Multiply,
        syn::BinOp::DivAssign(_) => B::Divide,
        syn::BinOp::RemAssign(_) => B::Modulo,
        syn::BinOp::BitXorAssign(_) => B::Xor,
        syn::BinOp::BitAndAssign(_) => B::And,
        syn::BinOp::BitOrAssign(_) => B::Or,
        syn::BinOp::ShlAssign(_) => B::ShiftLeft,
        syn::BinOp::ShrAssign(_) => B::ShiftRight,
        _ => return None,
    })
}
