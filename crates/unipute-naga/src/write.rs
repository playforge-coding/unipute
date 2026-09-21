//! Turning a validated naga module into shader output.
//!
//! Every writer here takes the same two inputs, a module and the validation
//! info that goes with it, so adding a target is a matter of adding a feature
//! and one function.

use unipute_ir::Target;

use crate::error::{Error, Result};

/// A validated module, paired with the type information naga's writers need.
pub struct Validated {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
}

impl core::fmt::Debug for Validated {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Validated")
            .field("entry_points", &self.module.entry_points.len())
            .finish_non_exhaustive()
    }
}

/// Runs naga's validator over a module.
///
/// The capabilities are the ones a plain compute kernel needs. Anything a
/// kernel can express today falls inside them.
pub fn validate(module: naga::Module) -> Result<Validated> {
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    match validator.validate(&module) {
        Ok(info) => Ok(Validated { module, info }),
        Err(error) => Err(Error::Validation(format!("{error:?}"))),
    }
}

#[cfg(any(
    feature = "wgsl",
    feature = "spv",
    feature = "msl",
    feature = "hlsl",
    feature = "glsl"
))]
fn write_error(target: Target, error: impl core::fmt::Display) -> Error {
    Error::Write {
        target,
        message: error.to_string(),
    }
}

/// Which targets this build can actually produce.
pub fn enabled_targets() -> Vec<Target> {
    let mut targets = Vec::new();
    if cfg!(feature = "wgsl") {
        targets.push(Target::Wgsl);
    }
    if cfg!(feature = "spv") {
        targets.push(Target::SpirV);
    }
    if cfg!(feature = "msl") {
        targets.push(Target::Msl);
    }
    if cfg!(feature = "hlsl") {
        targets.push(Target::Hlsl);
    }
    if cfg!(feature = "glsl") {
        targets.push(Target::Glsl);
    }
    targets
}

#[cfg(feature = "wgsl")]
pub fn wgsl(validated: &Validated) -> Result<String> {
    naga::back::wgsl::write_string(
        &validated.module,
        &validated.info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|error| write_error(Target::Wgsl, error))
}

#[cfg(feature = "spv")]
pub fn spirv(validated: &Validated) -> Result<Vec<u32>> {
    let options = naga::back::spv::Options::default();
    naga::back::spv::write_vec(&validated.module, &validated.info, &options, None)
        .map_err(|error| write_error(Target::SpirV, error))
}

#[cfg(feature = "msl")]
pub fn msl(validated: &Validated) -> Result<String> {
    use naga::back::msl;

    let entry_point = entry_point_name(validated)?;
    let options = msl::Options {
        // Without a slot map naga has nowhere to put the buffer bindings, so
        // let it assign them rather than refusing the module.
        fake_missing_bindings: true,
        ..Default::default()
    };
    let pipeline_options = msl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, entry_point)),
        ..Default::default()
    };
    let mut writer = msl::Writer::new(String::new());
    writer
        .write(
            &validated.module,
            &validated.info,
            &options,
            &pipeline_options,
        )
        .map_err(|error| write_error(Target::Msl, error))?;
    Ok(writer.finish())
}

#[cfg(feature = "hlsl")]
pub fn hlsl(validated: &Validated) -> Result<String> {
    use naga::back::hlsl;

    let entry_point = entry_point_name(validated)?;
    let options = hlsl::Options::default();
    let pipeline_options = hlsl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, entry_point)),
    };
    let mut output = String::new();
    let mut writer = hlsl::Writer::new(&mut output, &options, &pipeline_options);
    writer
        .write(&validated.module, &validated.info, None)
        .map_err(|error| write_error(Target::Hlsl, error))?;
    Ok(output)
}

/// Writes GLSL for OpenGL ES 3.10, the first version with compute shaders.
///
/// That profile also loads on a desktop OpenGL 4.3 or later context that has
/// `ARB_ES3_1_compatibility`, which most do. For a specific desktop version,
/// use [`glsl_with`].
#[cfg(feature = "glsl")]
pub fn glsl(validated: &Validated) -> Result<String> {
    glsl_with(validated, naga::back::glsl::Version::new_gles(310))
}

/// Writes GLSL for a particular version and profile.
///
/// The `version` is naga's own type, so anything naga's GLSL writer can
/// produce can be asked for here. Compute shaders need OpenGL ES 3.10 or
/// desktop OpenGL 4.30 at the least, and naga refuses an older one.
#[cfg(feature = "glsl")]
pub fn glsl_with(validated: &Validated, version: naga::back::glsl::Version) -> Result<String> {
    use naga::back::glsl;

    let entry_point = entry_point_name(validated)?;
    let options = glsl::Options {
        version,
        binding_map: glsl_binding_map(&validated.module)?,
        ..Default::default()
    };
    let pipeline_options = glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point,
        multiview: None,
    };
    let mut output = String::new();
    let mut writer = glsl::Writer::new(
        &mut output,
        &validated.module,
        &validated.info,
        &options,
        &pipeline_options,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .map_err(|error| write_error(Target::Glsl, error))?;
    writer
        .write()
        .map_err(|error| write_error(Target::Glsl, error))?;
    Ok(output)
}

/// Gives every resource its GLSL binding number.
///
/// GLSL has no bind groups. A buffer or uniform block carries one
/// `binding = N`, and without it the driver picks a slot and the host is left
/// looking slots up by block name, which naga generates. A GLSL front end such
/// as wgpu's refuses a block with no binding at all. So each resource is
/// written with its own binding number and the group is dropped.
///
/// That leaves one namespace where the kernel had several, so two resources
/// in different groups that share a binding number would land on top of each
/// other. That is reported rather than written, since the fix is a different
/// `index` on one of them.
#[cfg(feature = "glsl")]
fn glsl_binding_map(module: &naga::Module) -> Result<naga::back::glsl::BindingMap> {
    let mut map = naga::back::glsl::BindingMap::default();
    let mut taken = std::collections::BTreeMap::new();
    for (_, global) in module.global_variables.iter() {
        let Some(resource) = global.binding.as_ref() else {
            continue;
        };
        let name = global.name.as_deref().unwrap_or("an unnamed resource");
        if let Some(other) = taken.insert(resource.binding, name) {
            return Err(Error::Invalid(format!(
                "GLSL has no bind groups, so every resource needs its own binding number, \
                 but `{other}` and `{name}` both use binding {}",
                resource.binding
            )));
        }
        let slot = u8::try_from(resource.binding).map_err(|_| {
            Error::Invalid(format!(
                "GLSL binding numbers stop at 255, and `{name}` uses {}",
                resource.binding
            ))
        })?;
        map.insert(*resource, slot);
    }
    Ok(map)
}

#[cfg(any(feature = "msl", feature = "hlsl", feature = "glsl"))]
fn entry_point_name(validated: &Validated) -> Result<String> {
    validated
        .module
        .entry_points
        .first()
        .map(|entry_point| entry_point.name.clone())
        .ok_or_else(|| Error::Invalid("the module has no entry point".to_owned()))
}
