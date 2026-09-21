//! The wgpu side of running a kernel.
//!
//! Unipute stops at the shader. This is the code the guide's "Running a
//! kernel" chapter sketches, written out in full against wgpu, and it is what
//! the examples and the GPU tests run kernels through. It is not part of the
//! library: a crate depending on Unipute brings no graphics API with it, and
//! this file is where that line is drawn.
//!
//! Everything the host needs comes off the kernel type rather than being
//! written out a second time: `NAME` for the entry point, `WORKGROUP_SIZE` for
//! the dispatch arithmetic, `BINDINGS` for the layout, and one of the shader
//! constants for the module itself.

// Each example uses a subset of this, and the parts one example leaves alone
// would otherwise warn in that example's build.
#![allow(dead_code)]

use std::borrow::Cow;

use unipute::{Access, BindingInfo, Kernel, WgslKernel};
use wgpu::util::DeviceExt;
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// An open device, and the queue that feeds it.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

/// A compute pipeline for one kernel, with the layouts it was built from.
pub struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    /// One layout per bind group, indexed by group number. A group the kernel
    /// skips over still gets an empty layout, since the pipeline layout is a
    /// dense list.
    layouts: Vec<wgpu::BindGroupLayout>,
    bindings: &'static [BindingInfo],
    name: &'static str,
}

impl Gpu {
    /// Opens the first adapter wgpu can find.
    ///
    /// `None` means the machine has nothing wgpu can drive, which is a state
    /// for the caller to report rather than something to panic over. The
    /// `WGPU_BACKEND` and `WGPU_ADAPTER_NAME` environment variables wgpu reads
    /// are honoured, so a particular device can be picked from outside.
    pub fn open() -> Option<Gpu> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some(Gpu {
            info: adapter.get_info(),
            device,
            queue,
        })
    }

    /// The adapter and back end in use, for a line of output.
    pub fn describe(&self) -> String {
        format!("{} through {:?}", self.info.name, self.info.backend)
    }

    /// Builds a pipeline for a kernel from its compile time WGSL.
    pub fn pipeline<K: WgslKernel>(&self) -> Pipeline {
        self.pipeline_from::<K>(wgsl::<K>())
    }

    /// Builds a pipeline for a kernel from any shader source wgpu accepts.
    ///
    /// The layout comes from `BINDINGS`, which is the point of having it: add
    /// a parameter to the kernel and the host follows without an edit.
    pub fn pipeline_from<K: Kernel>(&self, source: wgpu::ShaderSource<'_>) -> Pipeline {
        self.pipeline_with_entry::<K>(source, Some(K::NAME))
    }

    /// The same, with the entry point spelled out.
    ///
    /// `NAME` is the entry point in every target but GLSL, where a compute
    /// shader's entry point is always `main` and the kernel's name does not
    /// survive. `None` lets wgpu take the one entry point the module has.
    pub fn pipeline_with_entry<K: Kernel>(
        &self,
        source: wgpu::ShaderSource<'_>,
        entry_point: Option<&str>,
    ) -> Pipeline {
        let name = K::NAME;
        let bindings = K::BINDINGS;

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source,
            });

        let groups = bindings
            .iter()
            .map(|binding| binding.group + 1)
            .max()
            .unwrap_or(0);
        let layouts: Vec<wgpu::BindGroupLayout> = (0..groups)
            .map(|group| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
                    .iter()
                    .filter(|binding| binding.group == group)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding: binding.binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: buffer_type(binding.access),
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(name),
                        entries: &entries,
                    })
            })
            .collect();

        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
                cache: None,
            });

        Pipeline {
            pipeline,
            layouts,
            bindings,
            name,
        }
    }

    /// A storage buffer holding `data`, readable back with [`Gpu::read`].
    pub fn storage<T: IntoBytes + Immutable>(&self, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// A uniform buffer holding one value.
    ///
    /// The buffer is padded to 16 bytes. Nothing here needs it for a scalar,
    /// but it is the alignment uniform data has to have once it grows into a
    /// struct, and paying it up front keeps every back end happy.
    pub fn uniform<T: IntoBytes + Immutable>(&self, value: &T) -> wgpu::Buffer {
        let mut bytes = value.as_bytes().to_vec();
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Replaces the start of a buffer with `value` before the next dispatch.
    pub fn write<T: IntoBytes + Immutable>(&self, buffer: &wgpu::Buffer, value: &T) {
        self.queue.write_buffer(buffer, 0, value.as_bytes());
    }

    /// Runs a kernel once over `workgroups` groups and waits for it.
    ///
    /// `buffers` are in parameter order, the same order as `BINDINGS`, and
    /// each one is bound where its binding says.
    pub fn dispatch(&self, pipeline: &Pipeline, buffers: &[&wgpu::Buffer], workgroups: [u32; 3]) {
        assert_eq!(
            buffers.len(),
            pipeline.bindings.len(),
            "`{}` takes {} buffers, one per parameter",
            pipeline.name,
            pipeline.bindings.len()
        );

        let bind_groups: Vec<wgpu::BindGroup> = pipeline
            .layouts
            .iter()
            .enumerate()
            .map(|(group, layout)| {
                let entries: Vec<wgpu::BindGroupEntry> = pipeline
                    .bindings
                    .iter()
                    .zip(buffers)
                    .filter(|(binding, _)| binding.group as usize == group)
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding.binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(pipeline.name),
                    layout,
                    entries: &entries,
                })
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(pipeline.name),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pipeline.name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            for (group, bind_group) in bind_groups.iter().enumerate() {
                pass.set_bind_group(group as u32, bind_group, &[]);
            }
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device should finish the dispatch");
    }

    /// Copies a buffer back to the host.
    pub fn read<T: FromBytes + Immutable + Copy>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read back"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, None);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("the staging buffer should map");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device should finish the copy");

        let view = slice
            .get_mapped_range()
            .expect("the staging buffer was mapped above");
        let values = <[T]>::ref_from_bytes(&view)
            .expect("the buffer holds a whole number of elements")
            .to_vec();
        drop(view);
        staging.unmap();
        values
    }
}

/// A kernel's compile time WGSL as a wgpu shader source.
pub fn wgsl<K: WgslKernel>() -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Wgsl(Cow::Borrowed(K::WGSL))
}

/// A kernel's compile time SPIR-V as a wgpu shader source.
#[cfg(feature = "spv")]
pub fn spirv<K: unipute::SpirvKernel>() -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::SpirV(Cow::Borrowed(K::SPIRV))
}

/// GLSL text as a wgpu compute shader source.
///
/// This takes text rather than a kernel because wgpu reads GLSL through
/// naga, and naga's reader only accepts desktop profiles 4.40 and up. The
/// `GLSL` constant on a kernel is OpenGL ES 3.10, which an OpenGL driver
/// takes and naga does not, so a host going through wgpu writes a desktop
/// profile at run time with `unipute::naga_backend::write::glsl_with`.
pub fn glsl(source: String) -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Glsl {
        shader: Cow::Owned(source),
        stage: wgpu::naga::ShaderStage::Compute,
        defines: &[],
    }
}

/// A kernel lowered to naga IR at run time, handed to wgpu as a module.
///
/// No shader text is involved at all. This is the shortest path from a
/// kernel to a pipeline, and only works because wgpu and unipute-naga share
/// one naga.
#[cfg(feature = "runtime")]
pub fn naga<K: Kernel>() -> wgpu::ShaderSource<'static> {
    let module = unipute::naga_backend::lower(&K::ir()).expect("the kernel should lower");
    wgpu::ShaderSource::Naga(Cow::Owned(module))
}

/// How many workgroups cover `elements` items along each axis.
pub fn workgroups(elements: [u32; 3], workgroup: [u32; 3]) -> [u32; 3] {
    [
        elements[0].div_ceil(workgroup[0]),
        elements[1].div_ceil(workgroup[1]),
        elements[2].div_ceil(workgroup[2]),
    ]
}

fn buffer_type(access: Access) -> wgpu::BufferBindingType {
    match access {
        Access::Uniform => wgpu::BufferBindingType::Uniform,
        Access::Read => wgpu::BufferBindingType::Storage { read_only: true },
        Access::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
    }
}
