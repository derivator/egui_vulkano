//! [egui](https://docs.rs/egui) rendering backend for [Vulkano](https://docs.rs/vulkano).
// #![warn(missing_docs)]
use std::collections::HashMap;
use std::default::Default;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui::epaint::{
    textures::TexturesDelta, ClippedPrimitive, ClippedShape, ImageData, ImageDelta, Primitive,
};
use egui::{Color32, Context, Rect, TextureId};
use vulkano::buffer::{BufferAccess, BufferSlice, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::SubpassContents::Inline;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, CopyImageInfo, BufferImageCopy, CommandBufferBeginError,
};
use vulkano::descriptor_set::{
    DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::immutable::ImmutableImageCreationError;
use vulkano::image::{
    ImageCreateFlags, ImageDimensions, ImageUsage, ImmutableImage, ImageLayout, MipmapsCount, ImageAccess
};
use vulkano::image::sys::ImageError;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState, BlendOp};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState, Viewport};
use vulkano::pipeline::graphics::{GraphicsPipeline, GraphicsPipelineCreationError};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sampler::{
    Filter, Sampler, SamplerCreateInfo, SamplerCreationError, SamplerMipmapMode,
};
mod shaders;

#[derive(Default, Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl From<&egui::epaint::Vertex> for Vertex {
    fn from(v: &egui::epaint::Vertex) -> Self {
        let convert = {
            |c: Color32| {
                [
                    c.r() as f32 / 255.0,
                    c.g() as f32 / 255.0,
                    c.b() as f32 / 255.0,
                    c.a() as f32 / 255.0,
                ]
            }
        };

        Self {
            pos: [v.pos.x, v.pos.y],
            uv: [v.uv.x, v.uv.y],
            color: convert(v.color),
        }
    }
}

vulkano::impl_vertex!(Vertex, pos, uv, color);

use thiserror::Error;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::image::view::{ImageView, ImageViewCreationError};
use vulkano::memory::allocator::AllocationCreationError;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::render_pass::Subpass;
use vulkano::command_buffer::RenderPassError;
use vulkano::command_buffer::CopyError;
use vulkano::command_buffer::PipelineExecutionError;

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum PainterCreationError {
    #[error(transparent)]
    CreatePipelineFailed(#[from] GraphicsPipelineCreationError),
    #[error(transparent)]
    CreateSamplerFailed(#[from] SamplerCreationError),
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum UpdateTexturesError {
    #[error(transparent)]
    CreateImageViewFailed(#[from] ImageViewCreationError),
    #[error(transparent)]
    BuildFailed(#[from] DescriptorSetCreationError),
    #[error(transparent)]
    Alloc(#[from] AllocationCreationError),
    #[error(transparent)]
    Copy(#[from] CopyError),
    #[error(transparent)]
    CreateImage(#[from] ImageError),
    #[error(transparent)]
    CommandBufferError(#[from] CommandBufferBeginError),
    #[error(transparent)]
    ImmutableImageCreateError(#[from] ImmutableImageCreationError),
    #[error("No Egui State")]
    NoEguiState
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    UpdateSetFailed(#[from] UpdateTexturesError),
    #[error(transparent)]
    CreateBuffersFailed(#[from] AllocationCreationError),
    #[error(transparent)]
    RenderPassError(#[from] RenderPassError),
    #[error(transparent)]
    PipelineError(#[from] PipelineExecutionError),
}

#[must_use = "You must use this to avoid attempting to modify a texture that's still in use"]
#[derive(PartialEq)]
/// You must use this to avoid attempting to modify a texture that's still in use.
pub enum UpdateTexturesResult {
    /// No texture will be modified in this frame.
    Unchanged,
    /// A texture will be modified in this frame,
    /// and you must wait for the last frame to finish before submitting the next command buffer.
    Changed,
}

/// Contains everything needed to render the gui.
pub struct Painter {
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Graphics pipeline used to render the gui.
    pub pipeline: Arc<GraphicsPipeline>,
    /// Texture sampler used to render the gui.
    pub sampler: Arc<Sampler>,
    images: HashMap<egui::TextureId, Arc<ImmutableImage>>,
    texture_sets: HashMap<egui::TextureId, Arc<PersistentDescriptorSet>>,
    texture_free_queue: Vec<egui::TextureId>,
}

impl Painter {
    /// Pass in the vulkano [`Device`], [`Queue`] and [`Subpass`]
    /// that you want to use to render the gui.
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        queue: Arc<Queue>,
        subpass: Subpass,
    ) -> Result<Self, PainterCreationError> {
        let pipeline = create_pipeline(device.clone(), subpass.clone())?;
        let sampler = create_sampler(device.clone())?;
        Ok(Self {
            queue,
            memory_allocator,
            descriptor_set_allocator,
            pipeline,
            sampler,
            images: Default::default(),
            texture_sets: Default::default(),
            texture_free_queue: Vec::new(),
        })
    }

    /// Uploads all newly created and modified textures to the GPU.
    /// Has to be called before entering the first render pass.  
    /// If the return value is [`UpdateTexturesResult::Changed`],
    /// a texture will be changed in this frame and you need to wait for the last frame to finish
    /// before submitting the command buffer for this frame.
    pub fn update_textures(
        &mut self,
        textures_delta: TexturesDelta,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<UpdateTexturesResult, UpdateTexturesError>
    {
        for texture_id in textures_delta.free {
            self.texture_free_queue.push(texture_id);
        }
    
        let mut result = UpdateTexturesResult::Unchanged;

        for (texture_id, delta) in &textures_delta.set {
            let image = if let Some(image) = self.images.remove(texture_id) {
                result = UpdateTexturesResult::Changed; //modifying an existing image that might be in use
                if delta.is_whole() {
                    create_immutable_image_full(
                        &self.queue,
                        &delta.image,
                        // &mut cbb,
                        builder,
                        self.memory_allocator.clone()
                    )?
                } else {
                    create_immutable_image_part(
                        &self.queue,
                        &delta,
                        &image,
                        // &mut cbb,
                        builder,
                        self.memory_allocator.clone()
                    )?
                }
            } else {
                create_immutable_image_full(
                    &self.queue,
                    &delta.image,
                    // &mut cbb,
                    builder,
                    self.memory_allocator.clone()
                )?
            };
            let layout = &self.pipeline.layout().set_layouts()[0];

            let set = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                layout.clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    ImageView::new_default(image.clone())?,
                    self.sampler.clone(),
                )],
            )?;

            self.texture_sets.insert(*texture_id, set);
            self.images.insert(*texture_id, image.clone());
        }

        Ok(result)
    }

    /// Free textures freed by egui, *after* drawing
    pub fn free_textures(&mut self) {
        for texture_id in &self.texture_free_queue {
            self.texture_sets.remove(texture_id);
            self.images.remove(texture_id);
        }

        self.texture_free_queue.clear();
    }

    /// Advances to the next rendering subpass and uses the [`ClippedShape`]s from [`egui::FullOutput`] to draw the gui.
    pub fn draw<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>,
        window_size_points: [f32; 2],
        egui_ctx: &Context,
        clipped_shapes: Vec<ClippedShape>,
        viewport: Viewport,
    ) -> Result<(), DrawError>
    where
        P: CommandBufferAllocator,
    {
        builder
            .next_subpass(Inline)?
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, [viewport]);

        let clipped_primitives: Vec<ClippedPrimitive> = egui_ctx.tessellate(clipped_shapes);
        let num_meshes = clipped_primitives.len();

        let mut verts = Vec::<Vertex>::with_capacity(num_meshes * 4);
        let mut indices = Vec::<u32>::with_capacity(num_meshes * 6);
        let mut clips = Vec::<Rect>::with_capacity(num_meshes);
        let mut texture_ids = Vec::<TextureId>::with_capacity(num_meshes);
        let mut offsets = Vec::<(usize, usize)>::with_capacity(num_meshes);

        for cm in clipped_primitives.iter() {
            let clip = cm.clip_rect;
            let mesh = match &cm.primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => {
                    continue; // callbacks not supported at the moment
                }
            };

            // Skip empty meshes
            if mesh.vertices.len() == 0 || mesh.indices.len() == 0 {
                continue;
            }

            offsets.push((verts.len(), indices.len()));
            texture_ids.push(mesh.texture_id);

            for v in mesh.vertices.iter() {
                verts.push(v.into());
            }

            for i in mesh.indices.iter() {
                indices.push(*i);
            }

            clips.push(clip);
        }
        offsets.push((verts.len(), indices.len()));

        // Return if there's nothing to render
        if clips.len() == 0 {
            return Ok(());
        }
        let ppp = egui_ctx.pixels_per_point();
        let (vertex_buf, index_buf) = self.create_buffers((verts, indices))?;
        for (idx, clip) in clips.iter().enumerate() {
            let mut scissors = Vec::with_capacity(1);
            let o = clip.min;
            scissors.push(Scissor {
                origin: [(o.x as u32), (o.y as u32)],
                dimensions: [(window_size_points[0] * ppp) as u32, (window_size_points[1] * ppp) as u32],
            });
            builder.set_scissor(0, scissors);

            let offset = offsets[idx];
            let end = offsets[idx + 1];

            let vb_slice = BufferSlice::from_typed_buffer_access(vertex_buf.clone())
                .slice(offset.0 as u64..end.0 as u64)
                .unwrap();
            let ib_slice = BufferSlice::from_typed_buffer_access(index_buf.clone())
                .slice(offset.1 as u64..end.1 as u64)
                .unwrap();

            let texture_set = self.texture_sets.get(&texture_ids[idx]);
            if texture_set.is_none() {
                continue; //skip if we don't have a texture
            }

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    texture_set.unwrap().clone(),
                )
                .bind_vertex_buffers(0, vb_slice.clone())
                .bind_index_buffer(ib_slice.clone())
                .push_constants(self.pipeline.layout().clone(), 0, window_size_points)
                .draw_indexed(ib_slice.len() as u32, 1, 0, 0, 0)?;
        }
        self.free_textures();
        Ok(())
    }

    /// Create vulkano CpuAccessibleBuffer objects for the vertices and indices
    fn create_buffers(
        &self,
        triangles: (Vec<Vertex>, Vec<u32>),
    ) -> Result<
        (
            Arc<CpuAccessibleBuffer<[Vertex]>>,
            Arc<CpuAccessibleBuffer<[u32]>>,
        ),
        AllocationCreationError,
    > {

        let mut vertex_buffer_usage = BufferUsage::empty();
        let mut index_buffer_usage = BufferUsage::empty();

        vertex_buffer_usage.vertex_buffer = true;
        index_buffer_usage.index_buffer = true;

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            vertex_buffer_usage,
            false,
            triangles.0.iter().cloned(),
        )
        .unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            index_buffer_usage,
            false,
            triangles.1.iter().cloned(),
        )
        .unwrap();

        Ok((vertex_buffer, index_buffer))
    }
}

/// Create a graphics pipeline with the shaders and settings necessary to render egui output
fn create_pipeline(
    device: Arc<Device>,
    subpass: Subpass,
) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
    let vs = shaders::vs::load(device.clone()).unwrap();
    let fs = shaders::fs::load(device.clone()).unwrap();

    let blend = AttachmentBlend {
        color_op: BlendOp::Add,
        color_source: BlendFactor::One,
        color_destination: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
        alpha_source: BlendFactor::One,
        alpha_destination: BlendFactor::OneMinusSrcAlpha,
    };

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend(blend))
        .render_pass(subpass)
        .build(device.clone())?;
    Ok(pipeline)
}

/// Create a texture sampler for the textures used by egui
fn create_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
    Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            ..Default::default()
        },
    )
}

fn create_immutable_image_full(
    queue: &Arc<Queue>,
    texture: &ImageData,
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    allocator: Arc<StandardMemoryAllocator>,
) -> Result<Arc<ImmutableImage>, ImmutableImageCreationError> {
    let dimensions = ImageDimensions::Dim2d {
        width: texture.width() as u32,
        height: texture.height() as u32,
        array_layers: 1,
    };

    let format = Format::R8G8B8A8_SRGB;

    let image_data = match texture {
        ImageData::Color(image) => image
            .pixels
            .iter()
            .flat_map(|c| c.to_array())
            .collect::<Vec<_>>(),
        ImageData::Font(image) => image
            .srgba_pixels(Some(1.0))
            .flat_map(|c| c.to_array())
            .collect::<Vec<_>>(),
    };

    let img_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            transfer_src: true,
            ..Default::default()
        },
        false,
        image_data,
    )?;


    let flags = ImageCreateFlags::empty();
    let layout = ImageLayout::ShaderReadOnlyOptimal;
    let usage = ImageUsage {
        transfer_dst: true,
        transfer_src: true,
        sampled: true,
        ..ImageUsage::empty()
    };
    let (image, initializer) = ImmutableImage::uninitialized(
        &allocator,
        dimensions,
        format,
        MipmapsCount::One,
        usage,
        flags,
        layout,
        vec![queue.queue_family_index()],
    )?;

    cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(img_buffer, initializer)).unwrap();

    Ok(image)
}

fn create_immutable_image_part(
    queue: &Arc<Queue>,
    delta: &ImageDelta,
    old: &Arc<ImmutableImage>,
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    allocator: Arc<StandardMemoryAllocator>,
) -> Result<Arc<ImmutableImage>, ImmutableImageCreationError> {

    let image_data = match &delta.image {
        ImageData::Color(image) => image
            .pixels
            .iter()
            .flat_map(|c| c.to_array())
            .collect::<Vec<_>>(),
        ImageData::Font(image) => image
            .srgba_pixels(Some(1.0))
            .flat_map(|c| c.to_array())
            .collect::<Vec<_>>(),
    };

    let img_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            transfer_src: true,
            ..Default::default()
        },
        false,
        image_data,
    )?;

    let flags = ImageCreateFlags::empty();
    let layout = ImageLayout::ShaderReadOnlyOptimal;

    let (image, initializer) = ImmutableImage::uninitialized(
        &allocator,
        old.dimensions(),
        old.format(),
        MipmapsCount::One,
        old.usage().clone(),
        flags,
        layout,
        vec![queue.queue_family_index()],
    )?;

    cbb.copy_image(CopyImageInfo::images(old.clone(), initializer.clone())).unwrap();

    let size = [delta.image.width() as u32, delta.image.height() as u32, 1];
    let offset = match delta.pos {
        None => [0, 0, 0],
        Some(pos) => [pos[0] as u32, pos[1] as u32, 0],
    };
    cbb.copy_buffer_to_image(CopyBufferToImageInfo {
        regions: [
            BufferImageCopy {
                image_extent: size,
                image_offset: offset,
                image_subresource: initializer.subresource_layers(),
                ..Default::default()
            }
        ].into(),
        ..CopyBufferToImageInfo::buffer_image(img_buffer, initializer)
    }).unwrap();
    Ok(image)
}