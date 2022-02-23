//! [egui](https://docs.rs/egui) rendering backend for [Vulkano](https://docs.rs/vulkano).
#![warn(missing_docs)]
use std::collections::HashMap;
use std::default::Default;
use std::sync::Arc;

use egui::epaint::{textures::TexturesDelta, ClippedMesh, ClippedShape, ImageData, ImageDelta};
use egui::{Color32, Context, Rect, TextureId};
use vulkano::buffer::{BufferAccess, BufferSlice, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::SubpassContents::Inline;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, CopyBufferImageError,
    DrawIndexedError, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
    DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{
    ImageCreateFlags, ImageCreationError, ImageDimensions, ImageUsage, StorageImage,
};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::{GraphicsPipeline, GraphicsPipelineCreationError};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreationError, SamplerMipmapMode,
};

mod shaders;

#[derive(Default, Debug, Clone)]
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
use vulkano::command_buffer::pool::CommandPoolBuilderAlloc;
use vulkano::image::view::{ImageView, ImageViewCreationError};
use vulkano::memory::DeviceMemoryAllocError;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::render_pass::Subpass;

#[derive(Error, Debug)]
pub enum PainterCreationError {
    #[error(transparent)]
    CreatePipelineFailed(#[from] GraphicsPipelineCreationError),
    #[error(transparent)]
    CreateSamplerFailed(#[from] SamplerCreationError),
}

#[derive(Error, Debug)]
pub enum UpdateTexturesError {
    #[error(transparent)]
    CreateImageViewFailed(#[from] ImageViewCreationError),
    #[error(transparent)]
    BuildFailed(#[from] DescriptorSetCreationError),
    #[error(transparent)]
    Alloc(#[from] DeviceMemoryAllocError),
    #[error(transparent)]
    Copy(#[from] CopyBufferImageError),
    #[error(transparent)]
    CreateImage(#[from] ImageCreationError),
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    UpdateSetFailed(#[from] UpdateTexturesError),
    #[error(transparent)]
    NextSubpassFailed(#[from] AutoCommandBufferBuilderContextError),
    #[error(transparent)]
    CreateBuffersFailed(#[from] DeviceMemoryAllocError),
    #[error(transparent)]
    DrawIndexedFailed(#[from] DrawIndexedError),
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
    device: Arc<Device>,
    queue: Arc<Queue>,
    /// Graphics pipeline used to render the gui.
    pub pipeline: Arc<GraphicsPipeline>,
    /// Texture sampler used to render the gui.
    pub sampler: Arc<Sampler>,
    images: HashMap<egui::TextureId, Arc<StorageImage>>,
    texture_sets: HashMap<egui::TextureId, Arc<PersistentDescriptorSet>>,
    texture_free_queue: Vec<egui::TextureId>,
}

impl Painter {
    /// Pass in the vulkano [`Device`], [`Queue`] and [`Subpass`]
    /// that you want to use to render the gui.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
    ) -> Result<Self, PainterCreationError> {
        let pipeline = create_pipeline(device.clone(), subpass.clone())?;
        let sampler = create_sampler(device.clone())?;
        Ok(Self {
            device,
            queue,
            pipeline,
            sampler,
            images: Default::default(),
            texture_sets: Default::default(),
            texture_free_queue: Vec::new(),
        })
    }

    fn write_image_delta<P>(
        &mut self,
        image: Arc<StorageImage>,
        delta: &ImageDelta,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>,
    ) -> Result<(), UpdateTexturesError>
    where
        P: CommandPoolBuilderAlloc,
    {
        let image_data = match &delta.image {
            ImageData::Color(image) => image
                .pixels
                .iter()
                .flat_map(|c| c.to_array())
                .collect::<Vec<_>>(),
            ImageData::Alpha(image) => image
                .pixels
                .iter()
                .flat_map(|&r| vec![r, r, r, r])
                .collect::<Vec<_>>(),
        };
        let img_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::transfer_source(),
            false,
            image_data,
        )?;

        let size = [delta.image.width() as u32, delta.image.height() as u32, 1];
        let offset = match delta.pos {
            None => [0, 0, 0],
            Some(pos) => [pos[0] as u32, pos[1] as u32, 0],
        };

        builder.copy_buffer_to_image_dimensions(img_buffer, image, offset, size, 0, 1, 0)?;
        Ok(())
    }

    /// Uploads all newly created and modified textures to the GPU.
    /// Has to be called before entering the first render pass.  
    /// If the return value is [`UpdateTexturesResult::Changed`],
    /// a texture will be changed in this frame and you need to wait for the last frame to finish
    /// before submitting the command buffer for this frame.
    pub fn update_textures<P>(
        &mut self,
        textures_delta: TexturesDelta,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>,
    ) -> Result<UpdateTexturesResult, UpdateTexturesError>
    where
        P: CommandPoolBuilderAlloc,
    {
        for texture_id in textures_delta.free {
            self.texture_free_queue.push(texture_id);
        }

        let mut result = UpdateTexturesResult::Unchanged;

        for (texture_id, delta) in &textures_delta.set {
            let image = if delta.is_whole() {
                let image = create_image(self.queue.clone(), &delta.image)?;
                let layout = &self.pipeline.layout().descriptor_set_layouts()[0];

                let set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new(image.clone())?,
                        self.sampler.clone(),
                    )],
                )?;

                self.texture_sets.insert(*texture_id, set);
                self.images.insert(*texture_id, image.clone());
                image
            } else {
                result = UpdateTexturesResult::Changed; //modifying an existing image that might be in use
                self.images[texture_id].clone()
            };
            self.write_image_delta(image, delta, builder)?;
        }

        Ok(result)
    }

    /// Free textures freed by egui, *after* drawing
    fn free_textures(&mut self) {
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
    ) -> Result<(), DrawError>
    where
        P: CommandPoolBuilderAlloc,
    {
        builder
            .next_subpass(Inline)?
            .bind_pipeline_graphics(self.pipeline.clone());

        let clipped_meshes: Vec<ClippedMesh> = egui_ctx.tessellate(clipped_shapes);
        let num_meshes = clipped_meshes.len();

        let mut verts = Vec::<Vertex>::with_capacity(num_meshes * 4);
        let mut indices = Vec::<u32>::with_capacity(num_meshes * 6);
        let mut clips = Vec::<Rect>::with_capacity(num_meshes);
        let mut texture_ids = Vec::<TextureId>::with_capacity(num_meshes);
        let mut offsets = Vec::<(usize, usize)>::with_capacity(num_meshes);

        for cm in clipped_meshes.iter() {
            let (clip, mesh) = (cm.0, &cm.1);

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

        let (vertex_buf, index_buf) = self.create_buffers((verts, indices))?;
        for (idx, clip) in clips.iter().enumerate() {
            let mut scissors = Vec::with_capacity(1);
            let o = clip.min;
            let (w, h) = (clip.width() as u32, clip.height() as u32);
            scissors.push(Scissor {
                origin: [(o.x as u32), (o.y as u32)],
                dimensions: [w, h],
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
                .bind_vertex_buffers(0, vb_slice.clone())
                .bind_index_buffer(ib_slice.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    texture_set.unwrap().clone(),
                )
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
        DeviceMemoryAllocError,
    > {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            triangles.0.iter().cloned(),
        )?;

        let index_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::index_buffer(),
            false,
            triangles.1.iter().cloned(),
        )?;

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

    let mut blend = AttachmentBlend::alpha();
    blend.color_source = BlendFactor::One;

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
    Sampler::start(device.clone())
        .mag_filter(Filter::Linear)
        .min_filter(Filter::Linear)
        .mipmap_mode(SamplerMipmapMode::Linear)
        .address_mode_u(SamplerAddressMode::ClampToEdge)
        .address_mode_v(SamplerAddressMode::ClampToEdge)
        .address_mode_w(SamplerAddressMode::ClampToEdge)
        .mip_lod_bias(0.0)
        .anisotropy(Some(1.0))
        .min_lod(0.0)
        .build()
}

/// Create a Vulkano image for the given egui texture
fn create_image(
    queue: Arc<Queue>,
    texture: &ImageData,
) -> Result<Arc<StorageImage>, ImageCreationError> {
    let dimensions = ImageDimensions::Dim2d {
        width: texture.width() as u32,
        height: texture.height() as u32,
        array_layers: 1,
    };

    let format = match texture {
        ImageData::Color(_) => Format::R8G8B8A8_SRGB,
        ImageData::Alpha(_) => Format::R8G8B8A8_UNORM,
    };

    let usage = ImageUsage {
        transfer_destination: true,
        sampled: true,
        storage: false,
        ..ImageUsage::none()
    };

    let image = StorageImage::with_usage(
        queue.device().clone(),
        dimensions,
        format,
        usage,
        ImageCreateFlags::none(),
        [queue.family()],
    )?;

    Ok(image)
}
