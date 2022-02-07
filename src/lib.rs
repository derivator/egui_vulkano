//! [egui](https://docs.rs/egui) rendering backend for [Vulkano](https://docs.rs/vulkano).

use std::sync::Arc;

use egui::{Color32, CtxRef, Rect};
use epaint::{ClippedMesh, ClippedShape};
use vulkano::buffer::{BufferSlice, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::SubpassContents::Inline;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, DrawIndexedError,
    PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
    PersistentDescriptorSet, WriteDescriptorSet, DescriptorSetCreationError
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageCreationError, ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::{
    GraphicsPipeline, GraphicsPipelineCreationError,
};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreationError, SamplerMipmapMode};
use vulkano::sync::{FlushError, GpuFuture};

mod shaders;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl From<&epaint::Vertex> for Vertex {
    fn from(v: &epaint::Vertex) -> Self {
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
pub enum UpdateSetError {
    #[error(transparent)]
    CreateTextureFailed(#[from] CreateTextureError),
    #[error(transparent)]
    CreateImageViewFailed(#[from] ImageViewCreationError),
    #[error(transparent)]
    BuildFailed(#[from] DescriptorSetCreationError),
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    UpdateSetFailed(#[from] UpdateSetError),
    #[error(transparent)]
    NextSubpassFailed(#[from] AutoCommandBufferBuilderContextError),
    #[error(transparent)]
    CreateBuffersFailed(#[from] DeviceMemoryAllocError),
    #[error(transparent)]
    DrawIndexedFailed(#[from] DrawIndexedError),
}

pub type EguiPipeline = GraphicsPipeline;

/// Contains everything needed to render the gui.
pub struct Painter {
    pub texture_version: u64,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pipeline: Arc<EguiPipeline>,
    pub subpass: Subpass,
    pub sampler: Arc<Sampler>,
    pub set: Option<Arc<PersistentDescriptorSet>>,
}

impl Painter {
    /// Pass in your vulkano `Device`, `Queue` and the `Subpass`
    /// that you want to use to render the gui.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
    ) -> Result<Self, PainterCreationError> {
        let pipeline = create_pipeline(device.clone(), subpass.clone())?;
        let sampler = create_sampler(device.clone())?;
        Ok(Self {
            texture_version: 0,
            device,
            queue,
            pipeline,
            subpass,
            sampler,
            set: None,
        })
    }

    fn update_set(&mut self, egui_ctx: &CtxRef) -> Result<(), UpdateSetError> {
        let texture = egui_ctx.font_image();
        if texture.version == self.texture_version {
            return Ok(());
        }
        self.texture_version = texture.version;

        let layout = &self.pipeline.layout().descriptor_set_layouts()[0];
        let image = create_font_texture(self.queue.clone(), texture)?;

        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                ImageView::new(image)?,
                self.sampler.clone()
            )],
        )?; 

        self.set = Some(set);
        Ok(())
    }

    /// Pass in the `ClippedShape`s that egui gives us to draw the gui.
    pub fn draw<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>,
        window_size_points: [f32; 2],
        egui_ctx: &CtxRef,
        clipped_shapes: Vec<ClippedShape>,
    ) -> Result<(), DrawError>
    where
        P: CommandPoolBuilderAlloc,
    {
        self.update_set(egui_ctx)?;
        builder.next_subpass(Inline)?
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.set.as_ref().unwrap().clone(),
            );
        let clipped_meshes: Vec<ClippedMesh> = egui_ctx.tessellate(clipped_shapes);
        let num_meshes = clipped_meshes.len();

        let mut verts = Vec::<Vertex>::with_capacity(num_meshes * 4);
        let mut indices = Vec::<u32>::with_capacity(num_meshes * 6);
        let mut clips = Vec::<Rect>::with_capacity(num_meshes);
        let mut offsets = Vec::<(usize, usize)>::with_capacity(num_meshes);

        for cm in clipped_meshes.iter() {
            let (clip, mesh) = (cm.0, &cm.1);

            // There's an incredibly weird edge case where epaint
            // will give us meshes with no actual content in them.
            // In that case, we skip rendering the mesh.
            // This also fixes a crash within vulkano that occurs
            // if we try to initialize a buffer with a length of 0
            // and then later try to slice into it (vulkano forces
            // a minimum size of 1 for all buffers, breaking an
            // assertion for self.size() / mem::size_of::<T>()).
            if mesh.vertices.len() == 0 || mesh.indices.len() == 0 {
                continue;
            }

            offsets.push((verts.len(), indices.len()));

            for v in mesh.vertices.iter() {
                verts.push(v.into());
            }

            for i in mesh.indices.iter() {
                indices.push(*i);
            }

            clips.push(clip);
        }
        offsets.push((verts.len(), indices.len()));

        // Small optimization: If there's nothing to render,
        // return here instead of taking time to create an
        // empty (1 byte) buffer.
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

            //let vb_slice = vb.clone().slice(offset.0..end.0).unwrap(); does not work
            let vb_slice = BufferSlice::from_typed_buffer_access(vertex_buf.clone())
                .slice(offset.0 as u64..end.0 as u64)
                .unwrap();
            let ib_slice = BufferSlice::from_typed_buffer_access(index_buf.clone())
                .slice(offset.1 as u64..end.1 as u64)
                .unwrap();

            builder.bind_vertex_buffers(0, vb_slice.clone())
                .bind_index_buffer(ib_slice.clone())
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    window_size_points
                )
                .draw_indexed(
                    ib_slice.len() as u32,
                    1,
                    0,
                    0,
                    0
                )?;
        }
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
) -> Result<Arc<EguiPipeline>, GraphicsPipelineCreationError> {
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
        .rasterization_state(
            RasterizationState::new()
                .cull_mode(CullMode::None)
        )
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend(blend))
        .render_pass(subpass)
        .build(device.clone())?;
    Ok(pipeline)
}

/// Create a texture sampler for the egui font texture
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

type EguiTexture = ImmutableImage;

#[derive(Debug, Error)]
pub enum CreateTextureError {
    #[error(transparent)]
    CreateImageFailed(#[from] ImageCreationError),
    #[error(transparent)]
    FlushFailed(#[from] FlushError),
}

/// Create an image containing the egui font texture
fn create_font_texture(
    queue: Arc<Queue>,
    texture: Arc<epaint::FontImage>,
) -> Result<Arc<EguiTexture>, CreateTextureError> {
    let dimensions = ImageDimensions::Dim2d {
        width: texture.width as u32,
        height: texture.height as u32,
        array_layers: 1,
    };

    let image_data = &texture
        .pixels
        .iter()
        .flat_map(|&r| vec![r, r, r, r])
        .collect::<Vec<_>>();

    let (image, image_future) = ImmutableImage::from_iter(
        image_data.iter().cloned(),
        dimensions,
        MipmapsCount::One,
        Format::R8G8B8A8_UNORM, // &texture.pixels uses linear color space
        queue,
    )?;

    image_future.flush()?;
    Ok(image)
}
