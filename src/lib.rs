use std::sync::Arc;

use egui::{Color32, CtxRef, Rect};
use epaint::{ClippedMesh, ClippedShape};
use vulkano::buffer::{BufferSlice, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::SubpassContents::Inline;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, DrawIndexedError, DynamicState,
};
use vulkano::descriptor::descriptor_set::{
    DescriptorSet, PersistentDescriptorSet, PersistentDescriptorSetBuildError,
    PersistentDescriptorSetError,
};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::image::Dimensions::Dim2d;
use vulkano::image::{ImageCreationError, ImmutableImage, MipmapsCount};
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Scissor;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineCreationError};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError};
use vulkano::sync::{FlushError, GpuFuture};

pub mod shaders;

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
use vulkano::memory::DeviceMemoryAllocError;

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
    IncorrectDefinition(#[from] PersistentDescriptorSetError),
    #[error(transparent)]
    BuildFailed(#[from] PersistentDescriptorSetBuildError),
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

pub type EguiPipeline<Rp> = GraphicsPipeline<
    SingleBufferDefinition<Vertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Rp,
>;

pub struct Painter<Rp: RenderPassAbstract> {
    pub texture_version: u64,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pipeline: Arc<EguiPipeline<Rp>>,
    pub subpass: Subpass<Rp>,
    pub sampler: Arc<Sampler>,
    pub set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}

impl<Rp: RenderPassAbstract + Clone + Send + Sync + 'static> Painter<Rp> {
    /// This is all you need to render your gui.
    /// Pass in your vulkano `Device`, `Queue` and the `Subpass`
    /// that you want to use to render the gui
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass<Rp>,
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
        let texture = egui_ctx.texture();
        if texture.version == self.texture_version {
            return Ok(());
        }
        self.texture_version = texture.version;

        let layout = self.pipeline.layout().descriptor_set_layout(0).unwrap();
        let image = create_font_texture(self.queue.clone(), texture)?;

        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(image, self.sampler.clone())?
                .build()?,
        );

        self.set = Some(set);
        Ok(())
    }

    /// Pass in the `ClippedShape`s that egui gives us to draw the gui
    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder,
        dynamic_state: &DynamicState,
        window_size_points: [f32; 2],
        egui_ctx: &CtxRef,
        clipped_shapes: Vec<ClippedShape>,
    ) -> Result<(), DrawError> {
        self.update_set(egui_ctx)?;
        builder.next_subpass(Inline)?;
        let clipped_meshes: Vec<ClippedMesh> = egui_ctx.tessellate(clipped_shapes);
        let num_meshes = clipped_meshes.len();
        let mut verts = Vec::<Vertex>::with_capacity(num_meshes * 4);
        let mut indices = Vec::<u32>::with_capacity(num_meshes * 6);
        let mut clips = Vec::<Rect>::with_capacity(num_meshes);
        let mut offsets = Vec::<(usize, usize)>::with_capacity(num_meshes);

        for cm in clipped_meshes.iter() {
            offsets.push((verts.len(), indices.len()));
            let (clip, mesh) = (cm.0, &cm.1);

            for v in mesh.vertices.iter() {
                verts.push(v.into());
            }

            for i in mesh.indices.iter() {
                indices.push(*i);
            }

            clips.push(clip);
        }
        offsets.push((verts.len(), indices.len()));

        let (vertex_buf, index_buf) = self.create_buffers((verts, indices))?;
        for (idx, clip) in clips.iter().enumerate() {
            let mut ds = dynamic_state.clone();
            let mut scissors = Vec::with_capacity(1);
            let o = clip.min;
            let (w, h) = (clip.width() as u32, clip.height() as u32);
            scissors.push(Scissor {
                origin: [o.x as i32, o.y as i32],
                dimensions: [w, h],
            });
            ds.scissors = Some(scissors);

            let offset = offsets[idx];
            let end = offsets[idx + 1];

            //let vb_slice = vb.clone().slice(offset.0..end.0).unwrap(); does not work
            let vb_slice = BufferSlice::from_typed_buffer_access(vertex_buf.clone())
                .slice(offset.0..end.0)
                .unwrap();
            let ib_slice = BufferSlice::from_typed_buffer_access(index_buf.clone())
                .slice(offset.1..end.1)
                .unwrap();

            builder.draw_indexed(
                self.pipeline.clone(),
                &ds,
                vb_slice,
                ib_slice,
                self.set.as_ref().unwrap().clone(),
                window_size_points,
                vec![],
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
fn create_pipeline<Rp: RenderPassAbstract>(
    device: Arc<Device>,
    subpass: Subpass<Rp>,
) -> Result<Arc<EguiPipeline<Rp>>, GraphicsPipelineCreationError> {
    let vs = shaders::vs::Shader::load(device.clone()).unwrap();
    let fs = shaders::fs::Shader::load(device.clone()).unwrap();

    let mut blend = AttachmentBlend::alpha_blending();
    blend.color_source = BlendFactor::One;

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_scissors_dynamic(1)
            .fragment_shader(fs.main_entry_point(), ())
            .cull_mode_disabled()
            .blend_collective(blend)
            .render_pass(subpass)
            .build(device.clone())?,
    );
    Ok(pipeline)
}

/// Create a texture sampler for the egui font texture
fn create_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
    Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Linear,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        0.0,
        1.0,
        0.0,
        0.0,
    )
}

type EguiTexture = ImmutableImage<Format>;

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
    texture: Arc<epaint::Texture>,
) -> Result<Arc<EguiTexture>, CreateTextureError> {
    let dimensions = Dim2d {
        width: texture.width as u32,
        height: texture.height as u32,
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
        //Format::R8G8B8A8Srgb,
        Format::R8G8B8A8Unorm, //TODO: verify what the right format is. For now this looks better.
        queue,
    )?;

    image_future.flush()?;
    Ok(image)
}
