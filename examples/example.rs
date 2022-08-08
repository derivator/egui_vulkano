// Copied from https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/triangle.rs
/// Differences:
/// * Set the correct color format for the swapchain
/// * Second renderpass to draw the gui
use std::collections::VecDeque;
use std::convert::TryInto;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, ColorImage, Ui};
use egui_vulkano::UpdateTexturesResult;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError};
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture};
use vulkano::{swapchain, sync};
use vulkano::instance::debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowBuilder};

pub struct FrameEndFuture(Box<dyn GpuFuture>);

impl FrameEndFuture {
    pub fn now(device: Arc<Device>) -> Self {
        Self(sync::now(device).boxed())
    }

    pub fn get(self) -> Box<dyn GpuFuture> {
        self.0
    }
}

impl AsMut<dyn GpuFuture> for FrameEndFuture {
    fn as_mut(&mut self) -> &mut (dyn GpuFuture + 'static) {
        self.0.as_mut()
    }
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    for layer in vulkano::instance::layers_list().unwrap() {
        println!("Available layer: {}", layer.name());
    }
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        enabled_layers: vec!["VK_LAYER_KHRONOS_validation".into()],
        ..Default::default()
    }).unwrap();
    let _callback = unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                println!("Debug callback: {:?}", msg.description);
            })),
        ).ok()
    };
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type,
    );

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("egui_vulkano demo")
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
        .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let image_format = Some(Format::B8G8R8A8_SRGB);
        let image_extent: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent,
                image_usage: ImageUsage::color_attachment(),
                composite_alpha,

                ..Default::default()
            },
        )
            .unwrap()
    };

    #[derive(Default, Debug, Clone, Copy, Pod, Zeroable)]
    #[repr(C)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-0.5, -0.25],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.25, -0.1],
                },
            ]
                .iter()
                .cloned(),
        )
            .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        passes: [
            { color: [color], depth_stencil: {}, input: [] },
            { color: [color], depth_stencil: {}, input: [] } // Create a second renderpass to draw egui
        ]
    )
        .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone().into(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;

    let mut previous_frame_end: Option<Box<dyn GpuFuture>> = Some(sync::now(device.clone()).boxed());

    //Set up everything need to draw the gui
    let window = surface.window();
    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(4096, window);

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    )
        .unwrap();

    //Set up some window to look at for the test

    let mut egui_test = egui_demo_lib::ColorTest::default();
    let mut demo_windows = egui_demo_lib::DemoWindows::default();
    let mut egui_bench = Benchmark::new(1000);
    let mut my_texture = egui_ctx.load_texture("my_texture", ColorImage::example());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent { event, .. } => {
                let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                if !egui_consumed_event {
                    // do your own event handling here
                };
            }
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: surface.window().inner_size().into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![ClearValue::Float([0.0, 0.0, 1.0, 1.0]).into()];
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                    .unwrap();

                let frame_start = Instant::now();
                egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));
                demo_windows.ui(&egui_ctx);

                egui::Window::new("Color test")
                    .vscroll(true)
                    .show(&egui_ctx, |ui| {
                        egui_test.ui(ui);
                    });

                egui::Window::new("Settings").show(&egui_ctx, |ui| {
                    egui_ctx.settings_ui(ui);
                });

                egui::Window::new("Benchmark")
                    .default_height(600.0)
                    .show(&egui_ctx, |ui| {
                        egui_bench.draw(ui);
                    });

                egui::Window::new("Texture test").show(&egui_ctx, |ui| {
                    ui.image(my_texture.id(), (200.0, 200.0));
                    if ui.button("Reload texture").clicked() {
                        // previous TextureHandle is dropped, causing egui to free the texture:
                        my_texture = egui_ctx.load_texture("my_texture", ColorImage::example());
                    }
                });

                // Get the shapes from egui
                let egui_output = egui_ctx.end_frame();
                let platform_output = egui_output.platform_output;
                egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

                let tex_future = egui_painter
                    .update_textures(egui_output.textures_delta)
                    .expect("egui texture error");

                // Do your usual rendering
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values,
                            ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len().try_into().unwrap(), 1, 0, 0)
                    .unwrap(); // Don't end the render pass yet

                // Build your gui

                // Automatically start the next render subpass and draw the gui
                let size = surface.window().inner_size().to_logical(surface.window().scale_factor());
                egui_painter
                    .draw(
                        &mut builder,
                        [size.width, size.height],
                        &egui_ctx,
                        egui_output.shapes,
                    )
                    .unwrap();

                egui_bench.push(frame_start.elapsed().as_secs_f64());

                // End the render pass as usual
                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();
                let mut future_mut = acquire_future.join(tex_future).boxed();
                if let Some(future) = previous_frame_end.take() {
                    future_mut = future_mut.join(future).boxed();
                }
                let future = future_mut
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        future.wait(None).unwrap();
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

pub struct Benchmark {
    capacity: usize,
    data: VecDeque<f64>,
}

impl Benchmark {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: VecDeque::with_capacity(capacity),
        }
    }

    pub fn draw(&self, ui: &mut Ui) {
        let iter = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| Value::new(i as f64, *v * 1000.0));
        let curve = Line::new(Values::from_values_iter(iter)).color(Color32::BLUE);
        let target = HLine::new(1000.0 / 60.0).color(Color32::RED);

        ui.label("Time in milliseconds that the gui took to draw:");
        Plot::new("plot")
            .view_aspect(2.0)
            .include_y(0)
            .show(ui, |plot_ui| {
                plot_ui.line(curve);
                plot_ui.hline(target)
            });
        ui.label("The red line marks the frametime target for drawing at 60 FPS.");
    }

    pub fn push(&mut self, v: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }
}
