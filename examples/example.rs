// Copied from https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/triangle.rs
/// Differences:
/// * Set the correct color format for the swapchain
/// * Second renderpass to draw the gui
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, FontDefinitions, Ui};
use egui_winit_platform::{Platform, PlatformDescriptor};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::swapchain::{AcquireError, ColorSpace, Swapchain, SwapchainCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{swapchain, sync, Version};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

fn main() {
    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, Version::V1_0, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name.as_ref().unwrap(),
        physical.properties().device_type.unwrap(),
    );

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1024, 768))
        .with_title("egui_vulkano demo")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Set the swapchain format to Srgb to get correct colors for egui
        assert!(&caps
            .supported_formats
            .contains(&(Format::B8G8R8A8Srgb, ColorSpace::SrgbNonLinear)));
        let format = Format::B8G8R8A8Srgb;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .build()
            .unwrap()
    };

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

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

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            passes: [
                { color: [color], depth_stencil: {}, input: [] },
                { color: [color], depth_stencil: {}, input: [] } // Create a second renderpass to draw egui
            ]
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    //Set up everything need to draw the gui
    let window = surface.window();
    let size = window.inner_size();
    let mut egui_platform = Platform::new(PlatformDescriptor {
        physical_width: size.width as u32,
        physical_height: size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    )
    .unwrap();

    //Set up some window to look at for the test

    let mut egui_test = egui_demo_lib::ColorTest::default();
    let mut egui_bench = Benchmark::new(1000);

    let start_time = Instant::now();
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        egui_platform.handle_event(&event); //Let the gui handle events
        if egui_platform.captures_event(&event) {
            return; //Egui wants to handle this event, so ignore it
        }

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
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    );
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

                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // Do your usual rendering
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        (),
                        (),
                        vec![],
                    )
                    .unwrap(); // Don't end the render pass yet

                // Build your gui
                egui_bench.push(last_frame.elapsed().as_secs_f64());
                last_frame = Instant::now();
                egui_platform.update_time(start_time.elapsed().as_secs_f64());
                egui_platform.begin_frame();

                egui::Window::new("Color test")
                    .scroll(true)
                    .show(&egui_platform.context(), |ui| {
                        let mut none = None;
                        egui_test.ui(ui, &mut none);
                    });

                egui::Window::new("Benchmark").default_height(600.0).show(
                    &egui_platform.context(),
                    |ui| {
                        egui_bench.draw(ui);
                    },
                );

                // Get the shapes from egui
                let (_output, clipped_shapes) = egui_platform.end_frame();

                // Automatically start the next render subpass and draw the gui
                egui_painter
                    .draw(
                        &mut builder,
                        &dynamic_state,
                        [size.width as f32, size.height as f32],
                        &egui_platform.context(),
                        clipped_shapes,
                    )
                    .unwrap();

                // End the render pass as usual
                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
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
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
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
        // As of egui 0.13.0 it is impossible to set the color of an HLine
        // see https://github.com/emilk/egui/issues/524
        // let red = Stroke::new(2.0, Color32::RED);
        let zero = HLine::new(0.0);
        let target = HLine::new(1000.0 / 60.0);

        let plot = Plot::new("plot")
            .height(300.0)
            .width(700.0)
            .line(curve)
            .hline(zero)
            .hline(target);

        ui.label("Time in milliseconds that each frame took to draw:");
        ui.add(plot);
        ui.label("The light blue line marks the frametime target for drawing at 60 FPS.");
    }

    pub fn push(&mut self, v: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }
}
