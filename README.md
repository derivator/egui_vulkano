# Vulkano backend for egui

[![Latest version](https://img.shields.io/crates/v/egui_vulkano.svg)](https://crates.io/crates/egui_vulkano)
[![Documentation](https://docs.rs/egui_vulkano/badge.svg)](https://docs.rs/egui_vulkano)

This is a drawing backend to use [egui](https://github.com/emilk/egui) with [Vulkano](https://github.com/vulkano-rs/vulkano).
It can be used with [egui-winit](https://github.com/emilk/egui/tree/master/egui-winit) for input handling.

## Usage

```rust
let mut egui_painter = egui_vulkano::Painter::new(
    device.clone(), // your Vulkano Device
    queue.clone(), // your Vulkano Queue
    Subpass::from(render_pass.clone(), 1).unwrap(), // subpass that you set up to render the gui
)
.unwrap();

// ...

// Get the shapes from egui
let egui_output = egui_ctx.end_frame();
let result = egui_painter.update_textures(egui_output.textures_delta, &mut builder).unwrap();
// If using egui-winit:
egui_winit.handle_platform_output(surface.window(), &egui_ctx, egui_output.platform_output);

// Do your own rendering ...

// Automatically start the next render subpass and draw the gui
egui_painter
    .draw(
        &mut builder, // your vulkano AutoCommandBufferBuilder
        [width, height], // window size
        &egui_ctx, // your egui Context
        egui_output.shapes // shapes from egui,
    )
    .unwrap();
```

Check the included working [example](examples/example.rs) for more info.

**Pull requests are welcome!**

## Credits
With inspiration from
[egui_winit_ash_vk_mem](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem),
[egui_sdl2_gl](https://github.com/ArjunNair/egui_sdl2_gl) and
[egui_glium](https://github.com/emilk/egui/tree/master/egui_glium).
