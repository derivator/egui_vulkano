/// The shaders used to render the gui

/// The vertex shader
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/vert.vert"
    }
}

/// The fragment shader
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/frag.frag"
    }
}
