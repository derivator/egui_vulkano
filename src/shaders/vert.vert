#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

layout(push_constant) uniform PushConstants { vec2 screen_size; } pushConstants;

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

void main() {
gl_Position =
      vec4(2.0 * pos.x / pushConstants.screen_size.x - 1.0,
           2.0 * pos.y / pushConstants.screen_size.y - 1.0, 0.0, 1.0);
  outColor = linear_from_srgba(color);
  outUV = uv;
}
