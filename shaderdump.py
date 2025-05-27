#!/usr/bin/env python3
import moderngl
import numpy as np
from PIL import Image
import argparse
import os

# === PARSE CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Render ShaderToy GLSL shader to a sequence of frames.  Supports any image format supported by the Pillow library (PNG, JPG, BMP, TIF, WEBP, PPM, GIF, HEIF/HEIC).")
parser.add_argument('-i', '--input', required=True, help='Path to GLSL shader file')
parser.add_argument('-o', '--output', default='./out%04d.png', help='Output path pattern (e.g. ./out%%04d.png)')
parser.add_argument('-fps', '--framesPerSecond', type=int, default=60, help='Frame rate (default: 60)')
parser.add_argument('-t0', '--startTime', type=float, default=0.0, help='Start time in seconds (default: 0.0)')
parser.add_argument('-t1', '--endTime', type=float, default=1.0, help='End time in seconds (default: 1.0)')
parser.add_argument('-W', '--width', type=int, default=1920, help='Output width (default: 1920)')
parser.add_argument('-H', '--height', type=int, default=1080, help='Output height (default: 1080)')
parser.add_argument('--supersample', type=int, default=2, help='Supersample factor (default: 2)')

args = parser.parse_args()

# === CONFIGURATION ===
WIDTH, HEIGHT = args.width, args.height
SUPERSAMPLE = args.supersample
RENDER_W, RENDER_H = WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE
SHADER_PATH = args.input
OUTPUT_PATTERN = args.output
START_TIME = args.startTime
END_TIME = args.endTime
FPS = args.framesPerSecond

# === CONTEXT SETUP ===
ctx = moderngl.create_standalone_context()
# ctx = moderngl.create_standalone_context(backend='egl')  # uncomment for headless version

# === SHADER SETUP ===
with open(SHADER_PATH) as f:
    user_code = f.read()

fragment_shader = f"""
#version 330
#extension GL_ARB_gpu_shader_fp64 : enable
#pragma optimize(on)
#pragma precision(high)

out vec4 fragColor;
uniform vec3 iResolution;
uniform float iTime;
uniform int iFrame;

{user_code}



void main() {{
    float f = float(iFrame) / 60.0;
    mainImage(fragColor, gl_FragCoord.xy);
    // Force the compiler to think f is used
    if (f < -1.0) {{
        fragColor = vec4(1.0);  // never executed, but f is "used"
    }}
}}
"""

vertex_shader = """
#version 330
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

# === COMPILE SHADER AND CREATE GEOMETRY ===
prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
quad = ctx.buffer(np.array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
], dtype='f4'))
vao = ctx.simple_vertex_array(prog, quad, 'in_vert')

# === CREATE BLIT PROGRAM FOR DOWNSAMPLING ===
blit_vs = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main() {
    uv = (in_vert + 1.0) / 2.0;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

blit_fs = """
#version 330
uniform sampler2D source;
in vec2 uv;
out vec4 fragColor;
void main() {
    fragColor = texture(source, uv);
}
"""

blit_prog = ctx.program(vertex_shader=blit_vs, fragment_shader=blit_fs)
blit_prog['source'].value = 0
blit_quad = ctx.buffer(np.array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
], dtype='f4'))
blit_vao = ctx.simple_vertex_array(blit_prog, blit_quad, 'in_vert')

# === CREATE BUFFERS ===
highres_tex = ctx.texture((RENDER_W, RENDER_H), 4)
highres_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
highres_fbo = ctx.framebuffer(color_attachments=[highres_tex])

downsample_tex = ctx.texture((WIDTH, HEIGHT), 4)
downsample_fbo = ctx.framebuffer(color_attachments=[downsample_tex])

# === MAIN RENDER LOOP ===
num_frames = int((END_TIME - START_TIME) * FPS)
output_dir = os.path.dirname(os.path.abspath(OUTPUT_PATTERN))
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

for frame in range(num_frames):
    t = START_TIME + frame / FPS

    # Set shader uniforms
    prog['iResolution'].value = (RENDER_W, RENDER_H, 1.0)
    prog['iTime'].value = t

    prog['iFrame'].value = frame

    # Render to supersampled target
    highres_fbo.use()
    ctx.viewport = (0, 0, RENDER_W, RENDER_H)
    vao.render()

    # Downsample via GPU
    downsample_fbo.use()
    ctx.viewport = (0, 0, WIDTH, HEIGHT)
    highres_tex.use(location=0)
    blit_vao.render()

    # Save frame
    data = downsample_fbo.read(components=3, alignment=1)
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    out_path = OUTPUT_PATTERN % frame
    img.save(out_path)
    print(f"Wrote {out_path}")
