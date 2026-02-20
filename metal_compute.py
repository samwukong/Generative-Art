"""
Metal GPU compute for particle tracing on Apple Silicon (#8).

Offloads the per-step particle advection (velocity sampling, attractor
forces, normalization, stepping, bounds checking) to the GPU via a
Metal compute kernel. Color computation and QPainter drawing remain
on CPU.

Falls back transparently to NumPy when Metal is unavailable:
- Non-macOS platform
- pyobjc-framework-Metal not installed
- No Metal-capable GPU detected

Usage:
    tracer = MetalTracer(width, height, vx_field, vy_field, attractors,
                         influence_radius)
    if tracer.gpu_available:
        results = tracer.step_gpu(xs, ys, lengths, active, step_size, max_length)
    else:
        # fall back to NumPy path in renderer
"""

import numpy as np
import platform

# ---------------------------------------------------------------------------
# Metal Shading Language kernel
# ---------------------------------------------------------------------------

MSL_KERNEL_SOURCE = """\
#include <metal_stdlib>
using namespace metal;

struct Attractor {
    float x;
    float y;
    float strength;
    float _pad;  // align to 16 bytes
};

struct Params {
    uint width;
    uint height;
    float step_size;
    float max_length;
    float influence_radius;
    uint num_attractors;
    uint num_particles;
};

// Each GPU thread traces one particle forward by one step.
// Input:  particle state (pos_x, pos_y, length, active) in state buffer
// Output: segment endpoints (old_x, old_y, new_x, new_y) in segments buffer
//         updated state written back to state buffer

kernel void trace_step(
    device const float* vx_field   [[buffer(0)]],
    device const float* vy_field   [[buffer(1)]],
    device const Attractor* attrs  [[buffer(2)]],
    constant Params& params        [[buffer(3)]],
    device float4* state           [[buffer(4)]],  // x, y, length, active
    device float4* segments        [[buffer(5)]],  // old_x, old_y, new_x, new_y
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.num_particles) return;

    float4 st = state[gid];
    float px = st.x;
    float py = st.y;
    float len = st.z;
    float is_active = st.w;

    if (is_active < 0.5f) {
        segments[gid] = float4(0.0f);
        return;
    }

    uint W = params.width;
    uint H = params.height;

    // Clamp to grid and sample velocity field
    int xi = clamp(int(px), 0, int(W) - 1);
    int yi = clamp(int(py), 0, int(H) - 1);
    // NumPy array[width, height] is row-major: index = x * height + y
    uint field_idx = uint(xi) * H + uint(yi);

    float vx = vx_field[field_idx];
    float vy = vy_field[field_idx];

    // Apply attractor/repulsor forces
    for (uint a = 0; a < params.num_attractors; a++) {
        float dx = attrs[a].x - px;
        float dy = attrs[a].y - py;
        float dist = sqrt(dx * dx + dy * dy) + 1e-6f;
        float falloff = exp(-dist / params.influence_radius);
        float strength = attrs[a].strength;
        vx += strength * falloff * dx / dist;
        vy += strength * falloff * dy / dist;
    }

    // Normalize velocity
    float vmag = sqrt(vx * vx + vy * vy);
    vmag = max(vmag, 1e-10f);
    vx /= vmag;
    vy /= vmag;

    // Advance particle
    float new_x = px + params.step_size * vx;
    float new_y = py + params.step_size * vy;

    // Segment length
    float seg_dx = new_x - px;
    float seg_dy = new_y - py;
    float seg_len = sqrt(seg_dx * seg_dx + seg_dy * seg_dy);
    len += seg_len;

    // Output segment (old -> new)
    segments[gid] = float4(px, py, new_x, new_y);

    // Bounds and max-length check
    if (new_x < 0.0f || new_x >= float(W) ||
        new_y < 0.0f || new_y >= float(H) ||
        len > params.max_length) {
        state[gid] = float4(new_x, new_y, len, 0.0f);
    } else {
        state[gid] = float4(new_x, new_y, len, 1.0f);
    }
}
"""


# ---------------------------------------------------------------------------
# MetalTracer class
# ---------------------------------------------------------------------------

class MetalTracer:
    """
    GPU-accelerated particle tracer using Apple Metal compute shaders.

    Handles device initialization, kernel compilation, buffer management,
    and provides a step_gpu() method that replaces the NumPy inner loop.
    """

    def __init__(self, width, height, vx_field, vy_field, attractors,
                 influence_radius):
        """
        Parameters:
        -----------
        width, height : int
            Canvas dimensions (must match vx_field/vy_field shape)
        vx_field, vy_field : ndarray (width, height)
            Normalized curl velocity components
        attractors : list of (x, y, strength)
            Attractor/repulsor points
        influence_radius : float
            Falloff radius for attractor forces
        """
        self.width = width
        self.height = height
        self.attractors = attractors
        self.influence_radius = influence_radius
        self.gpu_available = False

        # Flatten fields to contiguous float32 for Metal buffer
        self._vx_flat = np.ascontiguousarray(vx_field, dtype=np.float32).ravel()
        self._vy_flat = np.ascontiguousarray(vy_field, dtype=np.float32).ravel()

        if platform.system() != 'Darwin':
            return

        try:
            import Metal
            import ctypes

            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                return

            self._device = device
            self._queue = device.newCommandQueue()

            # Compile MSL kernel
            options = Metal.MTLCompileOptions.new()
            library, err = device.newLibraryWithSource_options_error_(
                MSL_KERNEL_SOURCE, options, None)
            if err is not None:
                print(f'  Metal kernel compile error: {err}')
                return

            func = library.newFunctionWithName_('trace_step')
            if func is None:
                print('  Metal: trace_step function not found')
                return

            pipeline, err = device.newComputePipelineStateWithFunction_error_(
                func, None)
            if err is not None:
                print(f'  Metal pipeline error: {err}')
                return

            self._pipeline = pipeline
            self._Metal = Metal
            self._ctypes = ctypes

            # Create static buffers for velocity fields
            shared = Metal.MTLResourceStorageModeShared
            self._vx_buf = device.newBufferWithBytes_length_options_(
                self._vx_flat.tobytes(), self._vx_flat.nbytes, shared)
            self._vy_buf = device.newBufferWithBytes_length_options_(
                self._vy_flat.tobytes(), self._vy_flat.nbytes, shared)

            # Pack attractors: (x, y, strength, pad) as float4 per attractor
            n_att = len(attractors)
            att_data = np.zeros((max(n_att, 1), 4), dtype=np.float32)
            for i, (ax, ay, astr) in enumerate(attractors):
                att_data[i] = [ax, ay, astr, 0.0]
            self._att_buf = device.newBufferWithBytes_length_options_(
                att_data.tobytes(), att_data.nbytes, shared)
            self._n_attractors = n_att

            self.gpu_available = True
            print(f'  Metal GPU: {device.name()} '
                  f'({pipeline.maxTotalThreadsPerThreadgroup()} threads/group)')

        except (ImportError, Exception) as e:
            # pyobjc not installed or other init failure — fall back silently
            pass

    def step_gpu(self, xs, ys, lengths, active, step_size, max_length):
        """
        Advance all active particles by one step on the GPU.

        Parameters:
        -----------
        xs, ys : ndarray (N,) float64
            Current particle positions (modified in-place on return)
        lengths : ndarray (N,) float64
            Cumulative trail lengths (modified in-place)
        active : ndarray (N,) bool
            Active flags (modified in-place)
        step_size : float
            Distance per trace step
        max_length : float
            Max allowed trail length

        Returns:
        --------
        (old_xs, old_ys, new_xs, new_ys, was_active) : tuple of ndarray
            Segment endpoints and per-particle active mask for this step.
            Only entries where was_active=True should be drawn.
        """
        Metal = self._Metal
        n = len(xs)
        shared = Metal.MTLResourceStorageModeShared

        # Pack particle state: (x, y, length, active) as float4
        state = np.zeros((n, 4), dtype=np.float32)
        state[:, 0] = xs.astype(np.float32)
        state[:, 1] = ys.astype(np.float32)
        state[:, 2] = lengths.astype(np.float32)
        state[:, 3] = active.astype(np.float32)

        state_buf = self._device.newBufferWithBytes_length_options_(
            state.tobytes(), state.nbytes, shared)

        # Output buffer for segments
        seg_bytes = n * 4 * 4  # N * float4
        seg_buf = self._device.newBufferWithLength_options_(seg_bytes, shared)

        # Params struct: width, height (uint), step_size, max_length,
        #                influence_radius (float), num_attractors, num_particles (uint)
        import struct
        params_data = struct.pack('IIfffII',
                                  self.width, self.height,
                                  float(step_size), float(max_length),
                                  float(self.influence_radius),
                                  self._n_attractors, n)
        params_buf = self._device.newBufferWithBytes_length_options_(
            params_data, len(params_data), shared)

        # Dispatch
        cmd_buf = self._queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)
        encoder.setBuffer_offset_atIndex_(self._vx_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._vy_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(self._att_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(params_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(state_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(seg_buf, 0, 5)

        threads = Metal.MTLSizeMake(n, 1, 1)
        max_tpg = self._pipeline.maxTotalThreadsPerThreadgroup()
        tpg = Metal.MTLSizeMake(min(256, max_tpg), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(threads, tpg)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Read back results
        state_out = np.frombuffer(
            state_buf.contents().as_buffer(state_buf.length()),
            dtype=np.float32).reshape(n, 4).copy()
        seg_out = np.frombuffer(
            seg_buf.contents().as_buffer(seg_buf.length()),
            dtype=np.float32).reshape(n, 4).copy()

        was_active = active.copy()

        # Update particle arrays in-place
        xs[:] = state_out[:, 0].astype(np.float64)
        ys[:] = state_out[:, 1].astype(np.float64)
        lengths[:] = state_out[:, 2].astype(np.float64)
        active[:] = state_out[:, 3] > 0.5

        old_xs = seg_out[:, 0].astype(np.float64)
        old_ys = seg_out[:, 1].astype(np.float64)
        new_xs = seg_out[:, 2].astype(np.float64)
        new_ys = seg_out[:, 3].astype(np.float64)

        return old_xs, old_ys, new_xs, new_ys, was_active
