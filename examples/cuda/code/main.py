from array import array
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import statistics

from cuda_driver import Cuda

HEADER = ".version 7.0\n.target sm_70\n.address_size 64\n"
VECTOR = r"""
.visible .entry vector_add(.param .u64 a, .param .u64 b, .param .u64 output, .param .u32 count) {
    .reg .pred p;
    .reg .b32 r<5>;
    .reg .b64 d<8>;
    .reg .f32 f<4>;
    ld.param.u64 d1, [a];
    ld.param.u64 d2, [b];
    ld.param.u64 d3, [output];
    ld.param.u32 r0, [count];
    mov.u32 r1, %ctaid.x;
    mov.u32 r2, %ntid.x;
    mov.u32 r3, %tid.x;
    mad.lo.u32 r4, r1, r2, r3;
    setp.ge.u32 p, r4, r0;
    @p bra END;
    mul.wide.u32 d4, r4, 4;
    add.u64 d5, d1, d4;
    add.u64 d6, d2, d4;
    add.u64 d7, d3, d4;
    ld.global.f32 f1, [d5];
    ld.global.f32 f2, [d6];
    add.rn.f32 f3, f1, f2;
    st.global.f32 [d7], f3;
END:
    ret;
}
"""
REDUCTION = r"""
.visible .entry reduce_sum(.param .u64 input, .param .u64 output, .param .u32 count) {
    .shared .align 4 .b8 scratch[1024];
    .reg .pred p<3>;
    .reg .b32 r<12>;
    .reg .b64 d<9>;
    .reg .f32 f<4>;
    ld.param.u64 d1, [input];
    ld.param.u64 d2, [output];
    ld.param.u32 r0, [count];
    mov.u32 r1, %tid.x;
    mov.u32 r2, %ntid.x;
    mov.u32 r3, %ctaid.x;
    mov.u32 r4, %nctaid.x;
    mad.lo.u32 r5, r3, r2, r1;
    mul.lo.u32 r6, r2, r4;
    mov.f32 f0, 0f00000000;
LOAD:
    setp.ge.u32 p0, r5, r0;
    @p0 bra SHARE;
    mul.wide.u32 d3, r5, 4;
    add.u64 d4, d1, d3;
    ld.global.f32 f1, [d4];
    add.rn.f32 f0, f0, f1;
    add.u32 r5, r5, r6;
    bra LOAD;
SHARE:
    mov.u32 r7, scratch;
    mad.lo.u32 r8, r1, 4, r7;
    st.shared.f32 [r8], f0;
    bar.sync 0;
    shr.u32 r9, r2, 1;
TREE:
    setp.eq.u32 p1, r9, 0;
    @p1 bra STORE;
    setp.ge.u32 p2, r1, r9;
    @p2 bra SYNC;
    add.u32 r10, r1, r9;
    mad.lo.u32 r11, r10, 4, r7;
    ld.shared.f32 f1, [r8];
    ld.shared.f32 f2, [r11];
    add.rn.f32 f3, f1, f2;
    st.shared.f32 [r8], f3;
SYNC:
    bar.sync 0;
    shr.u32 r9, r9, 1;
    bra TREE;
STORE:
    setp.ne.u32 p0, r1, 0;
    @p0 bra END;
    ld.shared.f32 f0, [r7];
    mul.wide.u32 d5, r3, 4;
    add.u64 d6, d2, d5;
    st.global.f32 [d6], f0;
END:
    ret;
}
"""


def measure(cuda, operation, repeats):
    for _ in range(3):
        operation()
    cuda.call("cuCtxSynchronize")
    return [cuda.time(operation) for _ in range(repeats)]


def vector_study(cuda, settings):
    function = cuda.function(cuda.module(HEADER + VECTOR), "vector_add")
    trials = []
    capacity = max(settings["sizes"])
    a, b, output = [cuda.allocate(capacity * 4) for _ in range(3)]
    for n in settings["sizes"]:
        generator = random.Random(settings["seed"])
        left = array("f", (generator.uniform(-1, 1) for _ in range(n)))
        right = array("f", (generator.uniform(-1, 1) for _ in range(n)))
        actual = array("f", [0]) * n
        expected = array("f", (x + y for x, y in zip(left, right)))
        cuda.upload(a, left)
        cuda.upload(b, right)
        args = (a, b, output, C.c_uint32(n))
        timings = measure(cuda, lambda: cuda.launch(function, args, (n + 255) // 256, 256), settings["repeats"])
        cuda.download(actual, output)
        error = max(abs(x - y) for x, y in zip(actual, expected))
        if error != 0:
            raise RuntimeError(f"Vector addition differs from rounded float32 reference: {error}")
        trials.append({"elements": n, "event_ms": timings, "median_event_ms": statistics.median(timings), "max_absolute_error": error})
    return {"metrics": {"max_absolute_error": max(row["max_absolute_error"] for row in trials),
                        **{f"median_event_ms_n{row['elements']}": row["median_event_ms"] for row in trials}},
            "details": {"device": cuda.device, "trials": trials,
                        "measurement_scope": "CUDA event intervals after three warmups. Transfers, compilation and CPU reference excluded; host submission/scheduling gaps may contribute. Not an end-to-end application benchmark."}}


def reduction_study(cuda, settings):
    function = cuda.function(cuda.module(HEADER + REDUCTION), "reduce_sum")
    repetitions = settings["triples"]
    n = repetitions * 3
    blocks = 64
    data = cuda.allocate(n * 4)
    partials = cuda.allocate(blocks * 4)
    output = cuda.allocate(4)
    actual = array("f", [0])
    trials = []
    for order in ("interleaved", "grouped"):
        values = (array("f", [1e8, 1, -1e8]) * repetitions if order == "interleaved"
                  else array("f", [1e8]) * repetitions + array("f", [1]) * repetitions + array("f", [-1e8]) * repetitions)
        reference = math.fsum(values)
        cuda.upload(data, values)
        for block in settings["block_sizes"]:
            first_args = (data, partials, C.c_uint32(n))
            second_args = (partials, output, C.c_uint32(blocks))

            def operation():
                cuda.launch(function, first_args, blocks, block)
                cuda.launch(function, second_args, 1, block)

            timings = measure(cuda, operation, settings["repeats"])
            cuda.download(actual, output)
            error = abs(actual[0] - reference)
            trials.append({"order": order, "threads_per_block": block, "blocks_first_pass": blocks,
                           "elements": n, "gpu_sum": actual[0], "reference_fsum": reference,
                           "absolute_error": error, "relative_error": error / abs(reference),
                           "event_ms_two_passes": timings})
    return {"metrics": {f"absolute_error_{row['order']}_block{row['threads_per_block']}": row["absolute_error"] for row in trials},
            "details": {"device": cuda.device, "trials": trials,
                        "interpretation_limit": "These fixed synthetic float32 inputs demonstrate numerical behavior, not general accuracy or performance rankings. Both reduction passes run on the GPU. The reference uses Python math.fsum over the same stored float32 values."}}


settings = json.loads(Path(os.environ["ARL_SETTINGS"]).read_text())
with Cuda() as cuda:
    if settings["study"] == "vector-add":
        result = vector_study(cuda, settings)
    elif settings["study"] == "reduction":
        result = reduction_study(cuda, settings)
    else:
        raise ValueError("Unknown study")
    print(json.dumps({"device": cuda.device, "metrics": result["metrics"]}, allow_nan=False), flush=True)
    Path(os.environ["ARL_OUTPUT_DIR"], "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
