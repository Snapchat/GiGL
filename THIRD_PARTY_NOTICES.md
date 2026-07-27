# Third-Party Notices

GiGL is distributed under the MIT License (see `LICENSE`). It depends on third-party software covered
by separate licenses; those licenses apply to the corresponding software, not to GiGL itself. This file
records dependencies that warrant explicit attribution beyond the dependency metadata in
`fossa-deps.yml`.

## GraphLearn-for-PyTorch (GLT)

- **Project:** https://github.com/alibaba/graphlearn-for-pytorch
- **Copyright:** Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
- **License:** Apache License, Version 2.0 — http://www.apache.org/licenses/LICENSE-2.0
- **Version used:** commit `88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9`

GiGL uses GLT in two distinct ways, which have different distribution implications:

1. **At runtime, as an installed dependency.** `gigl/scripts/install_glt.sh` builds a GLT wheel from the
   pinned commit and installs it into the user's environment. GiGL does not redistribute that wheel;
   each installation builds it from upstream source.

2. **At build time, as C++ headers.** `gigl-core` compiles against GLT's headers so that
   `gigl_core.ShmQueueProbe` can read the enqueue/dequeue counters of a GLT shared-memory channel
   using field offsets derived from GLT's own type definitions rather than hard-coded byte offsets.
   `gigl-core/CMakeLists.txt` fetches the pinned source at configure time via git; see
   `gigl-core/GLT_PIN.cmake`.

   No GLT code is compiled into or linked against the published `gigl-core` wheel. GLT's
   `shm_queue.h` supplies type and layout information only, and the single GLT translation unit GiGL
   compiles (`graphlearn_torch/csrc/shm_queue.cc`) is linked exclusively into C++ test binaries, which
   are not distributed. This is verifiable on a built extension:

   ```bash
   nm -C --defined-only gigl-core/.cache/cmake_build/shm_queue_probe*.so | grep graphlearn  # no output
   nm -C -u          gigl-core/.cache/cmake_build/shm_queue_probe*.so | grep graphlearn  # no output
   ```

   If a future change compiles GLT sources into a shipped artifact, that artifact would carry
   Apache-2.0 obligations and this notice must be revisited.
