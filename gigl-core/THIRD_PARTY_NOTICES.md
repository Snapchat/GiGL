# Third-Party Notices — gigl-core

`gigl-core` is part of GiGL and is distributed under the MIT License. This file records third-party
software that `gigl-core` is built against. Those licenses apply to that software, not to gigl-core.

The repository-wide notices file is `THIRD_PARTY_NOTICES.md` at the GiGL root; this copy exists so the
notice travels with the separately-published `gigl-core` wheel and sdist.

## GraphLearn-for-PyTorch (GLT)

- **Project:** https://github.com/alibaba/graphlearn-for-pytorch
- **Copyright:** Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
- **License:** Apache License, Version 2.0 — http://www.apache.org/licenses/LICENSE-2.0
- **Version used:** the commit pinned as `GIGL_GLT_COMMIT` in `GLT_PIN.cmake`

`gigl-core/CMakeLists.txt` fetches GLT's source at configure time and compiles against its C++ headers,
so that `gigl_core.ShmQueueProbe` can read a GLT shared-memory channel's queue counters using field
offsets derived from GLT's own type definitions instead of hard-coded byte offsets.

**No GLT code is compiled into or linked against this distribution.** GLT's `shm_queue.h` contributes
type and layout information only, and the single GLT translation unit GiGL compiles
(`graphlearn_torch/csrc/shm_queue.cc`) is linked exclusively into C++ test binaries, which are not
distributed. Verifiable on a built extension:

```bash
nm -C --defined-only <site-packages>/gigl_core/shm_queue_probe*.so | grep graphlearn  # no output
nm -C -u             <site-packages>/gigl_core/shm_queue_probe*.so | grep graphlearn  # no output
```

If a future change compiles GLT sources into a shipped artifact, that artifact would carry Apache-2.0
obligations and this notice must be revisited.
