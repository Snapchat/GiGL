# Canonical GraphLearn-for-PyTorch (GLT) pin for the gigl-core C++ build.
#
# WHY THIS FILE EXISTS
# --------------------
# gigl-core compiles against GLT's C++ headers (graphlearn_torch/include/*.h), but GLT's installed
# wheel ships only its Python tree -- setup.py sets
# `package_dir={'graphlearn_torch': 'graphlearn_torch/python'}`, so no headers and no csrc/ are
# packaged. gigl-core therefore has to acquire the GLT source itself.
#
# The header layout of `graphlearn_torch::ShmQueueMeta` is read at runtime out of a shared-memory
# segment created by the *installed GLT wheel*. If gigl-core compiles against commit X while the
# wheel was built from commit Y, queue-size readings are silently wrong -- no compile error, no
# import error. So the commit used here MUST equal the commit `gigl/scripts/install_glt.sh` builds
# the wheel from.
#
# That invariant is enforced, not merely documented: `tests/config_tests/dep_vars_check.py` asserts
# that GIGL_GLT_COMMIT below equals GLT_COMMIT_SHA in `gigl/dep_vars.env` (which install_glt.sh
# reads), and it runs as part of the `precondition_tests` make target -- before anything installs.
#
# WHY THE PIN IS DUPLICATED INSTEAD OF SHARED
# -------------------------------------------
# It cannot live in only one place:
#   * `gigl/dep_vars.env` is not copied into containers/Dockerfile.{cpu,cuda,dataflow}.base, yet
#     those images run install_py_deps.sh -> install_glt.sh. (This change adds the COPY, but
#     dep_vars.env is still outside gigl-core's sdist.)
#   * A gigl-core-local file is not reachable by the separately-installed `gigl` package's
#     post-install script.
# So: this file is canonical for the C++ build (and lands in gigl-core's sdist, keeping
# `uv build --wheel gigl-core/` self-contained), dep_vars.env is canonical for the installer, and
# dep_vars_check.py keeps them equal.
#
# HOW TO BUMP THE PIN
# -------------------
#   1. Update GIGL_GLT_COMMIT and GLT_COMMIT_SHA in gigl/dep_vars.env to the same value.
#   2. Recompute the header hash:
#        git -C <a graphlearn-for-pytorch checkout> show <sha>:graphlearn_torch/include/shm_queue.h \
#          | sha256sum                                          # -> GIGL_GLT_SHM_QUEUE_H_SHA256
#   3. Re-verify the ShmQueueMeta layout assertions in gigl-core/tests/glt_headers_test.cpp.
#   4. Run `make precondition_tests && make unit_test_cpp`.
#   5. Rebuild the base Docker images. containers/Dockerfile.src compiles gigl-core against the pin
#      above but runs on a *prebuilt* base image (DOCKER_LATEST_BASE_*_IMAGE_NAME_WITH_TAG in
#      gigl/dep_vars.env) whose GLT wheel was built from whatever the pin was at the time. Bumping the
#      pin without refreshing those images yields new headers over an older wheel -- the exact skew this
#      file exists to prevent, and one nothing currently detects at runtime.
#
# This file is listed in gigl-core/pyproject.toml `cache-keys` and in gigl-core/Makefile's build
# prerequisites, so editing it forces a CMake reconfigure and a rebuild of the extensions. Without
# that, a pin bump would leave a stale compiled probe in place.

# GLT commit that gigl-core compiles against. Must match GLT_COMMIT_SHA in gigl/dep_vars.env.
set(GIGL_GLT_COMMIT "88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9")

# Upstream git repository, fetched at GIGL_GLT_COMMIT.
#
# Git rather than a source tarball, mainly so the commit is the *only* thing to update when bumping the
# pin. Git objects are content-addressed, so GIGL_GLT_COMMIT is itself the integrity check and there is
# no archive checksum to recompute and keep in step. It also matches how
# gigl/scripts/install_glt.sh consumes the pin (git clone + git checkout), so both consumers of the pin
# work the same way.
#
# A secondary benefit: GitHub's auto-generated /archive/<sha>.tar.gz files are not contractually
# byte-stable (their 2023 compression change changed such hashes), so a pinned tarball checksum can go
# stale through no change of ours. That is unlikely enough not to drive the decision on its own.
set(GIGL_GLT_GIT_REPOSITORY "https://github.com/alibaba/graphlearn-for-pytorch.git")

# SHA256 of graphlearn_torch/include/shm_queue.h at GIGL_GLT_COMMIT.
#
# Checked against whatever source tree we end up using -- fetched or supplied via
# GIGL_GLT_SOURCE_DIR. For a git fetch this is belt-and-braces on top of git's own object
# verification; for an override it is the only check there is, since an arbitrary local checkout
# otherwise silently replaces the pinned source.
#
# Hashing the one header we actually compile against is deliberately narrower than comparing
# `git rev-parse HEAD`: it still passes for a checkout at a different commit whose header is
# byte-identical, which is the common and harmless case.
set(GIGL_GLT_SHM_QUEUE_H_SHA256 "c6b1d04e5b71c780fc56b726d824642eec6c50ad54d3f3680973ffb7bf5e1efc")

# Path, relative to the GLT source root, of the header whose hash is checked above.
set(GIGL_GLT_SHM_QUEUE_H_RELPATH "graphlearn_torch/include/shm_queue.h")
