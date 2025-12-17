# GiGL Changelog

All notable changes to this repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]


## [0.1.0] - Dec 18, 2025

### Changed
* Update deps / python to latest supported version - use UV for dep management by @svij-sc in https://github.com/Snapchat/GiGL/pull/414
    - `uv` now helps us manage our python environment, so Instead of doing `python -m ...`; we will need to do `uv run python -m ...`; subsequently tools like `black` can now be executed as `uv run black` and dont need to be installed at a global level or inside some unmanaged python env.
    - As long as you are in the gigl directory, uv will know how to work.
    - We eliminate the need for using pip to manage/install deps; we also eliminate the need to maintain different requirements files, there is just one now uv.lock
    - **Deprecation:** As a result of this change, we have deprecated use of host managed python environments and we let `uv` manage the virtual env. Consequentially, we don't need to support `make` targets like `initialize_environment`, `generate_..._requirements`, etc. have been removed.
        - **Manual action needed:** Remove your old conda environments `conda remove --name gnn --all -y`, run `make install_dev_deps` from inside GiGL directory.
* Update Vuln reporting process. by @svij-sc in https://github.com/Snapchat/GiGL/pull/272
* Update splitter names to have `Dist` prefix by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/378
* Update resource config wrapper / validation check to support graph store mode by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/383
* Remove GLT imports from our example code by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/389


### Added
* Enforce exported config is to GCS. by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/243
* Include script folder in the workbench image by @svij-sc in https://github.com/Snapchat/GiGL/pull/255
* Support / as an overload on Uri, like Pathlib by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/224
* Add `/help` command. by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/267
* Add HashedNodeSplitter by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/273
* Add ability to load, partition, and separate node labels with features by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/276
* Add KGE constants by @nshah171 in https://github.com/Snapchat/GiGL/pull/239
* Add some 'shared' KGE utils by @nshah171 in https://github.com/Snapchat/GiGL/pull/233
* Allow users to specifcy no accelerators when creating dev box by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/287
* Add custom datetime resolvers by @svij-sc in https://github.com/Snapchat/GiGL/pull/279
* Add experimental parsing libs to parse KGE configs by @nshah171 in https://github.com/Snapchat/GiGL/pull/235
* Use clamp with bincount approach by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/281
* Support GCP region override for vertex ai jobs by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/294
* Enable adding labels to VAI pipelines by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/300
* Enable Node Classification in Dataset and Dataloaders by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/283
* disable nb e2e tests on merge by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/308
* Add ability to resolve any yaml --> dataclass using utility + introduce ${git_hash:} resolver by @svij-sc in https://github.com/Snapchat/GiGL/pull/307
* e2e tests on python by @svij-sc in https://github.com/Snapchat/GiGL/pull/301
* Node Label Tech Debt [1/2]: TFRecord DataLoader returns labels separately by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/304
* Add google_cloud_pipeline_components package by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/324
* Migrate DistDataset to it's own file by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/309
* shutdown example inference loaders by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/326
* [KGE] Add core KGE modeling code by @nshah-sc in https://github.com/Snapchat/GiGL/pull/293
* Throw error in partitioner for re-registering by @swong3-sc in https://github.com/Snapchat/GiGL/pull/325
* Shard nodes before selecting labels in ABLPLoader by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/328
* Set cuda devices for example loops by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/329
* Create dist_ablp_neighborloader_test by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/332
* Add label_edge_type_to_message_passing_edge_type by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/333
* Unify handling of partitioning with no edge features by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/334
* Add assert_labels to check the expected label tensor is correct by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/335
* Bump GLT Version and add tests for fanning out around isolated nodes by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/280
* Simplify Dataset `build` function by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/315
* [KGE] add edge_dataset creation utils by @nshah-sc in https://github.com/Snapchat/GiGL/pull/323
* Allow users to be alerted when pipelines finish by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/320
* Add VertexAiGraphStoreConfig to support separate graph and compute clusters by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/337
* Add longer timeout to v1 trainer by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/348
* Allow users to specify custom master nodes for networking utils by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/339
* [KGE] Add checkpointing utils by @nshah-sc in https://github.com/Snapchat/GiGL/pull/342
* Add FeatureFlag for not populating embedding field by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/352
* Shard Data Preprocessor Enumeration by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/344
* Support multiple supervision edge types in ABLP Loader by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/336
* Locking Deps for Release by @svij-sc in https://github.com/Snapchat/GiGL/pull/322
* Add `VertexAIService.launch_graph_store_job`  by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/350
* Add `get_cluster_spec` to get cluster spec on VAI by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/358
* LightGCN for homogenous graphs by @swong3-sc in https://github.com/Snapchat/GiGL/pull/338
* Add Prediction Exporter by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/366
* migrate gigl.module -> gigl.nn by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/359
* Properly set VAI env vars by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/373
* Add get_graph_store_info to determine "Graph cluster" info from environment by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/355
* support setting scheduling_strategy by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/371
* Add BqUtils.copy_table to enable (more efficient?) table copies by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/377
* Add Distributed Model Parallel Tests for models_test.py by @swong3-sc in https://github.com/Snapchat/GiGL/pull/363
* Manually log VAI pipeline runs by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/384
* Gowalla BigQuery loader script by @swong3-sc in https://github.com/Snapchat/GiGL/pull/381
* Add `compute_cluster_local_world_size` to `VertexAiGraphStoreConfig` and `GraphStoreInfo` by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/380
* Add `create_test_process_group` and swap Loader tests away from DistContext by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/376
* Break up unit tests into separate runners for python and scala by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/361
* Add types to SortedDict and add tests by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/391
* Add remote_dataset for graph store mode by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/386

### Fixed
* Kmonte/fix install by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/270
* Fix BQ Query by @svij-sc in https://github.com/Snapchat/GiGL/pull/263
* Fix cost association for glt/trainer by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/290
* Fail early if invalid Node IDs are found prior partitioning by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/284
* Fix Cursor Rule Resolution by @svij-sc in https://github.com/Snapchat/GiGL/pull/312
* Fix -- Remove sort of feature keys from preprocessed metadata `FeatureSpecDict` by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/306
* Fix small issue with train/testing examples by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/331

### Misc
* Update TFRecordDataLoader TODO by @mkolodner-sc in https://github.com/Snapchat/GiGL/pull/269
* Update Vuln reporting process. by @svij-sc in https://github.com/Snapchat/GiGL/pull/272
* Update comment to reflect code by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/278
* Update default image used in create_dev_instance script by @svij-sc in https://github.com/Snapchat/GiGL/pull/295
* nit: update error msg - unsupported image by @svij-sc in https://github.com/Snapchat/GiGL/pull/302
* Update Instructions - GiGL wheel is public by @svij-sc in https://github.com/Snapchat/GiGL/pull/349
* make README prettier by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/241
* Update Tut Readme w/ hook and presenter information by @svij-sc in https://github.com/Snapchat/GiGL/pull/254
* More detailed Toy Example by @svij-sc in https://github.com/Snapchat/GiGL/pull/260
* format md by @svij-sc in https://github.com/Snapchat/GiGL/pull/261
* Try Adding cursor rules for python dev. by @svij-sc in https://github.com/Snapchat/GiGL/pull/282
* Add new code owners by @svij-sc in https://github.com/Snapchat/GiGL/pull/286
* Actually use test file pattern by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/356
* Fossa Analyze show logs by @svij-sc in https://github.com/Snapchat/GiGL/pull/357
* Add NGCF to license by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/382
* Bump unit test and integration test timeout :( by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/375
* Fix spelling error in export_test by @swong3-sc in https://github.com/Snapchat/GiGL/pull/321
* Swap back to using docker build from scripts/ by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/256
* Prettify the graph visualizations by @svij-sc in https://github.com/Snapchat/GiGL/pull/238
* Tutorial assets by @yliu2-sc in https://github.com/Snapchat/GiGL/pull/265
* Use `typing.Dict` for Beam type annotations. by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/275
* Lock pip-compile version + follow up (#279 ) regen reqs / images  by @svij-sc in https://github.com/Snapchat/GiGL/pull/289
* [KGE] Add hydra conf yamls  by @nshah-sc in https://github.com/Snapchat/GiGL/pull/292
* Revert "Use `typing.Dict` for Beam type annotations. (#275)" by @kmontemayor2-sc in https://github.com/Snapchat/GiGL/pull/327


### New Contributors
* @nshah171, @swong3-sc
