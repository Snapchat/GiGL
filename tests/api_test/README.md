# Api Test

This folder contains an end-to-end (E2E) orchestration API test. It is used to evaluate whether the KFP orchestration
and subsequent Docker compile APIs function as expected.

The test simulates how a customer would use GiGL—not as part of a developer workflow, but as a user consuming a released
version of GiGL. Specifically, it tests the scenario where a customer wants to package their custom code alongside a
released GiGL package and run a full pipeline.

In this setup, the folder includes the custom customer code. Whatever version of GiGL is available in the Python
interpreter at runtime will be used to run the pipeline, mimicking how an actual user would invoke the workflow.
