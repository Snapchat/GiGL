import datetime
import os
import pathlib
import shutil
import subprocess
from pathlib import Path

import gigl.env.dep_constants as dep_constants
from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents

logger = Logger()


class ScalaPackager:
    def package_and_upload_jar(
        self,
        local_jar_directory: LocalUri,
        compiled_jar_path: LocalUri,
        component: GiGLComponents,
        use_spark35: bool = False,
    ) -> LocalUri:
        scala_folder_name = "scala_spark35" if use_spark35 else "scala"
        scala_folder_path = (
            pathlib.Path(__file__).parent.resolve().parent / scala_folder_name
        )
        build_scala_jar_command = (
            f"cd {scala_folder_path} && sbt {component.value}/assembly"
        )
        logger.info(
            f"Building jar for {component.name} with: {build_scala_jar_command}"
        )
        process = subprocess.Popen(
            build_scala_jar_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )

        while (ret_code := process.poll()) is None:
            if process.stdout is None:
                continue
            for line in process.stdout:
                logger.info(line.decode())

        if ret_code != 0:
            raise RuntimeError(
                f"Failed building scala jar for {component.name}. See stack trace for details."
            )

        Path(local_jar_directory.uri).mkdir(parents=True, exist_ok=True)

        # Replace existing jar file in local directory
        if Path(local_jar_directory.uri).glob("*.jar"):
            for file in Path(local_jar_directory.uri).glob("*.jar"):
                os.remove(file)

        jar_file_path = (
            Path(local_jar_directory.uri)
            / f"{component.value}-{datetime.datetime.now().timestamp()}.jar"
        )
        Path(jar_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(compiled_jar_path.uri).rename(jar_file_path)

        logger.info(f"Moved generated jar: {compiled_jar_path} to {jar_file_path}")

        return LocalUri(jar_file_path)  # Return the local path to the jar file

    def package_subgraph_sampler(self, use_spark35: bool = False) -> LocalUri:
        component = GiGLComponents.SubgraphSampler
        return self.package_and_upload_jar(
            local_jar_directory=dep_constants.get_local_jar_directory(
                component=component, use_spark35=use_spark35
            ),
            compiled_jar_path=dep_constants.get_compiled_jar_path(
                component=component, use_spark35=use_spark35
            ),
            component=component,
            use_spark35=use_spark35,
        )

    def package_split_generator(self, use_spark35: bool = False) -> LocalUri:
        component = GiGLComponents.SplitGenerator
        return self.package_and_upload_jar(
            local_jar_directory=dep_constants.get_local_jar_directory(
                component=component, use_spark35=use_spark35
            ),
            compiled_jar_path=dep_constants.get_compiled_jar_path(
                component=component, use_spark35=use_spark35
            ),
            component=component,
            use_spark35=use_spark35,
        )


if __name__ == "__main__":
    packager = ScalaPackager()

    # Remove all existing jars
    dirs_to_delete = [
        dep_constants.get_local_jar_directory(
            component=GiGLComponents.SubgraphSampler, use_spark35=False
        ).uri,
        dep_constants.get_local_jar_directory(
            component=GiGLComponents.SplitGenerator, use_spark35=False
        ).uri,
        dep_constants.get_local_jar_directory(
            component=GiGLComponents.SubgraphSampler, use_spark35=True
        ).uri,
        dep_constants.get_local_jar_directory(
            component=GiGLComponents.SplitGenerator, use_spark35=True
        ).uri,
    ]
    for directory in dirs_to_delete:
        shutil.rmtree(directory, ignore_errors=True)

    packager.package_subgraph_sampler()
    packager.package_subgraph_sampler(use_spark35=True)
    packager.package_split_generator(use_spark35=True)
