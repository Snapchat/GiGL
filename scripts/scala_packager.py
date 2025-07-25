import datetime
import os
import pathlib
import shutil
from pathlib import Path

import gigl.env.dep_constants as dep_constants
from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import run_command_and_stream_stdout
from gigl.src.common.constants.components import GiGLComponents

logger = Logger()


class ScalaPackager:
    """
    A utility class for packaging the Scala JARs for different components.

    Methods:
        package_subgraph_sampler: Packages the Subgraph Sampler component.
        package_split_generator: Packages the Split Generator component.
    """

    def __package_and_upload_jar(
        self,
        local_jar_directory: LocalUri,
        compiled_jar_path: LocalUri,
        component: GiGLComponents,
        use_spark35: bool = False,
    ) -> LocalUri:
        """
        Packages and uploads a JAR file for a specified GiGL component.

        Args:
            local_jar_directory (LocalUri): The local directory to store the JAR file
            compiled_jar_path (LocalUri): The path to the compiled JAR file - i.e. where the JAR will be generated to generated.
            component (GiGLComponents): The component for which the JAR is being packaged;
                one off GiGLComponents.SubgraphSampler or GiGLComponents.SplitGenerator.
            use_spark35 (bool)

        Returns:
            LocalUri: The local URI of the packaged JAR file.
        """
        scala_folder_name = "scala_spark35" if use_spark35 else "scala"
        scala_folder_path = (
            pathlib.Path(__file__).parent.resolve().parent / scala_folder_name
        )

        # Both the `bash -c``and `if [-f ~/.profile ]; then . source ~/.profile; fi;`
        # are needed so this can run in juypter notebooks on jupyterlab.
        # Without this, the sbt command will not be found.
        # I'm not sure why, but jupyterlab will launch new processes as `sh` which doesn't
        # have the right path. *and* just running bash does not set the path!
        build_scala_jar_command = f"bash -c ' if [ -f ~/.profile ]; then source ~/.profile; fi; cd {scala_folder_path} && sbt {component.value}/assembly'"
        logger.info(
            f"Building jar for {component.name} with: {build_scala_jar_command}"
        )
        ret_code = run_command_and_stream_stdout(build_scala_jar_command)
        if ret_code is not None and ret_code != 0:
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
        """
        Packages the Subgraph Sampler component.

        Args:
            use_spark35 (bool): Whether to use the Spark 3.5 implementation. Defaults to False.

        Returns:
            LocalUri: The local URI of the packaged Subgraph Sampler JAR file.
        """
        component = GiGLComponents.SubgraphSampler
        return self.__package_and_upload_jar(
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
        """
        Packages the Split Generator component.

        Args:
            use_spark35 (bool): Whether to use the Spark 3.5 implementation. Defaults to False.

        Returns:
            LocalUri: The local URI of the packaged Split Generator JAR file.
        """
        component = GiGLComponents.SplitGenerator
        return self.__package_and_upload_jar(
            local_jar_directory=dep_constants.get_local_jar_directory(
                component=component, use_spark35=use_spark35
            ),
            compiled_jar_path=dep_constants.get_compiled_jar_path(
                component=component, use_spark35=use_spark35
            ),
            component=component,
            use_spark35=use_spark35,
        )

    def package_all(self) -> tuple[LocalUri, LocalUri, LocalUri]:
        """
        Packages all components. Returns the local URIs of the packaged JAR files.

        Returns:
            tuple[LocalUri, LocalUri, LocalUri]:
                LocalURI of the Spark 3.1 Subgraph Sampler JAR,
                LocalURI of the Spark 3.5 Subgraph Sampler JAR,
                LocalURI of the Spark 3.1 Split Generator JAR,
        """
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
        sgs_path = self.package_subgraph_sampler(use_spark35=False)
        sgs_spark35_path = self.package_subgraph_sampler(use_spark35=True)
        split_gen_path = self.package_split_generator(use_spark35=True)

        return (
            sgs_path,
            sgs_spark35_path,
            split_gen_path,
        )


if __name__ == "__main__":
    ScalaPackager().package_all()
