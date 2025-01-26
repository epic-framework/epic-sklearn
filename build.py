import os
import shutil
from pathlib import Path

from setuptools import Distribution
from pybind11.setup_helpers import Pybind11Extension, build_ext


def build():
    leven_root = 'epic/sklearn/metrics/leven'
    leven_ext = Pybind11Extension(
        name=leven_root.replace('/', '.'),
        sources=[f'{leven_root}/leven.cpp'],
        include_dirs=[leven_root],
    )

    distribution = Distribution(dict(
        name="epic-sklearn",
        ext_modules=[leven_ext],
    ))

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        destination = output.relative_to(cmd.build_lib)
        shutil.copyfile(output, destination)
        # Whoever (owner, group, others) has read permission also gets execution permission
        mode = os.stat(destination).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(destination, mode)


if __name__ == "__main__":
    build()
