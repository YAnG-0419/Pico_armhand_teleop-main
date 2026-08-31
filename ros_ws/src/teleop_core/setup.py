from setuptools import find_packages, setup


setup(
    name="teleop_core",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/teleop_core"]),
        ("share/teleop_core", ["package.xml"]),
        ("share/teleop_core/launch", ["launch/control.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="descfly",
    maintainer_email="descfly@example.com",
    description="Shared command contract and safety boundary for dual-FR3 teleoperation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "joint_splitter = teleop_core.joint_splitter:main",
            "safety_gateway = teleop_core.safety_gateway:main",
        ]
    },
)
