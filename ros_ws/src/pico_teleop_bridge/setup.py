from glob import glob

from setuptools import find_packages, setup


setup(
    name="pico_teleop_bridge",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/pico_teleop_bridge"]),
        ("share/pico_teleop_bridge", ["package.xml"]),
        ("share/pico_teleop_bridge/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="descfly",
    maintainer_email="descfly@example.com",
    description="PICO UDP adapter for the normalized teleoperation command channel.",
    license="MIT",
    entry_points={"console_scripts": ["bridge = pico_teleop_bridge.node:main"]},
)
