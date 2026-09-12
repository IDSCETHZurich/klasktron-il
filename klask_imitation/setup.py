from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "gymnasium",
    "imitation",
    "stable-baselines3>=2.4.0",
]

EXTRAS_REQUIRE = {
    "data": [
        "huggingface_hub>=0.30.0",
    ],
    "dev": [
        "black>=24.0.0",
        "ruff>=0.6.0",
    ],
}


setup(
    name="klask_imitation",
    version="0.1.0",
    description="KlaskTron benchmark environment, baselines, and release utilities.",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "klasktron-download-data=klasktron.data.huggingface:main",
        ],
    },
)
