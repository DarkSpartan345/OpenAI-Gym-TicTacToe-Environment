from setuptools import setup, find_packages

setup(
    name="gym_TicTacToe",
    version="0.1",
    description="Custom TicTacToe Environment for Reinforcement Learning",
    author="Ronny Alberto Rueda Méndez (basado en Mauro Luzzatto)",
    packages=find_packages(),
    install_requires=[
        "gymnasium",  # o "gym" si decides seguir usando gym clásico
        "numpy"
    ],
    include_package_data=True,
)