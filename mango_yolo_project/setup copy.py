from setuptools import setup, find_packages

setup(
    name="mango_yolo_trainer",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.0',
        'scikit-learn',
        'pyyaml',
        'opencv-python',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'mango-train=auto_train:main',
        ],
    },
)