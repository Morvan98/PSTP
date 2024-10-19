from setuptools import setup, find_packages

setup(
    name='PSTP',
    version='0.1.0',
    author='Mofan Feng',
    author_email='fmf.von@sjtu.edu.cn',
    description='Protein phase separation predictor and phase separation region predictor',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_project_name',
    packages=find_packages(),  # 自动查找你的项目包
    package_data={
        'pstp': ['model_weights/slide_nn_model_weights/mix/*',
                 'model_weights/slide_nn_model_weights/saps/*',
                 'model_weights/slide_nn_model_weights/pdps/*',],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.7,<3.10.0",
)