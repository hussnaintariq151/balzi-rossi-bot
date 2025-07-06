from setuptools import setup, find_packages

setup(
    name='balzi_rossi_bot',
    version='0.1.0',
    description='AI Assistant for the Balzi Rossi Archaeological Site and Museum',
    author='Hussnain Tariq',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.1",
        "langchain>=0.2.0",
        "openai>=1.33.0",
        "python-dotenv>=1.0.1",
        "httpx>=0.27.0",
        "jinja2>=3.1.3",
        "starlette>=0.37.2",
        "astrapy>=1.3.0",
        "uuid",
        "aiofiles>=23.2.1",  # for static file serving
        "pydub>=0.25.1",     # if you're generating audio files
        "ffmpeg-python>=0.2.0",  # optional: if using ffmpeg with pydub
        "speechrecognition>=3.10.0",  # optional voice support
        "whisper @ git+https://github.com/openai/whisper.git",  # if using Whisper for ASR
    ],
    entry_points={
        'console_scripts': [
            'balzi_rossi_bot=main:app',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Framework :: FastAPI",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
