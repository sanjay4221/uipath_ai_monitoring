from setuptools import setup, find_packages

setup(
    name="uipath_ai_monitoring",
    version="1.0.0",
    author="RPA AI Team",
    description="ML-based UiPath Log Analysis & Monitoring System with Groq AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas", "numpy", "scikit-learn", "joblib",
        "nltk", "groq", "fastapi", "uvicorn",
        "pyyaml", "python-dotenv", "rich", "loguru",
        "matplotlib", "seaborn",
    ],
)
