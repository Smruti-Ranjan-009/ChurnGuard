from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def get_requirements(file_path=ROOT / "requirements.txt"):
    return [
        requirement.strip()
        for requirement in Path(file_path).read_text(encoding="utf-8").splitlines()
        if requirement.strip() and not requirement.lstrip().startswith("#")
    ]

setup(
    name="telco-churn-guard",
    version="0.1.0",
    description="Telco customer churn classification application",
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={"test": ["pytest>=8.0,<9.0", "ruff>=0.11,<1.0"]},
    python_requires=">=3.10",
)