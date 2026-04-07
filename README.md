# Python-Template-Repo

Template repository for Python projects

## Project Introduction

### Description

This is a comprehensive Python project template that provides a solid foundation for new Python projects. It includes CI/CD workflows, pre-commit hooks, code formatting, linting, and testing infrastructure out of the box.

### Features

- **GitHub Actions CI/CD**: Automated testing on Python 3.10, 3.11, and 3.12
- **Pre-commit Hooks**: Automatic code formatting and linting before commits
- **VSCode Integration**: Auto-run tasks on folder open and manual development tasks
- **Code Quality Tools**: Black for formatting, Ruff for linting, MyPy for type checking
- **Testing Infrastructure**: Pytest with coverage reporting
- **Modern Python Packaging**: Uses `pyproject.toml` for project configuration

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Git

### Initial Setup

1. **Clone the repository** (if using as a template):
   ```bash
   git clone <repository-url>
   cd Python-Template-Repo
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Update project metadata**:
   - Edit `pyproject.toml` and change `name = "your-project-name"` to your actual project name
   - Update version, description, and other metadata as needed

## Development Commands

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov --cov-report=term
```

Run tests with HTML coverage report:
```bash
pytest --cov --cov-report=html
```

### Code Formatting

Format code with Black:
```bash
black .
```

Format and lint with Ruff:
```bash
ruff check --fix .
ruff format .
```

### Type Checking

Run MyPy type checker:
```bash
mypy src/
```

### Pre-commit Hooks

Run pre-commit hooks manually on all files:
```bash
pre-commit run --all-files
```

### VSCode Tasks

If using VSCode, you can use the following tasks (accessible via `Cmd+Shift+P` → "Tasks: Run Task"):

- **git fetch**: Automatically runs on folder open
- **pre-commit install**: Automatically runs on folder open
- **Format with black**: Format code using Black
- **Format and lint with ruff**: Format and lint code using Ruff
- **Run tests**: Execute pytest without coverage
- **Run tests with coverage**: Execute pytest with coverage reporting

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD workflow
├── .vscode/
│   └── tasks.json              # VSCode tasks configuration
├── src/                        # Source code directory
│   └── .gitkeep
├── tests/                      # Test directory
│   └── .gitkeep
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
└── requirements.txt            # Legacy requirements file (optional)
```

## CI/CD

The project includes a GitHub Actions workflow that:

- Runs on pushes and pull requests to `main` and `develop` branches
- Tests against Python 3.10, 3.11, and 3.12
- Installs dependencies from `pyproject.toml`
- Runs pre-commit hooks
- Executes pytest with coverage reporting

## Code Quality

This template includes several code quality tools:

- **Black**: Code formatter with 100 character line length
- **Ruff**: Fast Python linter with auto-fix capabilities
- **MyPy**: Static type checker (configured for Python 3.11)
- **Pre-commit Hooks**: Additional checks for trailing whitespace, YAML validation, private key detection, and more

## Contributing

1. Create a feature branch
2. Make your changes
3. Ensure all tests pass: `pytest`
4. Ensure code is formatted: `black .` and `ruff check --fix .`
5. Commit your changes (pre-commit hooks will run automatically)
6. Push to your branch and create a pull request

## License

[Add your license here]

## Developer Credits

[Add developer credits here]
