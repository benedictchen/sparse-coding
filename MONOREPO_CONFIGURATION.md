# 🚀 Modern Monorepo Configuration (2025)

## 🎯 **Recommended Setup: pnpm + Nx Hybrid Approach**

Based on latest 2025 best practices, this configuration provides:
- ✅ **Fast package management** (pnpm workspaces)
- ✅ **Intelligent task orchestration** (Nx)
- ✅ **Advanced caching** (local + distributed)
- ✅ **Python + JavaScript support** (polyglot monorepo)
- ✅ **Independent PyPI publishing** (per package)

---

## 📦 **Project Structure**

```
research-papers/                           ← ROOT MONOREPO
├── pnpm-workspace.yaml                    ← pnpm workspace config
├── nx.json                                ← Nx configuration
├── package.json                           ← Root package.json
├── pyproject.toml                         ← Root Python config
├── .gitignore                             ← Monorepo gitignore
├── .github/                               ← CI/CD workflows
│   └── workflows/
│       ├── ci.yml                         ← Main CI pipeline
│       └── publish.yml                    ← PyPI publishing
├── packages/                              ← All research packages
│   ├── sparse-coding/                     ← Individual package
│   │   ├── package.json                   ← JS dependencies (if any)
│   │   ├── pyproject.toml                 ← Python package config
│   │   ├── project.json                   ← Nx project config
│   │   └── sparse_coding/                 ← Python module
│   ├── reservoir-computing/
│   │   ├── package.json
│   │   ├── pyproject.toml
│   │   ├── project.json
│   │   └── reservoir_computing/
│   └── ... (7 more packages)
├── tools/                                 ← Build tools and scripts
│   ├── scripts/
│   │   ├── build.py                       ← Build automation
│   │   ├── test.py                        ← Test runner
│   │   └── publish.py                     ← PyPI publishing
└── shared/                                ← Shared utilities
    └── research-commons/                  ← Common research tools
```

---

## 🔧 **Configuration Files**

### **1. pnpm-workspace.yaml**
```yaml
packages:
  - 'packages/*'
  - 'shared/*'
  - 'tools/*'
```

### **2. Root package.json**
```json
{
  "name": "research-papers-monorepo",
  "private": true,
  "packageManager": "pnpm@9.0.0",
  "scripts": {
    "build": "nx run-many --target=build",
    "test": "nx run-many --target=test", 
    "lint": "nx run-many --target=lint",
    "publish": "nx run-many --target=publish",
    "graph": "nx graph",
    "affected:build": "nx affected --target=build",
    "affected:test": "nx affected --target=test"
  },
  "devDependencies": {
    "nx": "^18.0.0",
    "@nx/python": "^18.0.0",
    "@nx/js": "^18.0.0"
  }
}
```

### **3. nx.json** 
```json
{
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "targetDefaults": {
    "build": {
      "cache": true,
      "inputs": ["production", "^production"]
    },
    "test": {
      "cache": true,
      "inputs": ["default", "^production", "{workspaceRoot}/pytest.ini"]
    },
    "lint": {
      "cache": true,
      "inputs": ["default", "{workspaceRoot}/.pylintrc"]
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/?(*.)+(spec|test).[jt]s?(x)?(.snap)",
      "!{projectRoot}/tsconfig.spec.json",
      "!{projectRoot}/jest.config.[jt]s",
      "!{projectRoot}/src/test-setup.[jt]s",
      "!{projectRoot}/test-setup.[jt]s",
      "!{projectRoot}/**/*_test.py",
      "!{projectRoot}/tests/**/*"
    ],
    "sharedGlobals": []
  },
  "generators": {
    "@nx/python:package": {
      "unitTestRunner": "pytest"
    }
  }
}
```

### **4. Root pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["shared*"]

[tool.pytest.ini_options]
testpaths = ["packages", "shared"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "--verbose --tb=short"

[tool.ruff]
line-length = 88
target-version = "py38"

[tool.ruff.lint]
select = ["E", "F", "W", "C90", "I", "N", "UP", "S", "B", "A", "COM", "DTZ", "EM", "G", "PIE", "T20", "SIM", "ARG", "PTH", "PL", "R", "TRY"]
```

---

## 📋 **Individual Package Configuration**

### **packages/sparse-coding/project.json**
```json
{
  "name": "sparse-coding",
  "root": "packages/sparse-coding", 
  "projectType": "library",
  "sourceRoot": "packages/sparse-coding/sparse_coding",
  "targets": {
    "build": {
      "executor": "@nx/python:build",
      "options": {
        "outputPath": "dist/packages/sparse-coding",
        "buildDir": "packages/sparse-coding/dist"
      }
    },
    "test": {
      "executor": "@nx/python:test", 
      "options": {
        "testFile": "packages/sparse-coding/tests/"
      }
    },
    "lint": {
      "executor": "@nx/python:flake8",
      "options": {
        "outputFile": "reports/packages/sparse-coding/lint.txt"
      }
    },
    "publish": {
      "executor": "@nx/python:publish",
      "options": {
        "distPath": "dist/packages/sparse-coding",
        "registry": "https://upload.pypi.org/legacy/"
      },
      "dependsOn": ["build"]
    }
  },
  "tags": ["type:package", "scope:research", "domain:vision"]
}
```

### **packages/sparse-coding/pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sparse-coding"
version = "2.1.0"
description = "Biological vision principles for efficient representation learning from natural images"
readme = "README.md"
license = {file = "LICENSE"}
authors = [{name = "Benedict Chen", email = "benedict@benedictchen.com"}]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "scikit-learn>=1.0.0"
]
requires-python = ">=3.8"

[project.urls]
Homepage = "https://github.com/benedictchen/sparse-coding"
Repository = "https://github.com/benedictchen/sparse-coding"
Documentation = "https://github.com/benedictchen/sparse-coding#readme"

[tool.setuptools.packages.find]
where = ["."]
include = ["sparse_coding*"]

[tool.setuptools.package-data]
sparse_coding = ["*.md", "*.txt"]
```

---

## 🚀 **Setup Instructions**

### **1. Install pnpm + Nx**
```bash
# Install pnpm via corepack (recommended 2025 method)
corepack enable
corepack prepare pnpm@latest --activate

# Verify installation
pnpm --version
```

### **2. Initialize monorepo**
```bash
cd research-papers/
pnpm init

# Install Nx
pnpm add -D nx @nx/python @nx/js

# Initialize Nx
pnx nx init

# Create workspace config
echo "packages:
  - 'packages/*'
  - 'shared/*'
  - 'tools/*'" > pnpm-workspace.yaml
```

### **3. Install dependencies across all packages**
```bash
# Install all dependencies for all packages
pnpm install

# Add a dependency to specific package
pnpm add numpy --filter sparse-coding

# Add dev dependency to root
pnpm add -D pytest --filter root
```

---

## ⚡ **Development Workflow**

### **Daily Commands**
```bash
# Build all packages
pnpm nx run-many --target=build

# Test all packages  
pnpm nx run-many --target=test

# Build only affected packages (after changes)
pnpm nx affected --target=build

# Test specific package
pnpm nx run sparse-coding:test

# Publish specific package to PyPI
pnpm nx run sparse-coding:publish

# View project dependency graph
pnpm nx graph
```

### **CI/CD Integration**
```bash
# In CI pipeline
pnpm nx affected --target=test --base=origin/main
pnpm nx affected --target=build --base=origin/main
pnpm nx affected --target=publish --base=origin/main
```

---

## 🎯 **Key Advantages**

### **Performance Benefits**
- ✅ **pnpm**: 60% faster installs, 85MB total vs 130MB per project (npm)
- ✅ **Nx caching**: Skip rebuilding unchanged packages
- ✅ **Affected detection**: Only test/build what changed
- ✅ **Parallel execution**: Run tasks across packages simultaneously

### **Developer Experience**
- ✅ **Single command**: `pnpm install` for entire monorepo
- ✅ **Dependency management**: Shared dependencies automatically deduplicated
- ✅ **Task orchestration**: `nx run-many` for bulk operations
- ✅ **Visual feedback**: `nx graph` shows package relationships

### **Publishing & Deployment**
- ✅ **Independent versioning**: Each package has own version
- ✅ **Selective publishing**: Publish only changed packages  
- ✅ **Automated CI/CD**: GitHub Actions integration
- ✅ **PyPI ready**: Each package configured for PyPI publishing

---

## 🔧 **Migration Strategy**

### **Phase 1: Structure Setup**
1. Create monorepo structure with pnpm-workspace.yaml
2. Move existing packages to `packages/` directory
3. Configure Nx for each package

### **Phase 2: Dependency Management**
1. Consolidate shared dependencies to root
2. Configure package-specific dependencies
3. Test builds and imports

### **Phase 3: Automation**
1. Setup CI/CD pipelines 
2. Configure automated testing
3. Setup PyPI publishing workflows

---

## 📊 **Tool Comparison Summary**

| Tool | Best For | Performance | Learning Curve | 2025 Status |
|------|----------|-------------|----------------|-------------|
| **pnpm + Nx** ⭐ | Python/JS hybrid projects | Excellent | Medium | ✅ Recommended |
| Lerna v6 + Nx | Pure JS projects | Excellent | Easy | ✅ Good alternative |
| Rush (Microsoft) | Enterprise polyglot | Good | Steep | ✅ Enterprise focus |
| Turborepo | JS/TS performance focus | Excellent | Easy | ✅ Simple setup |

**🎯 Recommendation: pnpm + Nx** provides the best balance of performance, features, and Python support for your research packages in 2025.