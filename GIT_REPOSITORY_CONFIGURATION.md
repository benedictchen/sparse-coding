# 🔧 Git Repository Configuration Guide

## 📋 **CRITICAL: Agent Instructions for Git Repository Management**

This document provides essential configuration information for all agents working on this research project.

---

## 🎯 **Project Structure Overview**

```
research_papers/                    ← ROOT (master coordination only)
├── RESEARCH_PACKAGES/              ← Contains all individual packages
│   ├── sparse_coding/              ← INDIVIDUAL GIT REPO
│   ├── reservoir_computing/        ← INDIVIDUAL GIT REPO  
│   ├── information_bottleneck/     ← INDIVIDUAL GIT REPO
│   ├── holographic_memory/         ← INDIVIDUAL GIT REPO
│   ├── inductive_logic_programming/ ← INDIVIDUAL GIT REPO
│   ├── universal_learning/         ← INDIVIDUAL GIT REPO
│   ├── qualitative_reasoning/      ← INDIVIDUAL GIT REPO
│   ├── tensor_product_binding/     ← INDIVIDUAL GIT REPO
│   └── self_organizing_maps/       ← INDIVIDUAL GIT REPO
├── SHARED_UTILITIES/               ← Shared tools (part of root)
├── RULES.md                        ← Project governance
└── PROJECT_MAP.md                  ← Navigation guide
```

---

## 📦 **Individual Package Git Repository Mapping**

| Local Package Directory | GitHub Repository | Purpose |
|--------------------------|-------------------|---------|
| `sparse_coding/` | `benedictchen/sparse-coding` | 🧠 Biological vision principles - Olshausen & Field (1996) |
| `reservoir_computing/` | `benedictchen/reservoir-computing-benedictchen` | 🌊 Echo State Networks & Liquid State Machines - Jaeger/Maass |
| `information_bottleneck/` | `benedictchen/information-bottleneck` | 💧 Information theory foundation - Tishby (1999) |
| `holographic_memory/` | `benedictchen/holographic-memory` | 🌀 Vector symbolic architecture - Plate (1995) |
| `inductive_logic_programming/` | `benedictchen/inductive-logic-programming` | 🔍 Learning from examples - Muggleton (1991) |
| `universal_learning/` | `benedictchen/universal-learning` | 🧠 AIXI theoretical framework - Hutter (2005) |
| `qualitative_reasoning/` | `benedictchen/qualitative-reasoning` | 🤔 Commonsense physics - Forbus (1984) |
| `tensor_product_binding/` | `benedictchen/tensor-product-binding` | 🔗 Compositional representations - Smolensky (1990) |
| `self_organizing_maps/` | `benedictchen/self-organizing-maps` | 🗺️ Unsupervised learning - Kohonen (1982) |

---

## 🔧 **Git Configuration Commands for Each Package**

### **Initial Setup (when package needs git initialization):**
```bash
cd RESEARCH_PACKAGES/[package_name]
git init
git remote add origin https://github.com/benedictchen/[repo-name].git
git add .
git commit -m "Initial modularized codebase"
git push -u origin main
```

### **Daily Development Workflow:**
```bash
# Work within specific package
cd RESEARCH_PACKAGES/[package_name]

# Make changes to code
# ... development work ...

# Commit changes to package-specific repo
git add .
git commit -m "Descriptive commit message"
git push origin main
```

---

## ⚠️  **CRITICAL RULES FOR AGENTS**

### **✅ DO:**
1. **Work within individual package directories** when modifying code
2. **Commit to the appropriate package repository** for changes
3. **Use package-specific git repos** for all version control
4. **Check current directory** before running git commands
5. **Navigate to correct package** before committing changes

### **❌ NEVER:**
1. **Don't commit package changes to root repository**
2. **Don't mix changes across multiple packages** in single commit
3. **Don't work on packages from root directory**
4. **Don't assume git commands apply to root repo**
5. **Don't create cross-package dependencies** without coordination

---

## 🎯 **Agent Decision Tree for Git Operations**

```
🤔 Agent wants to make changes to code?
│
├── 📦 Changes are to a specific research package (sparse_coding, etc.)
│   ├── ✅ Navigate to RESEARCH_PACKAGES/[package_name]/
│   ├── ✅ Make changes within that directory
│   ├── ✅ Use git commands from within package directory
│   └── ✅ Commit to package-specific GitHub repository
│
└── 🏗️ Changes are to project structure (RULES.md, PROJECT_MAP.md, etc.)
    ├── ✅ Work from root directory
    ├── ✅ Coordinate changes across project
    └── ✅ Use root git repository (if needed for coordination)
```

---

## 🔍 **Verification Commands**

### **Check which git repo you're in:**
```bash
pwd                    # Show current directory
git remote -v         # Show git remote URLs
git status            # Show git repository status
```

### **Expected output for package directory:**
```bash
# From RESEARCH_PACKAGES/sparse_coding/
$ git remote -v
origin  https://github.com/benedictchen/sparse-coding.git (fetch)
origin  https://github.com/benedictchen/sparse-coding.git (push)
```

---

## 📊 **Package Status Tracking**

| Package | Git Status | Last Updated | Notes |
|---------|------------|-------------|-------|
| sparse_coding | ✅ Configured | 2025-09-01 | Modularized, duplicates archived |
| reservoir_computing | 🔧 Needs setup | TBD | Modularized with esn_modules/, lsm_modules/ |
| information_bottleneck | 🔧 Needs setup | TBD | Modularized with ib_modules/ |
| holographic_memory | 🔧 Needs setup | TBD | Modularized with hm_modules/ |
| inductive_logic_programming | 🔧 Needs setup | TBD | Modularized with ilp_modules/ |
| universal_learning | 🔧 Needs setup | TBD | Modularized with solomonoff_modules/ |
| qualitative_reasoning | 🔧 Needs setup | TBD | Modularized with qr_modules/ |
| tensor_product_binding | 🔧 Needs setup | TBD | Modularized with tpb_modules/ |
| self_organizing_maps | 🔧 Needs setup | TBD | Modularized with som_modules/ |

---

## 🚀 **PyPI Publishing Configuration**

Each package is designed for **independent PyPI publishing**:

- **Individual `pyproject.toml`** in each package directory
- **Separate versioning** for each package
- **Independent release cycles** 
- **Modular dependencies** between packages allowed but not required

---

## 📝 **Future Agent Reminders**

1. **Always check `pwd`** before git operations
2. **Each package is its own universe** for development purposes
3. **Cross-package changes** require coordination and multiple commits
4. **Package modularization** is complete - work within modular structure
5. **old_archive/ folders** contain superseded files ready for deletion
6. **Backup files (*_BACKUP.py)** are safety nets - don't delete without verification

---

**🎯 SUMMARY: Each research package is an independent git repository connected to its own GitHub repo. Always work within the specific package directory when making changes to that package's code.**