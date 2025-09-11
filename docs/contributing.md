# Contributing to Sparse Coding

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- pip or conda

### Installation for Development
```bash
# Clone the repository
git clone https://github.com/benedictchen/sparse-coding.git
cd sparse-coding

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies
```bash
pip install -e ".[dev,test,docs]"
```

This installs:
- **Testing**: pytest, pytest-cov, hypothesis
- **Linting**: ruff, black, mypy
- **Documentation**: mkdocs, mkdocs-material, mkdocstrings
- **Development**: pre-commit, build, twine

## Code Standards

### Style Guidelines
- **Black** for code formatting
- **Ruff** for linting and import sorting  
- **MyPy** for type checking
- **pytest** for testing

Run all checks:
```bash
# Format code
black src/ tests/

# Lint and fix
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ --cov=src/sparse_coding
```

### Research Implementation Standards

#### Mathematical Accuracy
All implementations must be research-accurate:
- Cite original papers in docstrings
- Include mathematical formulations
- Implement algorithms exactly as published
- Validate against reference implementations

Example:
```python
def fista_step(self, y: ArrayLike, D: ArrayLike, penalty: PenaltyProtocol) -> ArrayLike:
    """
    FISTA proximal gradient step.
    
    Research Foundation:
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    
    Mathematical Formulation:
    a_{k+1} = prox_{η·ψ}(y_k - η∇f(y_k))
    where f(a) = (1/2)||x - Da||²
    """
    gradient = self._compute_gradient(y, D)
    proximal_input = y - self.step_size * gradient
    return penalty.prox(proximal_input, self.step_size)
```

#### Configuration Design
Use dataclasses with validation:
```python
@dataclass
class SolverConfig:
    solver_type: SolverType
    max_iter: int = 1000
    tol: float = 1e-6
    adaptive_restart: bool = True
    
    def __post_init__(self):
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
```

#### Error Handling
Provide explicit error context:
```python
def create_penalty(penalty_type: str, **kwargs):
    try:
        return PENALTY_REGISTRY[penalty_type](**kwargs)
    except KeyError:
        available = list(PENALTY_REGISTRY.keys())
        raise ValueError(f"Unknown penalty '{penalty_type}'. Available: {available}")
    except TypeError as e:
        raise TypeError(f"Invalid parameters for {penalty_type}: {e}")
```

## Testing Requirements

### Test Coverage
- Minimum 90% test coverage
- 100% coverage for critical algorithms
- Property-based testing for mathematical functions

### Test Categories

#### Unit Tests
```python
def test_l1_penalty_mathematical_properties():
    """Test L1 penalty mathematical properties."""
    penalty = L1Penalty(lam=0.1)
    
    # Non-negativity
    a = np.random.randn(10)
    assert penalty.value(a) >= 0
    
    # Homogeneity
    assert abs(penalty.value(2*a) - 2*penalty.value(a)) < 1e-10
```

#### Integration Tests
```python
def test_sparse_coder_end_to_end():
    """Test complete sparse coding pipeline."""
    X = np.random.randn(64, 100)
    sc = SparseCoder(n_atoms=128, mode='l1')
    sc.fit(X, n_steps=10)
    
    codes = sc.encode(X)
    reconstructed = sc.decode(codes)
    
    assert codes.shape == (128, 100)
    assert reconstructed.shape == X.shape
```

#### Property-Based Tests
```python
@hypothesis.given(
    data=st.lists(st.floats(-10, 10), min_size=1, max_size=100),
    lam=st.floats(0.001, 1.0)
)
def test_l1_proximal_properties(data, lam):
    """Property-based testing for L1 proximal operator."""
    penalty = L1Penalty(lam=lam)
    a = np.array(data)
    
    # Proximal operator properties
    prox_result = penalty.prox(a, 1.0)
    assert prox_result.shape == a.shape
    assert np.all(np.abs(prox_result) <= np.abs(a))
```

## Documentation Standards

### Docstring Format
Use research-oriented docstrings:
```python
def solve(self, D: ArrayLike, X: ArrayLike, penalty: PenaltyProtocol) -> ArrayLike:
    """
    Solve sparse coding problem using FISTA algorithm.
    
    Research Foundation:
    Beck & Teboulle (2009) SIAM Journal on Imaging Sciences
    
    Mathematical Problem:
    minimize (1/2)||X - DA||²_F + ψ(A)
       A
    
    Parameters
    ----------
    D : ArrayLike, shape (n_features, n_atoms)
        Dictionary matrix with normalized columns
    X : ArrayLike, shape (n_features, n_samples)  
        Data matrix to encode
    penalty : PenaltyProtocol
        Penalty function implementing value() and prox()
        
    Returns
    -------
    ArrayLike, shape (n_atoms, n_samples)
        Sparse coefficient matrix
        
    Notes
    -----
    Convergence rate: O(1/k²) for convex penalty functions
    """
```

### API Documentation
- Use mkdocstrings for automatic API docs
- Include mathematical formulations
- Provide working code examples
- Link to research papers

## Contribution Workflow

### 1. Issue Creation
Before starting work:
- Check existing issues and pull requests
- Create issue describing the problem/enhancement
- Discuss approach with maintainers

### 2. Development Process
```bash
# Create feature branch
git checkout -b feature/algorithm-improvement

# Make changes following standards
# Add tests for all new functionality
# Update documentation

# Run full test suite
pytest tests/ --cov=src/sparse_coding --cov-report=html

# Check code quality
black src/ tests/
ruff check src/ tests/
mypy src/
```

### 3. Pull Request Requirements
- [ ] All tests pass
- [ ] Code coverage ≥ 90%
- [ ] Documentation updated
- [ ] Research citations included
- [ ] Mathematical accuracy verified
- [ ] Performance regression tests pass

### Pull Request Template
```markdown
## Description
Brief description of changes

## Research Foundation
- Paper citations for new algorithms
- Mathematical formulation changes
- Validation against reference implementations

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Property-based tests for mathematical functions
- [ ] Performance benchmarks run

## Documentation
- [ ] API documentation updated
- [ ] User guide sections added/modified
- [ ] Research foundations documented
```

## Algorithm Contribution Guidelines

### New Penalty Functions
1. Implement `PenaltyProtocol` interface
2. Provide mathematical formulation in docstring
3. Include proximal operator if available
4. Add comprehensive tests for mathematical properties
5. Benchmark against existing implementations

### New Solvers
1. Implement `SolverProtocol` interface
2. Include convergence analysis
3. Provide research citations
4. Test convergence guarantees
5. Compare performance with existing solvers

### Performance Improvements
1. Benchmark against current implementation
2. Maintain mathematical accuracy
3. Test with various problem sizes
4. Document performance characteristics
5. Ensure numerical stability

## Release Process

### Version Numbering
Follow semantic versioning:
- **MAJOR**: Breaking API changes
- **MINOR**: New features, algorithm additions
- **PATCH**: Bug fixes, documentation updates

### Release Checklist
- [ ] All tests pass on CI
- [ ] Documentation builds successfully
- [ ] Performance benchmarks run
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Git tag created
- [ ] PyPI package uploaded
- [ ] GitHub release created

## Getting Help

### Resources
- **Documentation**: [Read the Docs](https://sparse-coding.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/benedictchen/sparse-coding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/benedictchen/sparse-coding/discussions)

### Research Context
When contributing algorithm improvements:
1. Cite original research papers
2. Verify mathematical correctness
3. Compare with published results
4. Include convergence analysis
5. Test on standard datasets

### Code Review Process
1. Automatic checks (CI/CD pipeline)
2. Research accuracy review
3. Performance impact assessment
4. Documentation review
5. Final approval by maintainers

## Research Ethics

### Attribution Requirements
- Always cite original research papers
- Include author names and publication details
- Acknowledge algorithm sources in code
- Maintain research integrity

### Non-Commercial License
This project uses a custom non-commercial license:
- Research and educational use encouraged
- Commercial use requires permission
- Contributions welcome under same license
- See LICENSE file for full terms

## Support the Project

### Funding
- [GitHub Sponsors](https://github.com/sponsors/benedictchen)
- [PayPal Donations](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)

### Non-Financial Contributions
- Algorithm implementations
- Bug reports and fixes
- Documentation improvements
- Performance optimizations
- Research validation
- Community support

## Acknowledgments

This project implements algorithms from numerous research papers. See the research foundations documentation for complete citations and acknowledgments to the original algorithm developers.