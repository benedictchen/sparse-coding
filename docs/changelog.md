# Changelog

## [2.1.0] - 2025-01-11

### Added
- Complete MkDocs documentation site with Material theme
- Professional API documentation with comprehensive algorithm coverage
- Performance benchmarking suite with scikit-learn comparison
- GitHub Actions CI/CD pipeline with automated PyPI publishing
- Comprehensive contributing guidelines for developers

### Fixed
- `HAS_SCIPY` undefined variable causing decode() method failures
- Missing documentation pages breaking navigation links
- Import path issues in benchmark scripts
- Sparse matrix handling in decode() method

### Improved
- Documentation structure with user guides, API reference, and research foundations
- Error handling with explicit context and debugging information
- Performance optimization with parallel processing
- Code quality with comprehensive type annotations

## [2.0.1] - 2024-12-15

### Fixed
- Critical bug in transcoder mode nonlinear decoder training
- Memory leak in batch processing for large datasets
- Numerical stability issues in NCG optimization

### Changed
- Updated documentation with latest research references
- Improved error messages for better debugging

## [2.0.0] - 2024-11-30

### Added
- Transcoder mode with nonlinear decoder (2025 research)
- Adaptive atom management for dynamic dictionary sizing
- GPU acceleration support with CuPy integration
- Comprehensive monitoring and logging system
- Lambda annealing for improved convergence

### Changed
- **BREAKING**: Moved to src layout for better packaging
- **BREAKING**: Updated API for better consistency
- Improved numerical stability across all algorithms
- Enhanced parallel processing capabilities

### Removed
- **BREAKING**: Deprecated old API methods
- Legacy flat layout structure

## [1.2.0] - 2024-10-15

### Added
- Complete Olshausen & Field (1996) reproduction mode
- Homeostatic balancing mechanisms
- Dead atom detection and reinitialization
- Research-accurate parameter validation

### Fixed
- Convergence issues in log-prior optimization
- Memory usage in large dictionary scenarios
- Gradient clipping in NCG methods

### Improved
- Algorithm selection guidance
- Performance optimization for large datasets
- Documentation with mathematical foundations

## [1.1.1] - 2024-09-20

### Fixed
- Import errors in monitoring modules
- Compatibility with NumPy 1.24+
- Edge cases in dictionary normalization

### Changed
- Updated dependencies for better stability
- Improved test coverage to 95%

## [1.1.0] - 2024-08-10

### Added
- TensorBoard integration for training monitoring
- CSV export functionality for offline analysis
- Batch processing optimizations
- Sparse matrix support throughout

### Improved
- FISTA algorithm convergence rate
- Memory efficiency for large problems
- Documentation with usage examples

### Fixed
- Numerical precision issues in edge cases
- Parallel processing on Windows systems

## [1.0.0] - 2024-07-01

### Added
- Initial stable release
- Multiple sparse coding algorithms:
  - L1 regularization with FISTA
  - Log-prior with NCG optimization
  - Pure Olshausen & Field implementation
- MOD dictionary learning
- Comprehensive test suite
- Basic documentation

### Features
- Research-accurate implementations
- Production-ready performance
- Extensive parameter validation
- Cross-platform compatibility

## [0.9.0] - 2024-06-15

### Added
- Beta release for testing
- Core sparse coding functionality
- Dictionary learning algorithms
- Basic monitoring capabilities

### Known Issues
- Limited documentation
- Performance not fully optimized
- Some edge cases not handled

## [0.1.0] - 2024-05-01

### Added
- Initial development release
- Basic FISTA implementation
- Simple dictionary learning
- Proof of concept functionality

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, algorithm additions
- **PATCH**: Bug fixes, documentation updates

## Migration Guides

### Migrating from 1.x to 2.x

**Breaking Changes**:

1. **Src Layout**: Package moved to src layout structure
```python
# Old import (1.x)
from sparse_coding.sparse_coder import SparseCoder

# New import (2.x)
from sparse_coding import SparseCoder  # Simplified
```

2. **API Consistency**: Some method names standardized
```python
# Old API (1.x)
sc.learn_dictionary(X, iterations=30)

# New API (2.x)  
sc.fit(X, n_steps=30)  # Consistent with scikit-learn
```

3. **Configuration**: Parameter validation stricter
```python
# Old (1.x) - would accept invalid values
sc = SparseCoder(n_atoms=-10)  # Silently failed

# New (2.x) - explicit validation
sc = SparseCoder(n_atoms=128)  # ValueError if invalid
```

### Migrating from 0.x to 1.x

**Major Changes**:

1. **API Stabilization**: Method signatures finalized
2. **Algorithm Names**: Standardized naming convention
3. **Error Handling**: Comprehensive exception system
4. **Documentation**: Complete API reference added

## Contributors

- **Benedict Chen** - Lead Developer and Researcher
- **Community Contributors** - Bug reports, feature requests, testing

## License

This project is licensed under a Custom Non-Commercial License with Donation Requirements.

## Support

- **Documentation**: [https://sparse-coding.readthedocs.io](https://sparse-coding.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/benedictchen/sparse-coding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/benedictchen/sparse-coding/discussions)
- **Funding**: [GitHub Sponsors](https://github.com/sponsors/benedictchen)

## Acknowledgments

This implementation is based on foundational research by:

- **Olshausen, B. A., & Field, D. J. (1996)** - Original sparse coding algorithm
- **Beck, A., & Teboulle, M. (2009)** - FISTA optimization method  
- **Engan, K., Aase, S. O., & Husoy, J. H. (1999)** - MOD dictionary learning

We thank the research community for developing these fundamental algorithms that make modern sparse coding possible.