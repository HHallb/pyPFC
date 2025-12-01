# Security Policy

## Reporting a Vulnerability

We take the security of pyPFC seriously. If you believe you have found a security vulnerability in pyPFC, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send an email to the project maintainer with the following information:

- **Email**: [Your email here - replace with actual maintainer email]
- **Subject**: "Security vulnerability in pyPFC"

### What to Include

Please include the following information in your report:

1. **Description**: A clear description of the vulnerability
2. **Impact**: What an attacker could potentially do
3. **Steps to Reproduce**: Detailed steps to reproduce the vulnerability
4. **Affected Versions**: Which versions of pyPFC are affected
5. **Environment**: Operating system, Python version, PyTorch version, CUDA version
6. **Proof of Concept**: Code example demonstrating the vulnerability (if applicable)

## Security Best Practices

When using pyPFC, please follow these security best practices:

### Input Validation

- Validate all input parameters, especially those from external sources
- Be cautious when loading configuration files from untrusted sources
- Sanitize file paths to prevent directory traversal attacks

### Data Handling

- Be careful when processing simulation data from untrusted sources
- Validate grid dimensions and parameters to prevent memory exhaustion
- Use appropriate error handling to prevent information disclosure

### Dependencies

- Keep PyTorch and other dependencies updated to their latest secure versions
- Regularly audit your dependencies for known vulnerabilities
- Use virtual environments to isolate pyPFC installations

### Computational Resources

- Be aware of potential denial-of-service through resource exhaustion
- Set appropriate limits on simulation parameters (grid size, time steps, etc.)
- Monitor memory and CPU usage in production environments

## Known Security Considerations

### Memory Usage

- Large grid simulations can consume significant memory
- GPU memory limitations may cause crashes with oversized simulations
- Consider implementing memory checks before starting large simulations

### File I/O Operations

- Output files may contain sensitive simulation data
- Ensure proper file permissions when writing simulation results
- Be cautious about file paths in configuration files

### Numerical Stability

- Extreme parameter values may lead to numerical instabilities
- Validate physical parameters to ensure they are within reasonable ranges
- Implement bounds checking for critical simulation parameters

## Updates and Patches

Security updates will be:

- Released as soon as possible after verification
- Documented in the CHANGELOG.md
- Announced through GitHub releases
- Tagged with appropriate version numbers

## Contact

For security-related questions or concerns that are not vulnerabilities, you can:

- Check our documentation at [https://hhallb.github.io/pyPFC/](https://hhallb.github.io/pyPFC/)
- Review this security policy for updates

---

**Note**: This security policy may be updated from time to time. Please check back regularly for the most current version.
