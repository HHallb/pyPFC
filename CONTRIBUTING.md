# Contributing to pyPFC

Thank you for your interest in contributing to pyPFC. This document provides guidelines for contributing to the project, including bug reports and feature requests.

## Table of Contents

- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Performance Issues](#performance-issues)
- [Documentation Issues](#documentation-issues)
- [Questions and Support](#questions-and-support)

## Bug Reports

**Use the GitHub Issue Templates**: When reporting bugs, please use our [Bug Report template](https://github.com/HHallb/pyPFC/issues/new/choose) which will guide you through providing all necessary information.

The bug report template will ask for:

### Required Information

1. **pyPFC version**: Use `pip show pypfc`
2. **Operating System**: Windows, macOS, or Linux distribution
3. **Python version**: Use `python --version`
4. **PyTorch version**: Use `torch.__version__`
5. **GPU information** (if applicable): GPU model and CUDA version

### Bug Report Template

```markdown
**pyPFC Version:** 
**OS:** 
**Python Version:** 
**PyTorch Version:** 
**GPU/CUDA Version:** 

**Description:**
A clear and concise description of what the bug is.

**To Reproduce:**
Steps to reproduce the behavior:
1. 
2. 
3. 

**Expected Behavior:**
A clear description of what you expected to happen.

**Actual Behavior:**
What actually happened, including any error messages.

**Code Sample:**
Please provide a minimal code example that reproduces the issue:

**Additional Context:**
Add any other context about the problem here.
```

## Feature Requests

**Use the GitHub Issue Templates**: For feature requests, please use our [Feature Request template](https://github.com/HHallb/pyPFC/issues/new/choose) to explain your needs.

We welcome feature requests! The template will guide you to provide:

1. **Clear description** of the proposed feature
2. **Use case**: Why would this feature be useful?
3. **Implementation ideas** (if you have any)
4. **Examples** of how the feature would be used

## Performance Issues

For performance-related problems, use our [Performance Issue template](https://github.com/HHallb/pyPFC/issues/new/choose) which collects:

- Hardware specifications and benchmarking data
- Simulation parameters and memory usage
- Profiling information and optimization attempts

## Documentation Issues

To report documentation problems or suggest improvements, use our [Documentation template](https://github.com/HHallb/pyPFC/issues/new/choose) for:

- Missing or unclear documentation
- API reference improvements
- Tutorial and example requests
- Website or README issues

## Questions and Support

### Getting Help

- **Documentation**: [https://hhallb.github.io/pyPFC/](https://hhallb.github.io/pyPFC/)
- **GitHub Issues**: Use the issue templates for bug reports, feature requests, performance issues, and documentation
- **GitHub Discussions**: For questions and general discussion (if enabled)

### Before Asking for Help

1. Check the [documentation](https://hhallb.github.io/pyPFC/)
2. Try the examples in the `examples/` directory
3. Check that your environment meets the requirements

### When Asking Questions

For questions, you can:

- Create a [GitHub Issue](https://github.com/HHallb/pyPFC/issues/new/choose) using the appropriate template
- Check existing issues for similar problems

When asking questions:

- Provide a minimal code example
- Include error messages and stack traces
- Specify your environment (OS, Python version, etc.)
- Describe what you expected vs. what happened
