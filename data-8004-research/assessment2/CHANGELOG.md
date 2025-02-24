# Changelog
## [Unreleased] - 2025-02-24

### Added
- **Ethical Guidelines Section** (Added on 2025-02-24)  
  - Inserted a new section titled *Ethical Guidelines* immediately after the "Average Professionalism Ratings by Source" code chunk in the Statistical Analysis section.
  - The section covers topics including Informed Consent, Anonymity and Confidentiality, Data Protection and GDPR Compliance, Potential Risks, and Voluntary Participation & Data Sharing.
- **R Package Installation Instructions** (Added on 2025-02-24)  
  - Included inline instructions for installing essential R packages (`ggplot2`, `dplyr`, `tidyr`) to ensure reproducibility.
- **Commitizen Guidelines** (Added on 2025-02-24)  
  - Detailed steps for using Commitizen in the CONTRIBUTING.md file to generate standardized commit messages.
- **Environment Setup Enhancements** (Added on 2025-02-24)  
  - Updated environment setup instructions to include Microsoft Visual Studio, Visual Studio Code, and GitHub Copilot for enhanced development productivity.

### Changed
- **YAML Header & LaTeX Configuration** (Updated on 2025-02-24)  
  - Enhanced document settings with `keep-tex: true`, `documentclass: article`, and `geometry: "margin=1in"` for a consistent page layout.
  - Added LaTeX packages (`microtype`, `\sloppy`, and `\setlength{\emergencystretch}{3em}`) to improve text wrapping and typography.
  - Customized the `quote` environment using `etoolbox` to set quote blocks in a smaller Courier-style font (`\small\ttfamily`).
- **Email Sample Formatting** (Updated on 2025-02-24)  
  - Converted all email sample sections from fenced code blocks to quote blocks to ensure proper text wrapping and visual distinction.
- **Document Structure and Flow** (Updated on 2025-02-24)  
  - Reorganized sections for improved logical flow: Experimental Design, Ethical Guidelines, Statistical Analysis and Findings, followed by Discussion and Appendices.

### Fixed
- **Overfull Margin Issues** (Fixed on 2025-02-24)  
  - Resolved text overrunning issues by transitioning email samples to quote blocks and adjusting LaTeX settings.
- **R Package Loading Errors** (Fixed on 2025-02-24)  
  - Provided instructions for installing missing R packages to avoid errors during document rendering.

---

*This changelog reflects all modifications made on 2025-02-24 aimed at enhancing readability, ensuring reproducibility, and standardizing contributions throughout the project.*