# Contributing to AI vs. Human Email Professionalism Research Project

Thank you for considering contributing to this collaborative research project. We welcome your ideas, bug reports, feature requests, and code contributions!

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/your-repo/issues) page to report bugs or request new features.
- Provide a clear and descriptive title, including steps to reproduce issues if applicable.
- Label your issue appropriately (e.g., bug, enhancement, documentation).

### Submitting Pull Requests

1. **Fork the Repository:**  
   Fork the repository to your personal GitHub account and clone it locally.

2. **Create a New Branch:**  
   Create a branch for your changes:
   ```bash
   git checkout -b your-feature-or-bugfix
   ```

3. **Make Your Changes:**  
   - Follow the established coding styles:
     - For R code, follow tidyverse style guidelines.
     - For Quarto Markdown documents, maintain consistency with the YAML header and document structure.
   - Ensure that your changes do not conflict with existing formatting, especially for email samples and visualization code.

4. **Write Clear Commit Messages with Commitizen:**  
   We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to generate standardized commit messages.
   - If you haven't yet, install Commitizen globally:
     ```bash
     npm install -g commitizen
     ```
   - To initialize Commitizen in the repository (if not already set up), run:
     ```bash
     commitizen init cz-conventional-changelog --save-dev --save-exact
     ```
   - When committing your changes, use:
     ```bash
     git add .
     git cz
     ```
   This will prompt you to select a commit type and fill in a commit message that follows the conventional changelog standard.

5. **Submit a Pull Request (PR):**  
   - Push your branch to your fork:
     ```bash
     git push origin your-feature-or-bugfix
     ```
   - Open a PR against the main repository branch and reference any related issues.

## Code and Documentation Standards

- **YAML & LaTeX:**  
  Adhere to the document setup in the YAML header, ensuring margins, typography, and custom styling are maintained.

- **R Package Integration:**  
  Ensure any new code or dependencies (e.g., additional R packages) are documented in both the code and the repository setup instructions.  
  If needed, include installation instructions, for example:
  ```r
  install.packages("ggplot2")
  install.packages("dplyr")
  install.packages("tidyr")
  ```

- **Documentation:**  
  Update the README, CHANGELOG, or relevant documentation if your changes affect the project’s functionality or setup.

- **Testing & Rendering:**  
  Verify your changes by rendering the Quarto document:
  ```bash
  quarto render <your-document>.qmd
  ```

## Environment Setup

Before contributing, ensure you have the following installed:
- [Quarto](https://quarto.org/docs/get-started/)
- R with required packages (e.g., ggplot2, dplyr, tidyr)
- A LaTeX distribution (e.g., TeX Live on WSL) for PDF rendering
- [Node.js](https://nodejs.org/) and npm (for Commitizen)
- **Microsoft Visual Studio Code** or **Microsoft Visual Studio** (your preferred development environment)
- **GitHub Copilot** for enhanced code editing support

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). Please ensure your interactions and contributions are respectful and adhere to these guidelines.

## Questions & Support

If you have any questions or need support, please open an issue or join our discussion forums on GitHub Discussions.

Thank you for your contributions and for helping us improve this project!

*Happy Contributing!*