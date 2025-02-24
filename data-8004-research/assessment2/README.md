# AI vs. Human Email Professionalism Research Project

## Project Overview

This repository contains the complete research project examining whether users find AI-generated emails more professional than human-written ones. This collaborative study was conducted by Bogdan Bancescu, Célia Pallier, Charles Watson Ndethi Kibaki, Chingiz Saidov, and Christabel Camilleri as part of Assessment 2 for Research Methods and Tools.

Our research investigated user perceptions of professionalism in email communication by comparing human-written and AI-generated emails on identical topics. Through experimental design and quantitative analysis, we gathered data on how recipients rate professionalism and their ability to identify the source of emails.

## Initial Setup

### Document Preparation
1. All relevant assignment PDFs were uploaded to Claude Projects for reference and analysis
2. Requirements were extracted from course materials (Slides Week 4-6, Assessment 2 Guidelines)
3. The original collaborative Google Doc was referenced for core content and research structure
4. Email pairs (AI and human-written) were collected from each team member on various scenarios

### Environment Configuration
1. A GitHub repository was created for version control and collaboration
2. Quarto was selected as the document preparation system for its academic formatting capabilities
3. R packages were identified for statistical analysis and visualization

## Prompt Engineering Process

Multiple prompt iterations were used throughout the development process:

1. **Initial Research Outline**: Prompts were designed to extract and organize research elements from the original document
2. **Statistical Analysis**: Specialized prompts were created to generate analysis code for the survey data
3. **Visualization Engineering**: Custom prompts were developed to create effective data visualizations
4. **Ethical Guidelines**: Targeted prompts helped integrate comprehensive ethical considerations
5. **Reference Management**: Prompts were used to ensure proper academic citation formatting

## Technical Implementation

### Development Environment
1. **Visual Studio Code**: Used as the primary editor with Quarto extensions installed
2. **GitHub Copilot**: Leveraged for code assistance and suggestions during development
3. **R Integration**: R packages (ggplot2, dplyr, tidyr) were installed and configured
4. **Bibliography**: Created bibliography files (a2-references.bib) with APA citation style (apa.csl)

### Document Configuration

As documented in the repository CHANGELOG, the document underwent several technical enhancements:

1. **YAML Header Configuration**:
   - PDF rendering settings with proper margins and document class
   - LaTeX packages for typography optimization
   - Custom formatting for email examples

2. **LaTeX Enhancements**:
   - Added microtype package for enhanced text spacing
   - Implemented overflow handling with \sloppy and emergency stretch
   - Custom-styled quote environments for email samples

3. **R Package Integration**:
   - Configured setup chunk to load required statistical packages
   - Ensured reproducibility of all analyses

## Data Processing

### Survey Data Analysis
1. Raw data from the email professionalism survey was imported and cleaned
2. Statistical tests were performed to compare professionalism ratings
3. Multiple visualizations were created to represent the findings

### Visualization Development
1. Code chunks were designed for generating consistent plots
2. Enhanced labels, captions and adjusted axis limits were implemented
3. Visualizations were integrated directly into the document flow

### Email Sample Formatting
1. Consistent formatting was applied to all email examples
2. Quote blocks replaced code blocks for better text wrapping
3. Custom styling ensured visual distinction of email content

## Document Structure

The final document includes these main sections:

1. **Introduction**: Research question and hypothesis
2. **Methodology**: Variables, experimental design, and sampling strategy
3. **Ethical Guidelines**: Consent, confidentiality, and data protection
4. **Statistical Analysis**: Quantitative findings from the survey
5. **Results**: Key insights and patterns identified
6. **Discussion**: Interpretation and implications
7. **Appendices**: Email samples used in the study

## Best Practices

### Quarto Markdown Conventions
- Consistent section headings with proper hierarchy
- Code chunks with appropriate options and labels
- Cross-references for figures and tables
- Implementation based on best practices from the [Quarto documentation site](https://quarto.org/docs/guide/)

### Bibliography Management
- Properly formatted citation keys
- APA citation style implementation
- In-text citations linked to bibliography entries

### Visualization Integration
- High-resolution plots with appropriate dimensions
- Consistent styling across all visualizations
- Clear labeling and captioning

### Academic Formatting
- Structured abstract following academic conventions
- Proper section organization following research paper standards
- Professional typography and layout

## Repository Changes

All changes made throughout the project development are documented in the CHANGELOG.md file, which details:
- YAML header and LaTeX configuration enhancements
- Package installation updates
- Addition of the ethical guidelines section
- Statistical analysis and findings updates
- Email sample formatting changes
- Overall structural and style adjustments

This comprehensive documentation ensures reproducibility and transparency in the research process.