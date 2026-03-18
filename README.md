# Physics-Informed Neural Networks (PINNs) Documentation

A comprehensive Docusaurus site documenting the theory, methods, and research frontiers of Physics-Informed Neural Networks.

## Features

✨ **Complete Coverage**
- Foundational theory and canonical frameworks
- 65+ papers analyzed and synthesized
- Major PINN variants (VPINNs, cPINNs, XPINNs, B-PINNs, etc.)
- Neural operators and operator learning
- Training and optimization techniques
- 8 concrete research gaps suitable for novel contributions

🎯 **Structured Learning Path**
- Introduction → Foundations → Methods → Applications
- Quick-start coding examples
- Theory-to-practice integration

📚 **Literature Integration**
- Curated bibliography across sections
- Links to arXiv, DOI, and GitHub repositories
- Citation counts and impact metrics

🔬 **Research Frontiers**
- Detailed analysis of 8 open problems
- Connected to current methodological gaps
- Suitable for top-tier venue contributions

## Project Structure

```
pinns-docusaurus/
├── docs/
│   ├── introduction.md
│   ├── why-pinns.md
│   ├── quick-start.md
│   ├── foundations/
│   │   ├── introduction.md
│   │   ├── canonical-framework.md
│   │   ├── survey-overview.md
│   │   └── [core concepts]
│   ├── methods/
│   │   ├── variants.md
│   │   ├── vpinn.md
│   │   ├── cpinn-xpinn.md
│   │   └── [more methods]
│   ├── tools/
│   │   ├── deepxde.md
│   │   ├── modulus.md
│   │   └── [libraries]
│   ├── training/
│   │   ├── overview.md
│   │   ├── spectral-bias.md
│   │   ├── loss-balancing.md
│   │   └── [optimization techniques]
│   ├── applications/
│   │   ├── fluids.md
│   │   ├── solids.md
│   │   └── [domain applications]
│   ├── research/
│   │   ├── gaps-overview.md
│   │   ├── gap-1-multiphysics.md
│   │   ├── gap-2-ntk-theory.md
│   │   └── [all 8 gaps]
│   └── literature/
│       ├── foundational.md
│       ├── variants.md
│       └── [categorized references]
├── blog/
├── src/
│   ├── css/custom.css
│   ├── components/
│   └── pages/
├── static/
│   └── img/
├── docusaurus.config.js
├── sidebars.js
├── package.json
└── README.md
```

## Installation & Setup

### Prerequisites
- Node.js 16+ and npm/yarn

### Quick Start

1. **Clone or download the project**
   ```bash
   cd pinns-docusaurus
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start development server**
   ```bash
   npm run start
   # or
   yarn start
   ```

   The site will open at `http://localhost:3000`

4. **Build for production**
   ```bash
   npm run build
   ```

   Output will be in the `build/` directory.

## Available Scripts

- `npm start` — Start development server with hot reload
- `npm run build` — Build static site for deployment
- `npm run serve` — Serve production build locally
- `npm run clear` — Clear cache
- `npm run write-heading-ids` — Auto-generate heading IDs

## Customization

### Adding New Documentation

1. Create a new `.md` file in appropriate subdirectory
2. Add frontmatter with metadata:
   ```yaml
   ---
   id: unique-id
   title: Page Title
   sidebar_position: 2
   ---
   ```
3. Update `sidebars.js` if adding a new section
4. Reference the file in the navigation structure

### Styling

- Modify `src/css/custom.css` for global styles
- Use CSS variables defined in root for consistent theming
- Dark mode automatically supported via `:root` and `[data-theme='dark']`

### Configuration

Main settings in `docusaurus.config.js`:
- Site title, logo, favicon
- Navbar and footer links
- Social cards and metadata
- Syntax highlighting theme

## Deployment

### GitHub Pages

```bash
npm run deploy
```

Requires proper `docusaurus.config.js` configuration (organizationName, projectName, etc.)

### Vercel

1. Connect GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `build`
4. Deploy!

### Netlify

1. Connect GitHub repository
2. Build command: `npm run build`
3. Publish directory: `build`

### Self-hosted

1. Build: `npm run build`
2. Serve `build/` directory with static HTTP server
3. Example: `python -m http.server 8000` (Python) or `npx serve build` (Node.js)

## Content Organization

### Sections

- **Introduction**: Overview and value proposition
- **Foundations**: Core theory, automatic differentiation, loss functions
- **Methods**: Variants (VPINN, cPINN, XPINN, etc.), neural operators
- **Tools**: DeepXDE, Modulus, SciANN, other libraries
- **Training**: Optimization, spectral bias, loss balancing, NTK, causal training
- **Applications**: Fluids, solids, electromagnetics, inverse problems
- **Research Gaps**: 8 open problems with detailed analysis
- **Literature**: Categorized references (~65 papers)

## Key Resources

- **DeepXDE**: [deepxde.readthedocs.io](https://deepxde.readthedocs.io)
- **NVIDIA Modulus**: [docs.nvidia.com/modulus](https://docs.nvidia.com/modulus)
- **PINNpapers**: [github.com/idrl-lab/PINNpapers](https://github.com/idrl-lab/PINNpapers)
- **awesome-pinn**: [github.com/erfanhamdi/awesome-pinn](https://github.com/erfanhamdi/awesome-pinn)

## Contributing

To contribute to this documentation:

1. Fork or clone the repository
2. Create a branch for your changes
3. Edit or add `.md` files
4. Test locally: `npm start`
5. Submit a pull request

## Citation

If you use this documentation, please cite:

```bibtex
@misc{pinns-docusaurus,
  title={Physics-Informed Neural Networks: Comprehensive Documentation},
  author={PINN Research Community},
  year={2026},
  url={https://pinns.ai}
}
```

## License

This documentation is provided as an educational resource.

## Acknowledgments

- **Raissi, Perdikaris, Karniadakis** for the foundational PINN framework
- **Docusaurus team** for the excellent documentation platform
- **PINN community** for open-source libraries and research papers

---

**Last Updated**: March 2026  
**Literature Coverage**: ~65 papers through early 2026  
**Curated by**: PINN Research Community
