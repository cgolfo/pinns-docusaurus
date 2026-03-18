/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a docs page layout (if the docs plugin is enabled)
 - provide next/previous navigation

 The sidebars can be generated from the file structure or explicitly defined here.

 Create as many sidebars as you want.
 */

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: '🎯 Introduction',
      items: [
        'introduction',
        'why-pinns',
        'quick-start',
      ],
    },
    {
      type: 'category',
      label: '📚 Foundations',
      items: [
        'foundations/introduction',
        'foundations/canonical-framework',
        'foundations/survey-overview',
        {
          type: 'category',
          label: 'Core Concepts',
          items: [
            'foundations/auto-differentiation',
            'foundations/loss-functions',
            'foundations/optimization',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '🔬 Methods & Variants',
      items: [
        'methods/variants',
      ],
    },
    {
      type: 'category',
      label: '🏗️ Architectures & Tools',
      items: [
        'tools/deepxde',
        'tools/sciann',
        'tools/modulus',
      ],
    },
    {
      type: 'category',
      label: '⚙️ Training & Optimization',
      items: [
        'training/overview',
        'training/spectral-bias',
      ],
    },
    {
      type: 'category',
      label: '🎯 Applications',
      items: [
        'applications/fluids',
        'applications/solids',
      ],
    },
    {
      type: 'category',
      label: '🔍 Research Gaps & Future Directions',
      items: [
        'research/gaps-overview',
        'research/gap-1-multiphysics',
      ],
    },
    {
      type: 'category',
      label: '📖 Literature & References',
      items: [
        'literature/foundational',
      ],
    },
  ],
};

module.exports = sidebars;
