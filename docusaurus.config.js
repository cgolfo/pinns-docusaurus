// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physics-Informed Neural Networks',
  tagline: 'A Comprehensive Guide to PINNs: Embedding Physics into Deep Learning',
  favicon: 'img/favicon.ico',
  url: 'https://pinns.ai',
  baseUrl: '/',
  organizationName: 'pinns-community',
  projectName: 'pinns-docusaurus',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/pinns-community/pinns-docusaurus/tree/main/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/pinns-community/pinns-docusaurus/tree/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/pinns-social-card.png',
      navbar: {
        title: 'PINNs',
        logo: {
          alt: 'PINNs Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://github.com/idrl-lab/pinns',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Foundations',
                to: '/docs/foundations/introduction',
              },
              {
                label: 'Methods',
                to: '/docs/methods/variants',
              },
              {
                label: 'Research Gaps',
                to: '/docs/research/gaps',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Issues',
                href: 'https://github.com/idrl-lab/pinns/issues',
              },
              {
                label: 'Discussions',
                href: 'https://github.com/idrl-lab/pinns/discussions',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'arXiv',
                href: 'https://arxiv.org',
              },
              {
                label: 'DeepXDE',
                href: 'https://deepxde.readthedocs.io',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} PINN Research Community.`,
      },
      prism: {
        theme: require('prism-react-renderer/themes/github'),
        darkTheme: require('prism-react-renderer/themes/dracula'),
      },
    }),
};

module.exports = config;
