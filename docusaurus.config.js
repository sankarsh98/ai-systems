// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI Systems',
  tagline: 'How agents use tools, and how models run fast and cheap',
  favicon: 'img/favicon.ico',
  future: {v4: true},

  url: 'https://sankarsh98.github.io',
  baseUrl: '/ai-systems/',
  organizationName: 'sankarsh98',
  projectName: 'ai-systems',

  onBrokenLinks: 'throw',
  markdown: {hooks: {onBrokenMarkdownLinks: 'throw'}},

  i18n: {defaultLocale: 'en', locales: ['en']},

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/sankarsh98/ai-systems/tree/main/',
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],

  themeConfig: {
    colorMode: {respectPrefersColorScheme: true},
    navbar: {
      title: 'AI Systems',
      items: [
        {type: 'docSidebar', sidebarId: 'courseSidebar', position: 'left', label: 'Course'},
        {href: 'https://github.com/sankarsh98/ai-systems', label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'Built with Docusaurus. Code in the repo is runnable.',
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python'],
    },
  },
};

export default config;
