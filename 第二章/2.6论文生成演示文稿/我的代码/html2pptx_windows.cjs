// Windows薄适配；不修改固定版本官方Skill，也不生成内容或布局。
const path = require('path');
const { chromium } = require('playwright');
const launch = chromium.launch.bind(chromium);
chromium.launch = (options = {}) => launch({
  ...options,
  env: { ...process.env, ...(options.env || {}) },
});
const official = require(path.resolve(__dirname, '../skills/pptx/scripts/html2pptx.js'));
module.exports = (htmlFile, presentation, options) => official(
  path.resolve(htmlFile).replace(/\\/g, '/'), presentation, options,
);
