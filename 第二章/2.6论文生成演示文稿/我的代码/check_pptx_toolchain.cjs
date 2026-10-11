// 由用户的03检查脚本调用，只检查依赖与浏览器启动，不生成PPT。
const path = require('path');

async function main() {
  const root = path.resolve(__dirname, '..');
  const modules = {};
  for (const name of ['pptxgenjs', 'playwright', 'sharp', 'react', 'react-dom', 'react-icons']) {
    require(name);
    modules[name] = require.resolve(name);
  }
  require(path.join(root, 'skills', 'pptx', 'scripts', 'html2pptx.js'));
  const browser = await require('playwright').chromium.launch({ headless: true });
  try {
    const version = browser.version();
    process.stdout.write(JSON.stringify({ modules, official_js_loadable: true,
      chromium_launch_ok: true, chromium_version: version }));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  process.stderr.write(String(error.stack || error));
  process.exitCode = 1;
});
