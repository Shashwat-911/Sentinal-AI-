const { defineConfig, devices } = require('@playwright/test');
const port = Number(process.env.PLAYWRIGHT_PORT || 9102);
module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  globalTimeout: 5 * 60 * 1000,
  expect: { timeout: 10000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }], ['junit', { outputFile: 'test-results/junit.xml' }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'node scripts/playwright-server.cjs',
    url: `http://127.0.0.1:${port}/`,
    timeout: 60000,
    reuseExistingServer: false,
    env: { PLAYWRIGHT_PORT: String(port) },
  },
});
