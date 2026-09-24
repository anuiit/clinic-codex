import { defineConfig, devices } from 'playwright/test';

const baseURL = process.env.CLINIC_E2E_BASE_URL || 'http://localhost:7118';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // The app uses one shared Vite dev server plus IndexedDB-heavy flows.
  // Keep browser e2e deterministic locally and in CI; previous fully parallel
  // runs produced false-red page/context timeouts while the same suite passed
  // consistently with one worker.
  workers: 1,
  reporter: 'list',
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${new URL(baseURL).port} --strictPort`,
    url: baseURL,
    reuseExistingServer: true,
    timeout: 30000,
  },
});
