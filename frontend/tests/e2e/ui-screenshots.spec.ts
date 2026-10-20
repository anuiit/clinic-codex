// UI screenshot capture harness. Runs only with UI_SHOTS=1:
//   UI_SHOTS=1 npx playwright test ui-screenshots
// Screens land in test-results/ui-screens/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from 'playwright/test';
import { clearIndexedDbRecords, seedIndexedDbRecord } from './storageSeed';

test.skip(process.env.UI_SHOTS !== '1', 'screenshot harness: set UI_SHOTS=1 to run');

const OUT_DIR = fileURLToPath(new URL('../../test-results/ui-screens/', import.meta.url));
const REPO_IMAGE = fileURLToPath(new URL('../../src/test/fixtures/387_769v.jpg', import.meta.url));

function repoImageDataUrl(): string {
  const buffer = fs.readFileSync(REPO_IMAGE);
  return `data:image/jpeg;base64,${buffer.toString('base64')}`;
}

const classNames = ['atl', 'calli', 'cohuatl', 'pantli', 'tochtli', 'xihuitl_2'];

const trustTopK = [
  { class_name: 'atl', confidence: 0.82 },
  { class_name: 'calli', confidence: 0.63 },
  { class_name: 'tochtli', confidence: 0.41 },
  { class_name: 'pantli', confidence: 0.22 },
  { class_name: 'yauhtli', confidence: 0.12 },
];

const similarResults = {
  query: { bbox: [0, 0, 100, 100], mode: 'prototype' },
  best_match: { class_name: 'atl', similarity: 0.82, rejected: false },
  results: [
    { rank: 1, match_type: 'class_prototype', class_name: 'atl', class_label: null, similarity: 0.82, band: 'high', asset: '/samples/atl/a.png' },
    { rank: 2, match_type: 'class_prototype', class_name: 'calli', class_label: null, similarity: 0.63, band: 'high', asset: '/samples/calli/b.png' },
    { rank: 3, match_type: 'class_prototype', class_name: 'tochtli', class_label: null, similarity: 0.41, band: 'moderate', asset: '/samples/tochtli/c.png' },
    { rank: 4, match_type: 'class_prototype', class_name: 'yacatl', class_label: null, similarity: 0.31, band: 'low', asset: null },
    { rank: 5, match_type: 'class_prototype', class_name: 'pantli', class_label: null, similarity: 0.22, band: 'low', asset: '/samples/pantli/d.png' },
    { rank: 6, match_type: 'class_prototype', class_name: 'zacatl', class_label: null, similarity: 0.18, band: 'low', asset: null },
    { rank: 7, match_type: 'class_prototype', class_name: 'yauhtli', class_label: null, similarity: 0.12, band: 'low', asset: null },
  ],
};

const adminQueue = {
  status: 'ok',
  schema_version: 1,
  local_only: true,
  warning: 'local queue',
  counts: { total: 4, pending: 1, approved: 2, rejected: 1, trainable: 2 },
  analyses: [
    {
      analysis_id: 'admin-a',
      uploaded_at: '2026-06-30T20:00:00+00:00',
      image_path: '/tmp/admin-a/image.png',
      image_url: '/admin/annotations/admin-a/image',
      image_exists: true,
      elements: [
        {
          key: 'admin-a:0', analysis_id: 'admin-a', index: 0, class_name: 'atl',
          bbox: [10, 20, 30, 40], crop_path: '/tmp/admin-a/0.png',
          crop_url: '/admin/annotations/admin-a/0/crop', crop_exists: true,
          review_status: 'pending', trainable: false, source_fingerprint: 'p0',
          stale_decision: false, dataset_split: 'excluded', split_reason: 'pending_review',
        },
        {
          key: 'admin-a:1', analysis_id: 'admin-a', index: 1, class_name: 'calli',
          bbox: [40, 50, 60, 70], crop_path: '/tmp/admin-a/1.png',
          crop_url: '/admin/annotations/admin-a/1/crop', crop_exists: true,
          review_status: 'approved', trainable: true, source_fingerprint: 'p1',
          stale_decision: false, dataset_split: 'train', split_reason: 'trainable_hash_80_10_10',
        },
        {
          key: 'admin-a:2', analysis_id: 'admin-a', index: 2, class_name: 'tochtli',
          bbox: [70, 80, 90, 100], crop_path: '/tmp/admin-a/2.png',
          crop_url: '/admin/annotations/admin-a/2/crop', crop_exists: true,
          review_status: 'rejected', trainable: false, source_fingerprint: 'p2',
          stale_decision: false, dataset_split: 'excluded', split_reason: 'rejected_review',
        },
        {
          key: 'admin-a:3', analysis_id: 'admin-a', index: 3, class_name: 'pantli',
          bbox: [100, 110, 120, 130], crop_path: '/tmp/admin-a/3.png',
          crop_url: '/admin/annotations/admin-a/3/crop', crop_exists: true,
          review_status: 'approved', trainable: true, source_fingerprint: 'p3',
          stale_decision: false, dataset_split: 'val', split_reason: 'trainable_hash_80_10_10',
        },
      ],
    },
  ],
  diagnostics: [],
};

const trainingSummary = {
  status: 'ok',
  local_only: true,
  warning: 'local only',
  training_jobs_enabled: true,
  launch_allowed_for_request: true,
  launch_disabled_reasons: [],
  data: {
    total: 4, pending: 1, approved: 2, rejected: 1, trainable: 2,
    classes: ['calli', 'pantli'],
    per_class: { calli: 1, pantli: 1 },
    split_counts: { train: 1, val: 1, test: 0, excluded: 2 },
    diagnostics: [],
  },
  parameters: {
    editable: { dry_run: true, device: ['auto', 'cpu'], batch_size: { default: 16, min: 1, max: 256 } },
    script_env_defaults: { BATCH_SIZE: '16', DEVICE: 'auto' },
    config: { training: { num_epochs: 10 }, model: { backbone: 'dinov2_vits14' } },
  },
  paths: { script: '/repo/scripts/retrain.sh', runs_dir: '/repo/backend/training_runs', model_dir_override_active: false },
  artifacts: {
    approved_export_manifest: { path: '/repo/manifest.json', exists: true, sha256: 'abc' },
    model_registry: { status: 'ok', promoted_version: 'demo' },
  },
  latest_job: null,
};

async function mockApi(page: Page) {
  const repoImage = fs.readFileSync(REPO_IMAGE);
  await page.route('**/version', (route) =>
    route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ app_name: 'Clinic Codex', app_version: '0.1.0', model_version: '1.0.0' }),
    }),
  );
  await page.route('**/auth/me', (route) =>
    route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ auth_enabled: false, user: null }),
    }),
  );
  await page.route('**/classes', (route) =>
    route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ num_classes: 286, class_names: classNames }),
    }),
  );
  await page.route('**/trust', (route) =>
    route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        query: {},
        trust: {
          top1_class: 'atl', top1_similarity: 0.82, margin_to_second: 0.19,
          above_rejection_threshold: true, ambiguous: false, entropy: 0.4,
          top_k: trustTopK,
        },
      }),
    }),
  );
  await page.route('**/similar', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(similarResults) }),
  );
  await page.route('http://localhost:7117/samples/**', (route) =>
    route.fulfill({ status: 200, contentType: 'image/jpeg', body: repoImage }),
  );
  await page.route('**/save-annotation', (route) =>
    route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', analysis_id: 'shot-record', saved_count: 1, classes: ['atl'] }),
    }),
  );
  await page.route('**/admin/annotations', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(adminQueue) }),
  );
  await page.route(/.*\/admin\/annotations\/.*\/(image|crop)$/, (route) =>
    route.fulfill({ status: 200, contentType: 'image/jpeg', body: repoImage }),
  );
  await page.route('**/admin/training/summary', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(trainingSummary) }),
  );
  await page.route('**/admin/training/jobs/latest', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', job: null }) }),
  );
}

async function setTheme(page: Page, theme: 'dark' | 'light') {
  await page.evaluate((mode) => window.localStorage.setItem('clinic-codex-theme', mode), theme);
}

async function shot(page: Page, name: string) {
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT_DIR, `${name}.png`), fullPage: true });
}

async function seedAnalysis(page: Page) {
  const imageDataUrl = repoImageDataUrl();
  const size = await page.evaluate(
    (dataUrl) =>
      new Promise<[number, number]>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve([img.naturalWidth, img.naturalHeight]);
        img.onerror = () => reject(new Error('image load failed'));
        img.src = dataUrl;
      }),
    imageDataUrl,
  );
  const [w, h] = size;
  const record = {
    id: 'shot-record',
    imageName: '387_769v.jpg',
    imageDataUrl,
    timestamp: 1700000000000,
    result: {
      num_elements: 3,
      image_size: [w, h] as [number, number],
      elements: [
        {
          bbox: [w * 0.12, h * 0.15, w * 0.18, h * 0.2] as [number, number, number, number],
          class_name: 'atl', class_label: 0, confidence: 0.82, rejected: false,
          top_k: trustTopK,
        },
        {
          bbox: [w * 0.55, h * 0.3, w * 0.15, h * 0.18] as [number, number, number, number],
          class_name: 'calli', class_label: 1, confidence: 0.63, rejected: false,
          top_k: trustTopK.slice(1),
        },
        {
          bbox: [w * 0.35, h * 0.6, w * 0.2, h * 0.15] as [number, number, number, number],
          class_name: '', class_label: 2, confidence: 0.28, rejected: false,
          top_k: trustTopK.slice(2),
        },
      ],
    },
    annotations: {},
    annotationStatus: { 0: 'validated', 1: 'draft' },
  };
  await seedIndexedDbRecord(page, record);
  return record;
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test.use({ viewport: { width: 1440, height: 900 } });

for (const theme of ['dark', 'light'] as const) {
  test(`capture ${theme} screens`, async ({ page }) => {
    test.setTimeout(180000);
    await page.goto('/login');
    await setTheme(page, theme);
    await clearIndexedDbRecords(page);
    const record = await seedAnalysis(page);

    await page.goto('/login');
    await shot(page, `${theme}-01-login`);

    await page.goto('/');
    await expect(page.getByText('Analyseur de glyphes Codex')).toBeVisible();
    await shot(page, `${theme}-02-workspace-empty-list`);

    await page.goto(`/?analysis=${record.id}`);
    await expect(page.getByTestId('workspace-stage')).toBeVisible();
    await shot(page, `${theme}-03-workspace-analysis`);

    // Enable the archetype gallery flag, select a region: trust panel + thumbnails.
    await page.evaluate(() => window.localStorage.setItem('clinic.showArchetypes', '1'));
    await page.reload();
    await expect(page.getByTestId('workspace-stage')).toBeVisible();
    await page.getByRole('button', { name: /atl région 0/i }).click();
    await expect(page.getByTestId('workspace-focused-selected-card')).toBeVisible();
    await expect(page.locator('[data-archetype-thumb="atl"][data-has-asset="true"]')).toBeVisible();
    await shot(page, `${theme}-04-workspace-region-selected`);

    await page.getByRole('button', { name: 'Annoter la région' }).click();
    await expect(page.getByTestId('annotation-stage')).toBeVisible();
    await shot(page, `${theme}-05-annotation`);

    await page.goto('/admin/annotations/review');
    await expect(page.getByRole('tab', { name: /Trier/i })).toBeVisible();
    await shot(page, `${theme}-06-admin-review`);

    await page.getByRole('tab', { name: /Dataset/i }).click();
    await shot(page, `${theme}-07-admin-dataset`);

    await page.getByRole('tab', { name: /Entraîner/i }).click();
    await shot(page, `${theme}-08-admin-training`);

    await page.goto('/dev');
    await expect(page.getByText('Dev Lab')).toBeVisible();
    await shot(page, `${theme}-09-devlab`);

    // Embedding explorer with mixed neighbor assets (some null) to show the issue.
    await page.setInputFiles('input[type="file"]', REPO_IMAGE);
    await page.getByRole('button', { name: 'Find Neighbors' }).click();
    await expect(page.getByText(/Nearest Neighbors/)).toBeVisible();
    await shot(page, `${theme}-10-devlab-neighbors`);
  });
}
