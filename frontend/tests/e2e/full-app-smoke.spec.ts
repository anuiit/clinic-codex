import { expect, test, type Page, type Route } from 'playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { clearIndexedDbRecords, seedIndexedDbRecord } from './storageSeed';
import type { AdminTrainingSummary } from '../../src/types';

const onePxPng =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const smokeRecord = {
  id: 'full-smoke-record',
  imageName: 'full-smoke-glyph.png',
  imageDataUrl: onePxPng,
  timestamp: 1700000000000,
  result: {
    num_elements: 2,
    image_size: [400, 300] as [number, number],
    elements: [
      {
        bbox: [50, 60, 80, 70] as [number, number, number, number],
        class_name: 'glyph-a',
        class_label: 0,
        confidence: 0.91,
        rejected: false,
        top_k: [{ class_name: 'glyph-alpha', confidence: 0.88 }],
      },
      {
        bbox: [190, 120, 90, 75] as [number, number, number, number],
        class_name: '',
        class_label: 1,
        confidence: 0.32,
        rejected: false,
        top_k: [{ class_name: 'beta', confidence: 0.44 }],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated', 1: 'draft' },
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
          key: 'admin-a:0',
          revision: 0,
          analysis_id: 'admin-a',
          index: 0,
          class_name: 'atl',
          bbox: [10, 20, 30, 40],
          crop_path: '/tmp/admin-a/0.png',
          crop_url: '/admin/annotations/admin-a/0/crop',
          crop_exists: true,
          review_status: 'pending',
          trainable: false,
          source_fingerprint: 'p0',
          stale_decision: false,
          dataset_split: 'excluded',
          split_reason: 'pending_review',
        },
        {
          key: 'admin-a:1',
          revision: 0,
          analysis_id: 'admin-a',
          index: 1,
          class_name: 'bet',
          bbox: [40, 50, 60, 70],
          crop_path: '/tmp/admin-a/1.png',
          crop_url: '/admin/annotations/admin-a/1/crop',
          crop_exists: true,
          review_status: 'approved',
          trainable: true,
          source_fingerprint: 'p1',
          stale_decision: false,
          dataset_split: 'train',
          split_reason: 'trainable_hash_80_10_10',
        },
        {
          key: 'admin-a:2',
          revision: 0,
          analysis_id: 'admin-a',
          index: 2,
          class_name: 'rej',
          bbox: [70, 80, 90, 100],
          crop_path: '/tmp/admin-a/2.png',
          crop_url: '/admin/annotations/admin-a/2/crop',
          crop_exists: true,
          review_status: 'rejected',
          trainable: false,
          source_fingerprint: 'p2',
          stale_decision: false,
          dataset_split: 'excluded',
          split_reason: 'rejected_review',
        },
        {
          key: 'admin-a:3',
          revision: 0,
          analysis_id: 'admin-a',
          index: 3,
          class_name: 'gimel',
          bbox: [100, 110, 120, 130],
          crop_path: '/tmp/admin-a/3.png',
          crop_url: '/admin/annotations/admin-a/3/crop',
          crop_exists: true,
          review_status: 'approved',
          trainable: true,
          source_fingerprint: 'p3',
          stale_decision: false,
          dataset_split: 'val',
          split_reason: 'trainable_hash_80_10_10',
        },
      ],
    },
  ],
  diagnostics: [],
};

const trainingSummary: AdminTrainingSummary = {
  status: 'ok',
  local_only: true,
  warning: 'local only',
  training_jobs_enabled: true,
  launch_allowed_for_request: true,
  launch_disabled_reasons: [],
  training_snapshot: {
    mode: 'local_prior',
    configured: true,
    valid: true,
    snapshot_id: null,
    snapshot_manifest_sha256: null,
    row_count: 2,
    class_count: 2,
    live_annotation_count: 2,
    live_train_count: 2,
    ready_for_training: true,
    promotion_evaluation_ready: false,
    split_counts: null,
    paths: {},
    errors: [],
  },
  data: {
    total: 4,
    pending: 1,
    approved: 2,
    rejected: 1,
    trainable: 2,
    classes: ['bet', 'gimel'],
    per_class: { bet: 1, gimel: 1 },
    split_counts: { train: 1, val: 1, test: 0, excluded: 2 },
    diagnostics: [],
  },
  parameters: {
    editable: {
      dry_run: true,
      device: ['auto', 'cpu'],
      batch_size: { default: 16, min: 1, max: 256 },
    },
    script_env_defaults: { BATCH_SIZE: '16', DEVICE: 'auto' },
    config: { training: { num_epochs: 10 }, model: { backbone: 'dinov2_vits14' } },
  },
  paths: {
    script: '/repo/scripts/retrain.sh',
    runs_dir: '/repo/backend/training_runs',
    model_dir_override_active: false,
  },
  artifacts: {
    approved_export_manifest: { path: '/repo/manifest.json', exists: true, sha256: 'abc' },
    model_registry: { status: 'ok', promoted_version: 'demo' },
  },
  latest_job: null,
  training_snapshot: { data_revision: 'a'.repeat(64), valid: true },
};

async function mockApi(page: Page) {
  await page.route('**/admin/annotations/*/*/history', route => route.fulfill({
    json: { status: 'ok', revision: 0, history: [] },
  }));
  await page.route('**/admin/classes', route => route.fulfill({
    json: { revision: 'catalogue-1', classes: ['glyph-a', 'glyph-alpha', 'beta', 'atl'].map(class_name => ({
      class_name, class_label: null, status: 'active', counts: {pending: 0, approved: 0, rejected: 0}, trainable_count: 0,
    })) },
  }));
  await page.route('**/version', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        app_name: 'Clinic Codex',
        app_version: '0.1.0',
        model_version: '1.0.0',
      }),
    }),
  );
  await page.route('**/auth/me', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ auth_enabled: false, user: null }),
    }),
  );
  await page.route(url => ['/classes', '/annotation-classes'].includes(url.pathname), (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        num_classes: 4,
        class_names: ['glyph-a', 'glyph-alpha', 'beta', 'atl'],
      }),
    }),
  );
  await page.route('**/trust', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        query: {},
        trust: {
          top1_class: 'glyph-a',
          top1_similarity: 0.9,
          margin_to_second: 0.3,
          above_rejection_threshold: true,
          ambiguous: false,
          entropy: 0.1,
          top_k: [],
        },
      }),
    }),
  );
  await page.route('**/save-annotation', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        analysis_id: smokeRecord.id,
        saved_count: 1,
        classes: ['glyph-a'],
      }),
    }),
  );
  await page.route('**/admin/annotations', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(adminQueue) }),
  );
  await page.route(/.*\/admin\/annotations\/.*\/(image|crop)$/, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'image/png',
      body: Buffer.from(onePxPng.split(',')[1], 'base64'),
    }),
  );
  await page.route(/.*\/admin\/annotations\/.*\/(review|modify)$/, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }),
  );
  await page.route('**/admin/training/summary', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(trainingSummary) }),
  );
  await page.route('**/admin/training/jobs/latest', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', job: null }) }),
  );
  await page.route('**/admin/training/jobs', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        job: {
          run_id: 'dry-1',
          status: 'running',
          dry_run: true,
          device: 'auto',
          batch_size: 16,
          started_at: new Date().toISOString(),
          exit_code: null,
          log_tail: ['started'],
        },
      }),
    }),
  );
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test('browser Back keeps an unsaved annotation when leaving is declined', async ({ page }) => {
  await page.goto('/');
  await clearIndexedDbRecords(page);
  await seedIndexedDbRecord(page, smokeRecord);
  await page.goto(`/?analysis=${smokeRecord.id}`);
  await page.getByRole('button', { name: /glyph-a région 0/i }).click();
  await page.getByRole('button', { name: 'Annoter la région' }).click();
  await page.getByLabel(/Nommer l’élément 0/).fill('new-glyph');

  const dialogPromise = page.waitForEvent('dialog', { timeout: 3000 });
  await page.evaluate(() => window.history.back());
  const dialog = await dialogPromise;
  await dialog.dismiss();

  await expect(page).toHaveURL(new RegExp(`/annotate/${smokeRecord.id}\\?element=0$`));
  await expect(page.getByLabel(/Nommer l’élément 0/)).toHaveValue('new-glyph');
});

test('browser Back keeps a note draft before the field loses focus', async ({ page }) => {
  await page.goto('/');
  await clearIndexedDbRecords(page);
  await seedIndexedDbRecord(page, smokeRecord);
  await page.goto(`/?analysis=${smokeRecord.id}`);
  await page.getByRole('button', { name: /glyph-a région 0/i }).click();
  await page.getByRole('button', { name: 'Annoter la région' }).click();
  await page.getByTestId('annotation-element-note').fill('note en cours');

  const dialogPromise = page.waitForEvent('dialog', { timeout: 3000 });
  await page.evaluate(() => window.history.back());
  await (await dialogPromise).dismiss();

  await expect(page).toHaveURL(new RegExp(`/annotate/${smokeRecord.id}\\?element=0$`));
  await expect(page.getByTestId('annotation-element-note')).toHaveValue('note en cours');
});

test('browser Back keeps an unsaved admin correction when leaving is declined', async ({ page }) => {
  await page.goto('/');
  await page.goto('/admin/annotations/review');
  await page.getByRole('button', { name: /^corriger$/i }).click();
  await page.getByLabel("Nom de l'élément").fill('new-glyph');

  await page.getByRole('link', { name: /retour à l’analyse/i }).evaluate((node) => {
    const modifiedClick = new MouseEvent('click', { bubbles: true, cancelable: true, ctrlKey: true });
    modifiedClick.preventDefault();
    node.dispatchEvent(modifiedClick);
  });

  const dialogPromise = page.waitForEvent('dialog', { timeout: 3000 });
  await page.evaluate(() => window.history.back());
  const dialog = await dialogPromise;
  await dialog.dismiss();

  await expect(page).toHaveURL(/\/admin\/annotations\/review$/);
  await expect(page.getByLabel("Nom de l'élément")).toHaveValue('new-glyph');
});

test('classes can recover after a failed catalogue request', async ({ page }) => {
  let recovered = false;
  await page.route('**/admin/classes', route => {
    if (!recovered) return route.fulfill({ status: 503, body: 'Unavailable' });
    return route.fulfill({ json: {
      revision: 'catalogue-2',
      classes: [{ class_name: 'glyph-a', class_label: 0, status: 'active', counts: { pending: 0, approved: 1, rejected: 0 }, trainable_count: 1 }],
    } });
  });
  await page.goto('/admin/annotations/classes');
  await expect(page.getByRole('alert')).toContainText('Catalogue des classes indisponible');
  recovered = true;
  await page.getByRole('button', { name: 'Réessayer' }).click();
  await page.getByRole('region', { name: 'Gestion des classes' }).locator('details summary').click();
  await expect(page.getByText('glyph-a', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Réessayer' })).toHaveCount(0);
});

test('comparison shows two complete annotated pages and synchronized region details', async ({ page }) => {
  let regionCount = 2;
  let latestComparisonJob: unknown = null;
  const source = readFileSync(path.resolve('src/test/fixtures/387_769v.jpg'));
  const base = { class_name: 'atl', confidence: 0.88, rejected: false, top_k: [{ class_name: 'atl', confidence: 0.88 }] };
  const candidate = { class_name: 'calli', confidence: 0.74, rejected: false, top_k: [{ class_name: 'calli', confidence: 0.74 }] };
  await page.route('**/admin/training/models', route => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({ versions: [{ version_id: 'candidate-1', status: 'candidate' }] }),
  }));
  await page.route('**/admin/training/jobs/latest', route => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', job: latestComparisonJob }),
  }));
  await page.route('**/admin/training/comparisons', async route => {
    expect(route.request().postDataJSON().analysis_id).toBe('admin-a');
    const job = {
      run_id: 'compare-1', kind: 'comparison', model_version_id: 'candidate-1',
      status: 'succeeded', dry_run: false, device: 'cpu', batch_size: 16,
      comparison: {
        sample_count: regionCount, warnings: [],
        protocol: { evaluation_scope: 'ad_hoc', metrics_scope: 'ad_hoc', base_historical_independence: 'unknown', device: 'cpu' },
        metrics: {
          common: { support: 1, base_correct: 1, candidate_correct: 0, gains: 0, regressions: 1, unchanged: 0, both_wrong: 0 },
          new_classes: { support: 0, candidate_correct: 0 }, coverage: { base: 1, candidate: 1 },
          latency_ms: { base: 12, candidate: 13 }, by_scope: {}, top3: { base_correct: 1, candidate_correct: 0 }, per_class: {},
        },
        rows: [
          { sample_id: 's0', analysis_id: 'admin-a', index: 0, image_name: '387_769v.jpg',
            bbox: [120, 200, 110, 140], source_image_size: [750, 1210], source_image_sha256: 'same',
            scope: 'ad_hoc', review_status: 'approved', candidate_exposure: 'train',
            expected_class: 'atl', base_supported: true, base, candidate, outcome: 'regression',
            crop_url: '/admin/training/jobs/compare-1/samples/s0', source_image_url: '/admin/training/jobs/compare-1/pages/s0' },
          { sample_id: 's1', analysis_id: 'admin-a', index: 1, image_name: '387_769v.jpg',
            bbox: [360, 510, 90, 100], source_image_size: [750, 1210], source_image_sha256: 'same',
            scope: 'ad_hoc', review_status: 'pending', candidate_exposure: 'train',
            expected_class: null, base_supported: true, base, candidate: base, outcome: 'unchanged',
            crop_url: '/admin/training/jobs/compare-1/samples/s1', source_image_url: '/admin/training/jobs/compare-1/pages/s1' },
        ].slice(0, Math.min(regionCount, 2)).concat(regionCount === 10 ? Array.from({ length: 8 }, (_, offset) => ({
          sample_id: 's' + (offset + 2), analysis_id: 'admin-a', index: offset + 2, image_name: '387_769v.jpg',
          bbox: [40 + offset * 75, 650 + (offset % 3) * 130, 55, 60], source_image_size: [750, 1210], source_image_sha256: 'same',
          scope: 'ad_hoc', review_status: 'pending', candidate_exposure: 'train',
          expected_class: null, base_supported: true, base, candidate: base, outcome: 'unchanged',
          crop_url: '/admin/training/jobs/compare-1/samples/s' + (offset + 2), source_image_url: '/admin/training/jobs/compare-1/pages/s0',
        })) : []),
      },
    };
    latestComparisonJob = job;
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', job }) });
  });
  await page.route(/\/admin\/training\/jobs\/compare-1\/pages\//, route =>
    route.fulfill({ status: 200, contentType: 'image/jpeg', body: source }));
  await page.route(/\/admin\/training\/jobs\/compare-1\/samples\//, route =>
    route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.from(onePxPng.split(',')[1], 'base64') }));
  await page.goto('/admin/annotations/compare');
  await page.getByRole('button', { name: 'Comparer', exact: true }).click();
  const active = page.getByRole('region', { name: 'Modèle actif' });
  const challenger = page.getByRole('region', { name: 'Modèle candidat' });
  await expect(active.locator('[data-overlay-region="true"]')).toHaveCount(2);
  await expect(challenger.locator('[data-overlay-region="true"]')).toHaveCount(2);
  await expect(active.locator('[data-box-id="s1"] rect[stroke="#2563eb"]')).toHaveCount(1);
  expect((await active.boundingBox())!.height).toBeLessThan(650);
  const sourceImageBox = (await active.locator('img').first().boundingBox())!;
  expect(sourceImageBox.y + sourceImageBox.height).toBeLessThanOrEqual(page.viewportSize()!.height);
  await expect.poll(() => active.locator('img').first().evaluate(image => (image as HTMLImageElement).naturalWidth)).toBe(750);
  await expect.poll(() => challenger.locator('img').first().evaluate(image => (image as HTMLImageElement).naturalWidth)).toBe(750);
  await page.screenshot({ path: 'test-results/comparison-overview.png' });
  await page.getByRole('navigation', { name: 'Régions comparées' }).getByRole('button', { name: /Région #1/ }).click();
  await expect(active.locator('[data-box-id="s1"]')).toHaveAttribute('data-selected', 'true');
  await expect(challenger.locator('[data-box-id="s1"]')).toHaveAttribute('data-selected', 'true');
  await expect(page.getByText('Région #1')).toBeVisible();
  await page.getByRole('region', { name: 'Comparaison des modèles' }).evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: 'test-results/comparison-page.png', fullPage: true });
  regionCount = 1;
  await page.getByRole('button', { name: 'Comparer', exact: true }).click();
  await expect(active.locator('[data-overlay-region="true"]')).toHaveCount(1);
  await expect(challenger.locator('[data-overlay-region="true"]')).toHaveCount(1);
  await expect(page.getByText(/cette page n’a qu’une région enregistrée/i)).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Régions comparées' })).toHaveCount(0);
  regionCount = 10;
  await page.getByRole('button', { name: 'Comparer', exact: true }).click();
  await expect(active.locator('[data-overlay-region="true"]')).toHaveCount(10);
  await expect(challenger.locator('[data-overlay-region="true"]')).toHaveCount(10);
  await expect(page.getByRole('navigation', { name: 'Régions comparées' }).getByRole('button')).toHaveCount(10);
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/comparison-10-regions.png' });
  await page.getByRole('combobox', { name: 'Mettre en évidence' }).selectOption('regression');
  await expect(active.locator('[data-overlay-region="true"]')).toHaveCount(10);
  await expect(page.getByText(/Mise en évidence : 1\/10 régions/i)).toBeVisible();
  await page.getByRole('tab', { name: /Trier/i }).click();
  await page.getByRole('tab', { name: /Comparer/i }).click();
  await expect(active.locator('[data-overlay-region="true"]')).toHaveCount(10);
  await expect(challenger.locator('[data-overlay-region="true"]')).toHaveCount(10);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(active).toBeVisible();
  await expect(challenger).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/comparison-mobile.png' });
});

test('shipped page example creates a real browser history entry', async ({ page }) => {
  await page.route('**/segment', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      num_elements: 1,
      image_size: [750, 1210],
      elements: [{
        bbox: [20, 30, 40, 50],
        class_name: 'atl',
        class_label: 0,
        confidence: 0.8,
        rejected: false,
        top_k: [{ class_name: 'atl', confidence: 0.8 }],
      }],
    }),
  }));
  await page.goto('/');
  await clearIndexedDbRecords(page);
  await page.reload();

  await page.getByText('Essayer avec 3 images du corpus').click();
  await page.getByRole('button', { name: "Analyser l'exemple 387_769v.jpg" }).click();
  await expect(page.getByTestId('workspace-history-list').getByText('387_769v.jpg')).toBeVisible();
  await expect(page.getByTestId('workspace-stage').getByRole('img', { name: '387_769v.jpg' })).toBeVisible();
});

test('full app smoke: workspace, annotation, and admin tabs stay wired', async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', (error) => consoleErrors.push(error.message));

  await page.goto('/');
  await clearIndexedDbRecords(page);
  await seedIndexedDbRecord(page, smokeRecord);
  await page.goto(`/?analysis=${smokeRecord.id}`);

  await expect(page.getByRole('banner').getByRole('heading', { name: 'Analyse' })).toBeVisible();
  await expect(page.getByRole('banner').getByRole('button', { name: 'Importer une image' })).toBeVisible();
  await page.getByRole('banner').getByText('Options', { exact: true }).click();
  await expect(page.getByTestId('runtime-version')).toContainText(
    'Clinic Codex v0.1.0',
  );
  await expect(page.getByTestId('runtime-version')).toContainText(
    'Modèle v1.0.0',
  );
  await page.getByRole('banner').getByText('Options', { exact: true }).click();
  await expect(
    page.getByTestId('workspace-stage').getByRole('img', { name: smokeRecord.imageName }),
  ).toBeVisible();
  await expect(page.getByTestId('image-bbox-stage-box-0')).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-workspace.png' });

  await page.getByRole('button', { name: /glyph-a région 0/i }).click();
  await expect(page.getByTestId('workspace-focused-selected-card')).toBeVisible();
  await page.getByRole('button', { name: 'Annoter la région' }).click();
  await expect(page).toHaveURL(new RegExp(`/annotate/${smokeRecord.id}\\?element=0$`));

  await expect(
    page.getByTestId('annotation-stage').getByRole('img', { name: smokeRecord.imageName }),
  ).toBeVisible();
  await expect(page.getByTestId('runtime-version')).toContainText(
    'Modèle v1.0.0',
  );
  const renameInput = page.getByLabel(/Nommer l’élément 0/);
  await expect(renameInput).toBeVisible();
  await renameInput.fill('gl');
  const suggestionMenu = page.getByTestId('element-name-suggestions');
  await expect(suggestionMenu).toBeVisible();
  await expect(suggestionMenu).toContainText('glyph-alpha');
  await expect
    .poll(() => suggestionMenu.evaluate((node) => node.parentElement?.tagName.toLowerCase()))
    .toBe('body');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Ajuster à la vue' }).click();
  await expect(page.getByText('100%')).toBeVisible();
  await page.getByRole('button', { name: 'Zoom avant' }).click();
  await expect(page.getByText('125%')).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-annotation.png' });

  await page.goto('/admin/annotations/review');
  await expect(page.getByTestId('runtime-version')).toContainText(
    'Clinic Codex v0.1.0',
  );
  await expect(page.getByRole('tab', { name: /Trier/i })).toHaveAttribute('aria-selected', 'true');
  const triageList = page.getByRole('listbox', { name: /file de triage/i });
  await expect(triageList).toBeVisible();
  await expect(triageList.getByRole('option', { name: /atl/i })).toBeVisible();
  await expect(triageList.getByRole('option', { name: /rej/i })).toBeVisible();
  await expect(triageList.getByRole('option', { name: /bet/i })).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-review.png' });

  await page.getByRole('tab', { name: /Dataset/i }).click();
  await expect(page).toHaveURL(/\/admin\/annotations\/dataset/);
  const datasetList = page.getByRole('listbox', { name: /Vue dataset/i });
  await expect(datasetList).toBeVisible();
  await expect(datasetList.getByRole('option', { name: /bet/i })).toBeVisible();
  await expect(datasetList.getByRole('option', { name: /gimel/i })).toBeVisible();
  await expect(datasetList.getByText(/rej/)).toHaveCount(0);
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-dataset.png' });

  await page.getByRole('tab', { name: /Classes/i }).click();
  await expect(page).toHaveURL(/\/admin\/annotations\/classes/);
  await expect(page.getByRole('heading', { name: 'Classes disponibles' })).toBeVisible();
  await expect(page.getByRole('searchbox', { name: 'Rechercher une classe' })).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-classes.png' });

  await page.getByRole('tab', { name: /Entraîner/i }).click();
  await expect(page).toHaveURL(/\/admin\/annotations\/training/);
  await expect(page.getByRole('heading', { name: 'Créer un candidat', exact: true })).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') await page.screenshot({ path: 'test-results/ux-training.png' });
  await page.getByRole('button', { name: 'Vérifier la préparation', exact: true }).click();
  await expect(page.getByRole('button', { name: 'En cours…', exact: true })).toBeVisible();
  if (process.env.CAPTURE_FULL_APP_VISUALS === '1') {
    await page.screenshot({ path: 'test-results/ux-training-running.png' });
    const candidateJob = {
      run_id: 'candidate-run', kind: 'training', status: 'succeeded', dry_run: false,
      device: 'cpu', batch_size: 16, model_version_id: 'candidate-1',
      result: {
        unique_count: 2, duplicate_count: 0, updated_classes: ['bet', 'gimel'],
        base_correct: 1, active_correct: 1, candidate_correct: 2, train_count: 2,
        holdout: { support: 1, base_correct: 1, candidate_correct: 1 },
        generalization_validated: false,
      },
    };
    await page.route('**/admin/training/summary', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ ...trainingSummary, latest_training_job: candidateJob }),
    }));
    await page.getByRole('tab', { name: /Classes/i }).click();
    await page.getByRole('tab', { name: /Entraîner/i }).click();
    await expect(page.getByText(/Apprentissage : actuel 1\/2 · candidat 2\/2/i)).toBeVisible();
    await page.screenshot({ path: 'test-results/ux-training-candidate.png' });
  }

  expect(consoleErrors.filter((line) => !line.includes('favicon'))).toEqual([]);
});
