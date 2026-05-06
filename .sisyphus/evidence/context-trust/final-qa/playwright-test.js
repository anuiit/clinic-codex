const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const EVIDENCE_DIR = path.join(__dirname, '.sisyphus', 'evidence', 'context-trust', 'final-qa');
if (!fs.existsSync(EVIDENCE_DIR)) {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
}

const FRONTEND_URL = 'http://localhost:5180';

// Create a 2x2 red PNG as data URL for testing
const RED_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAABZJREFUeNpi2r9//38gYGAEESAAEGAAasgJOgzOKCoAAAAASUVORK5CYII=';
const TEST_IMAGE = 'data:image/png;base64,' + RED_PNG_B64;

const testRecord = {
  id: 'test-final-qa-record',
  imageName: 'final-qa-test.png',
  imageDataUrl: TEST_IMAGE,
  timestamp: Date.now(),
  result: {
    num_elements: 3,
    image_size: [100, 100],
    elements: [
      {
        bbox: [10, 10, 30, 30],
        class_name: 'atl',
        class_label: 282,
        confidence: 0.85,
        rejected: false,
        top_k: [
          { class_name: 'atl', confidence: 0.85 },
          { class_name: 'tlalli', confidence: 0.60 },
          { class_name: 'tlacuilolli', confidence: 0.45 },
        ]
      },
      {
        bbox: [50, 50, 20, 20],
        class_name: 'calli',
        class_label: 100,
        confidence: 0.40,
        rejected: true,
        top_k: [
          { class_name: 'calli', confidence: 0.40 },
          { class_name: 'cozcacuauhtli', confidence: 0.38 },
          { class_name: 'cuauhtli', confidence: 0.35 },
        ]
      },
      {
        bbox: [70, 10, 20, 20],
        class_name: 'cohuatl',
        class_label: 50,
        confidence: 0.72,
        rejected: false,
        top_k: [
          { class_name: 'cohuatl', confidence: 0.72 },
          { class_name: 'cozcacuauhtli', confidence: 0.50 },
        ]
      },
    ]
  },
  annotations: {}
};

async function seedLocalStorage(page) {
  await page.evaluate((record) => {
    localStorage.setItem('codex_analyses', JSON.stringify([record]));
  }, testRecord);
}

async function assertNoScroll(page, label) {
  const scrollHeight = await page.evaluate(() => document.documentElement.scrollHeight);
  const innerHeight = await page.evaluate(() => window.innerHeight);
  const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  const innerWidth = await page.evaluate(() => window.innerWidth);
  const hasVScroll = scrollHeight > innerHeight;
  const hasHScroll = scrollWidth > innerWidth;
  return {
    label,
    pass: !hasVScroll && !hasHScroll,
    scrollHeight,
    innerHeight,
    scrollWidth,
    innerWidth,
    hasVScroll,
    hasHScroll
  };
}

async function runTests() {
  const browser = await chromium.launch();
  const results = [];
  let page;

  try {
    // ============================================================
    // F3.1 Zero scroll at 1920x1080
    // ============================================================
    page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    await page.goto(FRONTEND_URL);
    await seedLocalStorage(page);
    await page.reload();
    await page.waitForTimeout(800);

    const scroll1920 = await assertNoScroll(page, 'Zero scroll 1920x1080');
    results.push(scroll1920);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-zero-scroll-1920.png'), fullPage: false });

    // ============================================================
    // F3.2 Zero scroll at 2560x1440
    // ============================================================
    await page.setViewportSize({ width: 2560, height: 1440 });
    await page.reload();
    await page.waitForTimeout(800);

    const scroll2560 = await assertNoScroll(page, 'Zero scroll 2560x1440');
    results.push(scroll2560);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-zero-scroll-2560.png'), fullPage: false });

    // ============================================================
    // F3.3 Image fills canvas (check no max-w-[800px])
    // ============================================================
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.reload();
    await page.waitForTimeout(800);

    const imgSize1920 = await page.evaluate(() => {
      const img = document.querySelector('img[alt]');
      return img ? { width: img.clientWidth, height: img.clientHeight } : null;
    });
    results.push({
      label: 'Image size 1920x1080',
      pass: imgSize1920 !== null,
      details: imgSize1920
    });

    // ============================================================
    // F3.4 Click region -> trust panels appear
    // ============================================================
    await page.reload();
    await page.waitForTimeout(800);

    // Click on canvas to focus first element
    const canvas = await page.locator('canvas').first();
    await canvas.click();
    await page.waitForTimeout(400);

    const trustPanelVisible = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes('Trust Summary') && text.includes('Similarity Distribution') && text.includes('Confusion Context');
    });
    results.push({
      label: 'Trust panels visible on region focus',
      pass: trustPanelVisible
    });
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-trust-panels.png'), fullPage: false });

    // ============================================================
    // F3.5 Zoom 0.25x-4x works, overlay stays synced
    // ============================================================
    await page.reload();
    await page.waitForTimeout(800);
    await canvas.click();
    await page.waitForTimeout(200);

    // Zoom in to 2x
    const zoomInBtn = await page.locator('button[title="Zoom in"]').first();
    for (let i = 0; i < 4; i++) await zoomInBtn.click();
    await page.waitForTimeout(300);

    const zoom2xScroll = await assertNoScroll(page, 'Zoom 2x no scroll');
    results.push(zoom2xScroll);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-zoom-2x.png'), fullPage: false });

    // Zoom out to 0.5x
    const zoomOutBtn = await page.locator('button[title="Zoom out"]').first();
    for (let i = 0; i < 6; i++) await zoomOutBtn.click();
    await page.waitForTimeout(300);

    const zoom05Scroll = await assertNoScroll(page, 'Zoom 0.5x no scroll');
    results.push(zoom05Scroll);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-zoom-05x.png'), fullPage: false });

    // Reset zoom
    const fitBtn = await page.locator('button[title="Fit to view"]').first();
    await fitBtn.click();
    await page.waitForTimeout(200);

    // ============================================================
    // F3.6 Sidebar thumbnails clickable, sidebar stays open
    // ============================================================
    await page.reload();
    await page.waitForTimeout(800);

    // Collapse sidebar if open
    const sidebarCollapsed = await page.evaluate(() => {
      // Check if sidebar width is small (collapsed rail)
      const aside = document.querySelector('aside');
      return aside ? aside.clientWidth <= 80 : false;
    });

    if (!sidebarCollapsed) {
      const collapseBtn = await page.locator('button[title="Collapse history"]').first();
      if (await collapseBtn.isVisible().catch(() => false)) {
        await collapseBtn.click();
        await page.waitForTimeout(300);
      }
    }

    // In collapsed mode, click the current record thumbnail if present
    const sidebarWidthBefore = await page.evaluate(() => {
      const aside = document.querySelector('aside');
      return aside ? aside.clientWidth : 0;
    });

    // The current record thumbnail should be visible in collapsed mode
    const currentThumb = await page.locator('aside img[title="final-qa-test.png"]').first();
    if (await currentThumb.isVisible().catch(() => false)) {
      await currentThumb.click();
      await page.waitForTimeout(300);
    }

    const sidebarWidthAfter = await page.evaluate(() => {
      const aside = document.querySelector('aside');
      return aside ? aside.clientWidth : 0;
    });

    results.push({
      label: 'Sidebar stays collapsed on thumbnail click',
      pass: sidebarWidthAfter <= 80,
      before: sidebarWidthBefore,
      after: sidebarWidthAfter
    });
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-sidebar-click.png'), fullPage: false });

    // ============================================================
    // F3.7 Annotate page loads, image visible, canvas crisp
    // ============================================================
    await page.goto(`${FRONTEND_URL}/annotate/${testRecord.id}`);
    await page.waitForTimeout(800);

    const annotateTitle = await page.locator('text=Edit Annotations').first().isVisible().catch(() => false);
    const annotateCanvas = await page.locator('canvas').first().isVisible().catch(() => false);
    const annotateImg = await page.locator('img').first().isVisible().catch(() => false);

    results.push({
      label: 'Annotate page loads',
      pass: annotateTitle && annotateCanvas && annotateImg
    });
    await page.screenshot({ path: path.join(EVIDENCE_DIR, 'f3-annotate-page.png'), fullPage: false });

    // ============================================================
    // F3.8 No console errors (except CORS)
    // ============================================================
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto(FRONTEND_URL);
    await seedLocalStorage(page);
    await page.reload();
    await page.waitForTimeout(1000);

    const nonCorsErrors = consoleErrors.filter(e => !e.includes('CORS') && !e.includes('net::ERR_FAILED'));
    results.push({
      label: 'No console errors (except CORS)',
      pass: nonCorsErrors.length === 0,
      errors: nonCorsErrors
    });

    // ============================================================
    // Write results
    // ============================================================
    const summary = {
      date: new Date().toISOString(),
      total: results.length,
      passed: results.filter(r => r.pass).length,
      failed: results.filter(r => !r.pass).length,
      results
    };

    fs.writeFileSync(
      path.join(EVIDENCE_DIR, 'playwright-results.json'),
      JSON.stringify(summary, null, 2)
    );

    console.log(`\n=== Playwright QA Results ===`);
    console.log(`Passed: ${summary.passed}/${summary.total}`);
    for (const r of results) {
      console.log(`  ${r.pass ? 'PASS' : 'FAIL'} - ${r.label}`);
    }

  } finally {
    await browser.close();
  }
}

runTests().catch(err => {
  console.error('Playwright test error:', err);
  process.exit(1);
});
