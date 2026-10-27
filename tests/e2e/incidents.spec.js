const { test, expect } = require('@playwright/test');
const path = require('node:path');
const sample = require('../../web_review/sample.json');
test.beforeEach(async ({ page }) => { await page.goto('/'); });

test('uploads incident exports, filters severity and reviews state', async ({ page }) => {
  await page.getByLabel('Incident JSON file').setInputFiles(path.join(__dirname, '../../web_review/sample.json'));
  await expect(page.getByRole('status')).toHaveText('2 incident(s) shown');
  await page.getByLabel('Severity filter').selectOption('HIGH');
  await expect(page.getByRole('status')).toHaveText('1 incident(s) shown');
  await expect(page.getByRole('heading', { name: /LONG_DWELL/ })).toHaveCount(0);
  await page.getByRole('button', { name: 'Acknowledge', exact:true }).click();
  await expect(page.locator('.state')).toHaveText('Acknowledged');
  await page.getByRole('button', { name: 'Dismiss', exact:true }).click();
  await expect(page.locator('.state')).toHaveText('Dismissed');
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download reviewed JSON' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('reviewed-incidents.json');
  const stream = await download.createReadStream();
  let content = ''; for await (const chunk of stream) content += chunk;
  expect(JSON.parse(content).find(r => r.severity === 'HIGH').dismissed).toBe(true);
});

test('invalid reports clear stale results and display a useful error', async ({ page }) => {
  await page.getByRole('button', { name: 'Load synthetic example' }).click();
  await expect(page.getByRole('status')).toContainText('2 incident');
  await page.getByLabel('Report JSON', {exact:true}).fill('{"summary":"incomplete"}');
  await page.getByRole('button', { name: 'Review reports', exact:true }).click();
  await expect(page.getByRole('alert')).toContainText('Missing or invalid field');
  await expect(page.getByRole('button', { name: 'Download reviewed JSON' })).toBeDisabled();
});

test('report text is displayed literally and imported paths are discarded', async ({ page, request }) => {
  const row = {...sample[0],track_id:1,summary:'<img src=x onerror="window.injected=true">',report_key:'../../escape',report_path:'secret'};
  await page.getByLabel('Report JSON', {exact:true}).fill(JSON.stringify(row));
  await page.getByRole('button', { name: 'Review reports', exact:true }).click();
  await expect(page.locator('article')).toContainText(row.summary);
  expect(await page.evaluate(() => window.injected)).toBeUndefined();
  const response = await request.post('/api/preview', {data:row});
  expect(response.ok()).toBeTruthy();
  const normalized=(await response.json()).reports[0];
  expect(normalized.report_key).toMatch(/^[a-f0-9]{32}$/);
  expect(normalized.report_path).toBeUndefined();
  expect(normalized.track_id).toBe(1);
});

test('review controls fit a mobile viewport', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.getByRole('button', {name:'Load synthetic example'}).click();
  await expect(page.locator('article')).toHaveCount(2);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
