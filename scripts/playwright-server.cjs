// Spawn without a shell so Python paths containing spaces work on Windows.
const { spawn } = require('node:child_process');
const python = process.env.PLAYWRIGHT_PYTHON || 'python';
const child = spawn(python, ['-u', 'scripts/playwright_server.py'], {
  stdio: 'inherit', env: { ...process.env, PYTHONUTF8: '1' },
});
child.on('error', error => { console.error('Cannot start Python:', error.message); process.exit(1); });
child.on('exit', code => process.exit(code ?? 1));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
