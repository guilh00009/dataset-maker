import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev the UI talks to `mixer serve` (cargo run -p mixer-cli -- serve) through this proxy.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8787' } },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
});
