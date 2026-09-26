import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../assistant/web/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
