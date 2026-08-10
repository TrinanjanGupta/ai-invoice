import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use VITE_API_URL env var if set (Docker), otherwise default to localhost:8000 (local dev)
const apiTarget = process.env.VITE_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/health': {
        target: apiTarget,
        changeOrigin: true,
      }
    }
  }
})
