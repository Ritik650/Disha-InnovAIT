import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base URL is configurable so docker-compose can route /api → http://api:8000
// (compose service name) while local dev keeps using http://localhost:8000.
const API_TARGET = process.env.VITE_API_BASE || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,    // listen on 0.0.0.0 so docker port-publish works
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
