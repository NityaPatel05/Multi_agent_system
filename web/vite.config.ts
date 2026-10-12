import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// /research is proxied to the FastAPI service so the browser never needs CORS.
// The API requires X-API-Key. The dev proxy adds it from the repo-root .env (API_KEY) so the key
// never ships in browser code (and EventSource, which cannot set headers, still works).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  return {
  plugins: [react(), tailwindcss()],
  // Inline (empty) PostCSS config stops Vite walking up to a stray ~/postcss.config.js (Tailwind v3).
  css: { postcss: { plugins: [] } },
  server: {
    port: 5173,
    proxy: {
      "/research": {
        target: `http://localhost:${env.API_HOST_PORT || 8000}`,
        headers: env.API_KEY ? { "X-API-Key": env.API_KEY } : {},
      },
    },
  },
  };
});
