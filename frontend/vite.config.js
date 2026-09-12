import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tailwind v4 via the Vite plugin: no postcss or tailwind.config needed (tailwindcss.com/docs/installation/using-vite)
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
});
