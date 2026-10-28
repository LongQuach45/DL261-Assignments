import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  site: "https://longquach45.github.io",
  base: "/DL261-Assignments",
  trailingSlash: "never",
  build: { format: "file" },
  vite: { plugins: [tailwindcss()] }
});
