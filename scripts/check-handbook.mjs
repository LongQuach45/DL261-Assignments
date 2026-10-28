import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = process.cwd();
const pages = [
  "dist/index.html",
  "dist/assignments/a1.html",
  "dist/assignments/a2.html",
  "dist/assignments/a3.html",
  "dist/ai-usage.html"
];
const assignmentHeadings = [
  "Problem statement",
  "Dataset description and EDA",
  "Methodology",
  "Experimental setup",
  "Results",
  "Comparison and discussion",
  "Error analysis",
  "Limitations and conclusion",
  "Assignment-specific AI Usage Disclosure"
];

for (const file of pages) {
  if (!existsSync(resolve(root, file))) throw new Error(`Missing built page: ${file}`);
}

const index = readFileSync(resolve(root, "dist/index.html"), "utf8");
for (const item of ["MobyDick", "CO3133", "Lê Thành Sách", "Le Quoc Binh", "Quach Kien Long", "Le Thanh Lam", "To Nha Vy"]) {
  if (!index.includes(item)) throw new Error(`Landing page is missing: ${item}`);
}
if (index.includes("/DL261-Assignmentsassignments")) throw new Error("Malformed GitHub Pages base-path link");
for (const route of ["/DL261-Assignments/assignments/a1.html", "/DL261-Assignments/assignments/a2.html", "/DL261-Assignments/assignments/a3.html", "/DL261-Assignments/ai-usage.html"]) {
  if (!index.includes(`href=\"${route}\"`)) throw new Error(`Landing page is missing route: ${route}`);
}

for (const page of ["a1", "a2", "a3"]) {
  const html = readFileSync(resolve(root, `dist/assignments/${page}.html`), "utf8");
  for (const heading of assignmentHeadings) {
    if (!html.includes(`>${heading}<`)) throw new Error(`${page} is missing required heading: ${heading}`);
  }
}

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
}

console.log("Handbook page checks passed.");
