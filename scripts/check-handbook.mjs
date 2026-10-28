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
for (const item of ["Group MobyDick", "Group name", "CO3133", "Le Thanh Sach", "Le Quoc Binh", "Quach Kien Long", "Le Thanh Lam", "To Nha Vy"]) {
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
  for (const identity of ["Group MobyDick", "Le Thanh Sach", "Le Quoc Binh", "Quach Kien Long", "Le Thanh Lam", "To Nha Vy"]) {
    if (!html.includes(identity)) throw new Error(`${page} is missing assignment identity: ${identity}`);
  }
}

for (const file of [
  "assignments/assignment-1/source/reproducibility.json",
  "assignments/assignment-2/source/reproducibility.json"
]) {
  const record = JSON.parse(readFileSync(resolve(root, file), "utf8"));
  if (!record.provenance || !record.dataset || !record.training || !record.environment || !record.artifacts) {
    throw new Error(`Incomplete reproducibility metadata: ${file}`);
  }
}

const a1 = readFileSync(resolve(root, "dist/assignments/a1.html"), "utf8");
for (const artifact of ["assignment-1-final-report.pdf", "assignment-1-presentation-slides.pdf", "https://youtu.be/Wbq5g0MIwDk"]) {
  if (!a1.includes(artifact)) throw new Error(`Assignment 1 page is missing deliverable: ${artifact}`);
}

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
}

console.log("Handbook page checks passed.");
