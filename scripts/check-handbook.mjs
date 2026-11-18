import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const root = process.cwd();
const a3Config = readFileSync(resolve(root, "config/deliverables.yml"), "utf8").split("\n  a3:\n")[1] ?? "";
// User-selected publication timing, not a handbook exemption.
const a3DisclosureDeferred = a3Config.includes('ai_usage: "Deferred until assignment finalization"');
if (a3DisclosureDeferred && /^\s+milestone:.*M3.*final/im.test(a3Config)) {
  throw new Error("Restore Assignment 3 disclosure when the assignment is finalized");
}
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
    if (page === "a3" && a3DisclosureDeferred && heading === "Assignment-specific AI Usage Disclosure") continue;
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

const a2 = readFileSync(resolve(root, "dist/assignments/a2.html"), "utf8");
for (const artifact of ["assignments/assignment-2/proposal/dataset.md", "assignment-2-final-report.pdf", "assignment-2-presentation-slides.pdf", "https://www.youtube.com/watch?v=HHNRGS3sljo"]) {
  if (!a2.includes(artifact)) throw new Error(`Assignment 2 page is missing deliverable: ${artifact}`);
}
if (!a2.includes(">Approved</span>")) {
  throw new Error("Assignment 2 page does not show the approved dataset status");
}

const a3 = readFileSync(resolve(root, "dist/assignments/a3.html"), "utf8");
for (const field of ["M1 proposal available", "Pending instructor approval", "Amazon Berkeley Objects", "6,353", "5,161 / 596 / 596", "CC BY 4.0", "DistilBERT", "No experimental results", "Final report · pending", "Slides · pending", "YouTube video · pending"]) {
  if (!a3.includes(field)) throw new Error(`Assignment 3 M1 page is missing: ${field}`);
}
const repository = "https://github.com/LongQuach45/DL261-Assignments/blob/main/";
for (const path of ["assignments/assignment-3/proposal/dataset.md", "assignments/assignment-3/source/README.md", ...(a3DisclosureDeferred ? [] : ["assignments/assignment-3/source/AI_USAGE.md"])]) {
  if (!existsSync(resolve(root, path)) || !a3.includes(`href="${repository}${path}"`)) {
    throw new Error(`Missing Assignment 3 artifact or rendered repository link: ${path}`);
  }
}
for (const file of ["dist/index.html", "dist/ai-usage.html"]) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (!a3DisclosureDeferred && !html.includes(`${repository}assignments/assignment-3/source/AI_USAGE.md`)) {
    throw new Error(`${file} is missing Assignment 3 AI disclosure`);
  }
  if (a3DisclosureDeferred && html.includes(`${repository}assignments/assignment-3/source/AI_USAGE.md`)) {
    throw new Error(`${file} links to the deferred Assignment 3 disclosure`);
  }
  if (/Assignment 3 has not started|no assignment-specific AI-use record yet/.test(html)) {
    throw new Error(`${file} retains a stale Assignment 3 status`);
  }
}

const proposalPath = "assignments/assignment-3/proposal/dataset.md";
const proposal = readFileSync(resolve(root, proposalPath), "utf8");
for (const section of ["1. Dataset and task", "2. Selected subset", "3. Distribution analysis and limitations", "4. Split and leakage prevention", "5. Preprocessing pipeline", "6. Models and controlled experiment", "7. Evaluation and error analysis", "8. Compute and reproducibility", ...(!a3DisclosureDeferred ? ["AI Usage Disclosure"] : []), "References"]) {
  if (!proposal.includes(`## ${section}`)) throw new Error(`A3 proposal is missing: ${section}`);
}
for (const field of ["Amazon Berkeley Objects", "CC BY 4.0", "LICENSE-CC-BY-4.0.txt", "Pending instructor approval", "StratifiedGroupKFold", "macro-F1", "Image only", "Text only", "Simple fusion", "Improved fusion", "Gate ablation"]) {
  if (!proposal.includes(field)) throw new Error(`A3 proposal is missing: ${field}`);
}
const cells = (label) => {
  const row = proposal.split("\n").find((line) => line.startsWith(`| ${label} |`));
  if (!row) throw new Error(`Missing proposal table row: ${label}`);
  return row.split("|").slice(1, -1).map((cell) => cell.trim());
};
const number = (cell) => Number(cell.replaceAll(",", ""));
const classes = ["Chair", "Grocery", "Rug", "Shoes", "Sofa"];
const totals = [0, 0, 0, 0];
for (const label of classes) {
  const row = cells(label);
  const values = row.slice(1).map(number);
  if (values.length !== 4 || values.some((n) => !Number.isInteger(n) || n <= 0) || values[0] !== values[1] + values[2] + values[3]) throw new Error(`Inconsistent A3 class counts: ${label}`);
  values.forEach((value, i) => totals[i] += value);
  for (const value of row) if (!a3.includes(`>${value}<`)) throw new Error(`A3 page/proposal table differs: ${label}/${value}`);
}
if (totals[0] < 5000 || classes.length < 5) throw new Error("A3 subset does not meet default classification thresholds");
const split = cells("Train / validation / test")[1].split(" / ").map(number);
if (split.some((value, i) => value !== totals[i + 1])) throw new Error("A3 summary split differs from class table");
if (number(cells("Selected pairs / product-family groups")[1].split(" / ")[0]) !== totals[0]) throw new Error("A3 paired count differs from class table");
const files = readdirSync(resolve(root, "assignments/assignment-3/proposal")).filter((name) => !name.startsWith(".")).sort();
if (files.join(",") !== "abo-samples.jpg,dataset.md") throw new Error("A3 proposal should contain only the proposal and illustration");
if (!proposal.includes("](abo-samples.jpg)") || !a3.includes("Two real catalog images") || !a3.includes("Images © Amazon.com")) throw new Error("A3 illustration or attribution missing");
const image = readFileSync(resolve(root, "assignments/assignment-3/proposal/abo-samples.jpg"));
if (image[0] !== 0xff || image[1] !== 0xd8) throw new Error("A3 illustration is not a JPEG");

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
  if (/assignment-[23]-dataset-proposal\.pdf|Formatted proposal PDF/.test(html)) throw new Error(`Obsolete proposal PDF reference in ${file}`);
}

console.log("Site content checks passed.");
