import { existsSync, readFileSync, readdirSync } from "node:fs";
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

const a2 = readFileSync(resolve(root, "dist/assignments/a2.html"), "utf8");
for (const artifact of ["assignments/assignment-2/proposal/dataset.md", "assignment-2-final-report.pdf", "assignment-2-presentation-slides.pdf", "https://www.youtube.com/watch?v=HHNRGS3sljo"]) {
  if (!a2.includes(artifact)) throw new Error(`Assignment 2 page is missing deliverable: ${artifact}`);
}
if (!a2.includes(">Approved</span>")) {
  throw new Error("Assignment 2 page does not show the approved dataset status");
}

const a3 = readFileSync(resolve(root, "dist/assignments/a3.html"), "utf8");
for (const field of ["M1 proposal available", "Pending instructor approval", "N24News", "7,500", "6,000 / 750 / 750", "Usage terms remain unresolved", "DistilBERT", "No experimental results", "Final report · pending", "Slides · pending", "YouTube video · pending"]) {
  if (!a3.includes(field)) throw new Error(`Assignment 3 M1 page is missing: ${field}`);
}
const repository = "https://github.com/LongQuach45/DL261-Assignments/blob/main/";
for (const path of ["assignments/assignment-3/proposal/dataset.md", "assignments/assignment-3/source/README.md", "assignments/assignment-3/source/AI_USAGE.md"]) {
  if (!existsSync(resolve(root, path)) || !a3.includes(`href="${repository}${path}"`)) {
    throw new Error(`Missing Assignment 3 artifact or rendered repository link: ${path}`);
  }
}
for (const file of ["dist/index.html", "dist/ai-usage.html"]) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (!html.includes(`${repository}assignments/assignment-3/source/AI_USAGE.md`)) {
    throw new Error(`${file} is missing Assignment 3 AI disclosure`);
  }
  if (/Assignment 3 has not started|no assignment-specific AI-use record yet/.test(html)) {
    throw new Error(`${file} retains a stale Assignment 3 status`);
  }
}

const proposalPath = "assignments/assignment-3/proposal/dataset.md";
const proposal = readFileSync(resolve(root, proposalPath), "utf8");
for (const section of ["1. Dataset and task", "2. Preliminary distribution and pair quality", "3. Subset, split, and leakage prevention", "4. Preprocessing and alignment", "5. Models and training plan", "6. Experiments and evaluation", "7. Compute and reproducibility", "8. Approval and limitations", "AI Usage Disclosure", "References"]) {
  if (!proposal.includes(`## ${section}`)) throw new Error(`A3 proposal is missing: ${section}`);
}
for (const field of ["N24News", "047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106", "cc7a3422e948a84fffcfe2f9f8c6162ec076a45f712188a6e31be382a9150e9c", "Pending instructor approval", "Training permission is not cleared", "1,250", "6,000", "750", "macro-F1", "Image only", "Text only", "Simple fusion", "Improved fusion", "Gate ablation"]) {
  if (!proposal.includes(field)) throw new Error(`A3 proposal is missing: ${field}`);
}
const summaryFields = ["Full metadata records / categories", "Records across selected categories", "Distinct image URLs / raw headlines", "Candidate representatives after filtering", "Metadata-selected pairs / categories", "Train / validation / test targets", "Decoded sample pairs", "Visual sample assessments"];
for (const field of summaryFields) {
  const row = proposal.split("\n").find((line) => line.startsWith(`| ${field} |`));
  if (!row || !a3.includes(row.split("|")[2].trim())) throw new Error(`A3 page/proposal summary differs: ${field}`);
}
const publicProposalFiles = readdirSync(resolve(root, "assignments/assignment-3/proposal"));
if (publicProposalFiles.length !== 1 || publicProposalFiles[0] !== "dataset.md") throw new Error("A3 proposal directory should contain only dataset.md");
if (!a3.includes("18 training pairs") || !a3.includes("training permission is not cleared")) throw new Error("A3 page omits visual review or permission limitation");
if (/CREMA-D|actor-disjoint|7,438|5,965/.test(a3)) throw new Error("A3 page retains the superseded task");
for (const path of ["/proposal/README.md", "/proposal/pair-review", "/proposal/usage-review", "/proposal/metadata-audit", "/proposal/manifest.csv", "/proposal/groups.csv", "/proposal/audit_metadata.py"]) {
  if (a3.includes(path)) throw new Error(`Obsolete public A3 evidence link: ${path}`);
}

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
  if (/assignment-[23]-dataset-proposal\.pdf|Formatted proposal PDF/.test(html)) throw new Error(`Obsolete proposal PDF reference in ${file}`);
}

console.log("Handbook page checks passed.");
