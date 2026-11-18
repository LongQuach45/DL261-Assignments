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

const a2 = readFileSync(resolve(root, "dist/assignments/a2.html"), "utf8");
for (const artifact of ["assignments/assignment-2/proposal/dataset.md", "assignment-2-final-report.pdf", "assignment-2-presentation-slides.pdf", "https://www.youtube.com/watch?v=HHNRGS3sljo"]) {
  if (!a2.includes(artifact)) throw new Error(`Assignment 2 page is missing deliverable: ${artifact}`);
}
if (!a2.includes(">Approved</span>")) {
  throw new Error("Assignment 2 page does not show the approved dataset status");
}

const a3 = readFileSync(resolve(root, "dist/assignments/a3.html"), "utf8");
for (const field of ["M1 proposal available", "Pending instructor approval", "CREMA-D", "7,438", "5,965 / 735 / 738", "No experimental results", "Final report · pending", "Slides · pending", "YouTube video · pending"]) {
  if (!a3.includes(field)) throw new Error(`Assignment 3 M1 page is missing: ${field}`);
}
const repository = "https://github.com/LongQuach45/DL261-Assignments/blob/main/";
for (const path of ["assignments/assignment-3/proposal/dataset.md", "assignments/assignment-3/proposal/README.md", "assignments/assignment-3/source/README.md", "assignments/assignment-3/source/AI_USAGE.md"]) {
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

const evidenceDir = "assignments/assignment-3/proposal";
const audit = JSON.parse(readFileSync(resolve(root, evidenceDir, "metadata-audit.json"), "utf8"));
const actors = JSON.parse(readFileSync(resolve(root, evidenceDir, "actors.json"), "utf8"));
const allActors = Object.values(actors).flat();
if (new Set(allActors).size !== allActors.length || allActors.length !== audit.actors) {
  throw new Error("Assignment 3 actor lists overlap or omit actors");
}
const manifestLines = readFileSync(resolve(root, evidenceDir, "manifest.csv"), "utf8").trimEnd().split("\n");
const manifestHeader = manifestLines.shift().split(",");
const records = manifestLines.map((line) => Object.fromEntries(line.split(",").map((value, i) => [manifestHeader[i], value])));
if (new Set(records.map((r) => r.recording_id)).size !== audit.matched_recording_ids) throw new Error("A3 manifest recording IDs differ from audit");
for (const record of records) {
  if (!actors[record.split]?.includes(record.actor_id)) throw new Error(`A3 actor leakage: ${record.recording_id}`);
}
const eligible = records.filter((r) => r.status === "eligible");
if (eligible.length !== audit.eligible_pairs || records.length - eligible.length !== audit.excluded_known_defects) throw new Error("A3 eligibility totals differ from audit");
for (const split of audit.splits) {
  const members = eligible.filter((r) => r.split === split.split);
  if (members.length !== split.paired_samples || actors[split.split].length !== split.actors) throw new Error(`A3 split count differs: ${split.split}`);
  for (const code of Object.keys(audit.class_counts)) {
    if (members.filter((r) => r.emotion_code === code).length !== split[code]) throw new Error(`A3 class count differs: ${split.split}/${code}`);
  }
}
if (Object.keys(audit.class_counts).length < 5 || eligible.length < 5000) throw new Error("A3 does not meet default classification thresholds");
for (const artifact of ["class-counts.csv", "split-counts.csv", "exclusions.csv", "audit_metadata.py"]) {
  if (!existsSync(resolve(root, evidenceDir, artifact))) throw new Error(`A3 missing evidence: ${artifact}`);
}

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
  if (/assignment-[23]-dataset-proposal\.pdf|Formatted proposal PDF/.test(html)) throw new Error(`Obsolete proposal PDF reference in ${file}`);
}

console.log("Handbook page checks passed.");
