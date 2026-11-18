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
for (const field of ["M1 proposal available", "Pending instructor approval", "N24News", "7,500", "6,000 / 750 / 750", "Usage terms remain unresolved", "DistilBERT", "No experimental results", "Final report · pending", "Slides · pending", "YouTube video · pending"]) {
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
const readCsv = (file) => {
  const lines = readFileSync(resolve(root, evidenceDir, file), "utf8").trimEnd().split("\n");
  const header = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    if (values.length !== header.length) throw new Error(`Unexpected evidence CSV layout: ${file}`);
    return Object.fromEntries(header.map((key, i) => [key, values[i]]));
  });
};
const records = readCsv("manifest.csv");
const groups = readCsv("groups.csv");
const exclusions = readCsv("exclusions.csv");
const classRows = readCsv("class-counts.csv");
const splitRows = readCsv("split-counts.csv");
if (records.length !== audit.metadata_selected_pairs || records.length !== 7500) throw new Error("A3 provisional selection count differs");
if (audit.dataset !== "N24News" || audit.selected_categories.length !== 6) throw new Error("A3 dataset/task metadata differs");
for (const key of ["image_id", "source_row_index", "group_id", "article_url_sha256", "image_url_sha256", "headline_normalized_sha256"]) {
  if (new Set(records.map((r) => r[key])).size !== records.length) throw new Error(`A3 duplicate selected ${key}`);
}
for (const record of records) {
  if (!audit.selected_categories.includes(record.category) || record.status !== "selected_pending_media") throw new Error("A3 manifest claims unexpected eligibility/category");
  if (!["train", "validation", "test"].includes(record.split)) throw new Error("A3 manifest has invalid partition");
}
if (groups.length !== audit.selected_category_records || new Set(groups.map((r) => r.image_id)).size !== groups.length) throw new Error("A3 source grouping coverage differs");
const selectedIds = new Set(records.map((r) => r.image_id));
const selectedGroups = groups.filter((r) => r.status === "selected_pending_media");
if (selectedGroups.length !== records.length || selectedGroups.some((r) => !selectedIds.has(r.image_id))) throw new Error("A3 grouping ledger and manifest differ");
if (exclusions.length !== audit.duplicate_records_removed + audit.conflicting_label_records) throw new Error("A3 exclusion counts differ");
for (const split of audit.splits) {
  const members = records.filter((r) => r.split === split.split);
  const csv = splitRows.find((r) => r.split === split.split);
  if (members.length !== split.paired_samples || Number(csv?.paired_samples) !== members.length) throw new Error(`A3 split count differs: ${split.split}`);
  for (const category of audit.selected_categories) {
    const n = members.filter((r) => r.category === category).length;
    if (n !== split[category] || n !== Number(csv[category])) throw new Error(`A3 class count differs: ${split.split}/${category}`);
  }
}
for (const category of audit.selected_categories) {
  const csv = classRows.find((r) => r.category === category);
  if (records.filter((r) => r.category === category).length !== audit.class_counts[category] ||
      Number(csv?.metadata_selected_pairs) !== audit.class_counts[category] ||
      Number(csv?.candidate_representatives) !== audit.candidate_counts[category]) throw new Error(`A3 class summary differs: ${category}`);
}
if (audit.selected_categories.length < 5 || records.length < 5000) throw new Error("A3 metadata-selected target is below default thresholds");
if (existsSync(resolve(root, evidenceDir, "actors.json"))) throw new Error("Obsolete A3 actor split artifact remains");
const probes = JSON.parse(readFileSync(resolve(root, evidenceDir, "image-probes.json"), "utf8"));
if (probes.source_commit !== audit.source_commit || probes.probes.length !== 6 ||
    new Set(probes.probes.map((r) => r.category)).size !== 6) throw new Error("A3 source-image probe coverage differs");
for (const probe of probes.probes) {
  if (probe.method !== "HEAD" || !selectedIds.has(probe.image_id)) throw new Error("A3 probe is not a selected metadata record");
}
if (/CREMA-D|actor-disjoint|7,438|5,965/.test(a3)) throw new Error("A3 page retains the superseded task");

for (const file of pages) {
  const html = readFileSync(resolve(root, file), "utf8");
  if (/\bTODO\b/i.test(html)) throw new Error(`Visible TODO in ${file}`);
  if (/assignment-[23]-dataset-proposal\.pdf|Formatted proposal PDF/.test(html)) throw new Error(`Obsolete proposal PDF reference in ${file}`);
}

console.log("Handbook page checks passed.");
