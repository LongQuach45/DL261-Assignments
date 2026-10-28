export const site = {
  name: "MobyDick",
  url: "https://longquach45.github.io/DL261-Assignments/",
  repository: "https://github.com/LongQuach45/DL261-Assignments",
  course: "Deep Learning and Its Applications",
  code: "CO3133",
  semester: "Semester-261",
  instructor: "Lê Thành Sách",
  institution: "Ho Chi Minh City University of Technology – VNU-HCM",
  faculty: "Faculty of Computer Science and Engineering"
} as const;

export const members = [
  { name: "Le Quoc Binh", id: "2352117", role: "Pipeline, CNN/ViT, extensions", github: "https://github.com/harbinger-ux" },
  { name: "Quach Kien Long", id: "2352698", role: "Data/EDA, Linear/MLP models", github: "https://github.com/LongQuach45" },
  { name: "Le Thanh Lam", id: "2252421", role: "LSTM/GRU and robustness", github: "https://github.com/lamlethanh5555" },
  { name: "To Nha Vy", id: "2353360", role: "Error analysis, XAI, report/pages", github: "https://github.com/vy175" }
] as const;

export const external = {
  a1Source: `${site.repository}/tree/main/assignments/assignment-1/source`,
  a1Readme: `${site.repository}/blob/main/assignments/assignment-1/source/README.md`,
  a1Results: `${site.repository}/tree/main/assignments/assignment-1/source/results`,
  a1Ai: `${site.repository}/blob/main/assignments/assignment-1/source/AI_USAGE.md`,
  a2Source: `${site.repository}/tree/main/assignments/assignment-2/source`,
  a2Readme: `${site.repository}/blob/main/assignments/assignment-2/source/README.md`,
  a2Results: `${site.repository}/tree/main/assignments/assignment-2/source/results`,
  a2Ai: `${site.repository}/blob/main/assignments/assignment-2/source/AI_USAGE.md`,
  a2Proposal: `${site.repository}/blob/main/assignments/assignment-2/proposal/dataset.md`,
  a2ProposalPdf: `${site.repository}/blob/main/assignments/assignment-2/proposal/assignment-2-dataset-proposal.pdf`,
  rootAi: `${site.repository}/blob/main/AI_USAGE.md`
} as const;

export const aiSummaries = [
  {
    assignment: "Assignment 1",
    tool: "Google Gemini (2026 release)",
    summary: "Four documented uses: EDA grids, confusion-matrix and CSV formatting, saliency and robustness visualizations, and report typesetting.",
    verification: "Verified against official Matplotlib, PyTorch, and scikit-learn documentation plus generated outputs.",
    href: external.a1Ai
  },
  {
    assignment: "Assignment 2",
    tool: "Google Gemini (2026 release)",
    summary: "Four documented uses: integrity auditing, CUDA measurements, Grad-CAM visualization, and qualitative/error-analysis figures.",
    verification: "Verified against official Python, PIL, PyTorch, and scikit-learn documentation plus manifests and result CSVs.",
    href: external.a2Ai
  },
  {
    assignment: "Course website",
    tool: "OpenAI Codex (model/version not recorded)",
    summary: "Used by Quach Kien Long for website structure, content presentation, and static-site implementation assistance.",
    verification: "Student review against the handbook, repository artifacts, local builds, and deployed-page checks.",
    href: external.rootAi
  }
] as const;
