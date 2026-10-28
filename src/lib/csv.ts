import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export type CsvRow = Record<string, string>;

export function readCsv(relativePath: string, requiredColumns: string[]): CsvRow[] {
  const raw = readFileSync(resolve(process.cwd(), relativePath), "utf8").trim();
  const [header, ...lines] = raw.split(/\r?\n/);
  const columns = header.split(",");
  for (const column of requiredColumns) {
    if (!columns.includes(column)) throw new Error(`${relativePath} is missing ${column}`);
  }
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(columns.map((column, index) => [column, values[index] ?? ""]));
  });
}

export function percent(value: string): string {
  return `${(Number(value) * 100).toFixed(2)}%`;
}
