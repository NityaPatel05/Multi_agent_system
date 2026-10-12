import type { Claim } from "./types";

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

/**
 * The backend strips links from the report on purpose, so citations live only in
 * claims[].source_ids. We re-attach them client-side: find each claim's sentence in the
 * markdown and append a `[n](#cite-<claimId>)` link that <Report> renders as a chip.
 * Claims whose text can't be located are returned as `unanchored` and listed separately.
 */
export function anchorClaims(markdown: string, claims: Claim[]) {
  let out = markdown;
  const unanchored: Claim[] = [];
  claims.forEach((c, i) => {
    const idx = out.indexOf(c.text.trim());
    if (idx >= 0 && c.text.trim().length > 12) {
      const end = idx + c.text.trim().length;
      out = `${out.slice(0, end)} [${i + 1}](#cite-${c.id})${out.slice(end)}`;
    } else {
      // loose fallback: same words ignoring punctuation/markdown differences
      const target = norm(c.text);
      const line = out.split("\n").findIndex((l) => target.length > 12 && norm(l).includes(target));
      if (line >= 0) {
        const lines = out.split("\n");
        lines[line] = `${lines[line]} [${i + 1}](#cite-${c.id})`;
        out = lines.join("\n");
      } else unanchored.push(c);
    }
  });
  return { markdown: out, unanchored };
}

export const host = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};
