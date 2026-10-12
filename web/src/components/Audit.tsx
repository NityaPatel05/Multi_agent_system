import type { EvidenceMap } from "../lib/types";
import { Card } from "./ui";

// The API does not return audit_log / token_usage yet, so this tab shows what the
// evidence map already carries: fallback_history and rejection reasons.
export function Audit({ map }: { map: EvidenceMap }) {
  return (
    <div className="space-y-4">
      <Card title="Fallback history">
        {map.fallback_history.length === 0 ? (
          <p className="text-sm text-stone-500">No fallbacks. Every call used its primary provider.</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-stone-500">
              <tr><th className="py-1">Node</th><th>Reason</th><th>Fell back to</th></tr>
            </thead>
            <tbody>
              {map.fallback_history.map((f, i) => (
                <tr key={i} className="border-t border-stone-200 dark:border-stone-800">
                  <td className="py-1 font-medium">{f.node}</td><td>{f.reason}</td><td>{f.fallback}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      <Card title="Rejected sources">
        {Object.keys(map.rejection_reasons).length === 0 ? (
          <p className="text-sm text-stone-500">None.</p>
        ) : (
          <ul className="list-disc pl-5 text-sm">
            {Object.entries(map.rejection_reasons).map(([id, why]) => <li key={id}><code>{id}</code>: {why}</li>)}
          </ul>
        )}
      </Card>
      <p className="text-xs text-stone-500">Per-call audit log and token cost per node will appear here once the API returns them.</p>
    </div>
  );
}
