const fs = require('fs');
const content = fs.readFileSync('codex-frontend/src/pages/WorkspacePage.tsx', 'utf-8');

const replacement = `
                  </div>

                  {focusedIdx !== null && currentRecord && (() => {
                    const el = currentRecord.result.elements[focusedIdx];
                    if (!el) return null;
                    return (
                      <div className="mt-4 shrink-0 flex flex-col gap-0 border-t border-stone-800 pt-4 overflow-y-auto max-h-[50%]">
                        {/* Panel 1: Trust Summary */}
                        <div className="mb-4 rounded-xl border border-stone-800 bg-stone-950/70 p-3">
                          <div className="flex items-center gap-2 mb-2">
                            <span className={\`w-2 h-2 rounded-full \${
                              el.confidence >= 0.60 ? 'bg-green-500' : 
                              el.confidence >= 0.35 ? 'bg-amber-500' : 'bg-red-500'
                            }\`} />
                            <span className="text-sm font-medium text-stone-200">
                              {el.rejected ? 'Unknown / Rejected' : el.class_name}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold text-stone-100">
                              {(el.confidence * 100).toFixed(1)}%
                            </span>
                            <span className="text-xs text-stone-500">confidence</span>
                          </div>
                          {el.rejected && (
                            <div className="mt-2 rounded-lg bg-red-400/10 border border-red-400/20 px-3 py-2 text-xs text-red-300">
                              Below recognition threshold (0.35). Human review recommended.
                            </div>
                          )}
                        </div>

                        {/* Panel 2: Top-k Breakdown + Margin */}
                        {el.top_k && (
                          <div className="mb-4 rounded-xl border border-stone-800 bg-stone-950/70 p-3">
                            <p className="text-xs uppercase tracking-wider text-stone-500 mb-3">Top Candidates</p>
                            {el.top_k.map((item, idx) => {
                              const pct = item.confidence * 100;
                              return (
                                <div key={idx} className="mb-2 last:mb-0">
                                  <div className="flex justify-between text-xs mb-1">
                                    <span className="text-stone-300">{item.class_name}</span>
                                    <span className="text-stone-500">{pct.toFixed(1)}%</span>
                                  </div>
                                  <div className="h-1.5 rounded-full bg-stone-800 overflow-hidden">
                                    <div 
                                      className={\`h-full rounded-full \${
                                        idx === 0 ? 'bg-amber-500' : 'bg-stone-600'
                                      }\`}
                                      style={{ width: \`\${pct}%\` }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                            {(() => {
                              if (el.top_k.length >= 2) {
                                const margin = el.top_k[0].confidence - el.top_k[1].confidence;
                                return (
                                  <div className={\`mt-3 text-xs px-2 py-1 rounded-lg inline-flex items-center gap-1.5 \${
                                    margin < 0.05 ? 'bg-amber-500/10 text-amber-400' : 'text-stone-500'
                                  }\`}>
                                    <span>Margin: {(margin * 100).toFixed(1)}%</span>
                                    {margin < 0.05 && <span>— Ambiguous</span>}
                                  </div>
                                );
                              }
                              return null;
                            })()}
                          </div>
                        )}

                        {/* Panel 3: Confusion-Class List */}
                        {el.top_k && el.top_k.length > 1 && (
                          <div className="mb-4 rounded-xl border border-stone-800 bg-stone-950/70 p-3">
                            <p className="text-xs uppercase tracking-wider text-stone-500 mb-3">Also Considered</p>
                            <div className="space-y-1.5">
                              {el.top_k.slice(1).map((item, idx) => (
                                <div key={idx} className="flex items-center justify-between text-sm">
                                  <span className="text-stone-300">{item.class_name}</span>
                                  <span className="text-xs text-stone-500">{(item.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}
`;

const updated = content.replace(
  /                  <\/div>\n                <\/section>/,
  replacement + '\n                </section>'
);

fs.writeFileSync('codex-frontend/src/pages/WorkspacePage.tsx', updated);
console.log('patched');
