const fs = require('fs');

let content = fs.readFileSync('codex-frontend/src/pages/WorkspacePage.tsx', 'utf8');

// Replace the right section container
const rightSectionStart = '<section className="flex h-[72vh] min-h-[520px] flex-col overflow-hidden rounded-[28px] border border-stone-800 bg-stone-900/80 p-5 lg:p-6">';
const newRightSectionStart = `<section className="flex h-[72vh] min-h-[520px] flex-col overflow-hidden rounded-[28px] border border-stone-800 bg-stone-900/80 sidebar-shell">
                  {focusedIdx !== null ? (
                    <div className="flex h-full flex-col">
                      <div className="flex items-center justify-between sidebar-header px-5 py-4 border-b border-stone-800">
                        <button 
                          onClick={() => setFocusedIdx(null)}
                          className="flex items-center gap-2 text-stone-400 hover:text-stone-100 transition-colors"
                        >
                          <ChevronLeft size={16} />
                          <span className="text-sm font-medium">Back to Regions</span>
                        </button>
                        <span className="text-xs font-semibold text-stone-500 uppercase tracking-wider">Region {focusedIdx}</span>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto sidebar-body p-5 space-y-6">
                        {(() => {
                          const element = currentRecord.result.elements[focusedIdx];
                          const topPrediction = element.confidence;
                          const runnerUp = element.top_k.length > 1 ? element.top_k[1].confidence : 0;
                          const margin = topPrediction - runnerUp;
                          
                          return (
                            <>
                              {/* Panel 1: Trust Summary */}
                              <div className="flex flex-col gap-3 rounded-2xl border border-stone-800 bg-stone-900/50 p-4">
                                <h4 className="text-sm font-medium text-stone-400">Trust Summary</h4>
                                <div className="flex items-center justify-between">
                                  <span className="text-xl font-bold text-stone-100 truncate">{element.class_name}</span>
                                  <span className={\`shrink-0 px-2.5 py-1 rounded-md text-sm font-bold \${
                                    element.rejected 
                                      ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
                                      : 'bg-green-500/10 text-green-400 border border-green-500/20'
                                  }\`}>
                                    {(element.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                                {element.rejected && (
                                  <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-amber-400 mt-2">
                                    <AlertCircle size={16} className="shrink-0 mt-0.5" />
                                    <div className="text-sm">
                                      <strong>Low Confidence Flag</strong>
                                      <p className="mt-1 text-xs opacity-80">This prediction is below the confidence threshold. Manual review recommended.</p>
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Panel 2: Top-k Breakdown + Margin */}
                              <div className="flex flex-col gap-4 rounded-2xl border border-stone-800 bg-stone-900/50 p-4">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-sm font-medium text-stone-400">Similarity Distribution</h4>
                                  <span className="text-xs text-stone-500">Margin: {(margin * 100).toFixed(1)}%</span>
                                </div>
                                <div className="space-y-3">
                                  {element.top_k.map((item, i) => (
                                    <div key={i} className="flex flex-col gap-1.5">
                                      <div className="flex justify-between text-xs">
                                        <span className={i === 0 ? "text-stone-200 font-medium" : "text-stone-400"}>{item.class_name}</span>
                                        <span className={i === 0 ? "text-stone-300 font-medium" : "text-stone-500"}>{(item.confidence * 100).toFixed(1)}%</span>
                                      </div>
                                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-stone-800">
                                        <div 
                                          className={\`h-full rounded-full \${i === 0 ? (element.rejected ? 'bg-amber-500' : 'bg-green-500') : 'bg-stone-600'}\`}
                                          style={{ width: \`\${item.confidence * 100}%\` }}
                                        />
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              {/* Panel 3: Confusion-Class List + Sample Thumbnails */}
                              <div className="flex flex-col gap-3 rounded-2xl border border-stone-800 bg-stone-900/50 p-4">
                                 <h4 className="text-sm font-medium text-stone-400">Confusion Context</h4>
                                 <p className="text-xs text-stone-500 mb-2">Close alternatives that share visual features with this region.</p>
                                 <div className="grid gap-3">
                                   {element.top_k.slice(1, 4).map((item, i) => (
                                     <div key={i} className="flex items-center gap-3 p-2 rounded-lg border border-stone-800/50 bg-stone-950/50">
                                        <div className="h-10 w-10 shrink-0 rounded bg-stone-800 flex items-center justify-center text-stone-600 text-xs border border-stone-700/50">
                                          IMG
                                        </div>
                                        <div className="flex flex-col">
                                          <span className="text-sm font-medium text-stone-300">{item.class_name}</span>
                                          <span className="text-xs text-stone-500">{(item.confidence * 100).toFixed(1)}% similarity</span>
                                        </div>
                                     </div>
                                   ))}
                                 </div>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                      
                      <div className="p-5 border-t border-stone-800 sidebar-header">
                         <button
                           type="button"
                           onClick={handleEditorHandoff}
                           className="w-full flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-4 py-3 text-sm font-bold text-stone-950 transition-all hover:bg-amber-400 active:scale-[0.98]"
                         >
                           <Edit3 size={16} />
                           {currentRecord.result.elements[focusedIdx].rejected ? 'Correct this element' : 'Annotate Region'}
                         </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col h-full p-5 lg:p-6">`;

const newRightSectionEnd = `</div>
                  )}
                </section>`;

if (content.includes(rightSectionStart)) {
  content = content.replace(rightSectionStart, newRightSectionStart);
  // Need to replace the closing tag of this section properly
  // This section ends at line 825 (</section>) right before </div></>
  const closingSectionRegex = /<\/section>\s*<\/div>\s*<\/>\s*\) :/g;
  content = content.replace(closingSectionRegex, `</div>
                  )}
                </section>
              </div>
            </>
          ) :`);
  fs.writeFileSync('codex-frontend/src/pages/WorkspacePage.tsx', content);
  console.log("Patched successfully");
} else {
  console.log("Could not find the target string to replace.");
}

