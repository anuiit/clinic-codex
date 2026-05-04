[2026-05-05] Task: Add keyboard navigation for region list

- Added tabIndex and onKeyDown handler to the region list container (.flex-1.space-y-1.overflow-y-auto.pr-2) in WorkspacePage.tsx.
- Behavior implemented:
  - ArrowDown / ArrowRight: move focus to next region (wraps)
  - ArrowUp / ArrowLeft: move focus to previous region (wraps)
  - Enter / Space: toggle expand on focused region
  - Escape: clear focus (sets focusedIdx to null)
  - Handlers run only when a currentRecord exists and there are elements
  - Keyboard events attached to list container (not global); container is focusable via tabIndex=0

- Verification:
  - lsp diagnostics: clean for WorkspacePage.tsx
  - npm run build in codex-frontend: success

- Notes / gotchas:
  - Ensure the panel container can receive keyboard focus in the layout; tabIndex=0 may need focus styles later for accessibility.
  - Did not add ARIA roles beyond ensuring the container is focusable; consider role="list" + role="listitem" on cards for screen readers in follow-up.
