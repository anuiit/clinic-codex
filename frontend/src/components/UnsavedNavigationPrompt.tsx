import type { RefObject } from "react";
import { unstable_usePrompt as usePrompt } from "react-router";

export function UnsavedNavigationPrompt({
  dirty,
  allowLeaveRef,
  message,
}: {
  dirty: boolean;
  allowLeaveRef: RefObject<boolean>;
  message: string;
}) {
  usePrompt({
    when: ({ currentLocation, nextLocation }) =>
      dirty && !allowLeaveRef.current &&
      `${currentLocation.pathname}${currentLocation.search}` !== `${nextLocation.pathname}${nextLocation.search}`,
    message,
  });
  return null;
}
