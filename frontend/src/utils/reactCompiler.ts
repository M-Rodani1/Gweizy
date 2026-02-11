import React from 'react';

export function compilerMemo<T extends React.ComponentType<any>>(
  Component: T,
  displayName?: string
): React.MemoExoticComponent<T> {
  const Memoized = React.memo(Component);
  Memoized.displayName = displayName || Component.displayName || Component.name || 'MemoComponent';
  return Memoized;
}
