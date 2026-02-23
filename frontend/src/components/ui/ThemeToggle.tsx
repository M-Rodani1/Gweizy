import React, { useState, useEffect } from 'react';
import { safeGetLocalStorageItem, safeSetLocalStorageItem } from '../../utils/safeStorage';

const ThemeToggle: React.FC = () => {
  const getPreferred = () => {
    if (typeof window === 'undefined') return true;
    const savedTheme = safeGetLocalStorageItem('theme');
    if (savedTheme === 'light') return false;
    if (savedTheme === 'dark') return true;
    return !window.matchMedia('(prefers-color-scheme: light)').matches;
  };

  const [isDark, setIsDark] = useState<boolean>(true);

  useEffect(() => {
    const initial = getPreferred();
    setIsDark(initial);
    document.documentElement.classList.toggle('light-mode', !initial);

    // Listen for OS preference changes when user has not overridden
    const mql = window.matchMedia('(prefers-color-scheme: light)');
    const handler = (event: MediaQueryListEvent) => {
      const savedTheme = safeGetLocalStorageItem('theme');
      if (!savedTheme) {
        const useDark = !event.matches;
        setIsDark(useDark);
        document.documentElement.classList.toggle('light-mode', !useDark);
      }
    };
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  const toggleTheme = () => {
    const newIsDark = !isDark;
    setIsDark(newIsDark);
    if (!newIsDark) {
      // Switching to light mode
      document.documentElement.classList.add('light-mode');
      safeSetLocalStorageItem('theme', 'light');
    } else {
      // Switching to dark mode
      document.documentElement.classList.remove('light-mode');
      safeSetLocalStorageItem('theme', 'dark');
    }
  };

  return (
    <button
      onClick={toggleTheme}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? (
        <>
          <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
              clipRule="evenodd"
            />
          </svg>
          <span className="text-xs text-gray-400">Light</span>
        </>
      ) : (
        <>
          <svg className="w-4 h-4 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
          </svg>
          <span className="text-xs text-gray-400">Dark</span>
        </>
      )}
    </button>
  );
};

export default ThemeToggle;
