/**
 * Lazy-loaded lucide-react icons wrapper
 *
 * This ensures React is fully initialized before lucide-react components are used.
 * Helps prevent "Cannot set properties of undefined (setting 'Children')" errors.
 */

import React from 'react';

// Wait for React to be fully available
let ReactReady = false;

// Check if React is ready
function ensureReactReady(): boolean {
  if (typeof window === 'undefined') return false;
  
  try {
    // Check if React and React.createElement are available
    const win = window as any;
    if (win.React && win.React.createElement) {
      ReactReady = true;
      return true;
    }
    
    // Try to access React via import if available
    if (typeof React !== 'undefined' && (React as any).createElement) {
      ReactReady = true;
      return true;
    }
    
    return false;
  } catch {
    return false;
  }
}

// Lazy import lucide-react only after React is ready
let lucideReact: any = null;

async function getLucideReact() {
  if (lucideReact) return lucideReact;
  
  // Wait for React to be ready
  if (!ensureReactReady()) {
    // Poll until React is ready (max 1 second)
    const startTime = Date.now();
    while (!ensureReactReady() && Date.now() - startTime < 1000) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
  
  // Now import lucide-react
  lucideReact = await import('lucide-react');
  return lucideReact;
}

/**
 * Lazy-load a lucide-react icon
 * Usage: const Icon = await lazyLucideIcon('Activity');
 */
export async function lazyLucideIcon(iconName: string): Promise<any> {
  const lucide = await getLucideReact();
  return lucide[iconName];
}

/**
 * Get multiple icons at once
 */
export async function lazyLucideIcons(iconNames: string[]): Promise<Record<string, any>> {
  const lucide = await getLucideReact();
  const icons: Record<string, any> = {};
  for (const name of iconNames) {
    if (lucide[name]) {
      icons[name] = lucide[name];
    }
  }
  return icons;
}

/**
 * Synchronous icon getter (use only after app is fully loaded)
 */
export function getLucideIconSync(iconName: string): any {
  if (!ReactReady || !lucideReact) {
    console.warn(`Attempting to get ${iconName} before lucide-react is ready`);
    return null;
  }
  return lucideReact[iconName];
}
