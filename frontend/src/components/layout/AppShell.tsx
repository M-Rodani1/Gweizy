import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, BarChart3, Settings2, BookOpen } from 'lucide-react';
import StickyHeader from '../StickyHeader';
import { useChain } from '../../contexts/ChainContext';
import { checkHealth } from '../../api/gasApi';

type NavItem = {
  label: string;
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
};

interface AppShellProps {
  children: React.ReactNode;
  activePath?: string;
}

const AppShell: React.FC<AppShellProps> = ({ children, activePath }) => {
  const location = useLocation();
  const { selectedChain, multiChainGas } = useChain();
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;
  const currentPath = activePath || location.pathname;

  const navItems: NavItem[] = useMemo(() => ([
    { label: 'Overview', to: '/app', icon: LayoutDashboard, description: 'AI pilot & personalization' },
    { label: 'Analytics', to: '/analytics', icon: BarChart3, description: 'Forecasts, trends, heatmaps' },
    { label: 'System', to: '/system', icon: Settings2, description: 'API, mempool, model status' },
    { label: 'Docs', to: '/docs', icon: BookOpen, description: 'How it works & API usage' }
  ]), []);

  useEffect(() => {
    let mounted = true;
    const runHealthCheck = async () => {
      const healthy = await checkHealth();
      if (mounted) setApiStatus(healthy ? 'online' : 'offline');
    };
    runHealthCheck();
    const interval = setInterval(runHealthCheck, 60000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen app-shell">
      <a href="#main-content" className="skip-nav">Skip to main content</a>
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <div className="flex">
        {/* Left rail */}
        <aside
          className="hidden md:block w-64 border-r border-gray-800 bg-gray-950/70 backdrop-blur-lg sticky top-[76px] h-[calc(100vh-76px)]"
          aria-label="Application navigation"
        >
          <nav className="p-4 space-y-2">
            {navItems.map(({ label, to, icon: Icon, description }) => {
              const active = currentPath === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`group block rounded-xl px-3 py-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 transition-all ${
                    active
                      ? 'bg-cyan-500/15 border border-cyan-500/30 text-white shadow-lg shadow-cyan-500/10'
                      : 'border border-transparent hover:border-gray-700 hover:bg-gray-900/60 text-gray-200'
                  }`}
                  aria-current={active ? 'page' : undefined}
                  role="link"
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-5 h-5 ${active ? 'text-cyan-300' : 'text-gray-400 group-hover:text-gray-200'}`} />
                    <div className="flex-1">
                      <div className="font-semibold leading-tight">{label}</div>
                      <p className="text-xs text-gray-500 group-hover:text-gray-400">{description}</p>
                    </div>
                  </div>
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Main content */}
        <main id="main-content" className="flex-1 px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppShell;
