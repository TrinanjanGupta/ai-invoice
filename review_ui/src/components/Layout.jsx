import { Outlet, NavLink } from 'react-router-dom'
import {
  FileText, Upload, List, Cpu, ChevronLeft, ChevronRight,
  Sun, Moon
} from 'lucide-react'
import { useState, useEffect } from 'react'
import axios from 'axios'

export default function Layout() {
  const [health, setHealth] = useState(null)
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem('sidebar_collapsed') === 'true'
    } catch {
      return false
    }
  })

  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('theme')
      if (saved) return saved
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    } catch {
      return 'light'
    }
  })

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    try {
      localStorage.setItem('theme', theme)
    } catch {}
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))
  }

  const toggleSidebar = () => {
    setCollapsed(prev => {
      const next = !prev
      try {
        localStorage.setItem('sidebar_collapsed', String(next))
      } catch {}
      return next
    })
  }

  useEffect(() => {
    const fetchHealth = () => {
      axios.get('/health').then(r => setHealth(r.data)).catch(() => {})
    }
    fetchHealth()
    const timer = setInterval(fetchHealth, 15000)
    return () => clearInterval(timer)
  }, [])

  const allActive = health?.yolo_loaded && health?.layoutlm_loaded && health?.ollama_available

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans antialiased">
      {/* Sidebar (Fixed to viewport height, never scrolls with the form) */}
      <aside
        className={`${
          collapsed ? 'w-[68px]' : 'w-60'
        } h-screen sticky top-0 flex flex-col justify-between bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 transition-all duration-300 ease-in-out flex-shrink-0 select-none z-30 shadow-xs`}
      >
        {/* Top Part: Logo & Navigation */}
        <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">
          {/* Logo & Header */}
          <div className={`py-4 border-b border-slate-100 dark:border-slate-800 flex items-center ${collapsed ? 'px-3 justify-center' : 'px-4 justify-between'}`}>
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm shadow-blue-500/20">
                <FileText size={17} className="text-white" />
              </div>
              {!collapsed && (
                <div className="min-w-0">
                  <div className="text-sm font-bold text-slate-900 dark:text-white tracking-tight leading-none">Invoice</div>
                  <div className="text-[11px] text-blue-600 dark:text-blue-400 font-semibold mt-0.5">Digitizer</div>
                </div>
              )}
            </div>

            {!collapsed && (
              <button
                onClick={toggleSidebar}
                className="p-1 rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                title="Collapse sidebar"
              >
                <ChevronLeft size={16} />
              </button>
            )}
          </div>

          {/* Navigation */}
          <nav className={`flex-1 ${collapsed ? 'px-2' : 'px-3'} py-4 space-y-1.5`}>
            <NavLink
              to="/"
              end
              title={collapsed ? 'Upload Invoice' : undefined}
              className={({ isActive }) =>
                `flex items-center ${collapsed ? 'justify-center px-0' : 'gap-3 px-3'} py-2.5 rounded-lg text-sm font-semibold transition-all
                 ${isActive
                   ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 shadow-xs'
                   : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-200'}`
              }
            >
              <Upload size={18} className="flex-shrink-0" />
              {!collapsed && <span>Upload Invoice</span>}
            </NavLink>

            <NavLink
              to="/invoices"
              title={collapsed ? 'All Invoices' : undefined}
              className={({ isActive }) =>
                `flex items-center ${collapsed ? 'justify-center px-0' : 'gap-3 px-3'} py-2.5 rounded-lg text-sm font-semibold transition-all
                 ${isActive
                   ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 shadow-xs'
                   : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-200'}`
              }
            >
              <List size={18} className="flex-shrink-0" />
              {!collapsed && <span>All Invoices</span>}
            </NavLink>
          </nav>
        </div>

        {/* Bottom Part: Dark Mode Toggle + Pipeline Status + Collapse Toggle */}
        <div className="flex-shrink-0 border-t border-slate-100 dark:border-slate-800">
          
          {/* ── Theme Switcher (Dark / Light Mode) ─────────────────────────── */}
          <div className={`${collapsed ? 'p-2 flex justify-center' : 'px-4 py-3'}`}>
            {collapsed ? (
              <button
                onClick={toggleTheme}
                className="w-10 h-10 rounded-lg flex items-center justify-center text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
                title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
              >
                {theme === 'dark' ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} className="text-slate-600" />}
              </button>
            ) : (
              <button
                onClick={toggleTheme}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-slate-100/80 dark:bg-slate-800/70 hover:bg-slate-200/80 dark:hover:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-300 transition-colors border border-slate-200/60 dark:border-slate-700/60 shadow-2xs"
                title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
              >
                <div className="flex items-center gap-2">
                  {theme === 'dark' ? (
                    <Sun size={15} className="text-amber-400" />
                  ) : (
                    <Moon size={15} className="text-slate-600 dark:text-slate-400" />
                  )}
                  <span>{theme === 'dark' ? 'Dark Mode' : 'Light Mode'}</span>
                </div>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400">
                  {theme === 'dark' ? 'On' : 'Off'}
                </span>
              </button>
            )}
          </div>

          {/* Pipeline status */}
          <div className={`border-t border-slate-100 dark:border-slate-800 ${collapsed ? 'p-2' : 'px-4 py-3'}`}>
            {!collapsed ? (
              <>
                <div className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2 flex items-center justify-between">
                  <span>Pipeline Status</span>
                  <span className={`w-2 h-2 rounded-full ${allActive ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`} />
                </div>
                {health ? (
                  <div className="space-y-1.5">
                    <StatusRow label="YOLOv8" ok={health.yolo_loaded} />
                    <StatusRow label="LayoutLMv3" ok={health.layoutlm_loaded} />
                    <StatusRow label="Ollama AI" ok={health.ollama_available} />
                  </div>
                ) : (
                  <div className="text-xs text-slate-400">Checking...</div>
                )}
              </>
            ) : (
              <div
                className="flex flex-col items-center py-1"
                title={`AI Pipeline: ${allActive ? 'All Models Active' : 'Partial/Checking'}`}
              >
                <div className="relative">
                  <Cpu size={18} className="text-slate-400 hover:text-blue-600 transition-colors" />
                  <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${allActive ? 'bg-emerald-500' : 'bg-amber-400'}`} />
                </div>
              </div>
            )}
          </div>

          {/* Bottom Toggle Bar */}
          <div className="p-2 border-t border-slate-100 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/60">
            <button
              onClick={toggleSidebar}
              className={`w-full flex items-center ${
                collapsed ? 'justify-center' : 'justify-between px-2.5'
              } py-2 rounded-lg text-xs font-semibold text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-all`}
              title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
            >
              {!collapsed && <span>Collapse Sidebar</span>}
              {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
          </div>
        </div>
      </aside>

      {/* Main content (Scrolls independently of sidebar) */}
      <main className="flex-1 h-screen overflow-y-auto min-w-0 bg-slate-50 dark:bg-slate-950">
        <Outlet />
      </main>
    </div>
  )
}

function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">{label}</span>
      <span className={`text-xs font-semibold flex items-center gap-1.5 ${ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-500 dark:text-amber-400'}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-400'}`}></span>
        {ok ? 'Active' : 'Fallback'}
      </span>
    </div>
  )
}

