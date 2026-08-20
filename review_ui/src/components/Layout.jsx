import { Outlet, NavLink } from 'react-router-dom'
import { FileText, Upload, List, Cpu, ChevronLeft, ChevronRight, CheckCircle2, AlertCircle } from 'lucide-react'
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
    axios.get('/health').then(r => setHealth(r.data)).catch(() => {})
  }, [])

  const allActive = health?.yolo_loaded && health?.layoutlm_loaded && health?.ollama_available

  return (
    <div className="min-h-screen flex bg-slate-50">
      {/* Sidebar */}
      <aside
        className={`${
          collapsed ? 'w-[68px]' : 'w-60'
        } bg-white border-r border-slate-200 flex flex-col transition-all duration-300 ease-in-out flex-shrink-0 select-none z-30 shadow-xs`}
      >
        {/* Logo & Header */}
        <div className={`py-4 border-b border-slate-100 flex items-center ${collapsed ? 'px-3 justify-center' : 'px-4 justify-between'}`}>
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm shadow-blue-500/20">
              <FileText size={17} className="text-white" />
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <div className="text-sm font-bold text-slate-900 tracking-tight leading-none">Invoice</div>
                <div className="text-[11px] text-blue-600 font-semibold mt-0.5">Digitizer</div>
              </div>
            )}
          </div>

          {!collapsed && (
            <button
              onClick={toggleSidebar}
              className="p-1 rounded-md text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
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
               ${isActive ? 'bg-blue-50 text-blue-700 shadow-xs' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`
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
               ${isActive ? 'bg-blue-50 text-blue-700 shadow-xs' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`
            }
          >
            <List size={18} className="flex-shrink-0" />
            {!collapsed && <span>All Invoices</span>}
          </NavLink>
        </nav>

        {/* Pipeline status */}
        <div className={`border-t border-slate-100 ${collapsed ? 'p-2' : 'p-4'}`}>
          {!collapsed ? (
            <>
              <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center justify-between">
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
              className="flex flex-col items-center py-2"
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
        <div className="p-2 border-t border-slate-100 bg-slate-50/60">
          <button
            onClick={toggleSidebar}
            className={`w-full flex items-center ${
              collapsed ? 'justify-center' : 'justify-between px-2.5'
            } py-2 rounded-lg text-xs font-semibold text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-all`}
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {!collapsed && <span>Collapse Sidebar</span>}
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto min-w-0">
        <Outlet />
      </main>
    </div>
  )
}

function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500 font-medium">{label}</span>
      <span className={`text-xs font-semibold flex items-center gap-1.5 ${ok ? 'text-emerald-600' : 'text-amber-500'}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-400'}`}></span>
        {ok ? 'Active' : 'Fallback'}
      </span>
    </div>
  )
}

