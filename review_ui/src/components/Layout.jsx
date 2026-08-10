import { Outlet, NavLink } from 'react-router-dom'
import { FileText, Upload, List, Cpu } from 'lucide-react'
import { useState, useEffect } from 'react'
import axios from 'axios'

export default function Layout() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    axios.get('/health').then(r => setHealth(r.data)).catch(() => {})
  }, [])

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-60 bg-white border-r border-gray-200 flex flex-col">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-gray-100">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center">
              <FileText size={16} className="text-white" />
            </div>
            <div>
              <div className="text-sm font-bold text-gray-900">Invoice</div>
              <div className="text-xs text-gray-400 -mt-0.5">Digitizer</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors
               ${isActive ? 'bg-brand-50 text-brand-700' : 'text-gray-600 hover:bg-gray-50'}`
            }
          >
            <Upload size={16} />
            Upload Invoice
          </NavLink>
          <NavLink
            to="/invoices"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors
               ${isActive ? 'bg-brand-50 text-brand-700' : 'text-gray-600 hover:bg-gray-50'}`
            }
          >
            <List size={16} />
            All Invoices
          </NavLink>
        </nav>

        {/* Pipeline status */}
        <div className="px-4 py-4 border-t border-gray-100">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Pipeline Status
          </div>
          {health ? (
            <div className="space-y-2">
              <StatusRow label="YOLO" ok={health.yolo_loaded} />
              <StatusRow label="LayoutLMv3" ok={health.layoutlm_loaded} />
              <StatusRow label={`Ollama (${health.ollama_model})`} ok={health.ollama_available} />
            </div>
          ) : (
            <div className="text-xs text-gray-400">Checking...</div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-xs font-medium ${ok ? 'text-green-600' : 'text-amber-500'}`}>
        {ok ? '● Active' : '○ Fallback'}
      </span>
    </div>
  )
}
