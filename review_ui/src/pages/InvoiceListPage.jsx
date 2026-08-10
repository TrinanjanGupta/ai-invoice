import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, AlertTriangle, CheckCircle, Clock, XCircle, RefreshCw } from 'lucide-react'
import axios from 'axios'

const STATUS_CONFIG = {
  done:       { icon: CheckCircle,   color: 'text-green-600',  bg: 'bg-green-50',  label: 'Done' },
  reviewed:   { icon: CheckCircle,   color: 'text-blue-600',   bg: 'bg-blue-50',   label: 'Reviewed' },
  processing: { icon: Clock,         color: 'text-amber-600',  bg: 'bg-amber-50',  label: 'Processing' },
  pending:    { icon: Clock,         color: 'text-gray-500',   bg: 'bg-gray-50',   label: 'Pending' },
  failed:     { icon: XCircle,       color: 'text-red-600',    bg: 'bg-red-50',    label: 'Failed' },
}

export default function InvoiceListPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchJobs = async () => {
    try {
      const { data } = await axios.get('/api/invoices?limit=50')
      setJobs(data.jobs)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">All Invoices</h1>
          <p className="text-gray-500 mt-1 text-sm">{jobs.length} invoice{jobs.length !== 1 ? 's' : ''} processed</p>
        </div>
        <button onClick={fetchJobs} className="btn-secondary">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48 text-gray-400">
          <RefreshCw size={20} className="spinner mr-2" /> Loading...
        </div>
      ) : jobs.length === 0 ? (
        <div className="card p-16 text-center text-gray-400">
          <FileText size={40} className="mx-auto mb-3 opacity-30" />
          <p className="font-medium">No invoices yet</p>
          <p className="text-sm mt-1">Upload your first invoice to get started</p>
          <button onClick={() => navigate('/')} className="btn-primary mt-4">
            Upload Invoice
          </button>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/60">
                <th className="text-left px-5 py-3 font-semibold text-gray-500 text-xs uppercase tracking-wide">File</th>
                <th className="text-left px-5 py-3 font-semibold text-gray-500 text-xs uppercase tracking-wide">Status</th>
                <th className="text-left px-5 py-3 font-semibold text-gray-500 text-xs uppercase tracking-wide">Confidence</th>
                <th className="text-left px-5 py-3 font-semibold text-gray-500 text-xs uppercase tracking-wide">Flags</th>
                <th className="text-left px-5 py-3 font-semibold text-gray-500 text-xs uppercase tracking-wide">Date</th>
                <th className="px-5 py-3" />
              </tr>
            </thead>
            <tbody>
              {jobs.map((job, i) => {
                const cfg = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending
                const Icon = cfg.icon
                const conf = job.overall_confidence != null
                  ? Math.round(job.overall_confidence * 100)
                  : null
                return (
                  <tr
                    key={job.job_id}
                    className={`border-b border-gray-50 hover:bg-brand-50/40 cursor-pointer transition-colors
                                ${i % 2 === 0 ? '' : 'bg-gray-50/30'}`}
                    onClick={() => navigate(`/invoices/${job.job_id}`)}
                  >
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 bg-brand-50 rounded-lg flex items-center justify-center flex-shrink-0">
                          <FileText size={14} className="text-brand-600" />
                        </div>
                        <span className="font-medium text-gray-800 truncate max-w-[200px]">
                          {job.filename}
                        </span>
                      </div>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className={`badge ${cfg.bg} ${cfg.color}`}>
                        <Icon size={10} className="mr-1" />
                        {cfg.label}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      {conf != null ? (
                        <div className="flex items-center gap-2">
                          <div className="conf-bar w-20">
                            <div
                              className={`conf-bar-fill ${conf >= 85 ? 'bg-green-500' : conf >= 65 ? 'bg-amber-400' : 'bg-red-400'}`}
                              style={{ width: `${conf}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500">{conf}%</span>
                        </div>
                      ) : <span className="text-gray-300">—</span>}
                    </td>
                    <td className="px-5 py-3.5">
                      {job.needs_review ? (
                        <span className="badge bg-amber-50 text-amber-700">
                          <AlertTriangle size={10} className="mr-1" />
                          Needs Review
                        </span>
                      ) : job.status === 'done' || job.status === 'reviewed' ? (
                        <span className="badge bg-green-50 text-green-700">
                          <CheckCircle size={10} className="mr-1" />
                          Validated
                        </span>
                      ) : null}
                    </td>
                    <td className="px-5 py-3.5 text-gray-400 text-xs">
                      {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <span className="text-brand-600 text-xs font-medium">View →</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
