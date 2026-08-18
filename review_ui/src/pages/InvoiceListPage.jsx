import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FileText, AlertTriangle, CheckCircle, Clock, XCircle, RefreshCw,
  Sparkles, UploadCloud, Cpu, Play
} from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'

const STATUS_CONFIG = {
  done:       { icon: CheckCircle,   color: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-200', label: 'Done' },
  reviewed:   { icon: CheckCircle,   color: 'text-blue-600',    bg: 'bg-blue-50 border-blue-200',       label: 'Reviewed' },
  processing: { icon: Clock,         color: 'text-amber-600',   bg: 'bg-amber-50 border-amber-200',     label: 'Processing' },
  pending:    { icon: Clock,         color: 'text-slate-500',   bg: 'bg-slate-50 border-slate-200',     label: 'Pending' },
  failed:     { icon: XCircle,       color: 'text-red-600',     bg: 'bg-red-50 border-red-200',         label: 'Failed' },
}

export default function InvoiceListPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [triggering, setTriggering] = useState(false)

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

  const fetchTrainingStatus = async () => {
    try {
      const { data } = await axios.get('/api/train/status')
      setTrainingStatus(data)
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchJobs()
    fetchTrainingStatus()
    const interval = setInterval(() => {
      fetchJobs()
      fetchTrainingStatus()
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  const startTraining = async (modelType) => {
    setTriggering(true)
    try {
      await axios.post(`/api/train/${modelType}`)
      toast.success(`${modelType.toUpperCase()} training started in background!`)
      fetchTrainingStatus()
    } catch (e) {
      toast.error('Training trigger failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const reviewedCount = jobs.filter(j => j.status === 'reviewed' || j.status === 'done').length

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto space-y-6">
      
      {/* ── Top Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">All Invoices</h1>
          <p className="text-slate-500 mt-0.5 text-xs md:text-sm">
            {jobs.length} invoice{jobs.length !== 1 ? 's' : ''} digitized &bull; {reviewedCount} verified ground-truth samples
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={fetchJobs} className="btn-secondary text-xs">
            <RefreshCw size={13} /> Refresh
          </button>
          <button onClick={() => navigate('/')} className="btn-primary text-xs">
            <UploadCloud size={14} /> Upload Invoice
          </button>
        </div>
      </div>

      {/* ── Model Training Dashboard Banner ────────────────────────────────── */}
      <div className="card p-5 bg-gradient-to-r from-slate-900 to-slate-800 text-white shadow-md">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-400">
              <Cpu size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold">AI Model Intelligence Hub</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Active
                </span>
              </div>
              <p className="text-xs text-slate-300 mt-0.5">
                YOLOv8: <span className="font-semibold text-emerald-400">{trainingStatus?.yolo_loaded ? 'Loaded (Custom pt)' : 'Default'}</span> &bull; 
                LayoutLMv3: <span className="font-semibold text-blue-300">{trainingStatus?.layoutlm_loaded ? 'Fine-tuned' : 'Zero-shot / LLM fallback'}</span>
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {trainingStatus?.is_training ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/30 text-amber-300 text-xs">
                <RefreshCw size={13} className="spinner" />
                <span>{trainingStatus.progress || 'Training model in progress...'}</span>
              </div>
            ) : (
              <>
                <button
                  type="button"
                  disabled={triggering || reviewedCount === 0}
                  onClick={() => startTraining('yolo')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white transition-all disabled:opacity-40"
                  title="Retrain YOLOv8 on all reviewed ground-truth bounding boxes"
                >
                  <Play size={12} fill="currentColor" /> Retrain YOLO
                </button>
                <button
                  type="button"
                  disabled={triggering || reviewedCount === 0}
                  onClick={() => startTraining('layoutlm')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-600 hover:bg-teal-500 active:bg-teal-700 text-white transition-all disabled:opacity-40"
                  title="Export reviewed invoices and fine-tune LayoutLMv3"
                >
                  <Sparkles size={12} /> Train LayoutLMv3
                </button>
              </>
            )}
          </div>
        </div>

        {trainingStatus?.last_trained && (
          <div className="mt-3 pt-3 border-t border-slate-700/60 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Last trained on: {trainingStatus.last_trained}</span>
            <span>Ground truth samples: {reviewedCount} available</span>
          </div>
        )}
      </div>

      {/* ── Table Content ──────────────────────────────────────────────────── */}
      {loading ? (
        <div className="flex items-center justify-center h-48 text-slate-400">
          <RefreshCw size={22} className="spinner mr-2 text-blue-600" /> Loading invoice database...
        </div>
      ) : jobs.length === 0 ? (
        <div className="card p-16 text-center text-slate-400">
          <FileText size={44} className="mx-auto mb-3 opacity-25 text-slate-500" />
          <p className="font-semibold text-slate-700 text-base">No Invoices Uploaded Yet</p>
          <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
            Upload your first invoice PDF or image to extract fields, inspect layout boxes, and verify data.
          </p>
          <button onClick={() => navigate('/')} className="btn-primary mt-5 text-xs">
            <UploadCloud size={14} /> Upload Invoice Now
          </button>
        </div>
      ) : (
        <div className="card overflow-hidden border-slate-200">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/80 text-slate-500 font-semibold uppercase tracking-wider">
                <th className="text-left px-5 py-3.5">Invoice File</th>
                <th className="text-left px-5 py-3.5">Status</th>
                <th className="text-left px-5 py-3.5">Confidence</th>
                <th className="text-left px-5 py-3.5">Review Flags</th>
                <th className="text-left px-5 py-3.5">Processed Date</th>
                <th className="px-5 py-3.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {jobs.map((job, i) => {
                const cfg = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending
                const Icon = cfg.icon
                const conf = job.overall_confidence != null
                  ? Math.round(job.overall_confidence * 100)
                  : null

                return (
                  <tr
                    key={job.job_id}
                    className={`hover:bg-blue-50/40 cursor-pointer transition-colors ${
                      i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'
                    }`}
                    onClick={() => navigate(`/invoices/${job.job_id}`)}
                  >
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center shrink-0 border border-blue-100">
                          <FileText size={15} />
                        </div>
                        <div className="min-w-0">
                          <span className="font-bold text-slate-900 truncate block max-w-[240px]">
                            {job.filename}
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono">
                            {job.job_id.slice(0, 8)}…
                          </span>
                        </div>
                      </div>
                    </td>

                    <td className="px-5 py-3.5">
                      <span className={`badge border ${cfg.bg} ${cfg.color}`}>
                        <Icon size={11} className="mr-1" />
                        {cfg.label}
                      </span>
                    </td>

                    <td className="px-5 py-3.5">
                      {conf != null ? (
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                conf >= 80 ? 'bg-emerald-500' : conf >= 65 ? 'bg-amber-400' : 'bg-red-400'
                              }`}
                              style={{ width: `${conf}%` }}
                            />
                          </div>
                          <span className="font-mono font-bold text-slate-700">{conf}%</span>
                        </div>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>

                    <td className="px-5 py-3.5">
                      {job.needs_review ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
                          <AlertTriangle size={11} /> Needs Review
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                          <CheckCircle size={11} /> Verified
                        </span>
                      )}
                    </td>

                    <td className="px-5 py-3.5 text-slate-500 font-mono">
                      {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                    </td>

                    <td className="px-5 py-3.5 text-right">
                      <span className="text-xs font-semibold text-blue-600 hover:text-blue-800">
                        View &rarr;
                      </span>
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
