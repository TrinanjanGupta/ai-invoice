import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FileText, AlertTriangle, CheckCircle, Clock, XCircle, RefreshCw,
  Sparkles, UploadCloud, Cpu, Play, Search, X, ChevronLeft, ChevronRight,
  ChevronsLeft, ChevronsRight, Filter, RotateCw, Trash2
} from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'

const STATUS_CONFIG = {
  reviewed:           { icon: CheckCircle,   color: 'text-emerald-700', bg: 'bg-emerald-50 border-emerald-200', label: 'Verified Ground Truth' },
  partially_reviewed: { icon: Clock,         color: 'text-amber-800',   bg: 'bg-amber-50 border-amber-200',     label: 'Partially Reviewed' },
  done:               { icon: FileText,      color: 'text-blue-700',    bg: 'bg-blue-50 border-blue-200',       label: 'AI Extracted (Unreviewed)' },
  processing:         { icon: Clock,         color: 'text-purple-600',  bg: 'bg-purple-50 border-purple-200',   label: 'Processing' },
  pending:            { icon: Clock,         color: 'text-slate-500',   bg: 'bg-slate-50 border-slate-200',     label: 'Pending' },
  failed:             { icon: XCircle,       color: 'text-red-600',     bg: 'bg-red-50 border-red-200',         label: 'Failed' },
}

const FILTER_TABS = [
  { id: 'active_learning_queue', label: '🔥 AI Learning Queue', icon: Sparkles, color: 'text-rose-600' },
  { id: 'non_pending',        label: 'Active & Done',              icon: FileText },
  { id: 'reviewed',           label: 'Verified Ground Truth',      icon: CheckCircle, color: 'text-emerald-600' },
  { id: 'partially_reviewed', label: 'Partially Reviewed',         icon: Clock,       color: 'text-amber-600' },
  { id: 'done',               label: 'AI Extracted (Unreviewed)',  icon: FileText,    color: 'text-blue-600' },
  { id: 'all',                label: 'All Invoices',               icon: FileText },
  { id: 'failed',             label: 'Failed',                     icon: XCircle,     color: 'text-red-600' },
]

export default function InvoiceListPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const [totalCount, setTotalCount] = useState(0)
  const [initialLoading, setInitialLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [triggering, setTriggering] = useState(false)
  const [rescanningIds, setRescanningIds] = useState(new Set())
  const [deletingIds, setDeletingIds] = useState(new Set())

  // Pagination & Filter States (Default: non_pending)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [statusFilter, setStatusFilter] = useState('non_pending')
  const [searchInput, setSearchInput] = useState('')
  const [activeSearch, setActiveSearch] = useState('')

  // State ref for background operations without effect invalidation
  const stateRef = useRef({ page: 1, pageSize: 20, statusFilter: 'non_pending', activeSearch: '' })
  stateRef.current = { page, pageSize, statusFilter, activeSearch }

  const refreshDebounceTimerRef = useRef(null)
  const globalEsRef = useRef(null)

  // Fetch jobs with silent background support
  const fetchJobs = useCallback(async (params = {}, isSilent = false) => {
    const cur = stateRef.current
    const p = params.page ?? cur.page
    const ps = params.pageSize ?? cur.pageSize
    const sf = params.statusFilter ?? cur.statusFilter
    const s = params.activeSearch !== undefined ? params.activeSearch : cur.activeSearch

    if (!isSilent) {
      setIsRefreshing(true)
    }

    try {
      const offset = (p - 1) * ps
      let url = `/api/invoices?limit=${ps}&offset=${offset}&status=${sf}`
      if (sf === 'active_learning_queue') {
        url = `/api/active-learning/queue?limit=${ps}`
      } else if (s && s.trim()) {
        url += `&search=${encodeURIComponent(s.trim())}`
      }

      const { data } = await axios.get(url)
      const jobList = sf === 'active_learning_queue' ? (data.queue || []) : (data.jobs || [])
      setJobs(jobList)
      setTotalCount(sf === 'active_learning_queue' ? (data.total_in_queue || jobList.length) : (data.total || 0))

      setRescanningIds(prev => {
        const next = new Set(prev)
        for (const job of jobList) {
          if (job.status !== 'processing' && next.has(job.job_id)) {
            next.delete(job.job_id)
          }
        }
        return next
      })
    } catch (e) {
      console.error('fetchJobs error:', e)
      if (!isSilent) {
        toast.error('Failed to load invoice list: ' + (e.response?.data?.detail || e.message))
      }
    } finally {
      setInitialLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  // Patch a single row in the jobs list from an SSE event payload
  const patchJob = useCallback((payload) => {
    const { job_id, status, stage, stage_index, stage_label, progress_pct } = payload

    if (status === 'deleted') {
      setJobs(prev => prev.filter(j => j.job_id !== job_id))
      setTotalCount(prev => Math.max(0, prev - 1))
      return
    }

    setJobs(prev => prev.map(j =>
      j.job_id === job_id
        ? { ...j, status: status || j.status, stage, stage_index, stage_label, progress_pct }
        : j
    ))

    // Auto-clear from rescanningIds when job finishes
    if (status === 'done' || status === 'reviewed' || status === 'failed') {
      setRescanningIds(prev => {
        const next = new Set(prev)
        next.delete(job_id)
        return next
      })

      // Debounce silent background refresh (2 seconds)
      if (refreshDebounceTimerRef.current) {
        clearTimeout(refreshDebounceTimerRef.current)
      }
      refreshDebounceTimerRef.current = setTimeout(() => {
        fetchJobs({}, true)
      }, 2000)
    }
  }, [fetchJobs])

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const { data } = await axios.get('/api/train/status')
      setTrainingStatus(data)
    } catch (e) {
      console.error(e)
    }
  }, [])

  // Initial load and filter change trigger
  useEffect(() => {
    fetchJobs({ page, pageSize, statusFilter, activeSearch }, false)
    fetchTrainingStatus()
  }, [page, pageSize, statusFilter, activeSearch, fetchJobs, fetchTrainingStatus])

  // Global SSE listener — single stable connection on mount
  useEffect(() => {
    const es = new EventSource('/api/stream/jobs')
    globalEsRef.current = es

    es.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        patchJob(payload)
      } catch { /* ignore malformed events */ }
    }

    es.onerror = () => {
      // EventSource auto-reconnects
    }

    const fallback = setInterval(() => {
      fetchJobs({}, true)
      fetchTrainingStatus()
    }, 30_000)

    return () => {
      es.close()
      clearInterval(fallback)
      if (refreshDebounceTimerRef.current) {
        clearTimeout(refreshDebounceTimerRef.current)
      }
    }
  }, [fetchJobs, patchJob, fetchTrainingStatus])

  // Search handlers
  const handleSearch = (e) => {
    if (e) e.preventDefault()
    const trimmed = searchInput.trim()
    setActiveSearch(trimmed)
    setPage(1)
  }

  const handleClearSearch = () => {
    setSearchInput('')
    setActiveSearch('')
    setPage(1)
  }

  // Filter tab change handler
  const handleFilterChange = (tabId) => {
    setStatusFilter(tabId)
    setPage(1)
  }

  // Page size change handler
  const handlePageSizeChange = (newSize) => {
    const size = parseInt(newSize, 10)
    setPageSize(size)
    setPage(1)
  }

  const autoAcceptHighConfidence = async () => {
    setTriggering(true)
    try {
      const { data } = await axios.post('/api/active-learning/auto-accept')
      toast.success(data.message || 'Auto-accepted high-confidence invoices!')
      fetchJobs({}, true)
    } catch (e) {
      toast.error('Auto-accept error: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const triggerChampionRetraining = async () => {
    setTriggering(true)
    try {
      toast.loading('Champion/Challenger auto-retraining started...', { id: 'retrain' })
      const { data } = await axios.post('/api/active-learning/auto-train?epochs=10')
      const f1 = data.champion_f1 ?? data.champion_accuracy ?? data.candidate_f1 ?? 0
      if (data.status === 'PROMOTED') {
        toast.success(`🏆 Candidate promoted! Entity F1: ${(f1 * 100).toFixed(1)}%`, { id: 'retrain' })
      } else {
        const candF1 = data.candidate_f1 ?? data.candidate_accuracy ?? 0
        toast.error(`❌ Model kept champion: Candidate F1 was ${(candF1 * 100).toFixed(1)}%`, { id: 'retrain' })
      }
      fetchTrainingStatus()
      fetchJobs({}, true)
    } catch (e) {
      toast.error('Retraining failed: ' + (e.response?.data?.detail || e.message), { id: 'retrain' })
    } finally {
      setTriggering(false)
    }
  }

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

  const reprocessAll = async () => {
    setTriggering(true)
    try {
      const { data } = await axios.post('/api/invoices/reprocess-all?only_unreviewed=true')
      toast.success(`Queued ${data.total_queued} invoice(s) for re-scanning!`)
      if (data.queued && data.queued.length > 0) {
        const newIds = new Set(data.queued.map(q => q.job_id))
        setRescanningIds(newIds)
        setJobs(prev => prev.map(j => newIds.has(j.job_id) ? { ...j, status: 'processing' } : j))
      }
      fetchJobs({}, true)
    } catch (e) {
      toast.error('Re-scan trigger failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const retryAllFailed = async () => {
    setTriggering(true)
    try {
      const { data } = await axios.post('/api/invoices/retry-all-failed')
      toast.success(`Queued ${data.total_retried} failed invoice(s) for retry at the end of queue!`)
      if (data.jobs && data.jobs.length > 0) {
        const newIds = new Set(data.jobs.map(q => q.job_id))
        setRescanningIds(newIds)
        setJobs(prev => prev.map(j => newIds.has(j.job_id) ? { ...j, status: 'processing' } : j))
      }
      fetchJobs({}, true)
    } catch (e) {
      toast.error('Retry failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const reprocessSingle = async (e, jobId) => {
    e.stopPropagation()
    setRescanningIds(prev => new Set(prev).add(jobId))
    setJobs(prev => prev.map(j => j.job_id === jobId ? { ...j, status: 'processing' } : j))
    try {
      await axios.post(`/api/invoices/${jobId}/reprocess`)
      toast.success('Invoice re-scan in progress...')
    } catch (e) {
      setRescanningIds(prev => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
      toast.error('Re-scan failed: ' + (e.response?.data?.detail || e.message))
      fetchJobs({}, true)
    }
  }

  const retrySingle = async (e, jobId) => {
    e.stopPropagation()
    setRescanningIds(prev => new Set(prev).add(jobId))
    setJobs(prev => prev.map(j => j.job_id === jobId ? { ...j, status: 'processing' } : j))
    try {
      await axios.post(`/api/invoices/${jobId}/retry`)
      toast.success('Invoice added to queue for retry!')
    } catch (e) {
      setRescanningIds(prev => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
      toast.error('Retry failed: ' + (e.response?.data?.detail || e.message))
      fetchJobs({}, true)
    }
  }

  const deleteSingle = async (e, jobId, filename) => {
    e.stopPropagation()
    if (!window.confirm(`Are you sure you want to delete "${filename || 'this invoice'}" from the queue?`)) {
      return
    }

    setDeletingIds(prev => new Set(prev).add(jobId))
    try {
      await axios.delete(`/api/invoices/${jobId}`)
      toast.success('Invoice deleted from queue')
      setJobs(prev => prev.filter(j => j.job_id !== jobId))
      setTotalCount(prev => Math.max(0, prev - 1))
    } catch (e) {
      toast.error('Failed to delete invoice: ' + (e.response?.data?.detail || e.message))
    } finally {
      setDeletingIds(prev => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
    }
  }

  const clearUnreviewedAndPending = async () => {
    if (!window.confirm("Are you sure you want to clear all unreviewed, pending, and failed invoices from the list?\n\nVerified Ground Truth invoices will NOT be deleted.")) {
      return
    }
    setTriggering(true)
    try {
      const { data } = await axios.post('/api/invoices/bulk-delete?clear_pending=true&clear_unreviewed=true&clear_failed=true')
      toast.success(data.message || `Cleared ${data.deleted_count} invoice(s) from queue!`)
      fetchJobs({}, false)
    } catch (e) {
      toast.error('Bulk clear failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const cancelProcessing = async () => {
    setTriggering(true)
    try {
      const { data } = await axios.post('/api/invoices/cancel-processing')
      toast.success(data.message || 'Stopped all invoice processing!')
      setRescanningIds(new Set())
      fetchJobs({}, false)
    } catch (e) {
      toast.error('Cancel failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTriggering(false)
    }
  }

  const verifiedCount = jobs.filter(j => j.status === 'reviewed').length
  const processingCount = jobs.filter(j => j.status === 'processing').length
  const failedCount = jobs.filter(j => j.status === 'failed').length
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize))

  return (
    <div className="p-4 md:p-8 max-w-[1600px] w-full mx-auto space-y-6">
      
      {/* ── Top Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-black text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
            <span>Invoice Pipeline & Ground Truth Review</span>
            <span className="text-xs font-mono font-normal px-2.5 py-0.5 rounded-full bg-blue-100 dark:bg-blue-950/80 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
              {totalCount} Total
            </span>
          </h1>
          <p className="text-xs md:text-sm text-slate-500 dark:text-slate-400 mt-1">
            Review AI extractions, verify invoice details for ground-truth training, or trigger model fine-tuning.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Stop / Cancel Processing Button */}
          {(processingCount > 0 || rescanningIds.size > 0) && (
            <button
              onClick={cancelProcessing}
              disabled={triggering}
              className="btn-secondary text-xs bg-red-600 hover:bg-red-700 text-white border-red-700 shadow-sm flex items-center gap-1.5 font-bold animate-pulse"
              title="Immediately cancel all ongoing invoice processing and re-scans"
            >
              <XCircle size={13} />
              <span>Stop Processing ({processingCount || rescanningIds.size})</span>
            </button>
          )}

          {failedCount > 0 && (
            <button
              onClick={retryAllFailed}
              disabled={triggering}
              className="btn-secondary text-xs bg-red-50 hover:bg-red-100 text-red-800 border-red-200 shadow-sm flex items-center gap-1.5"
              title="Retry all failed invoices by adding them to the queue"
            >
              <RotateCw size={13} className={triggering ? 'spinner text-red-600' : 'text-red-600'} />
              <span>Retry Failed ({failedCount})</span>
            </button>
          )}

          <button
            onClick={() => fetchJobs({}, false)}
            disabled={isRefreshing}
            className="btn-secondary text-xs shadow-sm"
          >
            <RefreshCw size={13} className={isRefreshing ? 'spinner text-blue-600' : ''} />
            <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>

          <button
            onClick={reprocessAll}
            disabled={triggering}
            className={`btn-secondary text-xs bg-amber-50 hover:bg-amber-100 text-amber-900 border-amber-200 transition-all shadow-sm ${
              triggering ? 'opacity-60 cursor-not-allowed' : ''
            }`}
            title="Re-run the updated AI pipeline on unreviewed invoices"
          >
            {triggering ? (
              <>
                <RefreshCw size={13} className="spinner text-amber-600" />
                <span>Re-scanning...</span>
              </>
            ) : (
              <>
                <Sparkles size={13} className="text-amber-600" />
                <span>Re-scan Unreviewed</span>
              </>
            )}
          </button>

          <button
            onClick={clearUnreviewedAndPending}
            disabled={triggering}
            className={`btn-secondary text-xs bg-rose-50 hover:bg-rose-100 text-rose-800 dark:bg-rose-950/40 dark:text-rose-300 border-rose-200 dark:border-rose-800 transition-all shadow-sm ${
              triggering ? 'opacity-60 cursor-not-allowed' : ''
            }`}
            title="Clear unreviewed, pending, and failed invoices so you can start a clean batch"
          >
            <Trash2 size={13} className="text-rose-600 dark:text-rose-400" />
            <span>Clear Unreviewed & Pending</span>
          </button>

          <button onClick={() => navigate('/')} className="btn-primary text-xs shadow-sm">
            <UploadCloud size={14} /> Upload Invoice
          </button>
        </div>
      </div>

      {/* ── Model Training Dashboard Banner ────────────────────────────────── */}
      <div className="card p-5 bg-gradient-to-r from-slate-900 to-slate-800 text-white shadow-md border-slate-700">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-400">
              <Cpu size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold tracking-wide">AI Model Intelligence Hub</span>
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
                  disabled={triggering}
                  onClick={autoAcceptHighConfidence}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white transition-all shadow disabled:opacity-40"
                  title="Batch auto-accept all invoices with confidence >= 85% and valid arithmetic"
                >
                  <CheckCircle size={12} /> ⚡ Auto-Accept (≥85%)
                </button>
                <button
                  type="button"
                  disabled={triggering}
                  onClick={triggerChampionRetraining}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white transition-all shadow disabled:opacity-40"
                  title="Train candidate model on Gold corrections and auto-promote if holdout accuracy improves"
                >
                  <Sparkles size={12} /> 🏆 Champion Auto-Train
                </button>
              </>
            )}
          </div>
        </div>

        {trainingStatus?.last_trained && (
          <div className="mt-3 pt-3 border-t border-slate-700/60 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Last trained on: {trainingStatus.last_trained}</span>
            <span>Ground truth samples in view: {verifiedCount}</span>
          </div>
        )}
      </div>

      {/* ── Search & Filter Controls Bar ───────────────────────────────────── */}
      <div className="card p-4 space-y-3.5 bg-white dark:bg-slate-900 shadow-sm border-slate-200 dark:border-slate-800">
        
        {/* Search Bar */}
        <form onSubmit={handleSearch} className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[260px]">
            <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500" />
            <input
              type="text"
              placeholder="Search by invoice file name or Job ID..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full pl-9 pr-8 py-2 bg-slate-50 dark:bg-slate-800/80 hover:bg-slate-100/70 dark:hover:bg-slate-800 focus:bg-white dark:focus:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs md:text-sm text-slate-800 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
            />
            {searchInput && (
              <button
                type="button"
                onClick={handleClearSearch}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-0.5 rounded-full"
                title="Clear search input"
              >
                <X size={14} />
              </button>
            )}
          </div>

          <button
            type="submit"
            className="btn-primary text-xs px-4 py-2 flex items-center gap-1.5 shrink-0 shadow-sm"
          >
            <Search size={13} />
            <span>Search</span>
          </button>

          {activeSearch && (
            <button
              type="button"
              onClick={handleClearSearch}
              className="btn-secondary text-xs px-3 py-2 flex items-center gap-1 text-slate-600 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 shrink-0"
              title="Reset search filter"
            >
              <X size={13} />
              <span>Clear Filter</span>
            </button>
          )}
        </form>

        {/* Status Filter Tabs */}
        <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full no-scrollbar">
            <span className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mr-1 flex items-center gap-1">
              <Filter size={11} /> Filter:
            </span>
            {FILTER_TABS.map((tab) => {
              const TabIcon = tab.icon
              const isActive = statusFilter === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => handleFilterChange(tab.id)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-sm ring-2 ring-blue-500/30'
                      : 'bg-slate-100 dark:bg-slate-800 hover:bg-slate-200/70 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white border border-transparent'
                  }`}
                >
                  <TabIcon size={12} className={isActive ? 'text-white' : tab.color || 'text-slate-500 dark:text-slate-400'} />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </div>

          {/* Page size dropdown */}
          <div className="flex items-center gap-2 shrink-0 text-xs text-slate-500 dark:text-slate-400">
            <span>Per page:</span>
            <select
              value={pageSize}
              onChange={(e) => handlePageSizeChange(e.target.value)}
              className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 font-medium cursor-pointer"
            >
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>
        </div>

      </div>

      {/* ── Table Content ──────────────────────────────────────────────────── */}
      {initialLoading && jobs.length === 0 ? (
        <div className="card p-16 flex flex-col items-center justify-center text-slate-400 space-y-2 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
          <RefreshCw size={24} className="spinner text-blue-600" />
          <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Loading invoice database...</span>
        </div>
      ) : jobs.length === 0 ? (
        <div className="card p-16 text-center text-slate-400 border-dashed bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
          <FileText size={44} className="mx-auto mb-3 opacity-25 text-slate-500" />
          <p className="font-semibold text-slate-700 dark:text-slate-200 text-base">No Matching Invoices Found</p>
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 max-w-sm mx-auto">
            {activeSearch || statusFilter !== 'all'
              ? 'No invoices match the current filter or search criteria. Try clearing search or selecting "All Invoices".'
              : 'Upload your first invoice PDF or image to extract fields, inspect layout boxes, and verify data.'}
          </p>
          <div className="mt-5 flex items-center justify-center gap-2">
            {(activeSearch || statusFilter !== 'all') && (
              <button
                onClick={() => {
                  setStatusFilter('all')
                  handleClearSearch()
                }}
                className="btn-secondary text-xs"
              >
                Reset All Filters
              </button>
            )}
            <button onClick={() => navigate('/')} className="btn-primary text-xs">
              <UploadCloud size={14} /> Upload Invoice Now
            </button>
          </div>
        </div>
      ) : (
        <div className="card overflow-hidden border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50/90 dark:bg-slate-800/80 text-slate-500 dark:text-slate-400 font-semibold uppercase tracking-wider text-[11px]">
                  <th className="text-center px-3.5 py-3.5 w-12 text-slate-400 dark:text-slate-500">#</th>
                  <th className="text-left px-5 py-3.5">Invoice File</th>
                  <th className="text-left px-5 py-3.5">Status</th>
                  <th className="text-left px-5 py-3.5">AI Fetched / Conf</th>
                  <th className="text-left px-5 py-3.5">Review Level</th>
                  <th className="text-left px-5 py-3.5">Created Date</th>
                  <th className="px-5 py-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {jobs.map((job, i) => {
                  const cfg = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending
                  const Icon = cfg.icon
                  const conf = job.overall_confidence != null
                    ? Math.round(job.overall_confidence * 100)
                    : null

                  const isScanning = rescanningIds.has(job.job_id) || job.status === 'processing'
                  const isDeleting = deletingIds.has(job.job_id)
                  const isFailed = job.status === 'failed'
                  const serialNo = (page - 1) * pageSize + i + 1

                  return (
                    <tr
                      key={job.job_id}
                      className={`cursor-pointer transition-all ${
                        isDeleting
                          ? 'opacity-40 pointer-events-none bg-red-50 dark:bg-red-950/40'
                          : isScanning
                          ? 'bg-amber-50/80 dark:bg-amber-950/40 border-l-4 border-l-amber-500 animate-pulse'
                          : isFailed
                          ? 'bg-red-50/20 dark:bg-red-950/20 hover:bg-red-50/40 dark:hover:bg-red-950/40'
                          : i % 2 === 0
                          ? 'bg-white dark:bg-slate-900 hover:bg-blue-50/40 dark:hover:bg-slate-800/60'
                          : 'bg-slate-50/30 dark:bg-slate-900/50 hover:bg-blue-50/40 dark:hover:bg-slate-800/60'
                      }`}
                      onClick={() => navigate(`/invoices/${job.job_id}`)}
                    >
                      {/* S.No. Serial Number */}
                      <td className="px-3.5 py-3.5 text-center text-slate-400 dark:text-slate-500 font-mono text-[11px] select-none font-medium">
                        {serialNo}
                      </td>

                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border ${
                            isFailed
                              ? 'bg-red-50 dark:bg-red-950 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800'
                              : isScanning
                              ? 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400 border-amber-300 dark:border-amber-700'
                              : 'bg-blue-50 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 border-blue-100 dark:border-blue-900'
                          }`}>
                            {isScanning ? (
                              <RefreshCw size={15} className="spinner text-amber-700 dark:text-amber-400" />
                            ) : isFailed ? (
                              <XCircle size={15} className="text-red-600 dark:text-red-400" />
                            ) : (
                              <FileText size={15} />
                            )}
                          </div>
                          <div className="min-w-0">
                            <span className="font-bold text-slate-900 dark:text-white truncate block max-w-[280px]">
                              {job.filename}
                            </span>
                            <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                              {job.job_id.slice(0, 8)}…
                            </span>
                          </div>
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        {isScanning ? (
                          <div className="flex flex-col gap-1">
                            <span className="badge border bg-amber-100/90 dark:bg-amber-950 border-amber-300 dark:border-amber-700 text-amber-900 dark:text-amber-300 flex items-center gap-1 font-bold w-fit">
                              <RefreshCw size={10} className="spinner text-amber-700 dark:text-amber-400" />
                              {job.stage_index > 0 ? `Stage ${job.stage_index}/6` : 'In Queue...'}
                            </span>
                            <span className="text-[11px] text-amber-800 dark:text-amber-300 truncate max-w-[180px] font-medium">
                              {job.stage_label || 'Processing document...'}
                            </span>
                          </div>
                        ) : (
                          <span className={`badge border ${cfg.bg} ${cfg.color} dark:bg-slate-800 dark:border-slate-700`}>
                            <Icon size={11} className="mr-1" />
                            {cfg.label}
                          </span>
                        )}
                      </td>

                      <td className="px-5 py-3.5">
                        {isScanning ? (
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-amber-200/60 dark:bg-amber-950 rounded-full h-1.5 overflow-hidden">
                              <div
                                className="bg-amber-600 dark:bg-amber-400 h-full rounded-full transition-all duration-500 ease-out"
                                style={{ width: `${job.progress_pct || 15}%` }}
                              />
                            </div>
                            <span className="font-mono font-bold text-amber-800 dark:text-amber-300 text-xs">
                              {job.progress_pct || 15}%
                            </span>
                          </div>
                        ) : conf != null ? (
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  conf >= 80 ? 'bg-emerald-500' : conf >= 65 ? 'bg-amber-400' : 'bg-red-400'
                                }`}
                                style={{ width: `${conf}%` }}
                              />
                            </div>
                            <span className="font-mono font-bold text-slate-700 dark:text-slate-300">{conf}%</span>
                          </div>
                        ) : (
                          <span className="text-slate-400 dark:text-slate-500">—</span>
                        )}
                      </td>

                      <td className="px-5 py-3.5">
                        {job.status === 'reviewed' ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/60 border border-emerald-200 dark:border-emerald-800 px-2.5 py-0.5 rounded-full">
                            <CheckCircle size={11} /> Ground Truth (Verified)
                          </span>
                        ) : job.status === 'partially_reviewed' ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 px-2.5 py-0.5 rounded-full">
                            <Clock size={11} /> Partially Reviewed
                          </span>
                        ) : job.status === 'failed' ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-950/60 border border-red-200 dark:border-red-800 px-2.5 py-0.5 rounded-full">
                            <XCircle size={11} /> Digitization Failed
                          </span>
                        ) : job.needs_review ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 px-2.5 py-0.5 rounded-full">
                            <AlertTriangle size={11} /> Needs Review (Low Conf)
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-950/60 border border-blue-200 dark:border-blue-800 px-2.5 py-0.5 rounded-full">
                            <FileText size={11} /> Unreviewed (AI Extracted)
                          </span>
                        )}
                      </td>

                      <td className="px-5 py-3.5 text-slate-500 dark:text-slate-400 font-mono text-[11px]">
                        {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                      </td>

                      <td className="px-5 py-3.5 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          
                          {/* Failed Retry Button */}
                          {isFailed && (
                            <button
                              disabled={isScanning}
                              onClick={(e) => retrySingle(e, job.job_id)}
                              className="px-2.5 py-1 text-[11px] font-bold rounded bg-red-600 hover:bg-red-500 text-white shadow-xs transition-all flex items-center gap-1"
                              title="Retry digitizing this invoice (adds to end of processing queue)"
                            >
                              <RotateCw size={11} className={isScanning ? 'spinner' : ''} />
                              <span>Retry</span>
                            </button>
                          )}

                          {/* Re-scan Button (for non-failed) */}
                          {!isFailed && (
                            <button
                              disabled={isScanning}
                              onClick={(e) => reprocessSingle(e, job.job_id)}
                              className={`px-2.5 py-1 text-[11px] font-medium rounded transition-all flex items-center gap-1 ${
                                isScanning
                                  ? 'bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700 cursor-not-allowed opacity-80'
                                  : 'bg-slate-100 dark:bg-slate-800 hover:bg-amber-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-amber-800 dark:hover:text-amber-400 border border-slate-200 dark:border-slate-700'
                              }`}
                              title={isScanning ? "Document is currently in processing queue" : "Re-scan this invoice with the latest AI pipeline"}
                            >
                              {isScanning ? (
                                <>
                                  <RefreshCw size={10} className="spinner text-amber-700 dark:text-amber-400" />
                                  <span>In Queue...</span>
                                </>
                              ) : (
                                <span>⚡ Re-scan</span>
                              )}
                            </button>
                          )}

                          {/* Delete / Cancel Button (Works for queued, failed, done) */}
                          <button
                            onClick={(e) => deleteSingle(e, job.job_id, job.filename)}
                            className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/60 rounded border border-transparent hover:border-red-200 dark:hover:border-red-800 transition-all ml-1"
                            title="Delete / Cancel invoice from queue"
                          >
                            <Trash2 size={13} />
                          </button>

                          <span className="text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 pl-1">
                            Review &rarr;
                          </span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* ── Pagination Footer ────────────────────────────────────────────── */}
          <div className="px-5 py-3.5 border-t border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/90 flex flex-wrap items-center justify-between gap-4">
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Showing <span className="font-semibold text-slate-800 dark:text-slate-200">{(page - 1) * pageSize + 1}</span> to{' '}
              <span className="font-semibold text-slate-800 dark:text-slate-200">{Math.min(page * pageSize, totalCount)}</span> of{' '}
              <span className="font-semibold text-slate-800 dark:text-slate-200">{totalCount}</span> invoices
            </div>

            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setPage(1)}
                disabled={page <= 1}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="First Page"
              >
                <ChevronsLeft size={14} />
              </button>

              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="Previous Page"
              >
                <ChevronLeft size={14} />
              </button>

              {/* Numbered Page Buttons */}
              <div className="flex items-center gap-1 px-1">
                {Array.from({ length: totalPages }, (_, idx) => idx + 1)
                  .filter(p => p === 1 || p === totalPages || Math.abs(p - page) <= 1)
                  .map((p, idx, arr) => {
                    const prevP = arr[idx - 1]
                    const showEllipsis = prevP && p - prevP > 1
                    return (
                      <span key={p} className="flex items-center">
                        {showEllipsis && <span className="px-1 text-slate-400 dark:text-slate-600 text-xs font-mono">…</span>}
                        <button
                          onClick={() => setPage(p)}
                          className={`min-w-[28px] h-7 px-2 text-xs font-semibold rounded transition-all ${
                            p === page
                              ? 'bg-blue-600 text-white shadow-sm'
                              : 'bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                          }`}
                        >
                          {p}
                        </button>
                      </span>
                    )
                  })}
              </div>

              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="Next Page"
              >
                <ChevronRight size={14} />
              </button>

              <button
                onClick={() => setPage(totalPages)}
                disabled={page >= totalPages}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="Last Page"
              >
                <ChevronsRight size={14} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
