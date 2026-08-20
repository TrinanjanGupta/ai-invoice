import { useCallback, useState, useRef, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import {
  Upload, FileText, AlertCircle, CheckCircle, Loader, RefreshCw,
  Layers, ArrowRight, ExternalLink, Check, Plus, RotateCw, Trash2
} from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'

const ACCEPTED = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/tiff': ['.tiff', '.tif'],
  'image/webp': ['.webp'],
}

const STEPS = [
  { id: 1, label: 'Pre-processing', desc: 'Deskew, denoise, normalise DPI' },
  { id: 2, label: 'Region Detection', desc: 'YOLOv8 identifies invoice zones' },
  { id: 3, label: 'OCR Extraction', desc: 'PaddleOCR reads text per region' },
  { id: 4, label: 'AI Understanding', desc: 'LayoutLMv3 maps fields' },
  { id: 5, label: 'LLM Fallback', desc: 'Ollama fills low-confidence fields' },
  { id: 6, label: 'Validation', desc: 'Rules engine checks math & formats' },
]

function formatFileSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export default function UploadPage() {
  const navigate = useNavigate()

  // Mode: 'idle' | 'single' | 'batch'
  const [mode, setMode] = useState('idle')

  // Single Upload State
  const [singleStatus, setSingleStatus] = useState('idle') // idle | uploading | polling | done | error
  const [singleFile, setSingleFile] = useState(null)
  const [singleJobId, setSingleJobId] = useState(null)
  const [singleActiveStep, setSingleActiveStep] = useState(0)
  const [singleProgressPct, setSingleProgressPct] = useState(10)
  const [singleStageLabel, setSingleStageLabel] = useState('Initializing...')
  const [singleError, setSingleError] = useState(null)
  const singleTargetRef = useRef(10)
  const singleEsRef = useRef(null)

  // Batch Upload State
  const [batchJobs, setBatchJobs] = useState([])
  const [batchUploading, setBatchUploading] = useState(false)
  const [recentJobs, setRecentJobs] = useState([])
  const [dbTotalCount, setDbTotalCount] = useState(0)
  const globalEsRef = useRef(null)

  // Fetch recent jobs and total count from database
  useEffect(() => {
    axios.get('/api/invoices?limit=5').then(({ data }) => {
      setRecentJobs(data.jobs || [])
      setDbTotalCount(data.total || 0)
    }).catch(() => {})
  }, [])

  // Smooth micro-increment ticker for single-file mode
  useEffect(() => {
    if (singleStatus !== 'polling' && singleStatus !== 'uploading') return
    const ticker = setInterval(() => {
      setSingleProgressPct((prev) => {
        const target = singleTargetRef.current
        if (prev < target) return Math.min(target, prev + 1)
        if (prev < 98 && prev < target + 12) return prev + 1
        return prev
      })
    }, 250)
    return () => clearInterval(ticker)
  }, [singleStatus])

  // Cleanup all SSE connections on unmount
  useEffect(() => {
    return () => {
      if (singleEsRef.current) singleEsRef.current.close()
      if (globalEsRef.current) globalEsRef.current.close()
    }
  }, [])

  // ── Global SSE Stream for Batch Mode ─────────────────────────────────────────
  const startGlobalSSE = useCallback(() => {
    if (globalEsRef.current) return

    const es = new EventSource('/api/stream/jobs')
    globalEsRef.current = es

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        setBatchJobs((prev) => {
          const next = prev.map((job) => {
            if (job.job_id !== data.job_id) return job

            const isFinished = ['done', 'reviewed', 'partially_reviewed', 'failed'].includes(data.status)
            return {
              ...job,
              status: data.status,
              stage: data.stage || job.stage,
              stage_index: data.stage_index ?? job.stage_index,
              stage_label: data.stage_label || job.stage_label,
              progress_pct: isFinished ? 100 : Math.max(job.progress_pct || 15, data.progress_pct || 15),
              overall_confidence: data.overall_confidence ?? job.overall_confidence,
              needs_review: data.needs_review ?? job.needs_review,
              error_message: data.error_message || job.error_message,
            }
          })
          try {
            sessionStorage.setItem('active_batch_jobs', JSON.stringify(next))
          } catch {}
          return next
        })
      } catch {
        /* ignore parse errors */
      }
    }

    es.onerror = () => {
      // EventSource automatically reconnects on network drop
    }
  }, [])

  // Restore active batch session from sessionStorage on reload
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem('active_batch_jobs')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMode('batch')
          setBatchJobs(parsed)
          startGlobalSSE()
          // Refresh latest statuses from API
          Promise.all(parsed.map(j =>
            axios.get(`/api/invoices/${j.job_id}`)
              .then(r => r.data)
              .catch(() => null)
          )).then(results => {
            setBatchJobs(prev => {
              const updated = prev.map(job => {
                const srv = results.find(r => r && r.job_id === job.job_id)
                if (!srv) return job
                const isFinished = ['done', 'reviewed', 'partially_reviewed', 'failed'].includes(srv.status)
                return {
                  ...job,
                  status: srv.status,
                  stage: srv.stage || job.stage,
                  stage_index: srv.stage_index ?? job.stage_index,
                  stage_label: srv.stage_label || job.stage_label,
                  progress_pct: isFinished ? 100 : Math.max(job.progress_pct || 15, srv.progress_pct || 15),
                  error_message: srv.error_message,
                  overall_confidence: srv.overall_confidence,
                }
              })
              try {
                sessionStorage.setItem('active_batch_jobs', JSON.stringify(updated))
              } catch {}
              return updated
            })
          })
        }
      }
    } catch {}
  }, [startGlobalSSE])

  // ── Handle Single File Upload ───────────────────────────────────────────────
  const handleSingleUpload = async (f) => {
    setMode('single')
    setSingleFile(f)
    setSingleError(null)
    setSingleStatus('uploading')
    setSingleActiveStep(0)
    setSingleProgressPct(10)
    singleTargetRef.current = 15
    setSingleStageLabel('Pre-processing: Initializing document...')

    try {
      const formData = new FormData()
      formData.append('file', f)
      const { data } = await axios.post('/api/invoices/upload', formData)
      setSingleJobId(data.job_id)
      setSingleStatus('polling')

      if (singleEsRef.current) singleEsRef.current.close()
      const es = new EventSource(`/api/invoices/${data.job_id}/stream`)
      singleEsRef.current = es

      es.onmessage = (e) => {
        const payload = JSON.parse(e.data)
        if (payload.stage_index != null && payload.stage_index > 0) {
          setSingleActiveStep(Math.min(STEPS.length - 1, payload.stage_index - 1))
        }
        if (payload.progress_pct != null && payload.progress_pct > singleTargetRef.current) {
          singleTargetRef.current = payload.progress_pct
        }
        if (payload.stage_label) {
          setSingleStageLabel(payload.stage_label)
        }

        if (payload.status === 'done' || payload.status === 'reviewed' || payload.status === 'partially_reviewed') {
          es.close()
          setSingleActiveStep(STEPS.length)
          singleTargetRef.current = 100
          setSingleProgressPct(100)
          setSingleStageLabel('Digitization Complete')
          setSingleStatus('done')
          toast.success('Invoice digitised successfully!')
          setTimeout(() => navigate(`/invoices/${data.job_id}`), 800)
        } else if (payload.status === 'failed') {
          es.close()
          setSingleStatus('error')
          setSingleError(payload.error_message || 'Processing failed')
          toast.error('Processing failed')
        }
      }
    } catch (err) {
      setSingleError(err.response?.data?.detail || 'Upload failed')
      setSingleStatus('error')
    }
  }

  // ── Handle Batch Upload ─────────────────────────────────────────────────────
  const handleBatchUpload = async (acceptedFiles) => {
    setMode('batch')
    setBatchUploading(true)

    // Pre-populate queue items
    const initialJobs = acceptedFiles.map((f, i) => ({
      job_id: `temp_${i}_${Date.now()}`,
      filename: f.name,
      fileSize: f.size,
      status: 'uploading',
      stage: 'preprocessing',
      stage_index: 1,
      stage_label: 'Uploading file...',
      progress_pct: 10,
    }))
    setBatchJobs(initialJobs)

    try {
      const formData = new FormData()
      acceptedFiles.forEach((f) => formData.append('files', f))

      const { data } = await axios.post('/api/invoices/upload-batch', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      // Update with server assigned job IDs
      const mappedJobs = data.jobs.map((srvJob, idx) => {
        const origFile = acceptedFiles[idx]
        return {
          job_id: srvJob.job_id,
          filename: srvJob.filename,
          fileSize: origFile ? origFile.size : 0,
          status: srvJob.status || 'processing',
          stage: 'preprocessing',
          stage_index: 1,
          stage_label: 'Queued in AI Pipeline...',
          progress_pct: 15,
        }
      })

      setBatchJobs(mappedJobs)
      setBatchUploading(false)
      toast.success(`Queued ${data.total_queued} invoices for AI decoding!`)

      // Start global stream to receive live progress
      startGlobalSSE()
    } catch (err) {
      setBatchUploading(false)
      toast.error('Batch upload failed: ' + (err.response?.data?.detail || err.message))
    }
  }

  // ── Dropzone Trigger ────────────────────────────────────────────────────────
  const onDrop = useCallback(
    async (acceptedFiles) => {
      if (!acceptedFiles.length) return

      if (acceptedFiles.length === 1) {
        handleSingleUpload(acceptedFiles[0])
      } else {
        handleBatchUpload(acceptedFiles)
      }
    },
    [startGlobalSSE]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 50,
    maxSize: 50 * 1024 * 1024,
    disabled: singleStatus === 'polling' || singleStatus === 'uploading' || batchUploading,
  })

  const resetUpload = () => {
    setMode('idle')
    setSingleStatus('idle')
    setSingleFile(null)
    setSingleError(null)
    setBatchJobs([])
    try {
      sessionStorage.removeItem('active_batch_jobs')
    } catch {}
    if (singleEsRef.current) singleEsRef.current.close()
    if (globalEsRef.current) globalEsRef.current.close()
    globalEsRef.current = null
  }

  // Calculate Batch Metrics
  const batchTotal = batchJobs.length
  const batchCompleted = batchJobs.filter((j) => ['done', 'reviewed', 'partially_reviewed'].includes(j.status)).length
  const batchProcessing = batchJobs.filter((j) => j.status === 'processing' || j.status === 'uploading').length
  const batchFailed = batchJobs.filter((j) => j.status === 'failed').length
  const batchOverallPct = batchTotal > 0 ? Math.round((batchCompleted / batchTotal) * 100) : 0

  const retryBatchJob = async (jobId) => {
    try {
      await axios.post(`/api/invoices/${jobId}/retry`)
      toast.success('Invoice re-queued for processing!')
      setBatchJobs(prev => prev.map(j =>
        j.job_id === jobId
          ? { ...j, status: 'processing', stage: 'preprocessing', stage_index: 1, stage_label: 'Queued in AI Pipeline...', progress_pct: 10, error_message: null }
          : j
      ))
    } catch (err) {
      toast.error('Retry failed: ' + (err.response?.data?.detail || err.message))
    }
  }

  const deleteBatchJob = async (jobId, filename) => {
    if (!window.confirm(`Remove "${filename || 'this invoice'}" from the queue?`)) return
    try {
      if (jobId && !jobId.startsWith('temp_')) {
        await axios.delete(`/api/invoices/${jobId}`)
      }
      setBatchJobs(prev => prev.filter(j => j.job_id !== jobId))
      toast.success('Invoice removed from queue')
    } catch (err) {
      toast.error('Failed to remove invoice: ' + (err.response?.data?.detail || err.message))
    }
  }

  return (
    <div className="p-4 md:p-8 max-w-5xl mx-auto space-y-6">
      
      {/* ── Top Header ───────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2.5">
            <Layers className="text-blue-600" size={24} /> Upload Invoices
          </h1>
          <p className="text-slate-500 mt-1 text-xs md:text-sm">
            Drag & drop single or multiple invoices (PDF, JPG, PNG, TIFF, WEBP). Our AI engine decodes them in parallel.
          </p>
        </div>

        {mode !== 'idle' && (
          <button
            onClick={resetUpload}
            className="btn-secondary text-xs flex items-center gap-1.5"
            title="Upload more invoices"
          >
            <Plus size={13} /> New Upload
          </button>
        )}
      </div>

      {/* ── Dropzone Area (Shown when idle or can add more) ───────────────────── */}
      {mode === 'idle' && (
        <div
          {...getRootProps()}
          className={`card p-10 md:p-12 flex flex-col items-center justify-center text-center cursor-pointer
                      border-2 border-dashed transition-all duration-200 bg-white dark:bg-slate-900
                      ${isDragActive ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/30 scale-[1.01]' : 'border-slate-300 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-50/60 dark:hover:bg-slate-800/40'}`}
        >
          <input {...getInputProps()} />
          <div className="w-16 h-16 rounded-2xl bg-blue-50 dark:bg-blue-950/50 border border-blue-100 dark:border-blue-900/60 flex items-center justify-center mb-4 text-blue-600 dark:text-blue-400 shadow-xs">
            <Upload size={28} />
          </div>

          <p className="font-bold text-slate-800 dark:text-slate-100 text-lg">
            {isDragActive ? 'Drop your invoices here' : 'Drag & drop invoice files here'}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
            or <span className="text-blue-600 dark:text-blue-400 font-semibold underline">browse from your computer</span> &bull; Multiple files supported
          </p>
          <div className="mt-4 flex items-center gap-2 text-[11px] text-slate-400 dark:text-slate-400 font-medium bg-slate-100/80 dark:bg-slate-800/80 px-3 py-1 rounded-full border border-slate-200 dark:border-slate-700">
            <span>PDF</span> &bull; <span>JPG / PNG</span> &bull; <span>TIFF</span> &bull; <span>WEBP</span> &bull; <span>Up to 50 files (max 50 MB each)</span>
          </div>
        </div>
      )}

      {/* ── Batch Upload Dashboard ───────────────────────────────────────────── */}
      {mode === 'batch' && (
        <div className="space-y-5">
          
          {/* Overall Batch Progress Card */}
          <div className="card p-5 md:p-6 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <span>Batch Digitization Queue</span>
                  <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                    {batchCompleted} / {batchTotal} Decoded
                  </span>
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Invoices are processed by the AI pipeline and added to your database automatically.
                </p>
              </div>

              <div className="flex items-center gap-2.5 text-xs">
                {batchProcessing > 0 && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-50 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800 font-medium">
                    <RefreshCw size={11} className="spinner text-amber-600 dark:text-amber-400" />
                    {batchProcessing} In Progress
                  </span>
                )}
                {batchCompleted > 0 && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 font-medium">
                    <CheckCircle size={11} />
                    {batchCompleted} Ready
                  </span>
                )}
                {batchFailed > 0 && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-50 dark:bg-red-950/60 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800 font-medium">
                    <AlertCircle size={11} />
                    {batchFailed} Failed
                  </span>
                )}
              </div>
            </div>

            {/* Overall Progress Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-600 dark:text-slate-400">Overall Batch Progress</span>
                <span className="font-mono text-blue-700 dark:text-blue-400 font-bold">{batchOverallPct}%</span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-blue-600 dark:bg-blue-500 h-full rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${batchOverallPct}%` }}
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-800">
              <button
                onClick={() => navigate('/invoices')}
                className="btn-primary text-xs flex items-center gap-1.5"
              >
                <span>View In All Invoices Table</span>
                <ArrowRight size={13} />
              </button>

              <button
                onClick={resetUpload}
                className="btn-secondary text-xs flex items-center gap-1 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <Plus size={13} /> Upload More Files
              </button>
            </div>
          </div>

          {/* File Queue List */}
          <div className="card overflow-hidden border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
            <div className="px-5 py-3.5 bg-slate-50/90 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-800 font-semibold text-xs text-slate-600 dark:text-slate-300 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-6 text-center text-slate-400">#</span>
                <span>Invoice Documents ({batchTotal})</span>
              </div>
              <span>Status & Stage</span>
            </div>

            <div className="divide-y divide-slate-100 dark:divide-slate-800 max-h-[520px] overflow-y-auto">
              {batchJobs.map((job, idx) => {
                const isDone = ['done', 'reviewed', 'partially_reviewed'].includes(job.status)
                const isFailed = job.status === 'failed'
                const isProcessing = job.status === 'processing' || job.status === 'uploading'

                return (
                  <div
                    key={job.job_id || idx}
                    className={`p-4 flex flex-wrap items-center justify-between gap-4 transition-colors ${
                      isDone
                        ? 'bg-white dark:bg-slate-900 hover:bg-emerald-50/20 dark:hover:bg-slate-800/40'
                        : isFailed
                        ? 'bg-red-50/30 dark:bg-red-950/20'
                        : 'bg-slate-50/40 dark:bg-slate-900/60 hover:bg-slate-50/80 dark:hover:bg-slate-800/30'
                    }`}
                  >
                    {/* File info with S.No. */}
                    <div className="flex items-center gap-3 min-w-[240px] max-w-sm">
                      <span className="font-mono text-slate-400 dark:text-slate-500 text-xs font-semibold w-6 text-center select-none shrink-0">
                        {idx + 1}
                      </span>
                      <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border ${
                        isDone
                          ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
                          : isFailed
                          ? 'bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800'
                          : 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800'
                      }`}>
                        {isDone ? (
                          <CheckCircle size={16} />
                        ) : isFailed ? (
                          <AlertCircle size={16} />
                        ) : (
                          <RefreshCw size={16} className="spinner text-blue-600 dark:text-blue-400" />
                        )}
                      </div>
                      <div className="min-w-0">
                        <p className="font-semibold text-slate-800 dark:text-slate-200 text-xs truncate" title={job.filename}>
                          {job.filename}
                        </p>
                        <p className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-0.5">
                          {formatFileSize(job.fileSize)} {job.job_id && !job.job_id.startsWith('temp_') && `• ${job.job_id.slice(0, 8)}…`}
                        </p>
                      </div>
                    </div>

                    {/* Stage & Progress Bar */}
                    <div className="flex-1 min-w-[200px] max-w-md">
                      <div className="flex items-center justify-between text-[11px] mb-1">
                        <span className={`font-medium truncate ${
                          isDone
                            ? 'text-emerald-700 dark:text-emerald-400'
                            : isFailed
                            ? 'text-red-600 dark:text-red-400'
                            : 'text-slate-700 dark:text-slate-300'
                        }`}>
                          {isDone ? 'AI Digitization Complete' : isFailed ? (job.error_message || 'Processing Error') : (job.stage_label || 'Processing...')}
                        </span>
                        <span className="font-mono font-bold text-slate-500 dark:text-slate-400 shrink-0 ml-2">
                          {job.progress_pct || 15}%
                        </span>
                      </div>

                      <div className="w-full bg-slate-200/70 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ease-out ${
                            isDone ? 'bg-emerald-500' : isFailed ? 'bg-red-500' : 'bg-blue-600 dark:bg-blue-500'
                          }`}
                          style={{ width: `${job.progress_pct || 15}%` }}
                        />
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center justify-end shrink-0 gap-2 min-w-[140px]">
                      {isDone ? (
                        <button
                          onClick={() => navigate(`/invoices/${job.job_id}`)}
                          className="btn-success text-xs py-1.5 px-3 flex items-center gap-1 shadow-xs"
                        >
                          <span>Review</span>
                          <ExternalLink size={12} />
                        </button>
                      ) : isFailed ? (
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => retryBatchJob(job.job_id)}
                            className="px-2.5 py-1 text-xs font-bold rounded bg-red-600 hover:bg-red-500 text-white flex items-center gap-1 shadow-xs transition-all"
                            title="Retry digitizing this failed invoice (adds to end of queue)"
                          >
                            <RotateCw size={11} />
                            <span>Retry</span>
                          </button>
                          <button
                            onClick={() => deleteBatchJob(job.job_id, job.filename)}
                            className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40 rounded"
                            title="Delete from queue"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-medium text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-2.5 py-1 rounded-md flex items-center gap-1.5">
                            <RefreshCw size={11} className="spinner text-blue-600 dark:text-blue-400" />
                            <span>In Queue</span>
                          </span>
                          <button
                            onClick={() => deleteBatchJob(job.job_id, job.filename)}
                            className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40 rounded"
                            title="Cancel / Delete from queue"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── Single File Pipeline Progress ────────────────────────────────────── */}
      {mode === 'single' && (singleStatus === 'uploading' || singleStatus === 'polling' || singleStatus === 'done') && (
        <div className="card p-6 fade-in shadow-sm border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-sm font-bold text-slate-900 dark:text-white">AI Digitization Pipeline</h2>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">{singleFile?.name} ({formatFileSize(singleFile?.size)})</p>
            </div>
            <span className="text-xs font-mono font-bold text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-950/60 px-2.5 py-0.5 rounded-full border border-blue-100 dark:border-blue-800">
              {singleProgressPct}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 mb-5 overflow-hidden">
            <div
              className="bg-blue-600 dark:bg-blue-500 h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${singleProgressPct}%` }}
            />
          </div>

          <div className="space-y-3.5">
            {STEPS.map((step, i) => {
              const isDone = i < singleActiveStep || singleStatus === 'done'
              const isActive = i === singleActiveStep && singleStatus !== 'done'
              return (
                <div key={step.id} className="flex items-start gap-3">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5
                                   transition-all duration-300
                                   ${isDone ? 'bg-emerald-500 shadow-xs' : isActive ? 'bg-blue-600 ring-4 ring-blue-100 dark:ring-blue-950' : 'bg-slate-100 dark:bg-slate-800'}`}>
                    {isDone ? (
                      <CheckCircle size={14} className="text-white" />
                    ) : isActive ? (
                      <Loader size={12} className="text-white spinner" />
                    ) : (
                      <span className="text-xs text-slate-400 dark:text-slate-500 font-medium">{step.id}</span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className={`text-sm font-medium transition-colors
                                     ${isDone ? 'text-emerald-800 dark:text-emerald-400' : isActive ? 'text-blue-700 dark:text-blue-300 font-semibold' : 'text-slate-400 dark:text-slate-500'}`}>
                        {step.label}
                      </p>
                      {isActive && (
                        <span className="text-[11px] font-medium text-blue-600 dark:text-blue-400 animate-pulse hidden sm:inline">
                          Running...
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      {isActive && singleStageLabel ? singleStageLabel : step.desc}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>

          {singleStatus === 'done' && (
            <p className="text-center text-sm text-emerald-600 dark:text-emerald-400 font-medium mt-5 animate-fade-in">
              ✓ Redirecting to review workspace...
            </p>
          )}
        </div>
      )}

      {/* Single Mode Error */}
      {mode === 'single' && singleStatus === 'error' && (
        <div className="card p-5 border-red-200 dark:border-red-900/60 bg-red-50 dark:bg-red-950/30 fade-in flex items-start gap-3">
          <AlertCircle size={18} className="text-red-500 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-semibold text-red-800 dark:text-red-300">Processing failed</p>
            <p className="text-sm text-red-600 dark:text-red-400 mt-1">{singleError}</p>
            <button
              onClick={resetUpload}
              className="text-xs text-red-700 dark:text-red-400 underline mt-2"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* ── Recent Invoices & Database Queue Banner (Shown when idle) ─────────── */}
      {mode === 'idle' && dbTotalCount > 0 && (
        <div className="card p-5 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs space-y-3.5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <div>
                <span className="text-xs font-bold text-slate-800 dark:text-slate-100 uppercase tracking-wider block">
                  Invoice Database & Live Queue ({dbTotalCount} Invoices)
                </span>
                <p className="text-[11px] text-slate-400 dark:text-slate-400 mt-0.5">
                  All previously uploaded and currently processing invoices are stored securely in your database.
                </p>
              </div>
            </div>
            <button
              onClick={() => navigate('/invoices')}
              className="btn-primary text-xs py-1.5 px-3.5 flex items-center gap-1.5 shadow-xs shrink-0"
            >
              <span>View All Invoices Table</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="divide-y divide-slate-100 dark:divide-slate-800 border border-slate-100 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-50/30 dark:bg-slate-800/30">
            {recentJobs.slice(0, 5).map((j, i) => (
              <div
                key={j.job_id}
                onClick={() => navigate(`/invoices/${j.job_id}`)}
                className="p-3 flex items-center justify-between text-xs hover:bg-blue-50/50 dark:hover:bg-slate-800/60 cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="font-mono text-slate-400 dark:text-slate-500 text-[11px] font-semibold w-5 text-center shrink-0">
                    {i + 1}
                  </span>
                  <FileText size={14} className="text-slate-400 dark:text-slate-400 shrink-0" />
                  <span className="font-semibold text-slate-800 dark:text-slate-200 truncate max-w-[280px]">
                    {j.filename}
                  </span>
                  <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono hidden sm:inline">
                    ({j.job_id.slice(0, 8)}…)
                  </span>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className={`badge text-[10px] px-2 py-0.5 border ${
                    j.status === 'reviewed' ? 'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800' :
                    j.status === 'processing' ? 'bg-amber-50 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border-amber-200 dark:border-amber-800' :
                    j.status === 'failed' ? 'bg-red-50 dark:bg-red-950/60 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800' :
                    'bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800'
                  }`}>
                    {j.status === 'reviewed' ? 'Verified' : j.status === 'processing' ? 'Processing' : j.status === 'failed' ? 'Failed' : 'AI Extracted'}
                  </span>
                  <span className="text-blue-600 dark:text-blue-400 font-semibold text-[11px] hover:underline">
                    Review &rarr;
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Feature Cards (Idle state) ────────────────────────────────────────── */}
      {mode === 'idle' && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1 fade-in">
          {[
            { icon: '🔍', title: 'YOLOv8 Zones', desc: 'Auto-detects invoice tables, totals, headers, and vendor boxes' },
            { icon: '🧠', title: 'LayoutLMv3 AI', desc: 'Spatial intelligence maps key-value fields with high accuracy' },
            { icon: '⚡', title: 'Batch Processing', desc: 'Drop multiple invoices and review them as they complete decoding' },
          ].map((c) => (
            <div key={c.title} className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
              <div className="text-2xl mb-2">{c.icon}</div>
              <p className="text-xs font-bold text-slate-800 dark:text-slate-100">{c.title}</p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
