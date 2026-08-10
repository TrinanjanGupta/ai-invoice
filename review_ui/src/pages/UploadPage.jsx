import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, AlertCircle, CheckCircle, Loader } from 'lucide-react'
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

export default function UploadPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState('idle') // idle | uploading | polling | done | error
  const [jobId, setJobId] = useState(null)
  const [activeStep, setActiveStep] = useState(0)
  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    if (!acceptedFiles.length) return
    const f = acceptedFiles[0]
    setFile(f)
    setError(null)
    setStatus('uploading')
    setActiveStep(0)

    try {
      const formData = new FormData()
      formData.append('file', f)
      const { data } = await axios.post('/api/invoices/upload', formData)
      setJobId(data.job_id)
      setStatus('polling')
      pollJob(data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed')
      setStatus('error')
    }
  }, [])

  const pollJob = (jid) => {
    let step = 0
    const interval = setInterval(async () => {
      // Advance steps visually
      if (step < STEPS.length - 1) {
        step++
        setActiveStep(step)
      }
      try {
        const { data } = await axios.get(`/api/invoices/${jid}`)
        if (data.status === 'done' || data.status === 'reviewed') {
          clearInterval(interval)
          setActiveStep(STEPS.length)
          setStatus('done')
          toast.success('Invoice digitised successfully!')
          setTimeout(() => navigate(`/invoices/${jid}`), 800)
        } else if (data.status === 'failed') {
          clearInterval(interval)
          setStatus('error')
          setError(data.error_message || 'Processing failed')
          toast.error('Processing failed')
        }
      } catch {
        // keep polling
      }
    }, 2200)
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
    disabled: status !== 'idle',
  })

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Invoice</h1>
        <p className="text-gray-500 mt-1 text-sm">
          Upload any invoice — PDF, JPG, PNG, TIFF, or WEBP. We'll extract all fields automatically.
        </p>
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`card p-10 flex flex-col items-center justify-center text-center cursor-pointer
                    border-2 border-dashed transition-all duration-200 mb-8
                    ${isDragActive ? 'dropzone-active border-brand-500' : 'border-gray-200 hover:border-brand-300 hover:bg-brand-50/30'}
                    ${status !== 'idle' ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="w-14 h-14 rounded-2xl bg-brand-50 flex items-center justify-center mb-4">
          <Upload size={24} className="text-brand-600" />
        </div>
        {file ? (
          <>
            <p className="font-semibold text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-400 mt-1">{(file.size / 1024).toFixed(0)} KB</p>
          </>
        ) : (
          <>
            <p className="font-semibold text-gray-700 text-lg">
              {isDragActive ? 'Drop it here' : 'Drag & drop your invoice'}
            </p>
            <p className="text-sm text-gray-400 mt-1">or click to browse</p>
            <p className="text-xs text-gray-300 mt-3">PDF · JPG · PNG · TIFF · WEBP · max 50 MB</p>
          </>
        )}
      </div>

      {/* Pipeline progress */}
      {(status === 'uploading' || status === 'polling' || status === 'done') && (
        <div className="card p-6 fade-in">
          <h2 className="text-sm font-semibold text-gray-700 mb-5">Processing Pipeline</h2>
          <div className="space-y-3">
            {STEPS.map((step, i) => {
              const isDone = i < activeStep
              const isActive = i === activeStep && status !== 'done'
              const isComplete = status === 'done'
              return (
                <div key={step.id} className="flex items-start gap-3">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5
                                   transition-all duration-300
                                   ${isDone || isComplete ? 'bg-green-500' : isActive ? 'bg-brand-600' : 'bg-gray-100'}`}>
                    {(isDone || isComplete) ? (
                      <CheckCircle size={14} className="text-white" />
                    ) : isActive ? (
                      <Loader size={12} className="text-white spinner" />
                    ) : (
                      <span className="text-xs text-gray-400 font-medium">{step.id}</span>
                    )}
                  </div>
                  <div>
                    <p className={`text-sm font-medium transition-colors
                                   ${isDone || isComplete ? 'text-green-700' : isActive ? 'text-brand-700' : 'text-gray-400'}`}>
                      {step.label}
                    </p>
                    <p className="text-xs text-gray-400">{step.desc}</p>
                  </div>
                </div>
              )
            })}
          </div>
          {status === 'done' && (
            <p className="text-center text-sm text-green-600 font-medium mt-5">
              ✓ Redirecting to review...
            </p>
          )}
        </div>
      )}

      {/* Error */}
      {status === 'error' && (
        <div className="card p-5 border-red-200 bg-red-50 fade-in flex items-start gap-3">
          <AlertCircle size={18} className="text-red-500 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-semibold text-red-800">Processing failed</p>
            <p className="text-sm text-red-600 mt-1">{error}</p>
            <button
              onClick={() => { setStatus('idle'); setFile(null); setError(null); }}
              className="text-xs text-red-700 underline mt-2"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Info cards */}
      {status === 'idle' && (
        <div className="grid grid-cols-3 gap-4 mt-8 fade-in">
          {[
            { icon: '🔍', title: 'Smart Detection', desc: 'YOLOv8 finds invoice regions automatically' },
            { icon: '🧠', title: 'AI Understanding', desc: 'LayoutLMv3 maps fields with spatial context' },
            { icon: '✅', title: 'Math Validation', desc: 'Every total, tax, and line item verified' },
          ].map(c => (
            <div key={c.title} className="card p-4">
              <div className="text-2xl mb-2">{c.icon}</div>
              <p className="text-sm font-semibold text-gray-800">{c.title}</p>
              <p className="text-xs text-gray-400 mt-1">{c.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
