import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  Save, Download, Eye, ArrowLeft, AlertTriangle,
  CheckCircle, RefreshCw, Trash2, Plus,
  FileText, Building, Landmark, ShoppingBag, MessageSquare,
  ChevronRight, ChevronLeft, Copy, Check, Sparkles, LayoutGrid,
  Columns, Maximize2, ZoomIn, ZoomOut, RotateCcw, RotateCw, ExternalLink,
  Play, Cpu, X
} from 'lucide-react'

// ── Number to Words Helper (Indian Numbering System) ──────────────────────────

function numberToWords(num) {
  if (!num || isNaN(num) || num <= 0) return ''
  const a = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
  const b = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']

  const inWords = (n) => {
    let str = ''
    if (n > 99) {
      str += a[Math.floor(n / 100)] + ' Hundred '
      n %= 100
    }
    if (n > 19) {
      str += b[Math.floor(n / 10)] + (n % 10 ? ' ' + a[n % 10] : '')
    } else if (n > 0) {
      str += a[n]
    }
    return str.trim()
  }

  let whole = Math.floor(num)
  let paise = Math.round((num - whole) * 100)
  let result = ''

  const crore = Math.floor(whole / 10000000)
  whole %= 10000000
  const lakh = Math.floor(whole / 100000)
  whole %= 100000
  const thousand = Math.floor(whole / 1000)
  whole %= 1000
  const hundred = whole

  if (crore) result += inWords(crore) + ' Crore '
  if (lakh) result += inWords(lakh) + ' Lakh '
  if (thousand) result += inWords(thousand) + ' Thousand '
  if (hundred) result += inWords(hundred) + ' '

  result = result.trim()
  if (result) result = 'Rupees ' + result
  if (paise > 0) {
    result += (result ? ' and ' : 'Rupees ') + inWords(paise) + ' Paise'
  }
  return result ? result + ' Only' : ''
}

// ── Standard Initial State Builder ──────────────────────────────────────────

function getEmptyForm() {
  return {
    meta: {
      invoiceNo: '',
      category: '',
      subcategory: '',
      date: '',
      placeOfSupply: '',
      dueDate: '',
    },
    client: {
      slsCode: '',
      name: '',
      addressLine1: '',
      addressLine2: '',
      phone: '',
      gstin: '',
    },
    company: {
      name: '',
      addressLine1: '',
      addressLine2: '',
      email: '',
      phone: '',
      gstin: '',
      pan: '',
    },
    bankDetails: {
      ifsc: '',
      branchName: '',
      bankName: '',
      accountName: '',
      accountNumber: '',
      confirmAccountNumber: '',
    },
    items: [
      {
        description: '',
        hsnSac: '',
        quantity: 1,
        unit: 'NOS',
        rate: 0,
        discount: 0,
        taxableValue: 0,
        cgstRate: 0,
        cgstAmount: 0,
        sgstRate: 0,
        sgstAmount: 0,
        igstRate: 0,
        igstAmount: 0,
      }
    ],
    totals: {
      taxableAmount: 0,
      totalDiscount: 0,
      netTaxable: 0,
      globalDiscount: 0,
      totalCgst: 0,
      totalSgst: 0,
      totalIgst: 0,
      roundOff: 0,
      grandTotal: 0,
      amountInWords: '',
    },
    remarks: '',
    certifiedRemarks: [
      'Certified that the particulars given above are true and correct.',
      'The applicable GST and other charges are accurately calculated.',
    ],
  }
}

// ── Wizard Steps Definition ──────────────────────────────────────────────────

const WIZARD_STEPS = [
  { id: 1, title: 'Bill To & Invoice Details', subtitle: 'Client information & invoice header metadata', icon: FileText },
  { id: 2, title: 'Biller & Bank Details', subtitle: 'Vendor profile, tax numbers & bank account info', icon: Building },
  { id: 3, title: 'Items & Totals', subtitle: 'Line items breakdown, taxes & financial totals', icon: ShoppingBag },
  { id: 4, title: 'Remarks & Certifications', subtitle: 'Notes, declarations & certified compliance', icon: MessageSquare },
  { id: 5, title: 'Review & Submit', subtitle: 'Validate, preview & export to Invoice Builder', icon: CheckCircle },
]

export default function ReviewPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()

  const [job, setJob] = useState(null)
  const [formData, setFormData] = useState(getEmptyForm())
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [copied, setCopied] = useState(false)

  // View modes
  const [isWizardMode, setIsWizardMode] = useState(true)
  const [currentStep, setCurrentStep] = useState(1)
  const [activeTab, setActiveTab] = useState('split') // 'edit' | 'split' | 'preview'
  const [docSource, setDocSource] = useState('original') // 'original' | 'rendered'
  const [previewHtml, setPreviewHtml] = useState('')

  // Document Viewer state
  const [docPage, setDocPage] = useState(0)
  const [docTotalPages, setDocTotalPages] = useState(1)
  const [zoomLevel, setZoomLevel] = useState(100)
  const [rotation, setRotation] = useState(0)

  // Training state
  const [showTrainModal, setShowTrainModal] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [isRescanning, setIsRescanning] = useState(false)
  const [rescanProgress, setRescanProgress] = useState({
    stage: 'preprocessing',
    stageIndex: 1,
    stageLabel: 'Pre-processing: Initializing document...',
    progressPct: 15,
  })
  const rescanTargetRef = useRef(15)
  const rescanEsRef = useRef(null)

  // Smooth rescan progress ticker (250ms, independent of SSE cadence)
  useEffect(() => {
    if (!isRescanning) return
    const ticker = setInterval(() => {
      setRescanProgress(prev => {
        const target = rescanTargetRef.current
        if (prev.progressPct < target) {
          return { ...prev, progressPct: Math.min(target, prev.progressPct + 1) }
        }
        // Creep forward up to 12% beyond last known target (covers slow OCR/LLM steps)
        if (prev.progressPct < 98 && prev.progressPct < (target + 12)) {
          return { ...prev, progressPct: prev.progressPct + 1 }
        }
        return prev
      })
    }, 300)
    return () => clearInterval(ticker)
  }, [isRescanning])

  // Cleanup rescan EventSource on unmount
  useEffect(() => {
    return () => { if (rescanEsRef.current) rescanEsRef.current.close() }
  }, [])


  // ── Auto-Calculate Totals ──────────────────────────────────────────────────

  const recalculate = useCallback((items, roundOff = 0) => {
    let taxableAmount = 0
    let totalDiscount = 0
    let netTaxable = 0
    let totalCgst = 0
    let totalSgst = 0
    let totalIgst = 0

    const updatedItems = items.map(it => {
      const q = parseFloat(it.quantity) || 0
      const r = parseFloat(it.rate) || 0
      const d = parseFloat(it.discount) || 0
      const cgstR = parseFloat(it.cgstRate) || 0
      const sgstR = parseFloat(it.sgstRate) || 0
      const igstR = parseFloat(it.igstRate) || 0

      const gross = q * r
      const taxable = Math.max(0, gross - d)
      const cgstA = (taxable * cgstR) / 100
      const sgstA = (taxable * sgstR) / 100
      const igstA = (taxable * igstR) / 100

      taxableAmount += gross
      totalDiscount += d
      netTaxable += taxable
      totalCgst += cgstA
      totalSgst += sgstA
      totalIgst += igstA

      return {
        ...it,
        taxableValue: Math.round(taxable * 100) / 100,
        cgstAmount: Math.round(cgstA * 100) / 100,
        sgstAmount: Math.round(sgstA * 100) / 100,
        igstAmount: Math.round(igstA * 100) / 100,
      }
    })

    const rawGrand = netTaxable + totalCgst + totalSgst + totalIgst + (parseFloat(roundOff) || 0)
    const grandTotal = Math.round(rawGrand * 100) / 100
    const inWords = numberToWords(grandTotal)

    return {
      items: updatedItems,
      totals: {
        taxableAmount: Math.round(taxableAmount * 100) / 100,
        totalDiscount: Math.round(totalDiscount * 100) / 100,
        netTaxable: Math.round(netTaxable * 100) / 100,
        globalDiscount: 0,
        totalCgst: Math.round(totalCgst * 100) / 100,
        totalSgst: Math.round(totalSgst * 100) / 100,
        totalIgst: Math.round(totalIgst * 100) / 100,
        roundOff: parseFloat(roundOff) || 0,
        grandTotal,
        amountInWords: inWords,
      }
    }
  }, [])

  // ── Load Invoice Data ──────────────────────────────────────────────────────

  const fetchJob = useCallback(async () => {
    try {
      const { data } = await axios.get(`/api/invoices/${jobId}`)
      setJob(data)

      // Fetch doc info
      try {
        const docRes = await axios.get(`/api/invoices/${jobId}/doc-info`)
        setDocTotalPages(docRes.data.pages || 1)
      } catch {}

      if (data.invoice_builder_data) {
        const bData = data.invoice_builder_data
        const calc = recalculate(bData.items || [], bData.totals?.roundOff || 0)
        setFormData({
          meta: { ...getEmptyForm().meta, ...(bData.meta || {}) },
          client: { ...getEmptyForm().client, ...(bData.client || {}) },
          company: { ...getEmptyForm().company, ...(bData.company || {}) },
          bankDetails: { ...getEmptyForm().bankDetails, ...(bData.bankDetails || {}) },
          items: calc.items.length > 0 ? calc.items : getEmptyForm().items,
          totals: { ...(bData.totals || {}), ...calc.totals },
          remarks: bData.remarks || '',
          certifiedRemarks: bData.certifiedRemarks || getEmptyForm().certifiedRemarks,
        })
      } else if (data.invoice) {
        const inv = data.invoice
        const vLines = (inv.vendor_address || '').split('\n')
        const bLines = (inv.buyer_address || '').split('\n')

        const rawItems = (inv.line_items || []).map(it => ({
          description: it.description || '',
          hsnSac: it.hsn_code || '',
          quantity: it.quantity || 1,
          unit: it.unit || 'NOS',
          rate: it.rate || 0,
          discount: it.discount || 0,
          taxableValue: it.amount || 0,
          cgstRate: it.cgst_rate || 0,
          cgstAmount: it.cgst_amount || 0,
          sgstRate: it.sgst_rate || 0,
          sgstAmount: it.sgst_amount || 0,
          igstRate: it.igst_rate || 0,
          igstAmount: it.igst_amount || 0,
        }))

        const calc = recalculate(rawItems.length > 0 ? rawItems : getEmptyForm().items, inv.round_off || 0)

        setFormData({
          meta: {
            invoiceNo: inv.invoice_number || '',
            category: inv.category || '',
            subcategory: inv.subcategory || '',
            date: inv.invoice_date || '',
            placeOfSupply: inv.place_of_supply || '',
            dueDate: inv.due_date || '',
          },
          client: {
            slsCode: inv.sls_code || '',
            name: inv.buyer_name || '',
            addressLine1: bLines[0] || '',
            addressLine2: bLines.slice(1).join('\n') || '',
            phone: inv.buyer_phone || '',
            gstin: inv.buyer_gstin || '',
          },
          company: {
            name: inv.vendor_name || '',
            addressLine1: vLines[0] || '',
            addressLine2: vLines.slice(1).join('\n') || '',
            email: inv.vendor_email || '',
            phone: inv.vendor_phone || '',
            gstin: inv.vendor_gstin || '',
            pan: inv.vendor_pan || '',
          },
          bankDetails: {
            ifsc: inv.ifsc_code || '',
            branchName: inv.branch_name || '',
            bankName: inv.bank_name || '',
            accountName: inv.account_name || inv.vendor_name || '',
            accountNumber: inv.account_number || '',
            confirmAccountNumber: inv.account_number || '',
          },
          items: calc.items,
          totals: {
            taxableAmount: inv.subtotal || calc.totals.taxableAmount,
            totalDiscount: inv.discount || calc.totals.totalDiscount,
            netTaxable: (inv.subtotal || calc.totals.taxableAmount) - (inv.discount || 0),
            globalDiscount: 0,
            totalCgst: inv.cgst || calc.totals.totalCgst,
            totalSgst: inv.sgst || calc.totals.totalSgst,
            totalIgst: inv.igst || calc.totals.totalIgst,
            roundOff: inv.round_off || 0,
            grandTotal: inv.grand_total || calc.totals.grandTotal,
            amountInWords: inv.amount_in_words || calc.totals.amountInWords,
          },
          remarks: inv.remarks || '',
          certifiedRemarks: getEmptyForm().certifiedRemarks,
        })
      }
    } catch {
      toast.error('Failed to load invoice')
    } finally {
      setLoading(false)
    }
  }, [jobId, recalculate])

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const { data } = await axios.get('/api/train/status')
      setTrainingStatus(data)
    } catch {}
  }, [])

  useEffect(() => {
    fetchJob()
    fetchTrainingStatus()
  }, [fetchJob, fetchTrainingStatus])

  // Polling
  useEffect(() => {
    if (job?.status === 'processing' || job?.status === 'pending') {
      const t = setTimeout(fetchJob, 3000)
      return () => clearTimeout(t)
    }
  }, [job, fetchJob])

  // ── Form Updaters ──────────────────────────────────────────────────────────

  const updateSection = (section, key, val) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: val,
      }
    }))
    setDirty(true)
  }

  const updateItem = (index, field, val) => {
    const updated = formData.items.map((it, idx) => {
      if (idx !== index) return it
      return { ...it, [field]: val }
    })
    const calc = recalculate(updated, formData.totals.roundOff)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals },
    }))
    setDirty(true)
  }

  const addItem = () => {
    const next = [
      ...formData.items,
      {
        description: '',
        hsnSac: '',
        quantity: 1,
        unit: 'NOS',
        rate: 0,
        discount: 0,
        taxableValue: 0,
        cgstRate: 0,
        cgstAmount: 0,
        sgstRate: 0,
        sgstAmount: 0,
        igstRate: 0,
        igstAmount: 0,
      }
    ]
    const calc = recalculate(next, formData.totals.roundOff)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals },
    }))
    setDirty(true)
  }

  const removeItem = (idx) => {
    if (formData.items.length <= 1) {
      toast.error('Invoice must contain at least 1 line item')
      return
    }
    const next = formData.items.filter((_, i) => i !== idx)
    const calc = recalculate(next, formData.totals.roundOff)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals },
    }))
    setDirty(true)
  }

  const updateRoundOff = (val) => {
    const calc = recalculate(formData.items, val)
    setFormData(prev => ({
      ...prev,
      totals: { ...prev.totals, ...calc.totals },
    }))
    setDirty(true)
  }

  // ── Save & Export Actions ──────────────────────────────────────────────────

  const save = async (asVerified = false) => {
    // Explicit boolean check prevents SyntheticEvent objects from being truthy
    const isTrulyVerified = asVerified === true
    setSaving(true)
    try {
      const { data } = await axios.patch(`/api/invoices/${jobId}`, {
        corrections: formData,
        is_verified: isTrulyVerified,
        status: isTrulyVerified ? 'reviewed' : 'partially_reviewed',
      })
      if (isTrulyVerified) {
        toast.success('Invoice verified & marked as ground truth! ✓')
      } else {
        toast.success('Draft progress saved as Partially Reviewed (unverified).')
      }
      setDirty(false)
      fetchJob()
    } catch (e) {
      toast.error('Save failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setSaving(false)
    }
  }

  const loadHtmlPreview = async (targetTab = 'preview') => {
    try {
      const { data } = await axios.get(`/api/invoices/${jobId}/html`)
      setPreviewHtml(data)
      setActiveTab(targetTab)
    } catch {
      toast.error('HTML Preview not available')
    }
  }

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(formData, null, 2))
    setCopied(true)
    toast.success('Copied clean Invoice Builder JSON to clipboard!')
    setTimeout(() => setCopied(false), 2500)
  }

  const downloadPdf = () => {
    window.open(`/api/invoices/${jobId}/pdf`, '_blank')
  }

  const triggerModelTraining = async (modelType) => {
    setTrainingInProgress(true)
    try {
      await axios.post(`/api/train/${modelType}`)
      toast.success(`${modelType.toUpperCase()} retraining triggered in background!`)
      fetchTrainingStatus()
    } catch (e) {
      toast.error('Training trigger failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setTrainingInProgress(false)
    }
  }

  const reprocessCurrentInvoice = async () => {
    // Close any previous rescan SSE connection
    if (rescanEsRef.current) rescanEsRef.current.close()

    rescanTargetRef.current = 15
    setIsRescanning(true)
    setRescanProgress({
      stage: 'preprocessing',
      stageIndex: 1,
      stageLabel: 'Pre-processing: Initializing document...',
      progressPct: 15,
    })
    try {
      await axios.post(`/api/invoices/${jobId}/reprocess`)
      toast.success('AI Pipeline re-scan started!')
      setJob(prev => prev ? { ...prev, status: 'processing' } : prev)

      // Open SSE stream — backend pushes events; no polling requests needed
      const es = new EventSource(`/api/invoices/${jobId}/stream`)
      rescanEsRef.current = es

      es.onmessage = (e) => {
        const data = JSON.parse(e.data)

        if (data.stage_index != null && data.stage_index > 0) {
          if ((data.progress_pct || 15) > rescanTargetRef.current) {
            rescanTargetRef.current = data.progress_pct || 15
          }
          setRescanProgress(prev => ({
            ...prev,
            stage: data.stage || 'processing',
            stageIndex: data.stage_index,
            stageLabel: data.stage_label || 'Processing document...',
          }))
        }

        if (data.status === 'done' || data.status === 'reviewed' || data.status === 'partially_reviewed') {
          es.close()
          rescanTargetRef.current = 100
          setRescanProgress(prev => ({ ...prev, stageLabel: 'Digitization Complete', progressPct: 100, stageIndex: 6 }))
          setTimeout(() => {
            setIsRescanning(false)
            fetchJob()
            toast.success('Invoice data refreshed with latest AI pipeline results!')
          }, 400)
        } else if (data.status === 'failed') {
          es.close()
          setIsRescanning(false)
          fetchJob()
          toast.error('Re-scan encountered an error.')
        }
      }

      es.onerror = () => {
        // EventSource auto-reconnects on transient network errors
      }
    } catch (e) {
      setIsRescanning(false)
      toast.error('Re-scan failed: ' + (e.response?.data?.detail || e.message))
    }
  }



  // ── Step Validation Checks ─────────────────────────────────────────────────

  const isStepValid = (stepId) => {
    if (stepId === 1) return Boolean(formData.client.name && formData.meta.invoiceNo && formData.meta.date)
    if (stepId === 2) return Boolean(formData.company.name && formData.bankDetails.ifsc && formData.bankDetails.accountNumber)
    if (stepId === 3) return formData.items.length > 0 && formData.totals.grandTotal > 0
    if (stepId === 4) return true
    return true
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 p-16">
        <RefreshCw size={24} className="spinner text-blue-600 mb-3" />
        <p className="font-semibold text-slate-700 text-sm">Loading Invoice Digitizer...</p>
      </div>
    )
  }

  const isReady = job && ['done', 'reviewed', 'partially_reviewed', 'failed'].includes(job.status)
  if (!isReady) {
    return (
      <div className="p-12 text-center text-slate-400">
        <RefreshCw size={28} className="spinner mx-auto mb-3 text-blue-600" />
        <p className="font-bold text-slate-800 text-base">Processing Invoice Intelligence...</p>
        <p className="text-xs text-slate-500 mt-1">Extracting layout, OCR text, and financial entities. This page will update automatically.</p>
      </div>
    )
  }

  const confidenceScore = Math.round((job.overall_confidence || 0) * 100)
  const previewImageUrl = `/api/invoices/${jobId}/preview-image?page=${docPage}&t=${dirty ? 'edit' : 'view'}`

  return (
    <div className="flex flex-col h-full bg-slate-50">
      
      {/* ── Top App Bar ──────────────────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between gap-4 flex-shrink-0 sticky top-0 z-30 shadow-sm">
        <div className="flex items-center gap-3 min-w-0">
          <button onClick={() => navigate('/invoices')} className="btn-secondary py-1.5 px-3">
            <ArrowLeft size={14} /> <span className="hidden sm:inline">Back</span>
          </button>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold text-slate-900 truncate max-w-xs md:max-w-md">{job.filename}</h1>
              <span
                className={`badge ${
                  job.status === 'reviewed'
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : job.status === 'partially_reviewed'
                    ? 'bg-amber-50 text-amber-800 border border-amber-200'
                    : 'bg-blue-50 text-blue-700 border border-blue-200'
                }`}
              >
                {job.status === 'reviewed'
                  ? '✓ Verified'
                  : job.status === 'partially_reviewed'
                  ? '⏳ Partially Reviewed'
                  : '🤖 AI Extracted'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">Job ID: {jobId.slice(0, 12)}</p>
          </div>
        </div>

        {/* Center: View Switcher */}
        <div className="hidden lg:flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200">
          <button
            type="button"
            onClick={() => setIsWizardMode(false)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              !isWizardMode ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <LayoutGrid size={13} />
            Standard View
          </button>
          <button
            type="button"
            onClick={() => setIsWizardMode(true)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              isWizardMode ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Sparkles size={13} className="text-amber-500" />
            🧙 Guided Wizard
          </button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          {/* Tab buttons */}
          <div className="flex bg-slate-100 rounded-lg p-1 gap-1 border border-slate-200">
            <button
              onClick={() => setActiveTab('edit')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                activeTab === 'edit' ? 'bg-white shadow-sm text-slate-900 font-bold' : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              Form
            </button>
            <button
              onClick={() => setActiveTab('split')}
              className={`hidden md:flex items-center gap-1 px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                activeTab === 'split' ? 'bg-white shadow-sm text-slate-900 font-bold' : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <Columns size={12} /> Split View
            </button>
            <button
              onClick={() => {
                if (docSource === 'rendered' && !previewHtml) {
                  loadHtmlPreview('preview')
                } else {
                  setActiveTab('preview')
                }
              }}
              className={`flex items-center gap-1 px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                activeTab === 'preview' ? 'bg-white shadow-sm text-slate-900 font-bold' : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <Eye size={12} /> Document
            </button>
          </div>

          <button
            disabled={isRescanning || job?.status === 'processing'}
            onClick={reprocessCurrentInvoice}
            className={`btn-secondary py-1.5 px-3 border transition-all ${
              isRescanning || job?.status === 'processing'
                ? 'bg-amber-100 text-amber-900 border-amber-300 cursor-not-allowed opacity-85'
                : 'text-amber-800 border-amber-200 bg-amber-50 hover:bg-amber-100'
            }`}
            title="Re-scan this invoice with the latest AI pipeline"
          >
            {isRescanning || job?.status === 'processing' ? (
              <>
                <RefreshCw size={13} className="spinner text-amber-700" />
                <span className="hidden lg:inline font-semibold">Re-scanning...</span>
              </>
            ) : (
              <>
                <Sparkles size={13} className="text-amber-600" />
                <span className="hidden lg:inline">Re-scan with AI</span>
              </>
            )}
          </button>

          <button
            onClick={() => setShowTrainModal(true)}
            className="btn-secondary py-1.5 px-3 text-purple-700 border-purple-200 bg-purple-50/50 hover:bg-purple-100/50"
            title="Train AI Models on Reviewed Invoices"
          >
            <Cpu size={13} /> <span className="hidden xl:inline">Train Models</span>
          </button>



          <button onClick={downloadPdf} title="Download rendered PDF invoice" className="btn-secondary py-1.5 px-3">
            <Download size={13} /> <span className="hidden md:inline">PDF</span>
          </button>
          <button onClick={copyJson} title="Copy JSON for Angular invoiceForm.patchValue" className="btn-secondary py-1.5 px-3">
            {copied ? <Check size={13} className="text-emerald-600" /> : <Copy size={13} />}
            <span className="hidden md:inline">{copied ? 'Copied!' : 'Copy JSON'}</span>
          </button>

          {/* Save Draft (Partial Review) */}
          <button
            onClick={() => save(false)}
            disabled={saving}
            className="btn-secondary py-1.5 px-3 text-xs border-amber-200 bg-amber-50 hover:bg-amber-100 text-amber-900 shadow-xs"
            title="Save changes as a draft without marking as verified ground truth"
          >
            {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700" />}
            <span>Save Partial</span>
          </button>

          {/* Full Verify & Confirm */}
          <button
            onClick={() => save(true)}
            disabled={saving}
            className="btn-success py-1.5 px-3.5 text-xs shadow-xs"
            title="Confirm all fields are correct and mark as verified ground truth for AI retraining"
          >
            {saving ? <RefreshCw size={13} className="spinner" /> : <CheckCircle size={13} />}
            <span>Verify & Complete</span>
          </button>
        </div>
      </header>


      {/* ── Active Scanning Banner ───────────────────────────────────────────── */}
      {(isRescanning || job?.status === 'processing') && (
        <div className="bg-slate-900 text-white px-6 py-2.5 flex flex-col md:flex-row md:items-center justify-between text-xs shadow-lg sticky top-[57px] z-20 border-b border-amber-500/40 gap-2.5 animate-fade-in">
          <div className="flex items-center gap-3 font-medium min-w-0">
            <div className="w-6 h-6 rounded-full bg-amber-500/20 border border-amber-500/50 flex items-center justify-center flex-shrink-0">
              <RefreshCw size={12} className="spinner text-amber-400" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-bold text-amber-300">
                  {rescanProgress.stageIndex > 0 ? `Stage ${rescanProgress.stageIndex}/6:` : 'AI Pipeline Active:'}
                </span>
                <span className="text-white font-semibold truncate">
                  {rescanProgress.stageLabel || 'Processing document...'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3 flex-shrink-0 self-end md:self-auto">
            <div className="w-36 bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
              <div
                className="bg-amber-400 h-full rounded-full transition-all duration-500 ease-out"
                style={{ width: `${rescanProgress.progressPct}%` }}
              />
            </div>
            <span className="text-[11px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2.5 py-0.5 rounded-full font-mono font-bold tracking-wide">
              {rescanProgress.progressPct}%
            </span>
          </div>
        </div>
      )}


      {/* ── Retrain Modal ────────────────────────────────────────────────────── */}
      {showTrainModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="card max-w-md w-full p-6 shadow-xl bg-white border-slate-200 animate-pop-in">
            <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
              <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
                <Cpu size={18} className="text-blue-600" />
                <span>Trigger AI Model Retraining</span>
              </div>
              <button onClick={() => setShowTrainModal(false)} className="text-slate-400 hover:text-slate-600">
                <X size={16} />
              </button>
            </div>

            <p className="text-xs text-slate-600 mb-4">
              Retrain your dedicated machine learning models directly using the verified invoices saved in your database.
            </p>

            <div className="space-y-3">
              <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-900">YOLOv8 Region Detector</h4>
                  <p className="text-[11px] text-slate-500">Learns visual bounding boxes for Header, Vendor, Items table, Totals, Bank details.</p>
                </div>
                <button
                  type="button"
                  disabled={trainingInProgress}
                  onClick={() => triggerModelTraining('yolo')}
                  className="btn-primary py-1.5 px-3 text-xs shrink-0"
                >
                  <Play size={11} fill="currentColor" /> Retrain
                </button>
              </div>

              <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-900">LayoutLMv3 Entity Classifier</h4>
                  <p className="text-[11px] text-slate-500">Fine-tunes token extraction on saved ground-truth fields.</p>
                </div>
                <button
                  type="button"
                  disabled={trainingInProgress}
                  onClick={() => triggerModelTraining('layoutlm')}
                  className="btn-success py-1.5 px-3 text-xs shrink-0"
                >
                  <Sparkles size={11} /> Fine-tune
                </button>
              </div>
            </div>

            {trainingStatus?.progress && (
              <div className="mt-4 p-2.5 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-800 flex items-center gap-2">
                <RefreshCw size={13} className="spinner text-blue-600 shrink-0" />
                <span>{trainingStatus.progress}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Main Content Area ────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-auto p-3 md:p-5 lg:p-6">
        <div className="w-full max-w-[1920px] mx-auto">
          
          {/* Fullscreen Document Tab */}
          {activeTab === 'preview' ? (
            <div className="card p-4 bg-white shadow-sm flex flex-col h-[calc(100vh-140px)]">
              <div className="flex flex-wrap justify-between items-center gap-3 mb-3 pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg border border-slate-200">
                  <button
                    onClick={() => setDocSource('original')}
                    className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                      docSource === 'original' ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600'
                    }`}
                  >
                    📄 Original Document
                  </button>
                  <button
                    onClick={() => {
                      setDocSource('rendered')
                      if (!previewHtml) loadHtmlPreview('preview')
                    }}
                    className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                      docSource === 'rendered' ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600'
                    }`}
                  >
                    🖨️ Standard Rendered
                  </button>
                </div>

                {docSource === 'original' && (
                  <div className="flex items-center gap-2">
                    {docTotalPages > 1 && (
                      <div className="flex items-center gap-1 bg-slate-100 px-2 py-1 rounded-lg text-xs font-semibold text-slate-700">
                        <button
                          disabled={docPage === 0}
                          onClick={() => setDocPage(p => Math.max(0, p - 1))}
                          className="p-0.5 hover:text-blue-600 disabled:opacity-30"
                        >
                          <ChevronLeft size={14} />
                        </button>
                        <span>Page {docPage + 1} of {docTotalPages}</span>
                        <button
                          disabled={docPage >= docTotalPages - 1}
                          onClick={() => setDocPage(p => Math.min(docTotalPages - 1, p + 1))}
                          className="p-0.5 hover:text-blue-600 disabled:opacity-30"
                        >
                          <ChevronRight size={14} />
                        </button>
                      </div>
                    )}

                    <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
                      <button
                        onClick={() => setZoomLevel(z => Math.max(40, z - 25))}
                        className="p-1 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                        title="Zoom Out"
                      >
                        <ZoomOut size={14} />
                      </button>
                      <span className="text-xs font-mono font-semibold px-1 text-slate-700 min-w-[42px] text-center">{zoomLevel}%</span>
                      <button
                        onClick={() => setZoomLevel(z => Math.min(350, z + 25))}
                        className="p-1 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                        title="Zoom In"
                      >
                        <ZoomIn size={14} />
                      </button>
                      <button
                        onClick={() => { setZoomLevel(100); setRotation(0) }}
                        className="p-1 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                        title="Reset Zoom (100%) & Rotation (0°)"
                      >
                        <RotateCcw size={12} />
                      </button>
                      <div className="w-[1px] h-3.5 bg-slate-300 mx-0.5" />
                      <button
                        onClick={() => setRotation(r => (r + 90) % 360)}
                        className={`p-1 rounded text-slate-600 hover:text-blue-600 hover:bg-white flex items-center gap-1 transition-all ${
                          rotation !== 0 ? 'text-blue-600 bg-blue-50 font-bold' : ''
                        }`}
                        title={`Rotate Image 90° clockwise (current: ${rotation}°)`}
                      >
                        <RotateCw size={13} />
                        {rotation !== 0 && <span className="text-[10px] font-mono">{rotation}°</span>}
                      </button>
                    </div>

                    <a
                      href={`/api/invoices/${jobId}/original`}
                      target="_blank"
                      rel="noreferrer"
                      className="btn-secondary py-1 px-2.5 text-xs"
                      title="Open raw file in new tab"
                    >
                      <ExternalLink size={12} /> Raw
                    </a>
                  </div>
                )}

                <button onClick={() => setActiveTab('split')} className="btn-secondary text-xs">
                  Back to Split View
                </button>
              </div>

              <div className="flex-1 w-full rounded-lg overflow-auto border border-slate-200 bg-slate-200/80 p-0 relative shadow-inner">
                {docSource === 'original' ? (
                  <div className="min-w-full min-h-full w-max h-max p-4 flex items-center justify-center">
                    <img
                      src={previewImageUrl}
                      alt="Original Scanned Invoice"
                      style={{
                        width: `${zoomLevel}%`,
                        minWidth: `${zoomLevel}%`,
                        maxWidth: 'none',
                        transform: `rotate(${rotation}deg)`,
                        transformOrigin: 'center center',
                        transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), width 0.1s ease-out',
                      }}
                      className="rounded-lg shadow-xl bg-white border border-slate-300 select-none block"
                    />
                  </div>
                ) : (
                  <iframe
                    srcDoc={previewHtml}
                    className="w-full h-full rounded-lg border-none bg-white"
                    title="Invoice HTML Preview"
                  />
                )}
              </div>
            </div>
          ) : (
            <div className={`grid ${activeTab === 'split' ? 'grid-cols-1 xl:grid-cols-12 2xl:grid-cols-12 gap-5' : 'grid-cols-1'}`}>
              
              {/* Form Column */}
              <div className={`${activeTab === 'split' ? 'xl:col-span-7 2xl:col-span-7' : 'w-full'} space-y-5`}>

                {/* Guided Stepper Header (When in Wizard Mode) */}
                {isWizardMode && (
                  <div className="card p-4 bg-white/95 border-slate-200 shadow-sm mb-2">
                    <div className="flex items-center justify-between gap-3 mb-3 px-1">
                      <div className="flex items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-blue-50 text-blue-600 border border-blue-100">
                          Step {currentStep} of {WIZARD_STEPS.length}
                        </span>
                        <h2 className="text-sm md:text-base font-bold text-slate-900 leading-none">
                          {WIZARD_STEPS[currentStep - 1].title}
                        </h2>
                        <span className="text-slate-300 hidden sm:inline">&bull;</span>
                        <span className="text-xs text-slate-400 hidden sm:inline">
                          {WIZARD_STEPS[currentStep - 1].subtitle}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <div className="w-20 md:w-28 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-600 to-teal-400 h-full rounded-full transition-all duration-500"
                            style={{ width: `${(currentStep / WIZARD_STEPS.length) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-teal-600 min-w-[32px] text-right">
                          {Math.round((currentStep / WIZARD_STEPS.length) * 100)}%
                        </span>
                      </div>
                    </div>

                    {/* Stepper Line and Nodes */}
                    <div className="wizard-stepper-bar px-2 md:px-6 pt-1">
                      <div className="wizard-step-line">
                        <div
                          className="wizard-step-line-progress"
                          style={{ width: `${((currentStep - 1) / (WIZARD_STEPS.length - 1)) * 100}%` }}
                        />
                      </div>

                      {WIZARD_STEPS.map(step => {
                        const active = currentStep === step.id
                        const completed = currentStep > step.id || (currentStep !== step.id && isStepValid(step.id))
                        return (
                          <div
                            key={step.id}
                            onClick={() => setCurrentStep(step.id)}
                            className={`wizard-step-item group ${active ? 'active' : ''} ${completed ? 'completed' : ''}`}
                          >
                            <div className="wizard-step-circle">
                              {completed ? (
                                <Check size={14} className="text-white animate-pop-in" />
                              ) : (
                                <span>{step.id}</span>
                              )}
                            </div>
                            <span className="text-[11px] font-semibold mt-1.5 text-center leading-tight tracking-tight text-slate-600 group-hover:text-slate-900 transition-colors max-w-[90px]">
                              {step.title.split('&')[0].trim()}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Validation / AI Confidence Banner */}
                <div className="card p-4 bg-gradient-to-r from-white via-white to-blue-50/40 border-blue-100 flex flex-wrap items-center justify-between gap-4 shadow-sm">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                      confidenceScore >= 80 ? 'bg-emerald-100 text-emerald-700' : confidenceScore >= 50 ? 'bg-blue-100 text-blue-700' : 'bg-amber-100 text-amber-700'
                    }`}>
                      <Sparkles size={20} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Field Detection & Fill Rate</span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                          confidenceScore >= 80 ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : confidenceScore >= 50 ? 'bg-blue-50 text-blue-700 border border-blue-200' : 'bg-amber-50 text-amber-700 border border-amber-200'
                        }`}>
                          {confidenceScore}%
                        </span>
                        <span className="text-[11px] font-semibold text-slate-400">
                          {confidenceScore >= 80 ? '(High Coverage)' : confidenceScore >= 50 ? '(Moderate Coverage)' : '(Partial Extraction)'}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 mt-0.5">
                        {confidenceScore >= 80
                          ? 'High extraction coverage — most invoice blocks and fields identified.'
                          : confidenceScore >= 50
                            ? 'Moderate extraction coverage — verify highlighted fields and line items.'
                            : 'Low extraction coverage — document format may require manual verification.'}
                      </p>
                    </div>
                  </div>

                  {job.review_reasons?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 max-w-md">
                      {job.review_reasons.slice(0, 3).map((r, i) => (
                        <span key={i} className="text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200 px-2 py-0.5 rounded-md flex items-center gap-1">
                          <AlertTriangle size={10} /> {r}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* ── STEP 1: BILL TO & INVOICE DETAILS ─────────────────────── */}
                {(!isWizardMode || currentStep === 1) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5 fade-in">
                    
                    {/* Bill To (Client) */}
                    <div className="card p-5 border-slate-200">
                      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                        <Building size={16} className="text-blue-600" />
                        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Bill To (Client)</h3>
                      </div>
                      <div className="space-y-3 text-xs">
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">SLS Code / Scheme</label>
                          <input
                            type="text"
                            className="field-input text-xs font-mono"
                            placeholder="e.g. WB-SLS-2024"
                            value={formData.client.slsCode}
                            onChange={e => updateSection('client', 'slsCode', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Client / Agency Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-semibold text-slate-900"
                            placeholder="Recipient Agency Name"
                            value={formData.client.name}
                            onChange={e => updateSection('client', 'name', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Address Line 1</label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="Building, Street, Area"
                            value={formData.client.addressLine1}
                            onChange={e => updateSection('client', 'addressLine1', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Address Line 2</label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="City, District, State, PIN"
                            value={formData.client.addressLine2}
                            onChange={e => updateSection('client', 'addressLine2', e.target.value)}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Phone</label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="+91 9876543210"
                              value={formData.client.phone}
                              onChange={e => updateSection('client', 'phone', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">GSTIN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono"
                              placeholder="19AAAAA0000A1Z5"
                              value={formData.client.gstin}
                              onChange={e => updateSection('client', 'gstin', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Invoice Meta */}
                    <div className="card p-5 border-slate-200">
                      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                        <FileText size={16} className="text-blue-600" />
                        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Invoice Details</h3>
                      </div>
                      <div className="space-y-3 text-xs">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Category <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="e.g. Services / Goods"
                              value={formData.meta.category}
                              onChange={e => updateSection('meta', 'category', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Subcategory <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="e.g. IT & Software"
                              value={formData.meta.subcategory}
                              onChange={e => updateSection('meta', 'subcategory', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Invoice Number <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-bold font-mono text-blue-700"
                            placeholder="INV-2024-001"
                            value={formData.meta.invoiceNo}
                            onChange={e => updateSection('meta', 'invoiceNo', e.target.value)}
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Invoice Date <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="DD/MM/YYYY"
                              value={formData.meta.date}
                              onChange={e => updateSection('meta', 'date', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Due Date</label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="DD/MM/YYYY"
                              value={formData.meta.dueDate}
                              onChange={e => updateSection('meta', 'dueDate', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Place of Supply (Supply to)</label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="19-West Bengal or State"
                            value={formData.meta.placeOfSupply}
                            onChange={e => updateSection('meta', 'placeOfSupply', e.target.value)}
                          />
                        </div>
                      </div>
                    </div>

                  </div>
                )}

                {/* ── STEP 2: BILLER & BANK DETAILS ────────────────────────── */}
                {(!isWizardMode || currentStep === 2) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5 fade-in">
                    
                    {/* Biller Details */}
                    <div className="card p-5 border-slate-200">
                      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                        <Building size={16} className="text-blue-600" />
                        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Biller Details (Vendor)</h3>
                      </div>
                      <div className="space-y-3 text-xs">
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Vendor / Company Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-bold text-slate-900"
                            placeholder="Vendor Registered Name (Max 40 chars)"
                            value={formData.company.name}
                            onChange={e => updateSection('company', 'name', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Address Line 1</label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="Address Line 1"
                            value={formData.company.addressLine1}
                            onChange={e => updateSection('company', 'addressLine1', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Address Line 2</label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="Address Line 2"
                            value={formData.company.addressLine2}
                            onChange={e => updateSection('company', 'addressLine2', e.target.value)}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Email</label>
                            <input
                              type="email"
                              className="field-input text-xs"
                              placeholder="vendor@company.com"
                              value={formData.company.email}
                              onChange={e => updateSection('company', 'email', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Phone</label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="+91 9876543210"
                              value={formData.company.phone}
                              onChange={e => updateSection('company', 'phone', e.target.value)}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">GSTIN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono"
                              placeholder="19AAAAA0000A1Z5"
                              value={formData.company.gstin}
                              onChange={e => updateSection('company', 'gstin', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">PAN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono"
                              placeholder="ABCDE1234F"
                              value={formData.company.pan}
                              onChange={e => updateSection('company', 'pan', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Bank Details */}
                    <div className="card p-5 border-slate-200">
                      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                        <Landmark size={16} className="text-blue-600" />
                        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Bank Details</h3>
                      </div>
                      <div className="space-y-3 text-xs">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">IFSC Code <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono font-semibold text-blue-700"
                              placeholder="SBIN0001234"
                              value={formData.bankDetails.ifsc}
                              onChange={e => updateSection('bankDetails', 'ifsc', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Branch Name <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs"
                              placeholder="Kolkata Main Branch"
                              value={formData.bankDetails.branchName}
                              onChange={e => updateSection('bankDetails', 'branchName', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Bank Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs"
                            placeholder="State Bank of India"
                            value={formData.bankDetails.bankName}
                            onChange={e => updateSection('bankDetails', 'bankName', e.target.value)}
                          />
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 mb-1">Account Beneficiary Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-semibold"
                            placeholder="Vendor Company Name"
                            value={formData.bankDetails.accountName}
                            onChange={e => updateSection('bankDetails', 'accountName', e.target.value)}
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Account Number <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs font-mono font-semibold"
                              placeholder="123456789012"
                              value={formData.bankDetails.accountNumber}
                              onChange={e => updateSection('bankDetails', 'accountNumber', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 mb-1">Confirm Account No <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs font-mono font-semibold"
                              placeholder="123456789012"
                              value={formData.bankDetails.confirmAccountNumber || formData.bankDetails.accountNumber}
                              onChange={e => updateSection('bankDetails', 'confirmAccountNumber', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                  </div>
                )}

                {/* ── STEP 3: ITEMS & TOTALS ────────────────────────────────── */}
                {(!isWizardMode || currentStep === 3) && (
                  <div className="space-y-5 fade-in">
                    
                    {/* Line Items Table */}
                    <div className="card p-5 border-slate-200">
                      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
                        <div className="flex items-center gap-2">
                          <ShoppingBag size={16} className="text-blue-600" />
                          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Line Items Table</h3>
                        </div>
                        <button
                          type="button"
                          onClick={addItem}
                          className="btn-secondary py-1 px-2.5 text-xs text-blue-600 border-blue-200 hover:bg-blue-50"
                        >
                          <Plus size={13} /> Add Item Row
                        </button>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/70">
                              <th className="py-2.5 px-2 text-left min-w-[180px]">Description</th>
                              <th className="py-2.5 px-2 text-left w-24">HSN/SAC</th>
                              <th className="py-2.5 px-2 text-right w-16">Qty</th>
                              <th className="py-2.5 px-2 text-left w-16">Unit</th>
                              <th className="py-2.5 px-2 text-right w-24">Rate (₹)</th>
                              <th className="py-2.5 px-2 text-right w-20">Disc (₹)</th>
                              <th className="py-2.5 px-2 text-right w-24 font-bold text-slate-700">Taxable (₹)</th>
                              <th className="py-2.5 px-2 text-right w-16">CGST %</th>
                              <th className="py-2.5 px-2 text-right w-16">SGST %</th>
                              <th className="py-2.5 px-2 text-right w-16">IGST %</th>
                              <th className="w-8"></th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {formData.items.map((item, idx) => (
                              <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                                <td className="py-1.5 px-1">
                                  <input
                                    type="text"
                                    className="field-input-sm w-full font-medium"
                                    placeholder="Item description"
                                    value={item.description}
                                    onChange={e => updateItem(idx, 'description', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="text"
                                    className="field-input-sm w-full font-mono text-center"
                                    placeholder="HSN"
                                    value={item.hsnSac}
                                    onChange={e => updateItem(idx, 'hsnSac', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    value={item.quantity}
                                    onChange={e => updateItem(idx, 'quantity', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <select
                                    className="field-input-sm w-full bg-white"
                                    value={item.unit || 'NOS'}
                                    onChange={e => updateItem(idx, 'unit', e.target.value)}
                                  >
                                    <option value="NOS">NOS</option>
                                    <option value="PCS">PCS</option>
                                    <option value="KG">KG</option>
                                    <option value="MTR">MTR</option>
                                    <option value="SET">SET</option>
                                    <option value="MONTH">MONTH</option>
                                  </select>
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    value={item.rate}
                                    onChange={e => updateItem(idx, 'rate', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono text-slate-500"
                                    value={item.discount}
                                    onChange={e => updateItem(idx, 'discount', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-2 text-right font-mono font-bold text-slate-800">
                                  ₹{item.taxableValue?.toFixed(2)}
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={item.cgstRate}
                                    onChange={e => updateItem(idx, 'cgstRate', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={item.sgstRate}
                                    onChange={e => updateItem(idx, 'sgstRate', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 px-1">
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={item.igstRate}
                                    onChange={e => updateItem(idx, 'igstRate', e.target.value)}
                                  />
                                </td>
                                <td className="py-1.5 pl-1 text-center">
                                  <button
                                    type="button"
                                    onClick={() => removeItem(idx)}
                                    className="p-1 text-slate-300 hover:text-red-500 transition-colors"
                                    title="Delete line item"
                                  >
                                    <Trash2 size={13} />
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Financial Totals Panel */}
                    <div className="card p-5 border-slate-200 bg-slate-50/40">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        
                        {/* Left: Amount in words & Round Off */}
                        <div className="space-y-3 text-xs">
                          <div>
                            <label className="block font-bold text-slate-600 uppercase tracking-wider mb-1">Amount in Words</label>
                            <div className="p-3 bg-white rounded-lg border border-slate-200 font-semibold text-blue-900 text-xs italic">
                              {formData.totals.amountInWords || 'Zero Rupees'}
                            </div>
                          </div>

                          <div className="flex items-center gap-3">
                            <label className="font-medium text-slate-600 w-24">Round Off (+/-):</label>
                            <input
                              type="number"
                              step="0.01"
                              className="field-input-sm w-32 font-mono text-right"
                              value={formData.totals.roundOff}
                              onChange={e => updateRoundOff(e.target.value)}
                            />
                          </div>
                        </div>

                        {/* Right: Breakdown Table */}
                        <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-2 text-xs">
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-500">Taxable Subtotal</span>
                            <span className="font-mono font-semibold text-slate-800">₹{formData.totals.taxableAmount?.toFixed(2)}</span>
                          </div>
                          {formData.totals.totalDiscount > 0 && (
                            <div className="flex justify-between py-1 border-b border-slate-100 text-amber-700">
                              <span>Total Discount</span>
                              <span className="font-mono font-semibold">-₹{formData.totals.totalDiscount?.toFixed(2)}</span>
                            </div>
                          )}
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-500">Net Taxable Amount</span>
                            <span className="font-mono font-semibold text-slate-800">₹{formData.totals.netTaxable?.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-500">Total CGST</span>
                            <span className="font-mono font-semibold text-slate-800">₹{formData.totals.totalCgst?.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-500">Total SGST</span>
                            <span className="font-mono font-semibold text-slate-800">₹{formData.totals.totalSgst?.toFixed(2)}</span>
                          </div>
                          {formData.totals.totalIgst > 0 && (
                            <div className="flex justify-between py-1 border-b border-slate-100">
                              <span className="text-slate-500">Total IGST</span>
                              <span className="font-mono font-semibold text-slate-800">₹{formData.totals.totalIgst?.toFixed(2)}</span>
                            </div>
                          )}
                          <div className="flex justify-between pt-2 text-sm font-bold text-slate-900 border-t border-slate-200">
                            <span>Grand Total (INR)</span>
                            <span className="font-mono text-base text-blue-700">₹{formData.totals.grandTotal?.toFixed(2)}</span>
                          </div>
                        </div>

                      </div>
                    </div>

                  </div>
                )}

                {/* ── STEP 4: REMARKS & CERTIFICATIONS ─────────────────────── */}
                {(!isWizardMode || currentStep === 4) && (
                  <div className="space-y-5 fade-in">
                    <div className="card p-5 border-slate-200 space-y-4">
                      <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
                        <MessageSquare size={16} className="text-blue-600" />
                        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Remarks & Declarations</h3>
                      </div>

                      <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">
                          Invoice Remarks / Notes (Max 50 words)
                        </label>
                        <textarea
                          rows={3}
                          className="field-input text-xs"
                          placeholder="Add any internal remarks, voucher notes, or terms here..."
                          value={formData.remarks}
                          onChange={e => setFormData(prev => ({ ...prev, remarks: e.target.value }))}
                        />
                      </div>

                      <div>
                        <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                          Standard Certified Remarks
                        </label>
                        <div className="space-y-2">
                          {formData.certifiedRemarks.map((remark, idx) => (
                            <div key={idx} className="flex items-start gap-2.5 p-2.5 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-700">
                              <CheckCircle size={14} className="text-emerald-600 shrink-0 mt-0.5" />
                              <span>{remark}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* ── STEP 5: REVIEW & SUBMIT ──────────────────────────────── */}
                {(!isWizardMode || currentStep === 5) && (
                  <div className="space-y-5 fade-in">
                    <div className="card p-6 border-slate-200 bg-white shadow-sm">
                      <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-100">
                        <div className="flex items-center gap-2">
                          <CheckCircle size={18} className="text-emerald-600" />
                          <h3 className="text-sm font-bold text-slate-900">Summary & Submission Checklist</h3>
                        </div>
                        <span className="badge bg-emerald-50 text-emerald-700 border border-emerald-200">Ready to Save</span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                          <p className="font-bold text-slate-800 uppercase tracking-wider text-[10px]">Client / Recipient</p>
                          <p className="font-bold text-slate-900">{formData.client.name || '—'}</p>
                          <p className="text-slate-500">GSTIN: {formData.client.gstin || 'Unregistered'}</p>
                          <p className="text-slate-500">SLS: {formData.client.slsCode || 'Standard'}</p>
                        </div>

                        <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                          <p className="font-bold text-slate-800 uppercase tracking-wider text-[10px]">Vendor / Biller</p>
                          <p className="font-bold text-slate-900">{formData.company.name || '—'}</p>
                          <p className="text-slate-500">GSTIN: {formData.company.gstin || '—'} | PAN: {formData.company.pan || '—'}</p>
                          <p className="text-slate-500">Bank: {formData.bankDetails.bankName} (A/C: {formData.bankDetails.accountNumber})</p>
                        </div>
                      </div>

                      <div className="mt-4 p-3.5 bg-blue-50/60 rounded-xl border border-blue-100 flex items-center justify-between">
                        <div>
                          <p className="text-xs font-bold text-blue-900">Invoice #{formData.meta.invoiceNo || 'DRAFT'}</p>
                          <p className="text-[11px] text-blue-700">Dated: {formData.meta.date} &bull; {formData.items.length} line items</p>
                        </div>
                        <div className="text-right">
                          <span className="text-[11px] text-slate-500 uppercase tracking-wider block">Grand Total</span>
                          <span className="text-base font-bold font-mono text-blue-700">₹{formData.totals.grandTotal?.toFixed(2)}</span>
                        </div>
                      </div>

                      {/* Action Bar */}
                      <div className="mt-6 pt-4 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <button onClick={downloadPdf} className="btn-secondary text-xs">
                            <Download size={13} /> PDF Invoice
                          </button>
                          <button onClick={copyJson} className="btn-secondary text-xs">
                            <Copy size={13} /> {copied ? 'Copied JSON!' : 'Copy to Invoice Builder'}
                          </button>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => save(false)}
                            disabled={saving}
                            className="btn-secondary text-xs px-4 py-2 text-amber-900 bg-amber-50 border-amber-200 hover:bg-amber-100"
                            title="Save your progress without marking as verified ground truth"
                          >
                            {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700" />}
                            <span>Save Partial</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => save(true)}
                            disabled={saving}
                            className="btn-success text-xs px-5 py-2 flex items-center gap-1.5"
                            title="Confirm all fields are correct and mark as verified ground truth"
                          >
                            {saving ? <RefreshCw size={14} className="spinner" /> : <CheckCircle size={14} />}
                            <span>Verify Ground Truth</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* ── Wizard Navigation Controls ─────────────────────────────── */}
                {isWizardMode && (
                  <div className="flex items-center justify-between pt-2 pb-6">
                    <button
                      type="button"
                      disabled={currentStep === 1}
                      onClick={() => setCurrentStep(prev => Math.max(1, prev - 1))}
                      className="btn-secondary px-4 py-2 text-xs disabled:opacity-30"
                    >
                      <ChevronLeft size={14} /> Previous Step
                    </button>

                    {currentStep < WIZARD_STEPS.length ? (
                      <button
                        type="button"
                        onClick={() => setCurrentStep(prev => Math.min(WIZARD_STEPS.length, prev + 1))}
                        className="btn-primary px-5 py-2 text-xs"
                      >
                        Next Step <ChevronRight size={14} />
                      </button>
                    ) : (
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => save(false)}
                          disabled={saving}
                          className="btn-secondary px-4 py-2 text-xs text-amber-900 bg-amber-50 border-amber-200 hover:bg-amber-100"
                          title="Save your changes so far without marking as verified ground truth"
                        >
                          {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700" />}
                          <span>Save as Partial</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => save(true)}
                          disabled={saving}
                          className="btn-success px-5 py-2 text-xs flex items-center gap-1.5"
                          title="Confirm all fields are correct and mark as verified ground truth"
                        >
                          {saving ? <RefreshCw size={14} className="spinner" /> : <CheckCircle size={14} />}
                          <span>Verify & Mark Done</span>
                        </button>
                      </div>
                    )}
                  </div>
                )}

              </div>

              {/* ── Split-Screen Document Preview Column ───────────────────── */}
              {activeTab === 'split' && (
                <div className="xl:col-span-5 2xl:col-span-5 sticky top-20 h-[calc(100vh-120px)] card p-3 bg-white shadow-sm flex flex-col">
                  
                  {/* Top Bar inside Document Pane */}
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100 flex-shrink-0">
                    <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
                      <button
                        onClick={() => setDocSource('original')}
                        className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${
                          docSource === 'original' ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        📄 Original File
                      </button>
                      <button
                        onClick={() => {
                          setDocSource('rendered')
                          if (!previewHtml) loadHtmlPreview('split')
                        }}
                        className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${
                          docSource === 'rendered' ? 'bg-white shadow-sm text-blue-600 font-bold' : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        🖨️ Standard HTML
                      </button>
                    </div>

                    {docSource === 'original' ? (
                      <div className="flex items-center gap-1">
                        {docTotalPages > 1 && (
                          <div className="flex items-center gap-0.5 text-[11px] font-semibold text-slate-600 mr-1">
                            <button
                              disabled={docPage === 0}
                              onClick={() => setDocPage(p => Math.max(0, p - 1))}
                              className="p-0.5 hover:text-blue-600 disabled:opacity-30"
                            >
                              <ChevronLeft size={12} />
                            </button>
                            <span>{docPage + 1}/{docTotalPages}</span>
                            <button
                              disabled={docPage >= docTotalPages - 1}
                              onClick={() => setDocPage(p => Math.min(docTotalPages - 1, p + 1))}
                              className="p-0.5 hover:text-blue-600 disabled:opacity-30"
                            >
                              <ChevronRight size={12} />
                            </button>
                          </div>
                        )}

                        <div className="flex items-center gap-0.5 bg-slate-100 p-0.5 rounded border border-slate-200">
                          <button
                            onClick={() => setZoomLevel(z => Math.max(40, z - 25))}
                            className="p-0.5 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                            title="Zoom Out"
                          >
                            <ZoomOut size={12} />
                          </button>
                          <span className="text-[10px] font-mono px-0.5 text-slate-700 min-w-[34px] text-center">{zoomLevel}%</span>
                          <button
                            onClick={() => setZoomLevel(z => Math.min(350, z + 25))}
                            className="p-0.5 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                            title="Zoom In"
                          >
                            <ZoomIn size={12} />
                          </button>
                          <button
                            onClick={() => { setZoomLevel(100); setRotation(0) }}
                            className="p-0.5 text-slate-600 hover:text-blue-600 rounded hover:bg-white"
                            title="Reset Zoom (100%) & Rotation (0°)"
                          >
                            <RotateCcw size={10} />
                          </button>
                          <div className="w-[1px] h-3 bg-slate-300 mx-0.5" />
                          <button
                            onClick={() => setRotation(r => (r + 90) % 360)}
                            className={`p-0.5 rounded text-slate-600 hover:text-blue-600 hover:bg-white flex items-center gap-0.5 transition-all ${
                              rotation !== 0 ? 'text-blue-600 bg-blue-50 font-bold' : ''
                            }`}
                            title={`Rotate Image 90° clockwise (current: ${rotation}°)`}
                          >
                            <RotateCw size={11} />
                            {rotation !== 0 && <span className="text-[9px] font-mono">{rotation}°</span>}
                          </button>
                        </div>

                        <button
                          onClick={() => setActiveTab('preview')}
                          className="text-slate-400 hover:text-blue-600 p-1"
                          title="Fullscreen Preview"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setActiveTab('preview')}
                        className="text-xs text-slate-500 hover:text-blue-600 flex items-center gap-1 font-medium p-1"
                        title="Fullscreen Preview"
                      >
                        <Maximize2 size={12} />
                      </button>
                    )}
                  </div>

                  {/* Document Frame */}
                  <div className="flex-1 w-full rounded-lg overflow-auto border border-slate-200 bg-slate-200/80 p-0 relative shadow-inner">
                    {docSource === 'original' ? (
                      <div className="min-w-full min-h-full w-max h-max p-3 flex items-center justify-center">
                        <img
                          src={previewImageUrl}
                          alt="Original Scanned Document"
                          style={{
                            width: `${zoomLevel}%`,
                            minWidth: `${zoomLevel}%`,
                            maxWidth: 'none',
                            transform: `rotate(${rotation}deg)`,
                            transformOrigin: 'center center',
                            transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), width 0.1s ease-out',
                          }}
                          className="rounded shadow-lg bg-white border border-slate-300 select-none block"
                        />
                      </div>
                    ) : (
                      <iframe
                        srcDoc={previewHtml}
                        className="w-full h-full border-none bg-white rounded"
                        title="Standard Rendered Invoice"
                      />
                    )}
                  </div>
                </div>
              )}

            </div>
          )}

        </div>
      </div>
    </div>
  )
}
