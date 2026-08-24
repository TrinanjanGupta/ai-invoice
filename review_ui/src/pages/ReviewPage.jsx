import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  Save, Download, Eye, ArrowLeft, AlertTriangle, AlertCircle, Upload,
  CheckCircle, RefreshCw, Trash2, Plus,
  FileText, Building, Landmark, ShoppingBag, MessageSquare,
  ChevronRight, ChevronLeft, Copy, Check, Sparkles, LayoutGrid,
  Columns, Maximize2, ZoomIn, ZoomOut, RotateCcw, RotateCw, ExternalLink,
  Play, Cpu, X, Layers
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

// ── Safe Currency & Number Formatter ─────────────────────────────────────────

function formatAmount(val) {
  if (val === null || val === undefined || val === '') return '0.00'
  const num = typeof val === 'number' ? val : parseFloat(val)
  return isNaN(num) ? '0.00' : num.toFixed(2)
}

// ── Standard Initial State Builder ──────────────────────────────────────────

function getEmptyForm() {
  return {
    meta: {
      invoiceNo: '',
      poNumber: '',
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
      paymentTerms: '',
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
      finalNetTaxable: 0,
      totalCgst: 0,
      totalSgst: 0,
      totalIgst: 0,
      globalCgstRate: 0,
      globalSgstRate: 0,
      globalIgstRate: 0,
      roundOff: 0,
      grandTotal: 0,
      amountInWords: '',
    },
    remarks: '',
    certifiedRemarks: [],
  }
}

// ── Standard Certificate Templates (Optional library for quick insertion) ───

const STANDARD_CERTIFICATE_TEMPLATES = [
  "Certified that the materials/articles as detailed in the invoice have been received in good condition and taken into stock vide Stock Entry No. ________ Dated ________.",
  "Certified that the services as detailed in the invoice have been satisfactorily rendered as per work order/agreement terms.",
  "Necessary budget provision exists under the appropriate Head of Account ________",
  "Special Remarks if any ________",
  "The claim has not been paid earlier.",
  "The quantity and specifications have been verified.",
  "The rates charged are as per approved order/work order/contract.",
]

// ── Default Columns Definition (Parity with Angular Invoice Builder) ─────────

const DEFAULT_COLUMNS = [
  { key: 'description', label: 'Description', type: 'text', removable: false, minWidth: '190px' },
  { key: 'hsnSac', label: 'HSN/SAC', type: 'text', width: '90px', removable: true },
  { key: 'quantity', label: 'Qty', type: 'number', width: '70px', removable: false },
  { key: 'unit', label: 'Unit', type: 'text', placeholder: 'NOS', width: '85px', removable: true },
  { key: 'rate', label: 'Rate (₹)', type: 'number', width: '95px', removable: false },
  { key: 'discount', label: 'Disc (₹)', type: 'number', width: '85px', removable: true },
  { key: 'taxableValue', label: 'Taxable Val (₹)', type: 'calc', width: '120px', removable: false },
]

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
  const [loadError, setLoadError] = useState(null)
  const [formData, setFormData] = useState(getEmptyForm())
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [copied, setCopied] = useState(false)

  // Columns Configuration (Dynamic Columns like Invoice Builder)
  const [columns, setColumns] = useState(DEFAULT_COLUMNS)
  const [showAddColumnModal, setShowAddColumnModal] = useState(false)
  const [newColumnName, setNewColumnName] = useState('')
  const [newColumnType, setNewColumnType] = useState('text')
  const [insertColIndex, setInsertColIndex] = useState(null)

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
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  const handleViewerMouseDown = (e) => {
    if (e.button !== 0) return // Left click only
    setIsDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }

  const handleViewerMouseMove = (e) => {
    if (!isDragging) return
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    })
  }

  const handleViewerMouseUp = () => {
    setIsDragging(false)
  }

  const resetView = () => {
    setPan({ x: 0, y: 0 })
    setZoomLevel(100)
    setRotation(0)
  }

  const fitWidth = () => {
    setPan({ x: 0, y: 0 })
    setZoomLevel(125)
  }

  const fitHeight = () => {
    setPan({ x: 0, y: 0 })
    setZoomLevel(90)
  }

  const handleViewerWheel = (e) => {
    // Zoom on wheel (Ctrl + Wheel or standard wheel over viewer)
    if (e.ctrlKey || e.metaKey || e.altKey) {
      e.preventDefault()
      const delta = e.deltaY < 0 ? 15 : -15
      setZoomLevel(z => Math.min(400, Math.max(30, z + delta)))
    }
  }

  // Bank verification & security
  const [showAccountNumber, setShowAccountNumber] = useState(false)
  const [showConfirmAccountNumber, setShowConfirmAccountNumber] = useState(false)
  const [accountVerificationStatus, setAccountVerificationStatus] = useState('pending')
  const [isVerifying, setIsVerifying] = useState(false)

  // Tax rate popup
  const [activeTaxRowIndex, setActiveTaxRowIndex] = useState(null)

  // Remarks & Certificates state
  const [newRemarkText, setNewRemarkText] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState('')

  // Training state
  const [showTrainModal, setShowTrainModal] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [trainingInProgress, setTrainingInProgress] = useState(false)
  const [isRescanning, setIsRescanning] = useState(false)
  const [rescanProgress, setRescanProgress] = useState({
    stage: 'preprocessing',
    stageIndex: 1,
    stageLabel: 'Pre-processing: Initializing document...',
    progressPct: 15,
  })
  const rescanTargetRef = useRef(15)
  const rescanEsRef = useRef(null)
  const formDataRef = useRef(formData)

  useEffect(() => {
    formDataRef.current = formData
  }, [formData])

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


  // ── Auto-Calculate Totals (Exact parity with InvoiceCalculationService) ────

  const recalculate = useCallback((items, roundOffManual = null, globalDiscountManual = null, globalRatesManual = null) => {
    let taxableAmount = 0
    let totalDiscount = 0
    let netTaxable = 0
    let hasItemLevelTax = false

    // First pass: Calculate item taxable and base totals
    const firstPassItems = (items || []).map(it => {
      const q = parseFloat(it.quantity) || 0
      const r = parseFloat(it.rate) || 0
      const d = parseFloat(it.discount) || 0
      const cgstR = parseFloat(it.cgstRate) || 0
      const sgstR = parseFloat(it.sgstRate) || 0
      const igstR = parseFloat(it.igstRate) || 0

      if (cgstR > 0 || sgstR > 0 || igstR > 0) {
        hasItemLevelTax = true
      }

      const gross = q * r
      const itemTaxable = Math.max(0, gross - d)

      taxableAmount += gross
      totalDiscount += d
      netTaxable += itemTaxable

      return {
        ...it,
        gross,
        baseTaxable: itemTaxable,
        cgstRate: cgstR,
        sgstRate: sgstR,
        igstRate: igstR,
      }
    })

    const globalDiscount = globalDiscountManual != null
      ? (parseFloat(globalDiscountManual) || 0)
      : (parseFloat(formDataRef.current?.totals?.globalDiscount) || 0)

    const effectiveGlobalDiscount = Math.min(globalDiscount, netTaxable)
    const discountRatio = netTaxable > 0 ? (effectiveGlobalDiscount / netTaxable) : 0

    let totalCgst = 0
    let totalSgst = 0
    let totalIgst = 0
    let finalNetTaxable = 0

    // Second pass: Calculate tax on apportioned taxable values
    const updatedItems = firstPassItems.map(it => {
      const apportionedDisc = it.baseTaxable * discountRatio
      const itemFinalTaxable = Math.max(0, it.baseTaxable - apportionedDisc)
      finalNetTaxable += itemFinalTaxable

      const cgstA = (itemFinalTaxable * it.cgstRate) / 100
      const sgstA = (itemFinalTaxable * it.sgstRate) / 100
      const igstA = (itemFinalTaxable * it.igstRate) / 100

      totalCgst += cgstA
      totalSgst += sgstA
      totalIgst += igstA

      return {
        ...it,
        taxableValue: Math.round(it.baseTaxable * 100) / 100,
        finalTaxableValue: Math.round(itemFinalTaxable * 100) / 100,
        cgstAmount: Math.round(cgstA * 100) / 100,
        sgstAmount: Math.round(sgstA * 100) / 100,
        igstAmount: Math.round(igstA * 100) / 100,
      }
    })

    const globalRates = globalRatesManual || {
      cgst: parseFloat(formDataRef.current?.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formDataRef.current?.totals?.globalSgstRate) || 0,
      igst: parseFloat(formDataRef.current?.totals?.globalIgstRate) || 0,
    }

    // Apply global rates if no item level tax is specified
    if (!hasItemLevelTax) {
      totalCgst = (finalNetTaxable * (globalRates.cgst || 0)) / 100
      totalSgst = (finalNetTaxable * (globalRates.sgst || 0)) / 100
      totalIgst = (finalNetTaxable * (globalRates.igst || 0)) / 100
    }

    const calculatedGrand = finalNetTaxable + totalCgst + totalSgst + totalIgst
    const roundedGrand = Math.round(calculatedGrand)
    const autoRoundOff = Math.round((roundedGrand - calculatedGrand) * 100) / 100
    const finalRoundOff = roundOffManual != null ? (parseFloat(roundOffManual) || 0) : autoRoundOff
    const grandTotal = Math.round((calculatedGrand + finalRoundOff) * 100) / 100
    const inWords = numberToWords(grandTotal)

    return {
      items: updatedItems,
      hasItemLevelTax,
      totals: {
        taxableAmount: Math.round(taxableAmount * 100) / 100,
        totalDiscount: Math.round(totalDiscount * 100) / 100,
        netTaxable: Math.round(netTaxable * 100) / 100,
        globalDiscount: Math.round(effectiveGlobalDiscount * 100) / 100,
        finalNetTaxable: Math.round(finalNetTaxable * 100) / 100,
        totalCgst: Math.round(totalCgst * 100) / 100,
        totalSgst: Math.round(totalSgst * 100) / 100,
        totalIgst: Math.round(totalIgst * 100) / 100,
        globalCgstRate: globalRates.cgst,
        globalSgstRate: globalRates.sgst,
        globalIgstRate: globalRates.igst,
        roundOff: finalRoundOff,
        grandTotal,
        amountInWords: inWords,
      }
    }
  }, [])

  // ── Load Invoice Data ──────────────────────────────────────────────────────

  const fetchJob = useCallback(async () => {
    try {
      setLoadError(null)
      const { data } = await axios.get(`/api/invoices/${jobId}`)
      setJob(data)

      // Fetch doc info
      try {
        const docRes = await axios.get(`/api/invoices/${jobId}/doc-info`)
        setDocTotalPages(docRes.data.pages || 1)
      } catch {}

      if (data.invoice_builder_data) {
        const bData = data.invoice_builder_data
        if (bData.columns && Array.isArray(bData.columns) && bData.columns.length > 0) {
          setColumns(bData.columns)
        }
        const globalDisc = parseFloat(bData.totals?.globalDiscount) || 0
        const globalRates = {
          cgst: parseFloat(bData.totals?.globalCgstRate) || 0,
          sgst: parseFloat(bData.totals?.globalSgstRate) || 0,
          igst: parseFloat(bData.totals?.globalIgstRate) || 0,
        }
        const calc = recalculate(bData.items || [], bData.totals?.roundOff != null ? bData.totals.roundOff : 0, globalDisc, globalRates)
        setFormData({
          meta: {
            ...getEmptyForm().meta,
            ...(bData.meta || {}),
            poNumber: bData.meta?.poNumber || bData.meta?.po_number || '',
          },
          client: { ...getEmptyForm().client, ...(bData.client || {}) },
          company: { ...getEmptyForm().company, ...(bData.company || {}) },
          bankDetails: {
            ...getEmptyForm().bankDetails,
            ...(bData.bankDetails || {}),
            paymentTerms: bData.bankDetails?.paymentTerms || bData.paymentTerms || '',
          },
          items: calc.items.length > 0 ? calc.items : getEmptyForm().items,
          totals: {
            ...(bData.totals || {}),
            ...calc.totals,
            globalDiscount: globalDisc,
            globalCgstRate: globalRates.cgst,
            globalSgstRate: globalRates.sgst,
            globalIgstRate: globalRates.igst,
          },
          remarks: bData.remarks || '',
          certifiedRemarks: Array.isArray(bData.certifiedRemarks) ? bData.certifiedRemarks : [],
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
          taxableValue: it.taxable_value != null ? it.taxable_value : (it.amount || 0),
          cgstRate: it.cgst_rate || 0,
          cgstAmount: it.cgst_amount || 0,
          sgstRate: it.sgst_rate || 0,
          sgstAmount: it.sgst_amount || 0,
          igstRate: it.igst_rate || 0,
          igstAmount: it.igst_amount || 0,
        }))

        const globalDisc = parseFloat(inv.global_discount) || 0
        const globalRates = {
          cgst: parseFloat(inv.global_cgst_rate) || 0,
          sgst: parseFloat(inv.global_sgst_rate) || 0,
          igst: parseFloat(inv.global_igst_rate) || 0,
        }
        const calc = recalculate(rawItems.length > 0 ? rawItems : getEmptyForm().items, inv.round_off != null ? inv.round_off : 0, globalDisc, globalRates)

        setFormData({
          meta: {
            invoiceNo: inv.invoice_number || '',
            poNumber: inv.po_number || '',
            category: inv.category || '',
            subcategory: inv.subcategory || '',
            date: inv.invoice_date || '',
            placeOfSupply: inv.place_of_supply || '',
            dueDate: inv.due_date || '',
          },
          client: {
            slsCode: inv.sls_code || '',
            name: inv.buyer_name || '',
            addressLine1: inv.buyer_address_line1 || (bLines[0] || ''),
            addressLine2: inv.buyer_address_line2 || (bLines.slice(1).join('\n') || ''),
            phone: inv.buyer_phone || '',
            gstin: inv.buyer_gstin || '',
          },
          company: {
            name: inv.vendor_name || '',
            addressLine1: inv.vendor_address_line1 || (vLines[0] || ''),
            addressLine2: inv.vendor_address_line2 || (vLines.slice(1).join('\n') || ''),
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
            paymentTerms: inv.payment_terms || '',
          },
          items: calc.items,
          totals: {
            taxableAmount: calc.totals.taxableAmount,
            totalDiscount: calc.totals.totalDiscount,
            netTaxable: calc.totals.netTaxable,
            globalDiscount: globalDisc,
            finalNetTaxable: calc.totals.finalNetTaxable,
            totalCgst: calc.totals.totalCgst,
            totalSgst: calc.totals.totalSgst,
            totalIgst: calc.totals.totalIgst,
            globalCgstRate: globalRates.cgst,
            globalSgstRate: globalRates.sgst,
            globalIgstRate: globalRates.igst,
            roundOff: calc.totals.roundOff,
            grandTotal: calc.totals.grandTotal,
            amountInWords: calc.totals.amountInWords,
          },
          remarks: inv.remarks || '',
          certifiedRemarks: Array.isArray(inv.certified_remarks) ? inv.certified_remarks : [],
        })
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load invoice'
      setLoadError(msg)
      toast.error(msg)
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
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(updated, formData.totals?.roundOff, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  const addItem = () => {
    const customDefaults = {}
    columns.forEach(col => {
      if (!['description', 'hsnSac', 'quantity', 'unit', 'rate', 'discount', 'taxableValue'].includes(col.key)) {
        customDefaults[col.key] = ''
      }
    })

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
        ...customDefaults,
      }
    ]
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(next, formData.totals?.roundOff, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  const removeItem = (idx) => {
    if (formData.items.length <= 1) {
      toast.error('Invoice must contain at least 1 line item')
      return
    }
    const next = formData.items.filter((_, i) => i !== idx)
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(next, formData.totals?.roundOff, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  const updateRoundOff = (val) => {
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(formData.items, val, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  // ── Column Management Handlers (Parity with Angular Invoice Builder) ──────

  const openAddColumnModal = (atIndex = null) => {
    setInsertColIndex(atIndex)
    setNewColumnName('')
    setNewColumnType('text')
    setShowAddColumnModal(true)
  }

  const confirmAddColumn = () => {
    const name = newColumnName.trim()
    if (!name) {
      toast.error('Please enter a column name')
      return
    }
    const key = name.toLowerCase().replace(/[^a-zA-Z0-9]/g, '')
    if (!key) {
      toast.error('Invalid column name')
      return
    }
    if (columns.some(c => c.key === key || c.label.toLowerCase() === name.toLowerCase())) {
      toast.error('A column with this name already exists')
      return
    }

    const newCol = {
      key,
      label: name,
      type: newColumnType || (name.toLowerCase().includes('date') ? 'date' : 'text'),
      removable: true,
      visible: true,
      width: '100px',
    }

    const nextCols = [...columns]
    if (insertColIndex !== null && insertColIndex !== undefined && insertColIndex >= 0) {
      nextCols.splice(insertColIndex + 1, 0, newCol)
    } else {
      const taxIdx = nextCols.findIndex(c => c.key === 'taxableValue')
      if (taxIdx > -1) {
        nextCols.splice(taxIdx, 0, newCol)
      } else {
        nextCols.push(newCol)
      }
    }

    setColumns(nextCols)
    setFormData(prev => ({
      ...prev,
      items: prev.items.map(it => ({ ...it, [key]: it[key] !== undefined ? it[key] : '' }))
    }))
    setDirty(true)
    setShowAddColumnModal(false)
    toast.success(`Column "${name}" added!`)
  }

  const removeCustomColumn = (colKey) => {
    const col = columns.find(c => c.key === colKey)
    if (!col || !col.removable) return
    setColumns(prev => prev.filter(c => c.key !== colKey))
    setFormData(prev => ({
      ...prev,
      items: prev.items.map(it => {
        const copy = { ...it }
        delete copy[colKey]
        return copy
      })
    }))
    setDirty(true)
    toast.success(`Removed column "${col.label}"`)
  }

  const updateColumnLabel = (colKey, newLabel) => {
    setColumns(prev => prev.map(c => c.key === colKey ? { ...c, label: newLabel } : c))
    setDirty(true)
  }

  const updateItemTaxRate = (idx, rateField, val) => {
    const rateVal = parseFloat(val) || 0
    const updated = (formData.items || []).map((it, i) => {
      if (i !== idx) return it
      return { ...it, [rateField]: rateVal }
    })
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(updated, formData.totals?.roundOff, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  const resetItemTax = (idx) => {
    const updated = (formData.items || []).map((it, i) => {
      if (i !== idx) return it
      return {
        ...it,
        sgstRate: 0,
        cgstRate: 0,
        igstRate: 0,
        sgstAmount: 0,
        cgstAmount: 0,
        igstAmount: 0,
      }
    })
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(updated, formData.totals?.roundOff, curDisc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
    setActiveTaxRowIndex(null)
  }

  const updateGlobalRate = (type, val) => {
    const rateVal = parseFloat(val) || 0
    const newRates = {
      cgst: type === 'cgst' ? rateVal : (parseFloat(formData.totals?.globalCgstRate) || 0),
      sgst: type === 'sgst' ? rateVal : (parseFloat(formData.totals?.globalSgstRate) || 0),
      igst: type === 'igst' ? rateVal : (parseFloat(formData.totals?.globalIgstRate) || 0),
    }
    const curDisc = parseFloat(formData.totals?.globalDiscount) || 0
    const calc = recalculate(formData.items, formData.totals?.roundOff, curDisc, newRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: curDisc, globalCgstRate: newRates.cgst, globalSgstRate: newRates.sgst, globalIgstRate: newRates.igst },
    }))
    setDirty(true)
  }

  const updateGlobalDiscount = (val) => {
    const disc = parseFloat(val) || 0
    const curRates = {
      cgst: parseFloat(formData.totals?.globalCgstRate) || 0,
      sgst: parseFloat(formData.totals?.globalSgstRate) || 0,
      igst: parseFloat(formData.totals?.globalIgstRate) || 0,
    }
    const calc = recalculate(formData.items, formData.totals?.roundOff, disc, curRates)
    setFormData(prev => ({
      ...prev,
      items: calc.items,
      totals: { ...prev.totals, ...calc.totals, globalDiscount: disc, globalCgstRate: curRates.cgst, globalSgstRate: curRates.sgst, globalIgstRate: curRates.igst },
    }))
    setDirty(true)
  }

  const addCertifiedRemark = (text) => {
    const clean = (text || '').trim()
    if (!clean) return
    setFormData(prev => {
      const current = prev.certifiedRemarks || []
      if (current.includes(clean)) return prev
      return { ...prev, certifiedRemarks: [...current, clean] }
    })
    setNewRemarkText('')
    setDirty(true)
  }

  const removeCertifiedRemark = (idx) => {
    setFormData(prev => ({
      ...prev,
      certifiedRemarks: (prev.certifiedRemarks || []).filter((_, i) => i !== idx)
    }))
    setDirty(true)
  }

  const updateCertifiedRemark = (idx, newText) => {
    setFormData(prev => {
      const next = [...(prev.certifiedRemarks || [])]
      next[idx] = newText
      return { ...prev, certifiedRemarks: next }
    })
    setDirty(true)
  }

  const clearAllCertifiedRemarks = () => {
    setFormData(prev => ({ ...prev, certifiedRemarks: [] }))
    setDirty(true)
  }

  // ── Save & Export Actions ──────────────────────────────────────────────────

  const save = async (asVerified = false) => {
    // Explicit boolean check prevents SyntheticEvent objects from being truthy
    const isTrulyVerified = asVerified === true
    setSaving(true)
    try {
      const payloadData = {
        ...formData,
        columns: columns,
      }
      const { data } = await axios.patch(`/api/invoices/${jobId}`, {
        corrections: payloadData,
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
    if (stepId === 1) return Boolean(formData?.client?.name && formData?.meta?.invoiceNo && formData?.meta?.date)
    if (stepId === 2) return Boolean(formData?.company?.name && formData?.bankDetails?.ifsc && formData?.bankDetails?.accountNumber)
    if (stepId === 3) return (formData?.items?.length || 0) > 0 && (formData?.totals?.grandTotal || 0) > 0
    if (stepId === 4) return true
    return true
  }

  // ── AI Fetched Percentage Calculation ──────────────────────────────────────
  const aiFetchedStats = useMemo(() => {
    const inv = job?.invoice || job?.extracted_invoice || job?.result || {}
    const rawItems = inv?.line_items || inv?.items || []
    
    // Core standard invoice fields extracted by the AI pipeline
    const aiFields = [
      { key: 'invoice_number', val: inv.invoice_number },
      { key: 'category', val: inv.category || inv.subcategory },
      { key: 'invoice_date', val: inv.invoice_date },
      { key: 'due_date', val: inv.due_date },
      { key: 'place_of_supply', val: inv.place_of_supply },
      { key: 'buyer_name', val: inv.buyer_name },
      { key: 'buyer_address', val: inv.buyer_address || inv.buyer_address_line1 },
      { key: 'buyer_phone', val: inv.buyer_phone },
      { key: 'buyer_gstin', val: inv.buyer_gstin },
      { key: 'vendor_name', val: inv.vendor_name },
      { key: 'vendor_address', val: inv.vendor_address || inv.vendor_address_line1 },
      { key: 'vendor_phone', val: inv.vendor_phone },
      { key: 'vendor_email', val: inv.vendor_email },
      { key: 'vendor_gstin', val: inv.vendor_gstin },
      { key: 'vendor_pan', val: inv.vendor_pan },
      { key: 'ifsc_code', val: inv.ifsc_code },
      { key: 'bank_name', val: inv.bank_name },
      { key: 'branch_name', val: inv.branch_name },
      { key: 'account_name', val: inv.account_name },
      { key: 'account_number', val: inv.account_number },
      { key: 'items', val: rawItems.length > 0 && rawItems[0]?.description ? 'ok' : null },
      { key: 'taxable_value', val: (inv.subtotal != null && inv.subtotal !== 0) ? 'ok' : (inv.taxable_amount != null && inv.taxable_amount !== 0) ? 'ok' : null },
      { key: 'grand_total', val: (inv.grand_total != null && inv.grand_total !== 0) ? 'ok' : null },
    ]

    const totalCount = aiFields.length
    const extractedCount = aiFields.filter(f => f.val != null && String(f.val).trim() !== '' && f.val !== '0').length
    const fieldRatio = Math.round((extractedCount / totalCount) * 100)

    let finalPct = fieldRatio
    const conf = job?.overall_confidence
    if (conf != null && conf > 0) {
      const confPct = Math.round(conf * 100)
      finalPct = Math.max(fieldRatio, Math.min(100, Math.round((fieldRatio * 0.6) + (confPct * 0.4))))
    }

    return {
      percentage: Math.max(5, Math.min(100, finalPct)),
      extractedCount,
      totalCount,
      confidence: Math.round((job?.overall_confidence || 0) * 100),
    }
  }, [job])

  // ── Real-Time Form Fill-Up Percentage Calculation ──────────────────────────
  const formFillStats = useMemo(() => {
    const fields = [
      // Client / Buyer
      { name: 'Client Name', val: formData?.client?.name, required: true },
      { name: 'Client Address', val: formData?.client?.addressLine1, required: false },
      { name: 'Client GSTIN', val: formData?.client?.gstin, required: false },
      { name: 'SLS Code', val: formData?.client?.slsCode, required: false },
      // Meta / Header
      { name: 'Invoice No', val: formData?.meta?.invoiceNo, required: true },
      { name: 'Invoice Date', val: formData?.meta?.date, required: true },
      { name: 'Due Date', val: formData?.meta?.dueDate, required: false },
      { name: 'Category', val: formData?.meta?.category, required: true },
      { name: 'Subcategory', val: formData?.meta?.subcategory, required: false },
      { name: 'Place of Supply', val: formData?.meta?.placeOfSupply, required: false },
      // Vendor / Company
      { name: 'Vendor Name', val: formData?.company?.name, required: true },
      { name: 'Vendor Address', val: formData?.company?.addressLine1, required: false },
      { name: 'Vendor Email', val: formData?.company?.email, required: false },
      { name: 'Vendor Phone', val: formData?.company?.phone, required: false },
      { name: 'Vendor GSTIN', val: formData?.company?.gstin, required: false },
      { name: 'Vendor PAN', val: formData?.company?.pan, required: false },
      // Bank
      { name: 'IFSC Code', val: formData?.bankDetails?.ifsc, required: true },
      { name: 'Bank Name', val: formData?.bankDetails?.bankName, required: true },
      { name: 'Branch Name', val: formData?.bankDetails?.branchName, required: false },
      { name: 'Account Name', val: formData?.bankDetails?.accountName, required: true },
      { name: 'Account Number', val: formData?.bankDetails?.accountNumber, required: true },
      // Items & Totals
      { name: 'Line Items', val: formData?.items?.length > 0 && formData?.items?.[0]?.description ? 'ok' : null, required: true },
      { name: 'Grand Total', val: formData?.totals?.grandTotal > 0 ? 'ok' : null, required: true },
    ]

    const totalCount = fields.length
    const filledCount = fields.filter(f => f.val != null && String(f.val).trim() !== '' && f.val !== 0).length
    const percentage = Math.round((filledCount / totalCount) * 100)

    const requiredFields = fields.filter(f => f.required)
    const requiredFilled = requiredFields.filter(f => f.val != null && String(f.val).trim() !== '' && f.val !== 0).length
    const isComplete = requiredFilled === requiredFields.length

    return {
      percentage: Math.max(0, Math.min(100, percentage)),
      filledCount,
      totalCount,
      requiredFilled,
      requiredTotal: requiredFields.length,
      isComplete,
    }
  }, [formData])

  const confidenceScore = Math.round((job?.overall_confidence || 0) * 100)
  const previewImageUrl = `/api/invoices/${jobId}/preview-image?page=${docPage}&t=${dirty ? 'edit' : 'view'}`
  const hasItemLevelTax = (formData?.items || []).some(it => (parseFloat(it?.cgstRate) > 0 || parseFloat(it?.sgstRate) > 0 || parseFloat(it?.igstRate) > 0))

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-slate-400 p-16">
        <RefreshCw size={28} className="spinner text-blue-600 mb-3" />
        <p className="font-bold text-slate-800 dark:text-slate-200 text-sm">Loading Invoice Digitizer...</p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 font-mono">Job: {jobId}</p>
      </div>
    )
  }

  if (loadError || !job) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] p-8 text-center max-w-md mx-auto">
        <div className="w-14 h-14 rounded-2xl bg-red-50 dark:bg-red-950/60 text-red-600 dark:text-red-400 flex items-center justify-center mb-4 border border-red-200 dark:border-red-800 shadow-sm">
          <AlertCircle size={28} />
        </div>
        <h2 className="text-base font-bold text-slate-900 dark:text-white mb-1.5">Invoice Document Not Found</h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-6 leading-relaxed">
          {loadError || `Job ${jobId} does not exist in the database or may have been deleted.`}
        </p>
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/invoices')} className="btn-primary py-2 px-4 text-xs flex items-center gap-1.5 shadow-sm">
            <ArrowLeft size={14} /> Back to Invoices
          </button>
          <button onClick={() => navigate('/')} className="btn-secondary py-2 px-4 text-xs flex items-center gap-1.5">
            <Upload size={14} /> Upload New
          </button>
        </div>
      </div>
    )
  }

  if (job?.status === 'deleted' && !job.invoice && !job.invoice_builder_data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] p-8 text-center max-w-md mx-auto">
        <div className="w-14 h-14 rounded-2xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center mb-4 border border-amber-200 dark:border-amber-800 shadow-sm">
          <Trash2 size={28} />
        </div>
        <h2 className="text-base font-bold text-slate-900 dark:text-white mb-1.5">Invoice Cleared from Queue</h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-6 leading-relaxed">
          This invoice ({job.filename || jobId}) was cleared from the queue and has no saved extraction data.
        </p>
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/invoices')} className="btn-primary py-2 px-4 text-xs flex items-center gap-1.5 shadow-sm">
            <ArrowLeft size={14} /> Back to Invoices
          </button>
          <button onClick={() => navigate('/')} className="btn-secondary py-2 px-4 text-xs flex items-center gap-1.5">
            <Upload size={14} /> Upload Invoice
          </button>
        </div>
      </div>
    )
  }

  const isReady = ['done', 'reviewed', 'partially_reviewed', 'failed'].includes(job.status)
  if (!isReady) {
    return (
      <div className="p-12 text-center text-slate-400">
        <RefreshCw size={28} className="spinner mx-auto mb-3 text-blue-600" />
        <p className="font-bold text-slate-800 dark:text-slate-200 text-base">Processing Invoice Intelligence...</p>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Extracting layout, OCR text, and financial entities. This page will update automatically.</p>
        <button onClick={() => navigate('/invoices')} className="btn-secondary mt-6 py-1.5 px-3 text-xs mx-auto flex items-center gap-1.5">
          <ArrowLeft size={13} /> Back to Invoices
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      
      {/* ── Top App Bar ──────────────────────────────────────────────────────── */}
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-4 md:px-6 py-3 flex items-center justify-between gap-4 flex-shrink-0 sticky top-0 z-30 shadow-xs">
        <div className="flex items-center gap-3 min-w-0">
          <button onClick={() => navigate('/invoices')} className="btn-secondary py-1.5 px-3">
            <ArrowLeft size={14} /> <span className="hidden sm:inline">Back</span>
          </button>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-sm font-bold text-slate-900 dark:text-white truncate max-w-xs md:max-w-md">{job.filename}</h1>
              <span
                className={`badge ${
                  job.status === 'reviewed'
                    ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                    : job.status === 'partially_reviewed'
                    ? 'bg-amber-50 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                    : 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                }`}
              >
                {job.status === 'reviewed'
                  ? '✓ Verified'
                  : job.status === 'partially_reviewed'
                  ? '⏳ Partially Reviewed'
                  : '🤖 AI Extracted'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 font-mono">Job ID: {jobId.slice(0, 12)}</p>
          </div>
        </div>

        {/* Center: View Switcher */}
        <div className="hidden lg:flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl border border-slate-200 dark:border-slate-700">
          <button
            type="button"
            onClick={() => setIsWizardMode(false)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              !isWizardMode
                ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <LayoutGrid size={13} />
            Standard View
          </button>
          <button
            type="button"
            onClick={() => setIsWizardMode(true)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              isWizardMode
                ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Sparkles size={13} className="text-amber-500" />
            🧙 Guided Wizard
          </button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          {/* Tab buttons */}
          <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1 gap-1 border border-slate-200 dark:border-slate-700">
            <button
              onClick={() => setActiveTab('edit')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                activeTab === 'edit'
                  ? 'bg-white dark:bg-slate-900 shadow-sm text-slate-900 dark:text-white font-bold'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
              }`}
            >
              Form
            </button>
            <button
              onClick={() => setActiveTab('split')}
              className={`hidden md:flex items-center gap-1 px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                activeTab === 'split'
                  ? 'bg-white dark:bg-slate-900 shadow-sm text-slate-900 dark:text-white font-bold'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
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
                activeTab === 'preview'
                  ? 'bg-white dark:bg-slate-900 shadow-sm text-slate-900 dark:text-white font-bold'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
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
                ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700 cursor-not-allowed opacity-85'
                : 'text-amber-800 dark:text-amber-300 border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 hover:bg-amber-100 dark:hover:bg-amber-900/50'
            }`}
            title="Re-scan this invoice with the latest AI pipeline"
          >
            {isRescanning || job?.status === 'processing' ? (
              <>
                <RefreshCw size={13} className="spinner text-amber-700 dark:text-amber-400" />
                <span className="hidden lg:inline font-semibold">Re-scanning...</span>
              </>
            ) : (
              <>
                <Sparkles size={13} className="text-amber-600 dark:text-amber-400" />
                <span className="hidden lg:inline">Re-scan</span>
              </>
            )}
          </button>

          {/* Save Draft (Partial Review) */}
          <button
            onClick={() => save(false)}
            disabled={saving}
            className="btn-secondary py-1.5 px-3 text-xs border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 hover:bg-amber-100 dark:hover:bg-amber-900/50 text-amber-900 dark:text-amber-300 shadow-xs"
            title="Save changes as a draft without marking as verified ground truth"
          >
            {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700 dark:text-amber-400" />}
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
        <div className="bg-slate-900 dark:bg-slate-950 text-white px-6 py-2.5 flex flex-col md:flex-row md:items-center justify-between text-xs shadow-lg sticky top-[57px] z-20 border-b border-amber-500/40 gap-2.5 animate-fade-in">
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

      {/* ── Fast-Review Active Learning Focus Banner ────────────────────────── */}
      {job && (
        <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white px-4 md:px-6 py-2.5 flex flex-wrap items-center justify-between gap-3 border-b border-indigo-500/30 text-xs shadow-md">
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold flex items-center gap-1">
              <Sparkles size={12} /> Active Learning Review
            </span>
            {job.template_id && (
              <span className="text-[11px] text-slate-300 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700 font-mono">
                Template: {job.template_id} {job.is_novel_template ? '🔥 (Novel Layout)' : '✓ (Known)'}
              </span>
            )}
            {job.review_reasons && job.review_reasons.length > 0 ? (
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-amber-400 font-semibold flex items-center gap-1">
                  <AlertTriangle size={12} /> Needs Attention:
                </span>
                {job.review_reasons.slice(0, 3).map((r, idx) => (
                  <span key={idx} className="bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded text-[11px] font-medium">
                    {r}
                  </span>
                ))}
              </div>
            ) : (
              <span className="text-emerald-400 font-medium flex items-center gap-1">
                <CheckCircle size={12} /> 100% High Confidence & Verified Math
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => save(true)}
              disabled={saving}
              className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-semibold rounded-lg shadow transition-all flex items-center gap-1.5 text-xs"
              title="Accept high confidence fields and mark as verified ground truth"
            >
              <CheckCircle size={12} /> Fast Accept Ground Truth
            </button>
          </div>
        </div>
      )}

      {/* ── Retrain Modal ────────────────────────────────────────────────────── */}
      {showTrainModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="card max-w-md w-full p-6 shadow-xl bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 animate-pop-in">
            <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 text-slate-900 dark:text-white font-bold text-sm">
                <Cpu size={18} className="text-blue-600" />
                <span>Trigger AI Model Retraining</span>
              </div>
              <button onClick={() => setShowTrainModal(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                <X size={16} />
              </button>
            </div>

            <p className="text-xs text-slate-600 dark:text-slate-400 mb-4">
              Retrain your dedicated machine learning models directly using the verified invoices saved in your database.
            </p>

            <div className="space-y-3">
              <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-900 dark:text-white">YOLOv8 Region Detector</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">Learns visual bounding boxes for Header, Vendor, Items table, Totals, Bank details.</p>
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

              <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-900 dark:text-white">LayoutLMv3 Entity Classifier</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">Fine-tunes token extraction on saved ground-truth fields.</p>
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
          </div>
        </div>
      )}

      {/* ── Main Content Area ────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-auto p-3 md:p-5 lg:p-6">
        <div className="w-full max-w-[1920px] mx-auto">
          
          {/* Fullscreen Document Tab */}
          {activeTab === 'preview' ? (
            <div className="card p-3 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 shadow-sm flex flex-col h-[calc(100vh-120px)]">
              <div className="flex flex-wrap justify-between items-center gap-2 mb-2 pb-2 border-b border-slate-100 dark:border-slate-800">
                <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
                  <button
                    onClick={() => setDocSource('original')}
                    className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                      docSource === 'original' ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold' : 'text-slate-600 dark:text-slate-400'
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
                      docSource === 'rendered' ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold' : 'text-slate-600 dark:text-slate-400'
                    }`}
                  >
                    🖨️ Standard Rendered
                  </button>
                </div>

                {docSource === 'original' && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    {docTotalPages > 1 && (
                      <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-lg text-xs font-semibold text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
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

                    <div className="flex items-center gap-0.5 bg-slate-100 dark:bg-slate-800 p-0.5 rounded-lg border border-slate-200 dark:border-slate-700">
                      <button
                        onClick={() => setZoomLevel(z => Math.max(30, z - 25))}
                        className="p-1 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                        title="Zoom Out"
                      >
                        <ZoomOut size={14} />
                      </button>
                      <span className="text-xs font-mono font-semibold px-1 text-slate-700 dark:text-slate-300 min-w-[42px] text-center">{zoomLevel}%</span>
                      <button
                        onClick={() => setZoomLevel(z => Math.min(400, z + 25))}
                        className="p-1 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                        title="Zoom In"
                      >
                        <ZoomIn size={14} />
                      </button>

                      <div className="w-[1px] h-3.5 bg-slate-300 dark:bg-slate-600 mx-0.5" />

                      <button
                        onClick={fitWidth}
                        className="px-2 py-0.5 text-[11px] font-semibold text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                        title="Fit Width"
                      >
                        Fit W
                      </button>
                      <button
                        onClick={fitHeight}
                        className="px-2 py-0.5 text-[11px] font-semibold text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                        title="Fit Height"
                      >
                        Fit H
                      </button>
                      <button
                        onClick={resetView}
                        className="p-1 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                        title="Reset Pan (0,0), Zoom (100%) & Rotation (0°)"
                      >
                        <RotateCcw size={12} />
                      </button>

                      <div className="w-[1px] h-3.5 bg-slate-300 dark:bg-slate-600 mx-0.5" />

                      <button
                        onClick={() => setRotation(r => (r + 90) % 360)}
                        className={`p-1 rounded text-slate-600 dark:text-slate-300 hover:text-blue-600 hover:bg-white dark:hover:bg-slate-700 flex items-center gap-1 transition-all ${
                          rotation !== 0 ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950 font-bold' : ''
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
                      className="btn-secondary py-1 px-2.5 text-xs text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
                      title="Open raw file in new tab"
                    >
                      <ExternalLink size={12} /> Raw
                    </a>
                  </div>
                )}

                <button onClick={() => setActiveTab('split')} className="btn-secondary text-xs text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800">
                  Back to Split View
                </button>
              </div>

              {/* Canvas with Pan & Dragging */}
              <div
                onMouseDown={handleViewerMouseDown}
                onMouseMove={handleViewerMouseMove}
                onMouseUp={handleViewerMouseUp}
                onMouseLeave={handleViewerMouseUp}
                onWheel={handleViewerWheel}
                className={`flex-1 w-full rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 dark:bg-slate-950 relative shadow-inner select-none ${
                  isDragging ? 'cursor-grabbing' : 'cursor-grab'
                }`}
              >
                {docSource === 'original' ? (
                  <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
                    <img
                      src={previewImageUrl}
                      alt="Original Scanned Invoice"
                      draggable={false}
                      style={{
                        width: `${zoomLevel}%`,
                        minWidth: `${zoomLevel}%`,
                        maxWidth: 'none',
                        transform: `translate3d(${pan.x}px, ${pan.y}px, 0) rotate(${rotation}deg)`,
                        transformOrigin: 'center center',
                        transition: isDragging ? 'none' : 'transform 0.12s cubic-bezier(0.4, 0, 0.2, 1), width 0.1s ease-out',
                        cursor: isDragging ? 'grabbing' : 'grab',
                        userSelect: 'none',
                      }}
                      className="rounded-lg shadow-2xl bg-white border border-slate-800 select-none block pointer-events-auto"
                    />

                    {/* Floating Pan & Zoom Hint */}
                    <div className="absolute bottom-3 right-3 bg-slate-900/85 backdrop-blur-xs text-[11px] text-slate-300 font-medium px-2.5 py-1 rounded-md border border-slate-700 pointer-events-none opacity-70">
                      🖐️ Drag to move • Ctrl+Scroll to zoom
                    </div>
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
            <div className="space-y-4">
              {/* ── Intelligence & Form Completion Summary Bar (Full Width across Top) ── */}
              <div className="card p-3 bg-white/95 dark:bg-slate-900/95 border-slate-200 dark:border-slate-800 shadow-xs flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-4 flex-wrap">
                  {/* AI Extraction Metric */}
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 flex items-center justify-center border border-blue-100 dark:border-blue-900/60 shrink-0">
                      <Sparkles size={15} />
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] font-bold text-slate-700 dark:text-slate-300">AI Fetched</span>
                        <span className="text-xs font-bold text-blue-600 dark:text-blue-400 font-mono">{aiFetchedStats.percentage}%</span>
                      </div>
                      <p className="text-[10px] text-slate-400 dark:text-slate-500">
                        {aiFetchedStats.extractedCount}/{aiFetchedStats.totalCount} fields detected {aiFetchedStats.confidence > 0 && `• ${aiFetchedStats.confidence}% conf`}
                      </p>
                    </div>
                  </div>

                  <div className="hidden sm:block w-[1px] h-7 bg-slate-200 dark:bg-slate-800" />

                  {/* Form Fill-up Metric */}
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center border shrink-0 ${
                      formFillStats.percentage >= 90
                        ? 'bg-emerald-50 dark:bg-emerald-950/80 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
                        : 'bg-amber-50 dark:bg-amber-950/80 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800'
                    }`}>
                      <CheckCircle size={15} />
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] font-bold text-slate-700 dark:text-slate-300">Form Fill-up</span>
                        <span className={`text-xs font-bold font-mono ${
                          formFillStats.percentage >= 90 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'
                        }`}>{formFillStats.percentage}%</span>
                      </div>
                      <p className="text-[10px] text-slate-400 dark:text-slate-500">
                        {formFillStats.filledCount}/{formFillStats.totalCount} fields completed • {formFillStats.requiredFilled}/{formFillStats.requiredTotal} required
                      </p>
                    </div>
                  </div>
                </div>

                {/* Right: Verification status badge */}
                <div className="flex items-center gap-2">
                  <span className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border ${
                    job.status === 'reviewed'
                      ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                      : job.status === 'partially_reviewed'
                      ? 'bg-amber-50 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                      : 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                  }`}>
                    {job.status === 'reviewed' ? '✓ Verified Ground Truth' : job.status === 'partially_reviewed' ? '⏳ Draft Progress' : '🤖 AI Extracted Data'}
                  </span>
                </div>
              </div>

              <div className={`grid ${activeTab === 'split' ? 'grid-cols-1 xl:grid-cols-12 2xl:grid-cols-12 gap-5' : 'grid-cols-1'}`}>
                
                {/* Form Column */}
                <div className={`${activeTab === 'split' ? 'xl:col-span-7 2xl:col-span-7' : 'w-full'} space-y-4`}>

                {/* Guided Stepper Header (When in Wizard Mode) */}
                {isWizardMode && (
                  <div className="card p-4 bg-white/95 dark:bg-slate-900/95 border-slate-200 dark:border-slate-800 shadow-sm mb-2">
                    <div className="flex items-center justify-between gap-3 mb-3 px-1">
                      <div className="flex items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-100 dark:border-blue-900/80">
                          Step {currentStep} of {WIZARD_STEPS.length}
                        </span>
                        <h2 className="text-sm md:text-base font-bold text-slate-900 dark:text-white leading-none">
                          {WIZARD_STEPS[currentStep - 1]?.title || 'Review Step'}
                        </h2>
                        <span className="text-slate-300 dark:text-slate-600 hidden sm:inline">&bull;</span>
                        <span className="text-xs text-slate-400 dark:text-slate-400 hidden sm:inline">
                          {WIZARD_STEPS[currentStep - 1]?.subtitle || ''}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <div className="w-20 md:w-28 bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-600 to-teal-400 h-full rounded-full transition-all duration-500"
                            style={{ width: `${(currentStep / WIZARD_STEPS.length) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-teal-600 dark:text-teal-400 min-w-[32px] text-right">
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
                            <span className="text-[11px] font-semibold mt-1.5 text-center leading-tight tracking-tight text-slate-600 dark:text-slate-400 group-hover:text-slate-900 dark:group-hover:text-white transition-colors max-w-[90px]">
                              {step.title.split('&')[0].trim()}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* ── STEP 1: BILL TO & INVOICE DETAILS ─────────────────────── */}
                {(!isWizardMode || currentStep === 1) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 fade-in">
                    
                    {/* Bill To (Client) */}
                    <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs">
                      <div className="flex items-center gap-2 pb-2 mb-2.5 border-b border-slate-100 dark:border-slate-800">
                        <Building size={15} className="text-blue-600 dark:text-blue-400" />
                        <h3 className="text-[11px] font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Bill To (Client)</h3>
                      </div>
                      <div className="space-y-2.5 text-xs">
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">SLS Code / Scheme</label>
                          <input
                            type="text"
                            className="field-input text-xs font-mono py-1.5"
                            placeholder="e.g. WB-SLS-2024"
                            value={formData.client.slsCode}
                            onChange={e => updateSection('client', 'slsCode', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Client / Agency Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-semibold text-slate-900 dark:text-white py-1.5"
                            placeholder="Recipient Agency Name"
                            value={formData.client.name}
                            onChange={e => updateSection('client', 'name', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Address Line 1</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="Building, Street, Area"
                            value={formData.client.addressLine1}
                            onChange={e => updateSection('client', 'addressLine1', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Address Line 2</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="City, District, State, PIN"
                            value={formData.client.addressLine2}
                            onChange={e => updateSection('client', 'addressLine2', e.target.value)}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Phone</label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="+91 9876543210"
                              value={formData.client.phone}
                              onChange={e => updateSection('client', 'phone', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">GSTIN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono py-1.5"
                              placeholder="19AAAAA0000A1Z5"
                              value={formData.client.gstin}
                              onChange={e => updateSection('client', 'gstin', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Invoice Meta */}
                    <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs">
                      <div className="flex items-center gap-2 pb-2 mb-2.5 border-b border-slate-100 dark:border-slate-800">
                        <FileText size={15} className="text-blue-600 dark:text-blue-400" />
                        <h3 className="text-[11px] font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Invoice Details</h3>
                      </div>
                      <div className="space-y-2.5 text-xs">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Category <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="e.g. Services / Goods"
                              value={formData.meta.category}
                              onChange={e => updateSection('meta', 'category', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Subcategory <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="e.g. IT & Software"
                              value={formData.meta.subcategory}
                              onChange={e => updateSection('meta', 'subcategory', e.target.value)}
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Invoice Number <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs font-bold font-mono text-blue-700 dark:text-blue-400 py-1.5"
                              placeholder="INV-2024-001"
                              value={formData.meta.invoiceNo}
                              onChange={e => updateSection('meta', 'invoiceNo', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">PO / Work Order No.</label>
                            <input
                              type="text"
                              className="field-input text-xs font-mono py-1.5"
                              placeholder="PO-2024-001"
                              value={formData.meta.poNumber || ''}
                              onChange={e => updateSection('meta', 'poNumber', e.target.value)}
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Invoice Date <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="DD/MM/YYYY"
                              value={formData.meta.date}
                              onChange={e => updateSection('meta', 'date', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Due Date</label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="DD/MM/YYYY"
                              value={formData.meta.dueDate}
                              onChange={e => updateSection('meta', 'dueDate', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Place of Supply (State)</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="19-West Bengal"
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
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 fade-in">
                    
                    {/* Biller Details */}
                    <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs">
                      <div className="flex items-center gap-2 pb-2 mb-2.5 border-b border-slate-100 dark:border-slate-800">
                        <Building size={15} className="text-blue-600 dark:text-blue-400" />
                        <h3 className="text-[11px] font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Biller Details (Vendor)</h3>
                      </div>
                      <div className="space-y-2.5 text-xs">
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Vendor / Company Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-bold text-slate-900 dark:text-white py-1.5"
                            placeholder="Vendor Registered Name (Max 40 chars)"
                            value={formData.company.name}
                            onChange={e => updateSection('company', 'name', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Address Line 1</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="Address Line 1"
                            value={formData.company.addressLine1}
                            onChange={e => updateSection('company', 'addressLine1', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Address Line 2</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="Address Line 2"
                            value={formData.company.addressLine2}
                            onChange={e => updateSection('company', 'addressLine2', e.target.value)}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Email</label>
                            <input
                              type="email"
                              className="field-input text-xs py-1.5"
                              placeholder="vendor@company.com"
                              value={formData.company.email}
                              onChange={e => updateSection('company', 'email', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Phone</label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="+91 9876543210"
                              value={formData.company.phone}
                              onChange={e => updateSection('company', 'phone', e.target.value)}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">GSTIN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono py-1.5"
                              placeholder="19AAAAA0000A1Z5"
                              value={formData.company.gstin}
                              onChange={e => updateSection('company', 'gstin', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">PAN</label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono py-1.5"
                              placeholder="ABCDE1234F"
                              value={formData.company.pan}
                              onChange={e => updateSection('company', 'pan', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Bank Details */}
                    <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs">
                      <div className="flex items-center gap-2 pb-2 mb-2.5 border-b border-slate-100 dark:border-slate-800">
                        <Landmark size={15} className="text-blue-600 dark:text-blue-400" />
                        <h3 className="text-[11px] font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Bank Details</h3>
                      </div>
                      <div className="space-y-2.5 text-xs">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">IFSC Code <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs uppercase font-mono font-semibold text-blue-700 dark:text-blue-400 py-1.5"
                              placeholder="SBIN0001234"
                              value={formData.bankDetails.ifsc}
                              onChange={e => updateSection('bankDetails', 'ifsc', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Branch Name <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs py-1.5"
                              placeholder="Kolkata Main Branch"
                              value={formData.bankDetails.branchName}
                              onChange={e => updateSection('bankDetails', 'branchName', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Bank Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="State Bank of India"
                            value={formData.bankDetails.bankName}
                            onChange={e => updateSection('bankDetails', 'bankName', e.target.value)}
                          />
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Account Beneficiary Name <span className="text-red-500">*</span></label>
                          <input
                            type="text"
                            className="field-input text-xs font-semibold py-1.5"
                            placeholder="Vendor Company Name"
                            value={formData.bankDetails.accountName}
                            onChange={e => updateSection('bankDetails', 'accountName', e.target.value)}
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Account Number <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs font-mono font-semibold py-1.5"
                              placeholder="123456789012"
                              value={formData.bankDetails.accountNumber}
                              onChange={e => updateSection('bankDetails', 'accountNumber', e.target.value)}
                            />
                          </div>
                          <div>
                            <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Confirm Account No <span className="text-red-500">*</span></label>
                            <input
                              type="text"
                              className="field-input text-xs font-mono font-semibold py-1.5"
                              placeholder="123456789012"
                              value={formData.bankDetails.confirmAccountNumber || formData.bankDetails.accountNumber}
                              onChange={e => updateSection('bankDetails', 'confirmAccountNumber', e.target.value)}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block font-medium text-slate-600 dark:text-slate-400 mb-1">Payment Terms</label>
                          <input
                            type="text"
                            className="field-input text-xs py-1.5"
                            placeholder="e.g. Net 30 / Due on receipt / Immediate"
                            value={formData.bankDetails.paymentTerms || ''}
                            onChange={e => updateSection('bankDetails', 'paymentTerms', e.target.value)}
                          />
                        </div>
                      </div>
                    </div>

                  </div>
                )}

                {/* ── STEP 3: ITEMS & TOTALS ────────────────────────────────── */}
                {(!isWizardMode || currentStep === 3) && (
                  <div className="space-y-5 fade-in">
                    
                    {/* Line Items Table */}
                    <div className="card p-5 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
                      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100 dark:border-slate-800">
                        <div className="flex items-center gap-2">
                          <ShoppingBag size={16} className="text-blue-600 dark:text-blue-400" />
                          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Line Items Table</h3>
                          <span className="text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-mono px-2 py-0.5 rounded-full border border-slate-200 dark:border-slate-700">
                            {formData.items.length} {formData.items.length === 1 ? 'item' : 'items'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => openAddColumnModal()}
                            className="btn-secondary py-1 px-2.5 text-xs text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-1"
                            title="Add a custom column to the line items table"
                          >
                            <Plus size={12} className="text-blue-600 dark:text-blue-400" /> Add Column
                          </button>
                          <button
                            type="button"
                            onClick={addItem}
                            className="btn-secondary py-1 px-2.5 text-xs text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800 hover:bg-blue-50 dark:hover:bg-blue-950/40 flex items-center gap-1"
                          >
                            <Plus size={12} /> Add Item Row
                          </button>
                        </div>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-semibold bg-slate-50/70 dark:bg-slate-800/50">
                              <th className="py-2.5 px-2 text-center w-8">#</th>
                              
                              {/* Dynamic Column Headers (Parity with Angular Invoice Builder) */}
                              {columns.map((col, colIdx) => (
                                <th
                                  key={col.key}
                                  className={`py-2 px-1.5 group relative align-middle ${
                                    col.type === 'number' || col.type === 'calc'
                                      ? 'text-right'
                                      : col.key === 'hsnSac' || col.key === 'unit'
                                      ? 'text-center'
                                      : 'text-left'
                                  }`}
                                  style={{ minWidth: col.minWidth || col.width || '80px', width: col.width }}
                                >
                                  <div className="flex items-center">
                                    <input
                                      type="text"
                                      value={col.label}
                                      onChange={e => updateColumnLabel(col.key, e.target.value)}
                                      className="bg-transparent border-transparent hover:border-slate-300 dark:hover:border-slate-700 focus:border-blue-500 focus:bg-white dark:focus:bg-slate-800 rounded px-1 w-full font-bold text-slate-700 dark:text-slate-200 outline-none transition-colors text-xs"
                                      style={{
                                        textAlign:
                                          col.type === 'number' || col.type === 'calc'
                                            ? 'right'
                                            : col.key === 'hsnSac' || col.key === 'unit'
                                            ? 'center'
                                            : 'left'
                                      }}
                                    />
                                  </div>

                                  {/* Column Action Controls (Hover Toolbar inside TH) */}
                                  <div className="absolute top-1 right-1 hidden group-hover:flex items-center gap-0.5 bg-white dark:bg-slate-800 rounded shadow-md border border-slate-200 dark:border-slate-700 p-0.5 z-20">
                                    <button
                                      type="button"
                                      onClick={() => openAddColumnModal(colIdx)}
                                      className="p-0.5 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-slate-700 rounded text-[10px]"
                                      title="Add Column After This"
                                    >
                                      <Plus size={11} />
                                    </button>
                                    {col.removable && (
                                      <button
                                        type="button"
                                        onClick={() => removeCustomColumn(col.key)}
                                        className="p-0.5 text-red-500 hover:bg-red-50 dark:hover:bg-slate-700 rounded text-[10px]"
                                        title="Remove Column"
                                      >
                                        <X size={11} />
                                      </button>
                                    )}
                                  </div>
                                </th>
                              ))}

                              <th className="w-8"></th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                            {formData.items.map((item, idx) => {
                              const itemTaxTotalRate = (item.sgstRate || 0) + (item.cgstRate || 0) + (item.igstRate || 0)
                              return (
                              <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/40 transition-colors group">
                                <td className="py-2 px-1 text-center font-mono text-slate-400 select-none">{idx + 1}</td>

                                {/* Dynamic Column Cells */}
                                {columns.map(col => {
                                  if (col.key === 'description') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="text"
                                          className="field-input-sm w-full font-medium"
                                          placeholder="Item description"
                                          value={item.description || ''}
                                          onChange={e => updateItem(idx, 'description', e.target.value)}
                                        />
                                      </td>
                                    )
                                  }

                                  if (col.key === 'hsnSac') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="text"
                                          className="field-input-sm w-full font-mono text-center"
                                          placeholder="HSN/SAC"
                                          value={item.hsnSac || ''}
                                          onChange={e => updateItem(idx, 'hsnSac', e.target.value)}
                                        />
                                      </td>
                                    )
                                  }

                                  if (col.key === 'quantity') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="number"
                                          min="0"
                                          className="field-input-sm w-full text-right font-mono"
                                          value={item.quantity}
                                          onChange={e => updateItem(idx, 'quantity', e.target.value)}
                                        />
                                      </td>
                                    )
                                  }

                                  {/* Unit: Text input with autocomplete suggestions (Exact parity with Invoice Builder) */}
                                  if (col.key === 'unit') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="text"
                                          list="unit-suggestions"
                                          className="field-input-sm w-full font-mono uppercase text-center"
                                          placeholder="NOS"
                                          value={item.unit || ''}
                                          onChange={e => updateItem(idx, 'unit', e.target.value.toUpperCase())}
                                        />
                                      </td>
                                    )
                                  }

                                  if (col.key === 'rate') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="number"
                                          min="0"
                                          className="field-input-sm w-full text-right font-mono"
                                          value={item.rate}
                                          onChange={e => updateItem(idx, 'rate', e.target.value)}
                                        />
                                      </td>
                                    )
                                  }

                                  if (col.key === 'discount') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-1">
                                        <input
                                          type="number"
                                          min="0"
                                          className="field-input-sm w-full text-right font-mono text-slate-500 dark:text-slate-400"
                                          value={item.discount}
                                          onChange={e => updateItem(idx, 'discount', e.target.value)}
                                        />
                                      </td>
                                    )
                                  }

                                  if (col.key === 'taxableValue') {
                                    return (
                                      <td key={col.key} className="py-1.5 px-2 text-right relative">
                                        <div className="flex items-center justify-end gap-1.5">
                                          <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                                            ₹{formatAmount(item.taxableValue)}
                                          </span>
                                          <button
                                            type="button"
                                            onClick={() => setActiveTaxRowIndex(activeTaxRowIndex === idx ? null : idx)}
                                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded border transition-colors ${
                                              itemTaxTotalRate > 0
                                                ? 'bg-blue-600 text-white border-blue-700 dark:bg-blue-600 dark:border-blue-500 shadow-xs'
                                                : 'bg-blue-50 text-blue-700 dark:bg-blue-950/80 dark:text-blue-300 border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900'
                                            }`}
                                            title="Edit item tax rates"
                                          >
                                            {itemTaxTotalRate > 0 ? `${itemTaxTotalRate}% Tax` : '+ Tax'}
                                          </button>
                                        </div>

                                        {/* Tax Popup */}
                                        {activeTaxRowIndex === idx && (
                                          <div className="absolute top-full right-0 mt-1 bg-white dark:bg-slate-800 rounded-lg shadow-2xl border border-slate-200 dark:border-slate-700 p-3.5 z-50 w-56 text-left animate-pop-in">
                                            <div className="flex justify-between items-center mb-2 pb-1 border-b border-slate-100 dark:border-slate-700">
                                              <span className="font-bold text-xs text-slate-700 dark:text-slate-200">Tax Rates (%)</span>
                                              <button onClick={() => setActiveTaxRowIndex(null)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                                                <X size={12} />
                                              </button>
                                            </div>

                                            <div className="space-y-2 text-xs">
                                              <div className="flex items-center justify-between">
                                                <label className="text-slate-600 dark:text-slate-400 font-medium">SGST %</label>
                                                <input
                                                  type="number"
                                                  className="field-input-sm w-20 text-right font-mono"
                                                  value={item.sgstRate}
                                                  onChange={e => updateItemTaxRate(idx, 'sgstRate', e.target.value)}
                                                />
                                              </div>
                                              <div className="flex items-center justify-between">
                                                <label className="text-slate-600 dark:text-slate-400 font-medium">CGST %</label>
                                                <input
                                                  type="number"
                                                  className="field-input-sm w-20 text-right font-mono"
                                                  value={item.cgstRate}
                                                  onChange={e => updateItemTaxRate(idx, 'cgstRate', e.target.value)}
                                                />
                                              </div>
                                              <div className="flex items-center justify-between">
                                                <label className="text-slate-600 dark:text-slate-400 font-medium">IGST %</label>
                                                <input
                                                  type="number"
                                                  className="field-input-sm w-20 text-right font-mono"
                                                  value={item.igstRate}
                                                  onChange={e => updateItemTaxRate(idx, 'igstRate', e.target.value)}
                                                />
                                              </div>
                                            </div>

                                            <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-700 flex justify-between">
                                              <button
                                                type="button"
                                                onClick={() => resetItemTax(idx)}
                                                className="text-[11px] text-red-500 hover:underline font-semibold"
                                              >
                                                Reset
                                              </button>
                                              <button
                                                type="button"
                                                onClick={() => setActiveTaxRowIndex(null)}
                                                className="btn-primary py-0.5 px-2.5 text-[11px]"
                                              >
                                                Done
                                              </button>
                                            </div>
                                          </div>
                                        )}
                                      </td>
                                    )
                                  }

                                  {/* Custom Added Columns */}
                                  return (
                                    <td key={col.key} className="py-1.5 px-1">
                                      {col.type === 'date' ? (
                                        <input
                                          type="date"
                                          className="field-input-sm w-full font-mono text-center"
                                          value={item[col.key] || ''}
                                          onChange={e => updateItem(idx, col.key, e.target.value)}
                                        />
                                      ) : col.type === 'number' ? (
                                        <input
                                          type="number"
                                          className="field-input-sm w-full text-right font-mono"
                                          placeholder="0"
                                          value={item[col.key] || ''}
                                          onChange={e => updateItem(idx, col.key, e.target.value)}
                                        />
                                      ) : (
                                        <input
                                          type="text"
                                          className="field-input-sm w-full font-medium"
                                          placeholder={col.label}
                                          value={item[col.key] || ''}
                                          onChange={e => updateItem(idx, col.key, e.target.value)}
                                        />
                                      )}
                                    </td>
                                  )
                                })}

                                <td className="py-1.5 pl-1 text-center">
                                  <button
                                    type="button"
                                    onClick={() => removeItem(idx)}
                                    className="p-1 text-slate-300 dark:text-slate-600 hover:text-red-500 transition-colors"
                                    title="Delete line item"
                                  >
                                    <Trash2 size={13} />
                                  </button>
                                </td>
                              </tr>
                            )})}
                          </tbody>
                        </table>
                      </div>

                      {/* Common Unit Suggestions Datalist */}
                      <datalist id="unit-suggestions">
                        <option value="NOS" />
                        <option value="PCS" />
                        <option value="KG" />
                        <option value="MTR" />
                        <option value="SET" />
                        <option value="LTR" />
                        <option value="BOX" />
                        <option value="PKT" />
                        <option value="BAGS" />
                        <option value="SQM" />
                        <option value="MONTH" />
                        <option value="HOURS" />
                        <option value="DAYS" />
                        <option value="JOB" />
                        <option value="LOT" />
                        <option value="LUMPSUM" />
                        <option value="MT" />
                        <option value="TON" />
                        <option value="PAIR" />
                        <option value="ROLL" />
                        <option value="DOZ" />
                      </datalist>
                    </div>

                    {/* Footer Split: Tax Breakdown & Financial Totals (Matching e-invoice/ui) */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      
                      {/* Left: Tax Breakdown & Amount in Words */}
                      <div className="lg:col-span-7 space-y-4">
                        {hasItemLevelTax && (
                          <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
                            <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                              Tax Breakdown (HSN / SAC Wise)
                            </h4>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-semibold bg-slate-50/70 dark:bg-slate-800/40">
                                    <th className="py-2 px-1 text-center w-6">#</th>
                                    <th className="py-2 px-1 text-left">Description</th>
                                    <th className="py-2 px-1 text-center w-16">HSN</th>
                                    <th className="py-2 px-1 text-right w-20">Taxable</th>
                                    <th className="py-2 px-1 text-center w-16">SGST</th>
                                    <th className="py-2 px-1 text-center w-16">CGST</th>
                                    <th className="py-2 px-1 text-center w-16">IGST</th>
                                    <th className="py-2 px-1 text-right w-20">Total Tax</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                                  {formData.items.map((it, idx) => {
                                    const totalTax = (it.cgstAmount || 0) + (it.sgstAmount || 0) + (it.igstAmount || 0)
                                    return (
                                      <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30">
                                        <td className="py-1.5 px-1 text-center text-slate-400 font-mono">{idx + 1}</td>
                                        <td className="py-1.5 px-1 font-medium text-slate-700 dark:text-slate-300 truncate max-w-[120px]">{it.description || '—'}</td>
                                        <td className="py-1.5 px-1 text-center font-mono text-slate-500">{it.hsnSac || '—'}</td>
                                        <td className="py-1.5 px-1 text-right font-mono font-bold">₹{formatAmount(it.taxableValue)}</td>
                                        <td className="py-1.5 px-1 text-center font-mono">{it.sgstRate || 0}% (₹{formatAmount(it.sgstAmount)})</td>
                                        <td className="py-1.5 px-1 text-center font-mono">{it.cgstRate || 0}% (₹{formatAmount(it.cgstAmount)})</td>
                                        <td className="py-1.5 px-1 text-center font-mono">{it.igstRate || 0}% (₹{formatAmount(it.igstAmount)})</td>
                                        <td className="py-1.5 px-1 text-right font-mono font-bold text-slate-800 dark:text-slate-200">
                                          ₹{formatAmount(totalTax)}
                                          {formData.totals.globalDiscount > 0 && (
                                            <span className="block text-[10px] text-slate-400 font-normal">
                                              (After Disc ₹{formatAmount((it.cgstAmount || 0) + (it.sgstAmount || 0) + (it.igstAmount || 0))})
                                            </span>
                                          )}
                                        </td>
                                      </tr>
                                    )
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Amount in Words */}
                        <div className="card p-4 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
                          <span className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider block mb-1">
                            Amount in Words
                          </span>
                          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700 font-semibold text-blue-900 dark:text-blue-300 text-xs italic">
                            {formData.totals?.amountInWords || 'Zero Rupees Only'}
                          </div>
                        </div>
                      </div>

                      {/* Right: Totals Container */}
                      <div className="lg:col-span-5 space-y-4">
                        <div className="card p-5 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 space-y-2.5 text-xs">
                          
                          {/* Global Tax Rates (Only visible if no item level tax is set) */}
                          {!hasItemLevelTax && (
                            <div className="p-3 mb-3 bg-slate-50 dark:bg-slate-800/70 rounded-lg border border-dashed border-slate-300 dark:border-slate-700">
                              <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
                                Global Tax Rates (%)
                              </span>
                              <div className="grid grid-cols-3 gap-2">
                                <div>
                                  <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-0.5">CGST %</label>
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={formData.totals?.globalCgstRate || 0}
                                    onChange={e => updateGlobalRate('cgst', e.target.value)}
                                  />
                                </div>
                                <div>
                                  <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-0.5">SGST %</label>
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={formData.totals?.globalSgstRate || 0}
                                    onChange={e => updateGlobalRate('sgst', e.target.value)}
                                  />
                                </div>
                                <div>
                                  <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-0.5">IGST %</label>
                                  <input
                                    type="number"
                                    className="field-input-sm w-full text-right font-mono"
                                    placeholder="0"
                                    value={formData.totals?.globalIgstRate || 0}
                                    onChange={e => updateGlobalRate('igst', e.target.value)}
                                  />
                                </div>
                              </div>
                            </div>
                          )}

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <span className="text-slate-500 dark:text-slate-400">Taxable Amount</span>
                            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">₹{formatAmount(formData.totals?.taxableAmount)}</span>
                          </div>

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800 text-slate-600 dark:text-slate-400">
                            <span>Discount</span>
                            <span className="font-mono font-semibold text-slate-600 dark:text-slate-400">(₹{formatAmount(formData.totals?.totalDiscount)})</span>
                          </div>

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800 font-semibold">
                            <span className="text-slate-700 dark:text-slate-300">Net Amount (Net Taxable)</span>
                            <span className="font-mono text-slate-900 dark:text-white">₹{formatAmount(formData.totals?.netTaxable)}</span>
                          </div>

                          {/* Global Discount Row */}
                          <div className="flex items-center justify-between py-1.5 border-b border-dashed border-slate-200 dark:border-slate-700">
                            <span className="text-slate-600 dark:text-slate-400 font-medium">Global Discount</span>
                            <div className="flex items-center gap-1">
                              <span className="text-slate-400">- ₹</span>
                              <input
                                type="number"
                                className="field-input-sm w-24 font-mono text-right"
                                placeholder="0.00"
                                value={formData.totals?.globalDiscount || ''}
                                onChange={e => updateGlobalDiscount(e.target.value)}
                              />
                            </div>
                          </div>

                          {/* Final Net Taxable */}
                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <div>
                              <span className="font-semibold text-slate-700 dark:text-slate-300 block">Final Net Taxable</span>
                              <span className="text-[10px] text-slate-400 font-normal">(After Global Discount)</span>
                            </div>
                            <span className="font-mono font-bold text-slate-900 dark:text-white">
                              ₹{formatAmount(formData.totals?.finalNetTaxable || (formData.totals?.netTaxable - (formData.totals?.globalDiscount || 0)))}
                            </span>
                          </div>

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <span className="text-slate-500 dark:text-slate-400">Total CGST</span>
                            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">₹{formatAmount(formData.totals?.totalCgst)}</span>
                          </div>

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <span className="text-slate-500 dark:text-slate-400">Total SGST</span>
                            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">₹{formatAmount(formData.totals?.totalSgst)}</span>
                          </div>

                          <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <span className="text-slate-500 dark:text-slate-400">Total IGST</span>
                            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">₹{formatAmount(formData.totals?.totalIgst)}</span>
                          </div>

                          {/* Round Off Input */}
                          <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-slate-800">
                            <span className="text-slate-500 dark:text-slate-400">Round Off</span>
                            <input
                              type="number"
                              step="0.01"
                              className="field-input-sm w-20 font-mono text-right"
                              value={formData.totals?.roundOff || 0}
                              onChange={e => updateRoundOff(e.target.value)}
                            />
                          </div>

                          {/* Grand Total */}
                          <div className="flex justify-between pt-3 text-sm font-bold text-slate-900 dark:text-white border-t-2 border-slate-900 dark:border-slate-100">
                            <span className="text-base">Grand Total</span>
                            <span className="font-mono text-lg text-blue-600 dark:text-blue-400">
                              Rs. {formatAmount(formData.totals?.grandTotal)}
                            </span>
                          </div>
                        </div>
                      </div>

                    </div>

                  </div>
                )}

                {/* ── STEP 4: REMARKS & CERTIFICATIONS ─────────────────────── */}
                {(!isWizardMode || currentStep === 4) && (
                  <div className="space-y-5 fade-in">
                    <div className="card p-5 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 space-y-4">
                      <div className="flex items-center gap-2 pb-3 border-b border-slate-100 dark:border-slate-800">
                        <MessageSquare size={16} className="text-blue-600 dark:text-blue-400" />
                        <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Remarks & Declarations</h3>
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400">
                            Invoice Remarks / Notes (Max 50 words)
                          </label>
                          <span className="text-[10px] text-slate-400 font-mono">
                            {(formData.remarks || '').trim() ? (formData.remarks || '').trim().split(/\s+/).length : 0} / 50 words
                          </span>
                        </div>
                        <textarea
                          rows={3}
                          className="field-input text-xs"
                          placeholder="Enter any internal remarks, voucher terms, or conditions..."
                          value={formData.remarks}
                          onChange={e => setFormData(prev => ({ ...prev, remarks: e.target.value }))}
                        />
                      </div>

                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                              Certified Remarks & Declarations
                            </label>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                              {(formData.certifiedRemarks || []).length} {formData.certifiedRemarks?.length === 1 ? 'entry' : 'entries'}
                            </span>
                          </div>
                          {(formData.certifiedRemarks || []).length > 0 && (
                            <button
                              type="button"
                              onClick={clearAllCertifiedRemarks}
                              className="text-xs text-rose-600 dark:text-rose-400 font-semibold hover:underline"
                            >
                              Clear All
                            </button>
                          )}
                        </div>

                        {/* List of dynamic certified remarks */}
                        <div className="space-y-2 mb-3">
                          {(formData.certifiedRemarks || []).length === 0 ? (
                            <div className="p-3 rounded-lg border border-dashed border-slate-300 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/30 text-center text-xs text-slate-500 dark:text-slate-400">
                              No declarations or certificates detected on this invoice. You can add custom remarks or pick from standard templates below.
                            </div>
                          ) : (
                            formData.certifiedRemarks.map((remark, idx) => (
                              <div
                                key={idx}
                                className="flex items-start gap-2.5 p-2.5 rounded-lg border bg-blue-50/50 dark:bg-blue-950/30 border-blue-200/80 dark:border-blue-800/70 text-xs text-slate-800 dark:text-slate-200"
                              >
                                <span className="text-[11px] font-mono font-bold text-blue-600 dark:text-blue-400 shrink-0 mt-1">
                                  #{idx + 1}
                                </span>
                                <textarea
                                  rows={2}
                                  value={remark}
                                  onChange={e => updateCertifiedRemark(idx, e.target.value)}
                                  className="flex-1 bg-transparent text-xs text-slate-800 dark:text-slate-200 border-none focus:outline-none resize-y"
                                />
                                <button
                                  type="button"
                                  onClick={() => removeCertifiedRemark(idx)}
                                  className="text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 p-1 shrink-0 transition-colors"
                                  title="Remove this certificate entry"
                                >
                                  <X size={14} />
                                </button>
                              </div>
                            ))
                          )}
                        </div>

                        {/* Add Certificate / Template Controls */}
                        <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-800/50 space-y-2.5">
                          <div className="flex gap-2">
                            <input
                              type="text"
                              className="field-input text-xs flex-1"
                              placeholder="Type a certificate, declaration or stock note to add..."
                              value={newRemarkText}
                              onChange={e => setNewRemarkText(e.target.value)}
                              onKeyDown={e => {
                                if (e.key === 'Enter') {
                                  e.preventDefault()
                                  addCertifiedRemark(newRemarkText)
                                }
                              }}
                            />
                            <button
                              type="button"
                              onClick={() => addCertifiedRemark(newRemarkText)}
                              disabled={!newRemarkText.trim()}
                              className="btn-primary text-xs px-3 py-1.5 shrink-0 disabled:opacity-40"
                            >
                              + Add
                            </button>
                          </div>

                          {/* Optional Template selector */}
                          <div className="flex items-center gap-2 pt-1 border-t border-slate-200/60 dark:border-slate-700/60">
                            <span className="text-[11px] text-slate-500 dark:text-slate-400 shrink-0 font-medium">
                              Insert Template:
                            </span>
                            <select
                              value={selectedTemplate}
                              onChange={e => {
                                const val = e.target.value
                                if (val) {
                                  addCertifiedRemark(val)
                                  setSelectedTemplate('')
                                }
                              }}
                              className="field-input text-xs py-1 flex-1 text-slate-600 dark:text-slate-300"
                            >
                              <option value="">-- Select a standard statutory / voucher template --</option>
                              {STANDARD_CERTIFICATE_TEMPLATES.map((tmpl, tIdx) => (
                                <option key={tIdx} value={tmpl}>
                                  {tmpl.length > 75 ? `${tmpl.slice(0, 75)}...` : tmpl}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                      </div>

                      {/* Authorized Signatory Line */}
                      <div className="pt-8 text-right">
                        <div className="inline-block border-t border-slate-400 dark:border-slate-600 pt-1.5 px-6 font-semibold text-xs text-slate-700 dark:text-slate-300">
                          Authorized Signatory
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* ── STEP 5: REVIEW & SUBMIT ──────────────────────────────── */}
                {(!isWizardMode || currentStep === 5) && (
                  <div className="space-y-5 fade-in">
                    <div className="card p-6 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
                      <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-100 dark:border-slate-800">
                        <div className="flex items-center gap-2">
                          <CheckCircle size={18} className="text-emerald-600 dark:text-emerald-400" />
                          <h3 className="text-sm font-bold text-slate-900 dark:text-white">Pre-Submission Summary Audit</h3>
                        </div>
                        <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                          Ready for Finalization
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="p-3.5 bg-slate-50 dark:bg-slate-800/60 rounded-xl border border-slate-200 dark:border-slate-700 space-y-1">
                          <p className="font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider text-[10px]">Client / Recipient</p>
                          <p className="font-bold text-slate-900 dark:text-white">{formData?.client?.name || '—'}</p>
                          <p className="text-slate-500 dark:text-slate-400">GSTIN: {formData?.client?.gstin || 'Unregistered'}</p>
                          <p className="text-slate-500 dark:text-slate-400">SLS: {formData?.client?.slsCode || 'Standard'}</p>
                        </div>

                        <div className="p-3.5 bg-slate-50 dark:bg-slate-800/60 rounded-xl border border-slate-200 dark:border-slate-700 space-y-1">
                          <p className="font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider text-[10px]">Vendor / Biller</p>
                          <p className="font-bold text-slate-900 dark:text-white">{formData?.company?.name || '—'}</p>
                          <p className="text-slate-500 dark:text-slate-400">GSTIN: {formData?.company?.gstin || '—'} | PAN: {formData?.company?.pan || '—'}</p>
                          <p className="text-slate-500 dark:text-slate-400">Bank: {formData?.bankDetails?.bankName || '—'} (A/C: {formData?.bankDetails?.accountNumber || '—'})</p>
                        </div>
                      </div>

                      <div className="mt-4 p-3.5 bg-blue-50/60 dark:bg-blue-950/40 rounded-xl border border-blue-100 dark:border-blue-900/60 flex items-center justify-between">
                        <div>
                          <p className="text-xs font-bold text-blue-900 dark:text-blue-300">Invoice #{formData?.meta?.invoiceNo || 'DRAFT'}</p>
                          <p className="text-[11px] text-blue-700 dark:text-blue-400">Dated: {formData?.meta?.date || '—'} &bull; {formData?.items?.length || 0} line items</p>
                        </div>
                        <div className="text-right">
                          <span className="text-[11px] text-slate-500 dark:text-slate-400 uppercase tracking-wider block">Grand Total</span>
                          <span className="text-base font-bold font-mono text-blue-700 dark:text-blue-400">₹{formatAmount(formData?.totals?.grandTotal)}</span>
                        </div>
                      </div>

                      {/* Action Bar */}
                      <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
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
                            className="btn-secondary text-xs px-4 py-2 text-amber-900 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800 hover:bg-amber-100"
                            title="Save your progress without marking as verified ground truth"
                          >
                            {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700 dark:text-amber-400" />}
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
                          className="btn-secondary px-4 py-2 text-xs text-amber-900 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800 hover:bg-amber-100"
                          title="Save your changes so far without marking as verified ground truth"
                        >
                          {saving ? <RefreshCw size={13} className="spinner" /> : <Save size={13} className="text-amber-700 dark:text-amber-400" />}
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
                <div className="xl:col-span-5 2xl:col-span-5 sticky top-16 h-[calc(100vh-76px)] card p-2.5 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 shadow-md flex flex-col">
                  
                  {/* Top Bar inside Document Pane */}
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100 dark:border-slate-800 flex-shrink-0">
                    <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-0.5 rounded-lg border border-slate-200 dark:border-slate-700">
                      <button
                        onClick={() => setDocSource('original')}
                        className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${
                          docSource === 'original'
                            ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold'
                            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
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
                          docSource === 'rendered'
                            ? 'bg-white dark:bg-slate-900 shadow-sm text-blue-600 dark:text-blue-400 font-bold'
                            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                        }`}
                      >
                        🖨️ Standard HTML
                      </button>
                    </div>

                    {docSource === 'original' ? (
                      <div className="flex flex-wrap items-center gap-1">
                        {docTotalPages > 1 && (
                          <div className="flex items-center gap-0.5 text-[11px] font-semibold text-slate-600 dark:text-slate-300 mr-1 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-700">
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

                        <div className="flex items-center gap-0.5 bg-slate-100 dark:bg-slate-800 p-0.5 rounded border border-slate-200 dark:border-slate-700">
                          <button
                            onClick={() => setZoomLevel(z => Math.max(30, z - 25))}
                            className="p-0.5 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                            title="Zoom Out"
                          >
                            <ZoomOut size={12} />
                          </button>
                          <span className="text-[10px] font-mono px-0.5 text-slate-700 dark:text-slate-300 min-w-[34px] text-center">{zoomLevel}%</span>
                          <button
                            onClick={() => setZoomLevel(z => Math.min(400, z + 25))}
                            className="p-0.5 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                            title="Zoom In"
                          >
                            <ZoomIn size={12} />
                          </button>

                          <div className="w-[1px] h-3 bg-slate-300 dark:bg-slate-600 mx-0.5" />

                          <button
                            onClick={fitWidth}
                            className="px-1.5 py-0.5 text-[10px] font-semibold text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                            title="Fit Width"
                          >
                            Fit W
                          </button>
                          <button
                            onClick={fitHeight}
                            className="px-1.5 py-0.5 text-[10px] font-semibold text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                            title="Fit Height"
                          >
                            Fit H
                          </button>
                          <button
                            onClick={resetView}
                            className="p-0.5 text-slate-600 dark:text-slate-300 hover:text-blue-600 rounded hover:bg-white dark:hover:bg-slate-700"
                            title="Reset Pan (0,0), Zoom (100%) & Rotation (0°)"
                          >
                            <RotateCcw size={11} />
                          </button>

                          <div className="w-[1px] h-3 bg-slate-300 dark:bg-slate-600 mx-0.5" />

                          <button
                            onClick={() => setRotation(r => (r + 90) % 360)}
                            className={`p-0.5 rounded text-slate-600 dark:text-slate-300 hover:text-blue-600 hover:bg-white dark:hover:bg-slate-700 flex items-center gap-0.5 transition-all ${
                              rotation !== 0 ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950 font-bold' : ''
                            }`}
                            title={`Rotate Image 90° clockwise (current: ${rotation}°)`}
                          >
                            <RotateCw size={11} />
                            {rotation !== 0 && <span className="text-[9px] font-mono">{rotation}°</span>}
                          </button>
                        </div>

                        <button
                          onClick={() => setActiveTab('preview')}
                          className="p-1 rounded text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                          title="Expand Full Document"
                        >
                          <Maximize2 size={13} />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setActiveTab('preview')}
                        className="p-1 rounded text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        title="Expand Full Document"
                      >
                        <Maximize2 size={13} />
                      </button>
                    )}
                  </div>

                  {/* Document Canvas Container with Interactive Pan & Drag */}
                  <div
                    onMouseDown={handleViewerMouseDown}
                    onMouseMove={handleViewerMouseMove}
                    onMouseUp={handleViewerMouseUp}
                    onMouseLeave={handleViewerMouseUp}
                    onWheel={handleViewerWheel}
                    className={`flex-1 w-full rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 dark:bg-slate-950 relative shadow-inner select-none ${
                      isDragging ? 'cursor-grabbing' : 'cursor-grab'
                    }`}
                  >
                    {docSource === 'original' ? (
                      <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
                        <img
                          src={previewImageUrl}
                          alt="Original Scanned Document"
                          draggable={false}
                          style={{
                            width: `${zoomLevel}%`,
                            minWidth: `${zoomLevel}%`,
                            maxWidth: 'none',
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) rotate(${rotation}deg)`,
                            transformOrigin: 'center center',
                            transition: isDragging ? 'none' : 'transform 0.12s cubic-bezier(0.4, 0, 0.2, 1), width 0.1s ease-out',
                            cursor: isDragging ? 'grabbing' : 'grab',
                            userSelect: 'none',
                          }}
                          className="rounded shadow-2xl bg-white border border-slate-800 select-none block pointer-events-auto"
                        />

                        {/* Floating Pan & Zoom Hint */}
                        <div className="absolute bottom-2 right-2 bg-slate-900/80 backdrop-blur-xs text-[10px] text-slate-400 font-medium px-2 py-0.5 rounded border border-slate-800 pointer-events-none opacity-60">
                          🖐️ Drag to move • Ctrl+Wheel to zoom
                        </div>
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
          </div>
        )}

      </div>
    </div>

      {/* ── Add Custom Column Modal (Parity with Angular Invoice Builder) ── */}
      {showAddColumnModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-800 w-full max-w-md p-5 space-y-4 animate-scale-in">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400">
                  <Plus size={16} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Add Table Column</h3>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">Create a dynamic custom field for line items</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowAddColumnModal(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              >
                <X size={16} />
              </button>
            </div>

            <form
              onSubmit={e => {
                e.preventDefault()
                confirmAddColumn()
              }}
              className="space-y-4"
            >
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Column Name / Header <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  autoFocus
                  required
                  placeholder="e.g. Batch No, Serial No, Mfg Date..."
                  className="field-input text-xs w-full font-medium"
                  value={newColumnName}
                  onChange={e => setNewColumnName(e.target.value)}
                />
              </div>

              {/* Quick Preset Suggestions */}
              <div>
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 mb-1.5 font-medium">
                  Quick Presets:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {['Batch No', 'Serial No', 'Item Code', 'Mfg Date', 'Expiry Date', 'Delivery Date', 'Remarks', 'PO Ref'].map(preset => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => {
                        setNewColumnName(preset)
                        if (preset.toLowerCase().includes('date')) {
                          setNewColumnType('date')
                        }
                      }}
                      className="px-2 py-0.5 text-[11px] rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-blue-50 hover:text-blue-600 dark:hover:bg-slate-700 transition border border-slate-200/60 dark:border-slate-700/60"
                    >
                      + {preset}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Column Field Type
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { type: 'text', label: 'Text', desc: 'Any words/codes' },
                    { type: 'number', label: 'Number', desc: 'Numeric values' },
                    { type: 'date', label: 'Date', desc: 'Calendar date' },
                  ].map(t => (
                    <button
                      key={t.type}
                      type="button"
                      onClick={() => setNewColumnType(t.type)}
                      className={`p-2 rounded-lg border text-left transition-all ${
                        newColumnType === t.type
                          ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300'
                          : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-400 hover:bg-slate-100'
                      }`}
                    >
                      <div className="text-xs font-bold">{t.label}</div>
                      <div className="text-[10px] opacity-75">{t.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="pt-2 flex justify-end items-center gap-2 border-t border-slate-100 dark:border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddColumnModal(false)}
                  className="btn-secondary py-1.5 px-3.5 text-xs text-slate-600 dark:text-slate-300"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary py-1.5 px-4 text-xs font-semibold flex items-center gap-1.5"
                >
                  <Plus size={13} />
                  <span>Add Column</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
