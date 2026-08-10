import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  Save, Download, Eye, ArrowLeft, AlertTriangle,
  CheckCircle, RefreshCw, ExternalLink, Trash2
} from 'lucide-react'

// ── Field editor ──────────────────────────────────────────────────────────────

function Field({ label, value, onChange, type = 'text', mono = false, wide = false }) {
  return (
    <div className={wide ? 'col-span-2' : ''}>
      <label className="block text-xs font-medium text-gray-500 mb-1">{label}</label>
      <input
        type={type}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        className={`field-input ${mono ? 'font-mono text-xs' : ''}`}
      />
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="card p-5 mb-4">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">{title}</h3>
      <div className="grid grid-cols-2 gap-4">{children}</div>
    </div>
  )
}

// ── Line items table ──────────────────────────────────────────────────────────

function LineItemsEditor({ items, onChange }) {
  const update = (idx, field, val) => {
    const next = items.map((it, i) => i === idx ? { ...it, [field]: val } : it)
    onChange(next)
  }
  const remove = (idx) => onChange(items.filter((_, i) => i !== idx))
  const add = () => onChange([...items, { description: '', quantity: 1, rate: 0, amount: 0 }])

  return (
    <div className="card p-5 mb-4">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Line Items</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100">
              <th className="text-left pb-2 text-xs text-gray-400 font-medium w-1/2">Description</th>
              <th className="text-right pb-2 text-xs text-gray-400 font-medium w-16">Qty</th>
              <th className="text-right pb-2 text-xs text-gray-400 font-medium w-24">Rate</th>
              <th className="text-right pb-2 text-xs text-gray-400 font-medium w-24">Amount</th>
              <th className="w-8" />
            </tr>
          </thead>
          <tbody>
            {items.length === 0 && (
              <tr><td colSpan={5} className="py-6 text-center text-xs text-gray-300">No line items extracted</td></tr>
            )}
            {items.map((item, idx) => (
              <tr key={idx} className="border-b border-gray-50">
                <td className="py-1.5 pr-2">
                  <input
                    className="field-input text-xs"
                    value={item.description || ''}
                    onChange={e => update(idx, 'description', e.target.value)}
                  />
                </td>
                <td className="py-1.5 px-1">
                  <input
                    className="field-input text-xs text-right font-mono"
                    type="number"
                    value={item.quantity || ''}
                    onChange={e => update(idx, 'quantity', e.target.value)}
                  />
                </td>
                <td className="py-1.5 px-1">
                  <input
                    className="field-input text-xs text-right font-mono"
                    type="number"
                    value={item.rate || ''}
                    onChange={e => update(idx, 'rate', e.target.value)}
                  />
                </td>
                <td className="py-1.5 px-1">
                  <input
                    className="field-input text-xs text-right font-mono"
                    type="number"
                    value={item.amount || ''}
                    onChange={e => update(idx, 'amount', e.target.value)}
                  />
                </td>
                <td className="py-1.5 pl-1">
                  <button
                    onClick={() => remove(idx)}
                    className="p-1 text-gray-300 hover:text-red-400 transition-colors"
                  >
                    <Trash2 size={13} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button onClick={add} className="mt-3 text-xs text-brand-600 hover:text-brand-800 font-medium">
        + Add line item
      </button>
    </div>
  )
}

// ── Validation panel ──────────────────────────────────────────────────────────

function ValidationPanel({ invoice }) {
  if (!invoice) return null
  const conf = Math.round((invoice.overall_confidence || 0) * 100)
  return (
    <div className="card p-5 mb-4">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Validation</h3>
      <div className="flex items-center gap-3 mb-4">
        <div className="conf-bar flex-1">
          <div
            className={`conf-bar-fill ${conf >= 85 ? 'bg-green-500' : conf >= 65 ? 'bg-amber-400' : 'bg-red-400'}`}
            style={{ width: `${conf}%` }}
          />
        </div>
        <span className="text-sm font-semibold text-gray-700">{conf}%</span>
      </div>

      {invoice.needs_review && invoice.review_reasons?.length > 0 ? (
        <div>
          <div className="flex items-center gap-1.5 text-amber-600 text-xs font-semibold mb-2">
            <AlertTriangle size={12} /> Issues found
          </div>
          <ul className="space-y-1">
            {invoice.review_reasons.map((r, i) => (
              <li key={i} className="text-xs text-amber-700 bg-amber-50 rounded px-2.5 py-1.5">
                {r}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 text-green-600 text-xs font-semibold">
          <CheckCircle size={12} /> All validation checks passed
        </div>
      )}
    </div>
  )
}

// ── Totals panel ─────────────────────────────────────────────────────────────

function TotalsPanel({ invoice, onChange }) {
  const num = (v) => v != null ? parseFloat(v) : null
  const fmt = (v) => v != null ? v.toFixed(2) : ''

  return (
    <div className="card p-5 mb-4">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Financials</h3>
      <div className="space-y-3">
        {[
          { key: 'subtotal', label: 'Subtotal' },
          { key: 'cgst', label: 'CGST' },
          { key: 'sgst', label: 'SGST' },
          { key: 'igst', label: 'IGST' },
          { key: 'tax_amount', label: 'Total Tax' },
          { key: 'discount', label: 'Discount' },
        ].map(({ key, label }) => (
          <div key={key} className="flex items-center justify-between gap-3">
            <span className="text-xs text-gray-500 w-28 flex-shrink-0">{label}</span>
            <input
              type="number"
              className="field-input text-xs text-right font-mono"
              value={invoice[key] != null ? invoice[key] : ''}
              onChange={e => onChange(key, e.target.value === '' ? null : parseFloat(e.target.value))}
            />
          </div>
        ))}
        <div className="border-t border-gray-100 pt-3 flex items-center justify-between gap-3">
          <span className="text-sm font-bold text-gray-800 w-28">Grand Total</span>
          <input
            type="number"
            className="field-input text-sm text-right font-mono font-bold text-brand-700"
            value={invoice.grand_total != null ? invoice.grand_total : ''}
            onChange={e => onChange('grand_total', e.target.value === '' ? null : parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  )
}

// ── Main Review Page ──────────────────────────────────────────────────────────

export default function ReviewPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [invoice, setInvoice] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [activeTab, setActiveTab] = useState('edit') // 'edit' | 'preview'
  const [previewHtml, setPreviewHtml] = useState('')

  const fetchJob = useCallback(async () => {
    try {
      const { data } = await axios.get(`/api/invoices/${jobId}`)
      setJob(data)
      if (data.invoice) setInvoice({ ...data.invoice })
    } catch {
      toast.error('Failed to load invoice')
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => { fetchJob() }, [fetchJob])

  // Poll if still processing
  useEffect(() => {
    if (job?.status === 'processing' || job?.status === 'pending') {
      const t = setTimeout(fetchJob, 3000)
      return () => clearTimeout(t)
    }
  }, [job, fetchJob])

  const setField = (key, val) => {
    setInvoice(prev => ({ ...prev, [key]: val }))
    setDirty(true)
  }

  const save = async () => {
    setSaving(true)
    try {
      await axios.patch(`/api/invoices/${jobId}`, { corrections: invoice })
      toast.success('Invoice saved')
      setDirty(false)
      fetchJob()
    } catch (e) {
      toast.error('Save failed: ' + (e.response?.data?.detail || e.message))
    } finally {
      setSaving(false)
    }
  }

  const loadPreview = async () => {
    try {
      const { data } = await axios.get(`/api/invoices/${jobId}/html`)
      setPreviewHtml(data)
      setActiveTab('preview')
    } catch {
      toast.error('Preview unavailable')
    }
  }

  const downloadPdf = () => {
    window.open(`/api/invoices/${jobId}/pdf`, '_blank')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 p-16">
        <RefreshCw size={20} className="spinner mr-2" /> Loading invoice...
      </div>
    )
  }

  if (!job || (job.status !== 'done' && job.status !== 'reviewed' && job.status !== 'failed')) {
    return (
      <div className="p-8 text-center text-gray-400">
        <RefreshCw size={20} className="spinner mx-auto mb-3" />
        <p className="font-medium">Processing invoice...</p>
        <p className="text-sm mt-1">This page will update automatically.</p>
      </div>
    )
  }

  if (job.status === 'failed') {
    return (
      <div className="p-8 max-w-xl mx-auto">
        <button onClick={() => navigate('/invoices')} className="btn-secondary mb-6">
          <ArrowLeft size={14} /> Back
        </button>
        <div className="card p-8 text-center border-red-200 bg-red-50">
          <AlertTriangle size={32} className="text-red-400 mx-auto mb-3" />
          <p className="font-semibold text-red-800">Processing Failed</p>
          <p className="text-sm text-red-600 mt-2">{job.error_message}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Top bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3 flex-shrink-0">
        <button onClick={() => navigate('/invoices')} className="btn-secondary">
          <ArrowLeft size={14} /> Back
        </button>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-900 truncate">{job.filename}</p>
          <p className="text-xs text-gray-400">Job {jobId.slice(0, 8)}…</p>
        </div>

        {/* Tabs */}
        <div className="flex bg-gray-100 rounded-lg p-1 gap-1">
          <button
            onClick={() => setActiveTab('edit')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all
                        ${activeTab === 'edit' ? 'bg-white shadow-sm text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
          >
            Edit Fields
          </button>
          <button
            onClick={loadPreview}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all
                        ${activeTab === 'preview' ? 'bg-white shadow-sm text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
          >
            <Eye size={11} /> Preview
          </button>
        </div>

        {dirty && (
          <span className="text-xs text-amber-600 font-medium bg-amber-50 px-2 py-1 rounded">
            Unsaved changes
          </span>
        )}
        <button onClick={downloadPdf} className="btn-secondary">
          <Download size={14} /> PDF
        </button>
        <button onClick={save} disabled={saving || !dirty} className="btn-primary">
          {saving ? <RefreshCw size={14} className="spinner" /> : <Save size={14} />}
          Save
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'preview' ? (
          <div className="p-6">
            <iframe
              srcDoc={previewHtml}
              className="w-full bg-white rounded-xl border border-gray-200 shadow-sm"
              style={{ minHeight: '900px' }}
              title="Invoice Preview"
            />
          </div>
        ) : (
          <div className="p-6 max-w-4xl mx-auto">
            <div className="grid grid-cols-3 gap-4">
              {/* Left: all editable fields */}
              <div className="col-span-2 space-y-0">
                <Section title="Invoice Info">
                  <Field label="Invoice Number" value={invoice?.invoice_number} onChange={v => setField('invoice_number', v)} />
                  <Field label="Invoice Date" value={invoice?.invoice_date} onChange={v => setField('invoice_date', v)} />
                  <Field label="Due Date" value={invoice?.due_date} onChange={v => setField('due_date', v)} />
                  <Field label="PO Number" value={invoice?.po_number} onChange={v => setField('po_number', v)} />
                  <Field label="Currency" value={invoice?.currency} onChange={v => setField('currency', v)} />
                </Section>

                <Section title="Vendor (Bill From)">
                  <Field label="Company Name" value={invoice?.vendor_name} onChange={v => setField('vendor_name', v)} wide />
                  <Field label="Address" value={invoice?.vendor_address} onChange={v => setField('vendor_address', v)} wide />
                  <Field label="GSTIN" value={invoice?.vendor_gstin} onChange={v => setField('vendor_gstin', v)} mono />
                  <Field label="PAN" value={invoice?.vendor_pan} onChange={v => setField('vendor_pan', v)} mono />
                  <Field label="Email" value={invoice?.vendor_email} onChange={v => setField('vendor_email', v)} type="email" />
                  <Field label="Phone" value={invoice?.vendor_phone} onChange={v => setField('vendor_phone', v)} />
                </Section>

                <Section title="Buyer (Bill To)">
                  <Field label="Company / Name" value={invoice?.buyer_name} onChange={v => setField('buyer_name', v)} wide />
                  <Field label="Address" value={invoice?.buyer_address} onChange={v => setField('buyer_address', v)} wide />
                  <Field label="GSTIN" value={invoice?.buyer_gstin} onChange={v => setField('buyer_gstin', v)} mono />
                </Section>

                <LineItemsEditor
                  items={invoice?.line_items || []}
                  onChange={items => { setField('line_items', items); setDirty(true) }}
                />

                <Section title="Payment Details">
                  <Field label="Bank Name" value={invoice?.bank_name} onChange={v => setField('bank_name', v)} />
                  <Field label="Account Number" value={invoice?.account_number} onChange={v => setField('account_number', v)} mono />
                  <Field label="IFSC Code" value={invoice?.ifsc_code} onChange={v => setField('ifsc_code', v)} mono />
                  <Field label="Payment Terms" value={invoice?.payment_terms} onChange={v => setField('payment_terms', v)} />
                </Section>
              </div>

              {/* Right sidebar */}
              <div className="col-span-1 space-y-0">
                <ValidationPanel invoice={invoice} />
                <TotalsPanel invoice={invoice} onChange={(k, v) => setField(k, v)} />

                {/* Quick JSON export */}
                <div className="card p-4">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Export</h3>
                  <div className="space-y-2">
                    <button
                      onClick={downloadPdf}
                      className="w-full btn-secondary text-xs justify-center"
                    >
                      <Download size={12} /> Download PDF
                    </button>
                    <button
                      onClick={() => {
                        const blob = new Blob([JSON.stringify(invoice, null, 2)], { type: 'application/json' })
                        const url = URL.createObjectURL(blob)
                        const a = document.createElement('a')
                        a.href = url
                        a.download = `invoice_${jobId.slice(0, 8)}.json`
                        a.click()
                        URL.revokeObjectURL(url)
                      }}
                      className="w-full btn-secondary text-xs justify-center"
                    >
                      <ExternalLink size={12} /> Export JSON
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
