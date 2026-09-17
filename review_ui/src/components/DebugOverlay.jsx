import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Layers, Cpu, Eye, EyeOff, AlertTriangle, CheckCircle, Info,
  Search, ChevronDown, ChevronUp, Sparkles, Crosshair, HelpCircle, X,
  Move, Crop, Pencil, MousePointer, Target, RefreshCw, Check, ArrowRight
} from 'lucide-react'

// ── Canonical Field Name Mapping ─────────────────────────────────────────────

export const FIELD_NAME_MAP = {
  invoice_number: { label: 'Invoice Number', section: 'meta', key: 'invoiceNo' },
  invoice_date: { label: 'Invoice Date', section: 'meta', key: 'date' },
  due_date: { label: 'Due Date', section: 'meta', key: 'dueDate' },
  po_number: { label: 'PO Number', section: 'meta', key: 'poNumber' },
  place_of_supply: { label: 'Place of Supply', section: 'meta', key: 'placeOfSupply' },
  vendor_name: { label: 'Vendor Name', section: 'company', key: 'name' },
  vendor_gstin: { label: 'Vendor GSTIN', section: 'company', key: 'gstin' },
  vendor_pan: { label: 'Vendor PAN', section: 'company', key: 'pan' },
  vendor_email: { label: 'Vendor Email', section: 'company', key: 'email' },
  vendor_phone: { label: 'Vendor Phone', section: 'company', key: 'phone' },
  vendor_address_line1: { label: 'Vendor Address', section: 'company', key: 'addressLine1' },
  buyer_name: { label: 'Buyer Name', section: 'client', key: 'name' },
  buyer_gstin: { label: 'Buyer GSTIN', section: 'client', key: 'gstin' },
  buyer_phone: { label: 'Buyer Phone', section: 'client', key: 'phone' },
  sls_code: { label: 'SLS Code', section: 'client', key: 'slsCode' },
  grand_total: { label: 'Grand Total', section: 'totals', key: 'grandTotal' },
  subtotal: { label: 'Taxable Amount (Subtotal)', section: 'totals', key: 'taxableAmount' },
  discount: { label: 'Total Discount', section: 'totals', key: 'totalDiscount' },
  cgst: { label: 'Total CGST', section: 'totals', key: 'totalCgst' },
  sgst: { label: 'Total SGST', section: 'totals', key: 'totalSgst' },
  igst: { label: 'Total IGST', section: 'totals', key: 'totalIgst' },
  bank_name: { label: 'Bank Name', section: 'bankDetails', key: 'bankName' },
  branch_name: { label: 'Branch Name', section: 'bankDetails', key: 'branchName' },
  account_name: { label: 'Account Name', section: 'bankDetails', key: 'accountName' },
  account_number: { label: 'Account Number', section: 'bankDetails', key: 'accountNumber' },
  ifsc_code: { label: 'IFSC Code', section: 'bankDetails', key: 'ifsc' },
}

export function getCanonicalFieldName(section, key) {
  for (const [canonical, meta] of Object.entries(FIELD_NAME_MAP)) {
    if (meta.section === section && meta.key === key) return canonical
  }
  return null
}

export function getFormFieldPath(canonicalName) {
  const meta = FIELD_NAME_MAP[canonicalName]
  if (!meta) return null
  return [meta.section, meta.key]
}

export function getFieldValue(formData, canonicalName) {
  const meta = FIELD_NAME_MAP[canonicalName]
  if (!meta || !formData) return ''
  const val = formData[meta.section]?.[meta.key]
  return val !== null && val !== undefined ? String(val) : ''
}

// ── Smart OCR Matching & Box Extraction ─────────────────────────────────────

function cleanForMatch(str) {
  if (!str) return ''
  return String(str)
    .toLowerCase()
    .replace(/[₹$€£,:\-\/\\.\s_#]/g, '')
    .trim()
}

/**
 * Searches OCR word boxes for matching text/numbers and returns the normalized bounding box [x1, y1, x2, y2].
 * Handles exact strings, cleaned numeric values, multi-word sequences, and substring containment.
 */
export function findOcrMatchesForValue(val, ocrBoxes = []) {
  if (!val || !ocrBoxes || ocrBoxes.length === 0) return null

  const rawStr = String(val).trim()
  if (!rawStr) return null

  const cleanedTarget = cleanForMatch(rawStr)
  if (!cleanedTarget) return null

  // 1. Single Token Exact or Cleaned Match
  for (const box of ocrBoxes) {
    if (!box.text || !box.bbox_norm) continue
    const cleanedBox = cleanForMatch(box.text)
    if (cleanedBox === cleanedTarget) {
      return {
        bbox_norm: [...box.bbox_norm],
        matchedText: box.text,
        confidence: box.confidence || 0.95,
        matchType: 'exact',
        source: box.source || 'ocr',
      }
    }
  }

  // 2. Multi-word Target: check sliding window of consecutive OCR tokens
  const targetWords = rawStr.toLowerCase().split(/\s+/).filter(Boolean)
  if (targetWords.length > 1) {
    for (let i = 0; i <= ocrBoxes.length - targetWords.length; i++) {
      const windowBoxes = ocrBoxes.slice(i, i + targetWords.length)
      const windowText = windowBoxes.map(b => (b.text || '').toLowerCase()).join(' ')
      const cleanedWindow = cleanForMatch(windowText)

      if (cleanedWindow === cleanedTarget || cleanedWindow.includes(cleanedTarget) || cleanedTarget.includes(cleanedWindow)) {
        let minX = 1000, minY = 1000, maxX = 0, maxY = 0
        let hasValid = false
        windowBoxes.forEach(b => {
          if (b.bbox_norm) {
            hasValid = true
            minX = Math.min(minX, b.bbox_norm[0])
            minY = Math.min(minY, b.bbox_norm[1])
            maxX = Math.max(maxX, b.bbox_norm[2])
            maxY = Math.max(maxY, b.bbox_norm[3])
          }
        })
        if (hasValid && maxX > minX && maxY > minY) {
          return {
            bbox_norm: [minX, minY, maxX, maxY],
            matchedText: windowBoxes.map(b => b.text).join(' '),
            confidence: 0.92,
            matchType: 'multi-token',
            source: windowBoxes[0]?.source || 'ocr',
          }
        }
      }
    }
  }

  // 3. Substring Containment Match (for codes or longer text lines)
  for (const box of ocrBoxes) {
    if (!box.text || !box.bbox_norm) continue
    const cleanedBox = cleanForMatch(box.text)
    if (cleanedBox.length >= 3 && (cleanedBox.includes(cleanedTarget) || (cleanedTarget.length >= 4 && cleanedTarget.includes(cleanedBox)))) {
      return {
        bbox_norm: [...box.bbox_norm],
        matchedText: box.text,
        confidence: 0.85,
        matchType: 'partial',
        source: box.source || 'ocr',
      }
    }
  }

  return null
}

/**
 * Extracts combined text from all OCR words located inside a given normalized bbox.
 */
export function getOcrTextInBox(bbox, ocrBoxes = []) {
  if (!bbox || !ocrBoxes || ocrBoxes.length === 0) return ''
  const [x1, y1, x2, y2] = bbox
  const inside = ocrBoxes.filter(b => {
    if (!b.bbox_norm) return false
    const [bx1, by1, bx2, by2] = b.bbox_norm
    const cx = (bx1 + bx2) / 2
    const cy = (by1 + by2) / 2
    return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2
  })

  inside.sort((a, b) => {
    const ay = a.bbox_norm[1]
    const by = b.bbox_norm[1]
    if (Math.abs(ay - by) > 15) return ay - by
    return a.bbox_norm[0] - b.bbox_norm[0]
  })

  return inside.map(b => b.text).join(' ').trim()
}

// ── SVG Debug Overlay Component with Interactive Editing ────────────────────

export function DebugOverlaySvg({
  debugData,
  showYolo = true,
  showOcr = true,
  showAnchors = true,
  showHandwriting = true,
  showFields = true,
  activeField = null,
  onSelectField = null,
  hoveredElement = null,
  setHoveredElement = null,
  isLocateMode = false,
  onPickOcrBox = null,
  isDrawMode = false,
  onUpdateFieldBbox = null,
}) {
  const svgRef = useRef(null)
  const [drawingRect, setDrawingRect] = useState(null)
  const [dragHandle, setDragHandle] = useState(null) // { type: 'nw'|'ne'|'sw'|'se'|'move', startX, startY, origBbox }
  const [liveBbox, setLiveBbox] = useState(null)

  // Active field evidence
  const activeEvidence = activeField && debugData?.field_evidence ? debugData.field_evidence[activeField] : null
  const currentBbox = liveBbox || (activeEvidence?.bbox_norm ? [...activeEvidence.bbox_norm] : null)

  // Clear liveBbox when activeField changes
  useEffect(() => {
    setLiveBbox(null)
  }, [activeField])

  // Convert browser mouse event to normalized 0-1000 coordinate space
  const getSvgCoordinates = useCallback((e) => {
    if (!svgRef.current) return { x: 0, y: 0 }
    const rect = svgRef.current.getBoundingClientRect()
    const x = Math.round(Math.max(0, Math.min(1000, ((e.clientX - rect.left) / rect.width) * 1000)))
    const y = Math.round(Math.max(0, Math.min(1000, ((e.clientY - rect.top) / rect.height) * 1000)))
    return { x, y }
  }, [])

  // Canvas Mouse Down: either start Draw Box or start dragging an active handle
  const handleSvgMouseDown = (e) => {
    if (e.button !== 0) return // Left click only

    if (isDrawMode && activeField) {
      e.preventDefault()
      e.stopPropagation()
      const { x, y } = getSvgCoordinates(e)
      setDrawingRect({ startX: x, startY: y, curX: x, curY: y })
    }
  }

  // Canvas Mouse Move: update drawing rectangle or handle drag
  const handleSvgMouseMove = (e) => {
    const { x, y } = getSvgCoordinates(e)

    if (drawingRect) {
      e.preventDefault()
      setDrawingRect(prev => prev ? { ...prev, curX: x, curY: y } : null)
      return
    }

    if (dragHandle && currentBbox) {
      e.preventDefault()
      const dx = x - dragHandle.startX
      const dy = y - dragHandle.startY
      const [ox1, oy1, ox2, oy2] = dragHandle.origBbox

      let nx1 = ox1, ny1 = oy1, nx2 = ox2, ny2 = oy2

      switch (dragHandle.type) {
        case 'move':
          nx1 = Math.max(0, Math.min(1000 - (ox2 - ox1), ox1 + dx))
          ny1 = Math.max(0, Math.min(1000 - (oy2 - oy1), oy1 + dy))
          nx2 = nx1 + (ox2 - ox1)
          ny2 = ny1 + (oy2 - oy1)
          break
        case 'nw':
          nx1 = Math.max(0, Math.min(ox2 - 5, ox1 + dx))
          ny1 = Math.max(0, Math.min(oy2 - 5, oy1 + dy))
          break
        case 'ne':
          nx2 = Math.max(ox1 + 5, Math.min(1000, ox2 + dx))
          ny1 = Math.max(0, Math.min(oy2 - 5, oy1 + dy))
          break
        case 'sw':
          nx1 = Math.max(0, Math.min(ox2 - 5, ox1 + dx))
          ny2 = Math.max(oy1 + 5, Math.min(1000, oy2 + dy))
          break
        case 'se':
          nx2 = Math.max(ox1 + 5, Math.min(1000, ox2 + dx))
          ny2 = Math.max(oy1 + 5, Math.min(1000, oy2 + dy))
          break
      }

      setLiveBbox([nx1, ny1, nx2, ny2])
    }
  }

  // Canvas Mouse Up: finalize drawn box or handle drag
  const handleSvgMouseUp = (e) => {
    if (drawingRect && activeField && onUpdateFieldBbox) {
      const minX = Math.min(drawingRect.startX, drawingRect.curX)
      const maxX = Math.max(drawingRect.startX, drawingRect.curX)
      const minY = Math.min(drawingRect.startY, drawingRect.curY)
      const maxY = Math.max(drawingRect.startY, drawingRect.curY)

      if (maxX - minX >= 6 && maxY - minY >= 6) {
        const finalBox = [minX, minY, maxX, maxY]
        setLiveBbox(finalBox)
        onUpdateFieldBbox(activeField, finalBox)
      }
      setDrawingRect(null)
    }

    if (dragHandle && liveBbox && activeField && onUpdateFieldBbox) {
      onUpdateFieldBbox(activeField, liveBbox)
      setDragHandle(null)
    } else {
      setDragHandle(null)
    }
  }

  const startHandleDrag = (type, e) => {
    e.preventDefault()
    e.stopPropagation()
    const { x, y } = getSvgCoordinates(e)
    setDragHandle({
      type,
      startX: x,
      startY: y,
      origBbox: currentBbox ? [...currentBbox] : [0, 0, 0, 0],
    })
  }

  if (!debugData) return null

  const isInteractive = isDrawMode || isLocateMode || !!dragHandle

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      className={`absolute inset-0 w-full h-full z-10 select-none ${
        isInteractive ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'
      }`}
      style={{ overflow: 'visible' }}
      onMouseDown={handleSvgMouseDown}
      onMouseMove={handleSvgMouseMove}
      onMouseUp={handleSvgMouseUp}
      onMouseLeave={handleSvgMouseUp}
    >
      <defs>
        <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#f43f5e" floodOpacity="0.9" />
        </filter>
        <filter id="glow-gold" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#f59e0b" floodOpacity="0.8" />
        </filter>
        <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#38bdf8" floodOpacity="0.7" />
        </filter>
      </defs>

      {/* ── Mode Banner Overlay on Top of Canvas ── */}
      {isLocateMode && (
        <g className="pointer-events-none">
          <rect x="0" y="0" width="1000" height="36" fill="rgba(15, 23, 42, 0.88)" rx="4" />
          <text x="500" y="24" fill="#38bdf8" fontSize="16" fontWeight="bold" textAnchor="middle">
            🎯 PICK MODE: Click any OCR box on the document to anchor location for {activeField || 'field'}
          </text>
        </g>
      )}

      {isDrawMode && (
        <g className="pointer-events-none">
          <rect x="0" y="0" width="1000" height="36" fill="rgba(15, 23, 42, 0.88)" rx="4" />
          <text x="500" y="24" fill="#a855f7" fontSize="16" fontWeight="bold" textAnchor="middle">
            ✏️ DRAW MODE: Click and drag a rectangle to set coordinates for {activeField || 'field'}
          </text>
        </g>
      )}

      {/* 1. 🟦 YOLO Visual Regions */}
      {showYolo && !isLocateMode && debugData.yolo_regions?.map((reg, idx) => {
        const [x1, y1, x2, y2] = reg.bbox_norm || [0, 0, 0, 0]
        const w = Math.max(3, x2 - x1)
        const h = Math.max(3, y2 - y1)
        const isHovered = hoveredElement?.type === 'yolo' && hoveredElement.index === idx

        return (
          <g
            key={`yolo-${idx}`}
            className="pointer-events-auto cursor-pointer"
            onMouseEnter={() => setHoveredElement && setHoveredElement({ type: 'yolo', index: idx, data: reg })}
            onMouseLeave={() => setHoveredElement && setHoveredElement(null)}
          >
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill={isHovered ? "rgba(56, 189, 248, 0.22)" : "rgba(56, 189, 248, 0.08)"}
              stroke="#0284c7"
              strokeWidth={isHovered ? 2.2 : 1.4}
              strokeDasharray="4 2"
              rx={2}
            />
            <rect
              x={x1}
              y={Math.max(0, y1 - 15)}
              width={Math.min(w, reg.label.length * 6.5 + 10)}
              height="13"
              fill="#0369a1"
              rx="2"
            />
            <text
              x={x1 + 3}
              y={Math.max(9, y1 - 5)}
              fill="#ffffff"
              fontSize="8"
              fontWeight="bold"
              fontFamily="monospace"
            >
              {reg.label}
            </text>
          </g>
        )
      })}

      {/* 2. 🟩 OCR Word / Line Bounding Boxes */}
      {(showOcr || isLocateMode) && debugData.ocr_boxes?.map((ocr, idx) => {
        const [x1, y1, x2, y2] = ocr.bbox_norm || [0, 0, 0, 0]
        const w = Math.max(2, x2 - x1)
        const h = Math.max(2, y2 - y1)
        const isHovered = hoveredElement?.type === 'ocr' && hoveredElement.index === idx

        return (
          <g
            key={`ocr-${idx}`}
            className="pointer-events-auto cursor-pointer"
            onClick={(e) => {
              if (isLocateMode && onPickOcrBox) {
                e.stopPropagation()
                onPickOcrBox(ocr)
              }
            }}
            onMouseEnter={() => setHoveredElement && setHoveredElement({ type: 'ocr', index: idx, data: ocr })}
            onMouseLeave={() => setHoveredElement && setHoveredElement(null)}
          >
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill={
                isLocateMode
                  ? (isHovered ? "rgba(56, 189, 248, 0.45)" : "rgba(34, 197, 94, 0.18)")
                  : (isHovered ? "rgba(34, 197, 94, 0.35)" : "rgba(34, 197, 94, 0.06)")
              }
              stroke={
                isLocateMode
                  ? (isHovered ? "#38bdf8" : "#22c55e")
                  : (isHovered ? "#16a34a" : "#22c55e")
              }
              strokeWidth={isLocateMode ? (isHovered ? 2.5 : 1.4) : (isHovered ? 1.8 : 0.8)}
              rx={1}
            />
          </g>
        )
      })}

      {/* 3. 🟨 TIE Anchors */}
      {showAnchors && !isLocateMode && debugData.tie_anchors?.map((anc, idx) => {
        const [x1, y1, x2, y2] = anc.bbox_norm || [0, 0, 0, 0]
        const w = Math.max(3, x2 - x1)
        const h = Math.max(3, y2 - y1)
        const isHovered = hoveredElement?.type === 'anchor' && hoveredElement.index === idx

        return (
          <g
            key={`anc-${idx}`}
            className="pointer-events-auto cursor-pointer"
            onMouseEnter={() => setHoveredElement && setHoveredElement({ type: 'anchor', index: idx, data: anc })}
            onMouseLeave={() => setHoveredElement && setHoveredElement(null)}
          >
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill={isHovered ? "rgba(234, 179, 8, 0.4)" : "rgba(234, 179, 8, 0.2)"}
              stroke="#ca8a04"
              strokeWidth={isHovered ? 2.5 : 1.6}
              rx={2}
            />
            <rect
              x={x1}
              y={Math.max(0, y1 - 13)}
              width={Math.min(w, anc.anchor.length * 6 + 8)}
              height="11"
              fill="#a16207"
              rx="2"
            />
            <text
              x={x1 + 2}
              y={Math.max(8, y1 - 4)}
              fill="#ffffff"
              fontSize="7.5"
              fontWeight="bold"
            >
              ⚓ {anc.anchor}
            </text>
          </g>
        )
      })}

      {/* 4. 🟪 Handwriting Patches */}
      {showHandwriting && !isLocateMode && debugData.handwriting_regions?.map((hw, idx) => {
        const [x1, y1, x2, y2] = hw.bbox_norm || [0, 0, 0, 0]
        const w = Math.max(3, x2 - x1)
        const h = Math.max(3, y2 - y1)
        const isHovered = hoveredElement?.type === 'handwriting' && hoveredElement.index === idx

        return (
          <g
            key={`hw-${idx}`}
            className="pointer-events-auto cursor-pointer"
            onMouseEnter={() => setHoveredElement && setHoveredElement({ type: 'handwriting', index: idx, data: hw })}
            onMouseLeave={() => setHoveredElement && setHoveredElement(null)}
          >
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill={isHovered ? "rgba(168, 85, 247, 0.3)" : "rgba(168, 85, 247, 0.15)"}
              stroke="#9333ea"
              strokeWidth={2}
              strokeDasharray="4 2"
              rx={3}
            />
            <rect
              x={x1}
              y={Math.max(0, y1 - 14)}
              width="46"
              height="12"
              fill="#7e22ce"
              rx="2"
            />
            <text
              x={x1 + 3}
              y={Math.max(9, y1 - 5)}
              fill="#ffffff"
              fontSize="8"
              fontWeight="bold"
            >
              ✍️ TrOCR
            </text>
          </g>
        )
      })}

      {/* 5. 🟥 Field Provenance BBoxes (Non-Active Fields) */}
      {showFields && Object.entries(debugData.field_evidence || {}).map(([fn, ev]) => {
        if (!ev.bbox_norm || fn === activeField) return null
        const [x1, y1, x2, y2] = ev.bbox_norm
        const w = Math.max(3, x2 - x1)
        const h = Math.max(3, y2 - y1)
        const isHovered = hoveredElement?.type === 'field' && hoveredElement.fieldName === fn

        return (
          <g
            key={`field-${fn}`}
            className="pointer-events-auto cursor-pointer"
            onClick={() => onSelectField && onSelectField(fn)}
            onMouseEnter={() => setHoveredElement && setHoveredElement({ type: 'field', fieldName: fn, data: ev })}
            onMouseLeave={() => setHoveredElement && setHoveredElement(null)}
          >
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill={isHovered ? "rgba(244, 63, 94, 0.25)" : "rgba(244, 63, 94, 0.12)"}
              stroke="#fb7185"
              strokeWidth={1.6}
              rx={2}
            />
            <rect
              x={x1}
              y={Math.max(0, y1 - 15)}
              width={Math.min(130, fn.length * 6.5 + 8)}
              height="13"
              fill="#9f1239"
              rx="2"
            />
            <text
              x={x1 + 3}
              y={Math.max(9, y1 - 5)}
              fill="#ffffff"
              fontSize="8"
              fontWeight="bold"
            >
              {fn}
            </text>
          </g>
        )
      })}

      {/* 6. 🎯 Active Selected Field with Interactive Resize / Move Handles */}
      {activeField && currentBbox && (
        <g key={`active-${activeField}`} className="pointer-events-auto">
          {(() => {
            const [x1, y1, x2, y2] = currentBbox
            const w = Math.max(4, x2 - x1)
            const h = Math.max(4, y2 - y1)
            const handleSize = 8

            return (
              <>
                {/* Main Highlighted Rect with Move Capability */}
                <rect
                  x={x1}
                  y={y1}
                  width={w}
                  height={h}
                  fill="rgba(244, 63, 94, 0.32)"
                  stroke="#e11d48"
                  strokeWidth={2.8}
                  strokeDasharray={dragHandle ? "3 3" : undefined}
                  rx={2}
                  filter="url(#glow-red)"
                  style={{ cursor: 'move' }}
                  onMouseDown={(e) => startHandleDrag('move', e)}
                />

                {/* Field Label Banner with Move Cursor */}
                <rect
                  x={x1}
                  y={Math.max(0, y1 - 18)}
                  width={Math.min(180, activeField.length * 7 + 36)}
                  height="16"
                  fill="#be123c"
                  rx="3"
                  style={{ cursor: 'move' }}
                  onMouseDown={(e) => startHandleDrag('move', e)}
                />
                <text
                  x={x1 + 4}
                  y={Math.max(11, y1 - 6)}
                  fill="#ffffff"
                  fontSize="9.5"
                  fontWeight="bold"
                  style={{ cursor: 'move', userSelect: 'none' }}
                  onMouseDown={(e) => startHandleDrag('move', e)}
                >
                  📍 {activeField}
                </text>

                {/* 4 Corner Resize Handles */}
                {/* NW */}
                <rect
                  x={x1 - handleSize / 2}
                  y={y1 - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#ffffff"
                  stroke="#be123c"
                  strokeWidth={2}
                  rx={1}
                  style={{ cursor: 'nwse-resize' }}
                  onMouseDown={(e) => startHandleDrag('nw', e)}
                />
                {/* NE */}
                <rect
                  x={x2 - handleSize / 2}
                  y={y1 - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#ffffff"
                  stroke="#be123c"
                  strokeWidth={2}
                  rx={1}
                  style={{ cursor: 'nesw-resize' }}
                  onMouseDown={(e) => startHandleDrag('ne', e)}
                />
                {/* SW */}
                <rect
                  x={x1 - handleSize / 2}
                  y={y2 - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#ffffff"
                  stroke="#be123c"
                  strokeWidth={2}
                  rx={1}
                  style={{ cursor: 'nesw-resize' }}
                  onMouseDown={(e) => startHandleDrag('sw', e)}
                />
                {/* SE */}
                <rect
                  x={x2 - handleSize / 2}
                  y={y2 - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#ffffff"
                  stroke="#be123c"
                  strokeWidth={2}
                  rx={1}
                  style={{ cursor: 'nwse-resize' }}
                  onMouseDown={(e) => startHandleDrag('se', e)}
                />
              </>
            )
          })()}
        </g>
      )}

      {/* 7. ✏️ Drawing Box Live Preview */}
      {drawingRect && (
        <g className="pointer-events-none">
          {(() => {
            const minX = Math.min(drawingRect.startX, drawingRect.curX)
            const maxX = Math.max(drawingRect.startX, drawingRect.curX)
            const minY = Math.min(drawingRect.startY, drawingRect.curY)
            const maxY = Math.max(drawingRect.startY, drawingRect.curY)
            const w = maxX - minX
            const h = maxY - minY

            return (
              <>
                <rect
                  x={minX}
                  y={minY}
                  width={w}
                  height={h}
                  fill="rgba(168, 85, 247, 0.35)"
                  stroke="#a855f7"
                  strokeWidth={2.4}
                  strokeDasharray="4 3"
                  rx={2}
                />
                <text
                  x={minX + 4}
                  y={Math.max(12, minY - 4)}
                  fill="#c084fc"
                  fontSize="9"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  [{minX}, {minY}, {maxX}, {maxY}]
                </text>
              </>
            )
          })()}
        </g>
      )}
    </svg>
  )
}

// ── Toolbar Layer Controls Component ────────────────────────────────────────

export function DebugLayerControls({
  isDebugMode,
  setIsDebugMode,
  debugData,
  isLoading,
  showYolo,
  setShowYolo,
  showOcr,
  setShowOcr,
  showAnchors,
  setShowAnchors,
  showHandwriting,
  setShowHandwriting,
  showFields,
  setShowFields,
  activeField,
  setActiveField,
  isLocateMode = false,
  setIsLocateMode = null,
  isDrawMode = false,
  setIsDrawMode = null,
}) {
  const summary = debugData?.summary || {}

  if (!isDebugMode) {
    return (
      <button
        type="button"
        onClick={() => setIsDebugMode(true)}
        className="px-2.5 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700"
        title="Activate Invoice Debug Mode (Visual YOLO regions, OCR boxes, TIE anchors, Evidence Inspector, Location Editing)"
      >
        <Cpu size={13} className="text-indigo-500 dark:text-indigo-400" />
        <span>Debug Mode</span>
      </button>
    )
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Active Debug Toggle */}
      <button
        type="button"
        onClick={() => {
          setIsDebugMode(false)
          if (setIsLocateMode) setIsLocateMode(false)
          if (setIsDrawMode) setIsDrawMode(false)
        }}
        className="px-2.5 py-1 rounded-md text-xs font-bold flex items-center gap-1.5 transition-all bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm ring-1 ring-indigo-400"
      >
        <Cpu size={13} className="animate-pulse text-amber-300" />
        <span>Debug Active</span>
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
      </button>

      {/* Interactive Location Mode Quick Toggles */}
      {setIsLocateMode && (
        <button
          type="button"
          onClick={() => {
            setIsLocateMode(!isLocateMode)
            if (setIsDrawMode) setIsDrawMode(false)
          }}
          className={`px-2 py-0.5 rounded text-xs font-bold flex items-center gap-1 transition-all ${
            isLocateMode
              ? 'bg-sky-500 text-white shadow-md ring-2 ring-sky-300'
              : 'bg-slate-800 hover:bg-slate-700 text-sky-300 border border-sky-500/40'
          }`}
          title="Click any text box on the document to anchor location for the active field"
        >
          <Target size={12} className={isLocateMode ? 'animate-spin' : ''} />
          <span>Pick Location</span>
        </button>
      )}

      {setIsDrawMode && (
        <button
          type="button"
          onClick={() => {
            setIsDrawMode(!isDrawMode)
            if (setIsLocateMode) setIsLocateMode(false)
          }}
          className={`px-2 py-0.5 rounded text-xs font-bold flex items-center gap-1 transition-all ${
            isDrawMode
              ? 'bg-purple-600 text-white shadow-md ring-2 ring-purple-300'
              : 'bg-slate-800 hover:bg-slate-700 text-purple-300 border border-purple-500/40'
          }`}
          title="Drag a custom bounding box rectangle directly on the document"
        >
          <Crop size={12} />
          <span>Draw Box</span>
        </button>
      )}

      {/* Layer Visibility Pills */}
      <div className="flex items-center gap-1 bg-slate-900/90 backdrop-blur-xs px-2 py-0.5 rounded-md border border-slate-700 text-[11px] text-white">
        <span className="text-slate-400 font-semibold uppercase text-[10px] mr-1 flex items-center gap-1">
          <Layers size={11} /> Layers:
        </span>

        {/* 🟦 YOLO */}
        <button
          type="button"
          onClick={() => setShowYolo(!showYolo)}
          className={`px-1.5 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
            showYolo ? 'bg-sky-500/30 text-sky-300 border border-sky-500/50' : 'text-slate-500 opacity-60'
          }`}
          title="Toggle YOLO Detected Visual Regions"
        >
          <span className="w-2 h-2 rounded-xs bg-sky-400 inline-block" />
          <span>YOLO ({summary.yolo_regions_count ?? 0})</span>
        </button>

        {/* 🟩 OCR */}
        <button
          type="button"
          onClick={() => setShowOcr(!showOcr)}
          className={`px-1.5 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
            showOcr ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-500/50' : 'text-slate-500 opacity-60'
          }`}
          title="Toggle Full-Page OCR Word & Line Boxes"
        >
          <span className="w-2 h-2 rounded-xs bg-emerald-400 inline-block" />
          <span>OCR ({summary.ocr_words_count ?? 0})</span>
        </button>

        {/* 🟨 TIE Anchors */}
        <button
          type="button"
          onClick={() => setShowAnchors(!showAnchors)}
          className={`px-1.5 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
            showAnchors ? 'bg-amber-500/30 text-amber-300 border border-amber-500/50' : 'text-slate-500 opacity-60'
          }`}
          title="Toggle TIE Matched Spatial Anchors"
        >
          <span className="w-2 h-2 rounded-xs bg-amber-400 inline-block" />
          <span>Anchors ({summary.tie_anchors_count ?? 0})</span>
        </button>

        {/* 🟪 Handwriting */}
        <button
          type="button"
          onClick={() => setShowHandwriting(!showHandwriting)}
          className={`px-1.5 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
            showHandwriting ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50' : 'text-slate-500 opacity-60'
          }`}
          title="Toggle Detected Handwriting Patches"
        >
          <span className="w-2 h-2 rounded-xs bg-purple-400 inline-block" />
          <span>HW ({summary.handwriting_count ?? 0})</span>
        </button>

        {/* 🟥 Field Provenance */}
        <button
          type="button"
          onClick={() => setShowFields(!showFields)}
          className={`px-1.5 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
            showFields ? 'bg-rose-500/30 text-rose-300 border border-rose-500/50' : 'text-slate-500 opacity-60'
          }`}
          title="Toggle Extracted Field Provenance BBoxes"
        >
          <span className="w-2 h-2 rounded-xs bg-rose-500 inline-block" />
          <span>Fields ({summary.fields_located_count ?? 0})</span>
        </button>
      </div>

      {/* Disagreements Alert Badge */}
      {summary.disagreements_count > 0 && (
        <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[11px] font-semibold flex items-center gap-1 animate-pulse">
          <AlertTriangle size={12} />
          {summary.disagreements_count} Disagreement(s)
        </span>
      )}
    </div>
  )
}

// ── Hovered Element Inspection Tooltip ──────────────────────────────────────

export function DebugHoverTooltip({ hoveredElement }) {
  if (!hoveredElement) return null
  const { type, data, fieldName } = hoveredElement

  return (
    <div className="absolute top-3 left-3 z-40 bg-slate-950/95 text-white border border-slate-700 shadow-2xl rounded-lg p-2.5 max-w-xs text-xs pointer-events-none backdrop-blur-md animate-in fade-in zoom-in-95 duration-100">
      {type === 'yolo' && (
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 mb-1.5">
            <span className="font-bold text-sky-400 flex items-center gap-1">🟦 YOLO Region</span>
            <span className="text-[10px] font-mono text-slate-400">{Math.round((data.confidence || 0) * 100)}%</span>
          </div>
          <div className="font-semibold text-slate-200 text-sm">{data.label}</div>
          <div className="text-[10px] text-slate-400 font-mono mt-1">BBox: [{data.bbox_norm?.join(', ')}]</div>
        </div>
      )}

      {type === 'ocr' && (
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 mb-1.5">
            <span className="font-bold text-emerald-400 flex items-center gap-1">🟩 OCR Token</span>
            <span className="text-[10px] font-mono text-slate-400">{Math.round((data.confidence || 0) * 100)}%</span>
          </div>
          <div className="font-mono text-slate-100 bg-slate-900 p-1.5 rounded border border-slate-800 text-xs break-all">
            "{data.text}"
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono mt-1">
            <span>Source: {data.source || 'ocr'}</span>
            <span>Line {data.line_no}</span>
          </div>
          <div className="text-[9px] text-slate-500 font-mono mt-0.5">BBox: [{data.bbox_norm?.join(', ')}]</div>
        </div>
      )}

      {type === 'anchor' && (
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 mb-1.5">
            <span className="font-bold text-amber-400 flex items-center gap-1">🟨 TIE Spatial Anchor</span>
            <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1 rounded">Anchor</span>
          </div>
          <div className="font-semibold text-amber-200">{data.anchor}</div>
          <div className="text-[10px] text-slate-400 font-mono mt-1">BBox: [{data.bbox_norm?.join(', ')}]</div>
        </div>
      )}

      {type === 'handwriting' && (
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 mb-1.5">
            <span className="font-bold text-purple-400 flex items-center gap-1">🟪 Handwriting Zone</span>
            <span className="text-[10px] text-purple-300">TrOCR</span>
          </div>
          <div className="text-slate-200 text-xs">Handwritten or stamp patch routed to specialized OCR.</div>
        </div>
      )}

      {type === 'field' && (
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 mb-1.5">
            <span className="font-bold text-rose-400 flex items-center gap-1">🟥 Field Provenance</span>
            <span className="text-[10px] font-mono text-slate-300">{Math.round((data.confidence || 0) * 100)}%</span>
          </div>
          <div className="font-semibold text-rose-200">{fieldName}</div>
          <div className="text-white font-bold text-sm mt-0.5">{data.selected_value}</div>
          <div className="text-[11px] text-slate-300 mt-1 italic">
            {data.selection_reason || `Source: ${data.selected_source}`}
          </div>
          <div className="text-[9px] text-slate-400 font-mono mt-0.5">BBox: [{data.bbox_norm?.join(', ')}]</div>
        </div>
      )}
    </div>
  )
}

// ── Field Evidence & Interactive Location Inspector ─────────────────────────

export function FieldEvidenceInspector({
  debugData,
  activeField,
  setActiveField,
  formData,
  onUpdateFieldBbox,
  onUpdateFormField,
  isLocateMode = false,
  setIsLocateMode = null,
  isDrawMode = false,
  setIsDrawMode = null,
  onClose,
}) {
  const [selectedFieldKey, setSelectedFieldKey] = useState(activeField || 'grand_total')
  const [manualCoords, setManualCoords] = useState('')
  const [isEditingCoords, setIsEditingCoords] = useState(false)

  // Keep selectedFieldKey aligned with activeField
  useEffect(() => {
    if (activeField) setSelectedFieldKey(activeField)
  }, [activeField])

  const evidenceMap = debugData?.field_evidence || {}
  const allFields = Object.keys(FIELD_NAME_MAP)
  const currentEvidence = evidenceMap[selectedFieldKey]
  const currentBbox = currentEvidence?.bbox_norm

  // Current value in form
  const formValue = formData ? getFieldValue(formData, selectedFieldKey) : ''

  // Text inside the current bounding box
  const textInBox = currentBbox && debugData?.ocr_boxes ? getOcrTextInBox(currentBbox, debugData.ocr_boxes) : ''

  // Run "Snap to Form Value"
  const handleSnapToFormValue = () => {
    if (!formValue) return
    const match = findOcrMatchesForValue(formValue, debugData?.ocr_boxes || [])
    if (match && onUpdateFieldBbox) {
      onUpdateFieldBbox(selectedFieldKey, match.bbox_norm, match.matchedText)
    }
  }

  // Commit manual coordinates
  const handleSaveCoords = () => {
    const parts = manualCoords.split(',').map(s => parseInt(s.trim(), 10))
    if (parts.length === 4 && parts.every(n => !isNaN(n) && n >= 0 && n <= 1000)) {
      if (onUpdateFieldBbox) {
        onUpdateFieldBbox(selectedFieldKey, parts)
      }
      setIsEditingCoords(false)
    }
  }

  return (
    <div className="card p-3.5 bg-slate-900 text-white border border-indigo-500/40 shadow-xl rounded-lg space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
            <Sparkles size={14} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider">Field Evidence & Location Inspector</h4>
            <p className="text-[11px] text-slate-400">Verifiable model reasoning & interactive bounding-box modification</p>
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Field Selector Dropdown / Pills */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] text-slate-400 font-semibold">Select Field:</span>
        <select
          value={selectedFieldKey}
          onChange={e => {
            const val = e.target.value
            setSelectedFieldKey(val)
            if (setActiveField) setActiveField(val)
          }}
          className="bg-slate-800 border border-slate-700 text-white text-xs font-semibold rounded px-2.5 py-1 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
        >
          {allFields.map(fn => {
            const hasBox = !!evidenceMap[fn]?.bbox_norm
            return (
              <option key={fn} value={fn}>
                {hasBox ? '📍 ' : '⚪ '} {FIELD_NAME_MAP[fn]?.label || fn}
              </option>
            )
          })}
        </select>

        {/* Quick Jump Buttons for Critical Fields */}
        {['grand_total', 'invoice_number', 'invoice_date', 'vendor_name'].map(k => (
          <button
            key={k}
            type="button"
            onClick={() => {
              setSelectedFieldKey(k)
              if (setActiveField) setActiveField(k)
            }}
            className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-all ${
              selectedFieldKey === k
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
            }`}
          >
            {k.replace('_', ' ')}
          </button>
        ))}
      </div>

      {/* ── Bounding Box & Re-anchoring Controls Card ── */}
      <div className="p-3 bg-slate-950/80 rounded-lg border border-indigo-500/30 space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Crosshair size={13} className="text-rose-400" />
              Document Location:
            </span>
            {currentBbox ? (
              <span className="font-mono text-xs text-rose-300 bg-rose-950/60 px-2 py-0.5 rounded border border-rose-500/40">
                [{currentBbox.join(', ')}]
              </span>
            ) : (
              <span className="text-xs text-amber-400 italic">No bounding box assigned</span>
            )}
          </div>

          {/* Location Modification Actions */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* 1. Snap to Form Value */}
            <button
              type="button"
              onClick={handleSnapToFormValue}
              disabled={!formValue}
              className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-[11px] font-semibold flex items-center gap-1 shadow-sm transition-all"
              title={`Find "${formValue}" in OCR text and snap bounding box automatically`}
            >
              <RefreshCw size={11} />
              <span>Snap to Form Value</span>
            </button>

            {/* 2. Pick from Document */}
            {setIsLocateMode && (
              <button
                type="button"
                onClick={() => {
                  setIsLocateMode(!isLocateMode)
                  if (setIsDrawMode) setIsDrawMode(false)
                }}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold flex items-center gap-1 transition-all ${
                  isLocateMode
                    ? 'bg-sky-500 text-white ring-2 ring-sky-300 shadow-md'
                    : 'bg-slate-800 hover:bg-slate-700 text-sky-300 border border-sky-500/40'
                }`}
                title="Click any text box on the invoice to anchor this field"
              >
                <Target size={11} className={isLocateMode ? 'animate-spin' : ''} />
                <span>{isLocateMode ? 'Click on Document...' : 'Pick from Doc'}</span>
              </button>
            )}

            {/* 3. Draw Custom Box */}
            {setIsDrawMode && (
              <button
                type="button"
                onClick={() => {
                  setIsDrawMode(!isDrawMode)
                  if (setIsLocateMode) setIsLocateMode(false)
                }}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold flex items-center gap-1 transition-all ${
                  isDrawMode
                    ? 'bg-purple-600 text-white ring-2 ring-purple-300 shadow-md'
                    : 'bg-slate-800 hover:bg-slate-700 text-purple-300 border border-purple-500/40'
                }`}
                title="Drag a rectangle on the document canvas to define coordinates"
              >
                <Crop size={11} />
                <span>{isDrawMode ? 'Drawing on Doc...' : 'Draw Box'}</span>
              </button>
            )}

            {/* 4. Manual Coords */}
            <button
              type="button"
              onClick={() => {
                if (!isEditingCoords) {
                  setManualCoords(currentBbox ? currentBbox.join(', ') : '100, 100, 300, 140')
                }
                setIsEditingCoords(!isEditingCoords)
              }}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium border border-slate-700"
              title="Edit [x1, y1, x2, y2] manually"
            >
              <Pencil size={11} />
            </button>

            {/* 5. Clear Box */}
            {currentBbox && onUpdateFieldBbox && (
              <button
                type="button"
                onClick={() => onUpdateFieldBbox(selectedFieldKey, null)}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-rose-950 text-rose-400 hover:text-rose-300 text-[11px] font-medium border border-slate-700 hover:border-rose-700 transition-colors"
                title="Clear bounding box for this field"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Manual Coords Input Row */}
        {isEditingCoords && (
          <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
            <span className="text-[11px] text-slate-400 font-mono">BBox [x1, y1, x2, y2]:</span>
            <input
              type="text"
              value={manualCoords}
              onChange={e => setManualCoords(e.target.value)}
              placeholder="e.g. 150, 200, 450, 240"
              className="bg-slate-900 border border-slate-700 text-white font-mono text-xs px-2 py-1 rounded flex-1 focus:ring-1 focus:ring-indigo-500"
            />
            <button
              type="button"
              onClick={handleSaveCoords}
              className="btn-primary text-xs px-2.5 py-1"
            >
              Apply
            </button>
          </div>
        )}

        {/* Text in Bounding Box vs Form Value comparison */}
        {currentBbox && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-800/80">
            <div className="bg-slate-900 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">Value in Form</span>
              <span className="text-white font-bold block truncate mt-0.5">
                {formValue || <span className="text-slate-500 italic">Empty</span>}
              </span>
            </div>

            <div className="bg-slate-900 p-2 rounded border border-slate-800 flex items-center justify-between gap-2">
              <div className="min-w-0 flex-1">
                <span className="text-[10px] text-emerald-400 uppercase tracking-wider block font-semibold">OCR Text in Box</span>
                <span className="text-emerald-300 font-semibold block truncate mt-0.5">
                  {textInBox ? `"${textInBox}"` : <span className="text-slate-500 italic">No OCR words inside box</span>}
                </span>
              </div>
              {textInBox && textInBox !== formValue && onUpdateFormField && (
                <button
                  type="button"
                  onClick={() => onUpdateFormField(selectedFieldKey, textInBox)}
                  className="px-2 py-1 rounded bg-emerald-700 hover:bg-emerald-600 text-white text-[10px] font-bold flex items-center gap-1 shrink-0 transition-colors shadow-xs"
                  title="Copy the OCR text inside this box into the form input"
                >
                  <ArrowRight size={10} /> Sync to Form
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Selected Field Evidence Breakdown */}
      {currentEvidence ? (
        <div className="space-y-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {/* Selected Value */}
            <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">Selected Value</span>
              <span className="text-base font-bold text-emerald-400 block truncate mt-0.5">
                {currentEvidence.selected_value || 'None'}
              </span>
            </div>

            {/* Decision Source */}
            <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">Decision Source</span>
              <span className="text-xs font-bold text-indigo-300 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-500/40 inline-block mt-1">
                {currentEvidence.selected_source}
              </span>
            </div>

            {/* Confidence */}
            <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">Confidence Score</span>
              <span className="text-xs font-bold text-amber-300 font-mono block mt-1">
                {Math.round((currentEvidence.confidence || 0) * 100)}%
              </span>
            </div>

            {/* Validation Rule Status */}
            <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">Validation Check</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded inline-block mt-1 ${
                currentEvidence.validation_status === 'passed'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
              }`}>
                {currentEvidence.validation_status === 'passed' ? '✓ PASS' : '⚠️ FLAGGED'}
              </span>
            </div>
          </div>

          {/* Selection Reason Explanation */}
          <div className="bg-slate-900/90 p-2 rounded border border-slate-800 flex items-start gap-2 text-xs">
            <Info size={14} className="text-sky-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-300">Selection Reasoning: </span>
              <span className="text-slate-400">{currentEvidence.selection_reason}</span>
            </div>
          </div>

          {/* Competing Candidates Comparison Table */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wide flex items-center gap-1.5">
                <Crosshair size={12} className="text-rose-400" />
                Competing Model Candidates ({currentEvidence.candidates?.length || 1})
              </span>
              {currentEvidence.disagreement_score > 0.05 && (
                <span className="text-[10px] text-amber-400 font-semibold bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/30">
                  Disagreement: {Math.round(currentEvidence.disagreement_score * 100)}%
                </span>
              )}
            </div>

            <div className="overflow-x-auto rounded border border-slate-800">
              <table className="w-full text-[11px] text-left">
                <thead className="bg-slate-900 text-slate-400 uppercase text-[10px]">
                  <tr>
                    <th className="py-1.5 px-2.5 font-semibold">Engine / Source</th>
                    <th className="py-1.5 px-2.5 font-semibold">Extracted Value</th>
                    <th className="py-1.5 px-2.5 font-semibold">Confidence</th>
                    <th className="py-1.5 px-2.5 font-semibold">Page</th>
                    <th className="py-1.5 px-2.5 font-semibold">Bounding Box</th>
                    <th className="py-1.5 px-2.5 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {(currentEvidence.candidates || []).map((c, cIdx) => {
                    const isSelected = String(c.value) === String(currentEvidence.selected_value)
                    const cBox = c.bbox_norm || c.bbox
                    return (
                      <tr key={cIdx} className={isSelected ? "bg-emerald-950/30" : "hover:bg-slate-900/60"}>
                        <td className="py-1.5 px-2.5 font-semibold font-mono text-indigo-300">
                          {c.source || currentEvidence.selected_source}
                        </td>
                        <td className="py-1.5 px-2.5 font-bold text-white">
                          {c.value}
                        </td>
                        <td className="py-1.5 px-2.5 font-mono text-amber-300">
                          {Math.round((c.confidence || currentEvidence.confidence || 0) * 100)}%
                        </td>
                        <td className="py-1.5 px-2.5 text-slate-400">
                          P.{c.page || 1}
                        </td>
                        <td className="py-1.5 px-2.5 text-slate-500 font-mono text-[10px]">
                          {cBox ? `[${cBox.join(',')}]` : 'N/A'}
                        </td>
                        <td className="py-1.5 px-2.5">
                          {isSelected ? (
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                              Selected ✓
                            </span>
                          ) : (
                            <button
                              type="button"
                              onClick={() => {
                                if (cBox && onUpdateFieldBbox) {
                                  onUpdateFieldBbox(selectedFieldKey, cBox, c.value)
                                }
                                if (onUpdateFormField) {
                                  onUpdateFormField(selectedFieldKey, c.value)
                                }
                              }}
                              className="text-[10px] font-semibold text-indigo-400 hover:text-indigo-300 underline"
                            >
                              Choose This
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-3 text-center text-xs text-slate-400 bg-slate-950/40 rounded border border-slate-800">
          No model predictions recorded for this field yet. You can use "Snap to Form Value", "Pick from Doc", or "Draw Box" to define its location.
        </div>
      )}
    </div>
  )
}
