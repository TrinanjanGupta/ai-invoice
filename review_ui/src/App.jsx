import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/Layout'
import UploadPage from './pages/UploadPage'
import InvoiceListPage from './pages/InvoiceListPage'
import ReviewPage from './pages/ReviewPage'

export default function App() {
  return (
    <>
      <Toaster position="top-right" toastOptions={{ duration: 4000 }} />
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UploadPage />} />
          <Route path="/invoices" element={<InvoiceListPage />} />
          <Route path="/invoices/:jobId" element={<ReviewPage />} />
        </Route>
      </Routes>
    </>
  )
}
