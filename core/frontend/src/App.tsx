// frontend/src/App.tsx
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuthStore } from "./lib/hooks/useAuth";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import DatasetDetail from "./pages/DatasetDetail";
import Pipelines from "./pages/Pipelines";
import PipelineDetail from "./pages/PipelineDetail";
import Models from "./pages/Models";
import ModelDetail from "./pages/ModelDetail";
import FeatureStore from "./pages/FeatureStore";
import Inference from "./pages/Inference";
import Reports from "./pages/Reports";
import Monitoring from "./pages/Monitoring";
import Login from "./pages/Login";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((s: any) => s.isAuthenticated);
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="datasets/:id" element={<DatasetDetail />} />
        <Route path="pipelines" element={<Pipelines />} />
        <Route path="pipelines/:id" element={<PipelineDetail />} />
        <Route path="models" element={<Models />} />
        <Route path="models/:id" element={<ModelDetail />} />
        <Route path="features" element={<FeatureStore />} />
        <Route path="inference" element={<Inference />} />
        <Route path="reports" element={<Reports />} />
        <Route path="monitoring" element={<Monitoring />} />
      </Route>
    </Routes>
  );
}
