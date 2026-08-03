// frontend/src/pages/Reports.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { FileText, Plus, Download, Clock, BarChart3 } from "lucide-react";
import toast from "react-hot-toast";

const reportTypes = [
  { value: "dataset_profile", label: "Dataset Profile" },
  { value: "data_quality", label: "Data Quality" },
  { value: "model_performance", label: "Model Performance" },
  { value: "pipeline_execution", label: "Pipeline Execution" },
  { value: "feature_analysis", label: "Feature Analysis" },
];

export default function Reports() {
  const [showGenerate, setShowGenerate] = useState(false);
  const queryClient = useQueryClient();

  const { data: reports, isLoading } = useQuery({
    queryKey: ["reports"],
    queryFn: () => apiClient.get("/reports").then((r) => r.data),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Reports</h1>
          <p className="text-gray-500 mt-1">Generated analysis reports</p>
        </div>
        <button
          onClick={() => setShowGenerate(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Generate Report
        </button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-5 bg-gray-200 rounded w-1/2 mb-3" />
              <div className="h-4 bg-gray-200 rounded w-full" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reports?.map((report: any) => (
            <div key={report.id} className="card">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold">{report.title}</h3>
                </div>
                <span className="badge badge-info">{report.report_type}</span>
              </div>
              <p className="text-sm text-gray-500 mb-3">{report.summary}</p>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {new Date(report.generated_at).toLocaleString()}
                </span>
                {report.exports && Object.keys(report.exports).length > 0 && (
                  <span className="flex items-center gap-1 text-agri-600">
                    <Download className="w-3 h-3" />
                    {Object.keys(report.exports).join(", ").toUpperCase()}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && reports?.length === 0 && (
        <div className="text-center py-12">
          <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="font-medium">No reports generated</h3>
          <p className="text-gray-500 text-sm mt-1">
            Generate your first report to see insights
          </p>
        </div>
      )}

      {showGenerate && (
        <GenerateReportModal onClose={() => setShowGenerate(false)} />
      )}
    </div>
  );
}

function GenerateReportModal({ onClose }: { onClose: () => void }) {
  const [reportType, setReportType] = useState("dataset_profile");
  const [datasetId, setDatasetId] = useState("");
  const [modelId, setModelId] = useState("");
  const queryClient = useQueryClient();

  const generateMutation = useMutation({
    mutationFn: (data: any) => apiClient.post("/reports/generate", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reports"] });
      toast.success("Report generated");
      onClose();
    },
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md">
        <h2 className="font-semibold mb-4">Generate Report</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            generateMutation.mutate({
              report_type: reportType,
              dataset_id: datasetId || undefined,
              model_id: modelId || undefined,
            });
          }}
          className="space-y-4"
        >
          <div>
            <label className="block text-sm font-medium mb-1">
              Report Type
            </label>
            <select
              value={reportType}
              onChange={(e) => setReportType(e.target.value)}
              className="input-field"
            >
              {reportTypes.map((rt) => (
                <option key={rt.value} value={rt.value}>
                  {rt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Dataset ID (optional)
            </label>
            <input
              type="text"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="input-field"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Model ID (optional)
            </label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="input-field"
            />
          </div>
          <div className="flex gap-3 justify-end">
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={generateMutation.isPending}
              className="btn-primary"
            >
              Generate
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
