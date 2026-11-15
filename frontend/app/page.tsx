"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/utils";
import { CardSkeleton, ChartSkeleton, TableSkeleton } from "@/components/common/Skeleton";
import { EmptyState, ErrorState } from "@/components/common/EmptyState";
import IndiaMapWrapper from "@/components/map/IndiaMapWrapper";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from "recharts";
import { Activity, MapPin, TrendingUp, Award } from "lucide-react";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "map", label: "Map" },
  { id: "models", label: "Models" },
  { id: "forecast", label: "Forecast" },
  { id: "predict", label: "Predict" },
  { id: "importance", label: "Features" },
];

export default function Dashboard() {
  const [tab, setTab] = useState("overview");
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [selectedYear, setSelectedYear] = useState<number>(2030);

  const { data: states, isLoading: statesLoading } = useQuery({
    queryKey: ["states"],
    queryFn: api.states,
  });

  const { data: latestEnrollment } = useQuery({
    queryKey: ["latestEnrollment"],
    queryFn: api.latestEnrollment,
  });

  const { data: modelComparison } = useQuery({
    queryKey: ["modelComparison"],
    queryFn: api.modelComparison,
  });

  const { data: bestModel } = useQuery({
    queryKey: ["bestModel"],
    queryFn: api.bestModel,
  });

  const { data: allForecasts } = useQuery({
    queryKey: ["allForecasts"],
    queryFn: api.allForecasts,
  });

  const { data: forecast2030 } = useQuery({
    queryKey: ["forecast2030"],
    queryFn: api.forecast2030,
  });

  const { data: availableYears } = useQuery({
    queryKey: ["availableYears"],
    queryFn: api.availableYears,
  });

  const { data: yearForecast } = useQuery({
    queryKey: ["yearForecast", selectedYear],
    queryFn: () => api.forecastByYear(selectedYear),
    enabled: selectedYear >= 2020 && selectedYear <= 2030,
  });

  const { data: featureImportance } = useQuery({
    queryKey: ["featureImportance"],
    queryFn: api.featureImportance,
  });

  const enrollmentMap = useMemo(() => {
    const m: Record<string, number> = {};
    latestEnrollment?.forEach((p) => {
      m[p.state] = p.enrollment;
    });
    return m;
  }, [latestEnrollment]);

  const forecastMap = useMemo(() => {
    const m: Record<string, number> = {};
    const data = yearForecast || forecast2030;
    if (Array.isArray(data)) {
      data.forEach((p: any) => {
        m[p.state] = p.xgboost ?? p.forecast;
      });
    }
    return m;
  }, [yearForecast, forecast2030]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-surface/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground">
              Higher Education Enrollment Forecasting
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-warning/20 text-warning border border-warning/30">BETA</span>
            </h1>
            <p className="text-xs text-muted">AISHE · Indian States · XGBoost Model · Forecast to 2030</p>
          </div>
          <div className="flex items-center gap-2">
            {bestModel && (
              <span className="text-xs px-3 py-1 rounded-full bg-accent/10 text-accent border border-accent/20">
                Best: {bestModel.model}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Tabs */}
      <nav className="border-b border-border bg-surface/30">
        <div className="max-w-7xl mx-auto px-6 flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === t.id
                  ? "border-accent text-accent"
                  : "border-transparent text-muted hover:text-foreground"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {tab === "overview" && (
          <OverviewTab
            states={states}
            bestModel={bestModel}
            modelComparison={modelComparison}
            latestEnrollment={latestEnrollment}
          />
        )}
        {tab === "map" && (
          <MapTab
            enrollmentMap={enrollmentMap}
            forecastMap={forecastMap}
            selectedState={selectedState}
            onSelectState={setSelectedState}
            selectedYear={selectedYear}
            onYearChange={setSelectedYear}
            availableYears={availableYears?.years || []}
          />
        )}
        {tab === "models" && (
          <ModelsTab modelComparison={modelComparison} />
        )}
        {tab === "forecast" && (
          <ForecastTab forecast2030={forecast2030} />
        )}
        {tab === "importance" && (
          <ImportanceTab featureImportance={featureImportance} />
        )}
        {tab === "predict" && <PredictTab />}
      </main>
    </div>
  );
}

function OverviewTab({ states, bestModel, modelComparison, latestEnrollment }: any) {
  const totalEnrollment = latestEnrollment?.reduce((s: number, p: any) => s + p.enrollment, 0) || 0;
  const topStates = [...(latestEnrollment || [])].sort((a: any, b: any) => b.enrollment - a.enrollment).slice(0, 5);

  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard icon={MapPin} label="States" value={states?.length || 0} />
        <KpiCard icon={Activity} label="Total Enrollment (2019)" value={formatNumber(totalEnrollment)} />
        <KpiCard icon={Award} label="Best Model" value={bestModel?.model || "—"} />
        <KpiCard icon={TrendingUp} label="Best MAE" value={bestModel ? formatNumber(bestModel.metrics.MAE) : "—"} />
      </div>

      {/* Top states */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-sm font-medium mb-4">Top 5 States by Enrollment (2019)</h3>
          {topStates.map((s: any, i: number) => (
            <div key={s.state} className="flex items-center justify-between py-2 border-b border-border last:border-0">
              <span className="text-sm">{i + 1}. {s.state}</span>
              <span className="text-sm font-mono text-muted">{formatNumber(s.enrollment)}</span>
            </div>
          ))}
        </div>

        <div className="card">
          <h3 className="text-sm font-medium mb-4">Model Performance Summary</h3>
          {modelComparison ? (
            <div className="space-y-2">
              {modelComparison.slice(0, 6).map((m: any) => (
                <div key={m.Model} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                  <span className="text-sm">{m.Model}</span>
                  <span className="text-xs font-mono text-muted">R² {m.R2.toFixed(4)}</span>
                </div>
              ))}
            </div>
          ) : (
            <CardSkeleton />
          )}
        </div>
      </div>
    </div>
  );
}

function MapTab({ enrollmentMap, forecastMap, selectedState, onSelectState, selectedYear, onYearChange, availableYears }: any) {
  const [view, setView] = useState<"enrollment" | "forecast">("enrollment");
  const data = view === "enrollment" ? enrollmentMap : forecastMap;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="text-lg font-medium">India — State-Level View</h2>
        <div className="flex items-center gap-3">
          {view === "forecast" && (
            <select
              value={selectedYear}
              onChange={(e) => onYearChange(Number(e.target.value))}
              className="px-3 py-1.5 text-xs rounded-lg bg-surface-2 text-foreground border border-border"
            >
              {(availableYears || []).map((y: number) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => setView("enrollment")}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                view === "enrollment" ? "bg-accent text-white" : "bg-surface-2 text-muted hover:text-foreground"
              }`}
            >
              Enrollment 2019
            </button>
            <button
              onClick={() => setView("forecast")}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                view === "forecast" ? "bg-accent text-white" : "bg-surface-2 text-muted hover:text-foreground"
              }`}
            >
              Forecast (XGBoost)
            </button>
          </div>
        </div>
      </div>

      <div className="card p-2">
        <IndiaMapWrapper
          data={data}
          onStateClick={onSelectState}
          selectedState={selectedState}
          legendTitle={view === "enrollment" ? "Enrollment 2019" : `Forecast ${selectedYear}`}
        />
      </div>

      {selectedState && (
        <div className="card">
          <h3 className="text-sm font-medium mb-2">Selected: {selectedState}</h3>
          <p className="text-xs text-muted">
            {view === "enrollment"
              ? `Enrollment 2019: ${formatNumber(data[selectedState] || 0)}`
              : `XGBoost Forecast ${selectedYear}: ${formatNumber(data[selectedState] || 0)}`}
          </p>
        </div>
      )}
    </div>
  );
}

function ModelsTab({ modelComparison }: any) {
  if (!modelComparison) return <TableSkeleton />;

  const chartData = modelComparison.map((m: any) => ({
    name: m.Model,
    MAE: m.MAE,
    R2: m.R2,
  }));

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-sm font-medium mb-4">Model Comparison — MAE</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
            <XAxis type="number" stroke="#6b6b7b" fontSize={11} />
            <YAxis dataKey="name" type="category" stroke="#6b6b7b" fontSize={11} width={120} />
            <Tooltip
              contentStyle={{ background: "#1a1a24", border: "1px solid #2a2a3a", borderRadius: 8 }}
              labelStyle={{ color: "#f4f4f5" }}
            />
            <Bar dataKey="MAE" fill="#6366f1" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3 className="text-sm font-medium mb-4">Detailed Metrics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 text-muted font-medium">Model</th>
                <th className="text-right py-2 text-muted font-medium">MAE</th>
                <th className="text-right py-2 text-muted font-medium">RMSE</th>
                <th className="text-right py-2 text-muted font-medium">MAPE</th>
                <th className="text-right py-2 text-muted font-medium">R²</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((m: any) => (
                <tr key={m.Model} className="border-b border-border last:border-0">
                  <td className="py-2 font-medium">{m.Model}</td>
                  <td className="text-right py-2 font-mono text-xs">{m.MAE.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td className="text-right py-2 font-mono text-xs">{m.RMSE.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td className="text-right py-2 font-mono text-xs">{m.MAPE.toFixed(2)}%</td>
                  <td className="text-right py-2 font-mono text-xs">{m.R2.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ForecastTab({ forecast2030 }: any) {
  if (!forecast2030) return <TableSkeleton />;

  // Aggregate by year
  const byYear: Record<number, number> = {};
  // We only have 2030 snapshot, so let's fetch all years
  // For now, show the 2030 forecasts per state

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-sm font-medium mb-4">XGBoost Forecast 2030 by State (Beta)</h3>
        <p className="text-xs text-muted mb-4">
          Ensemble model (XGBoost + ARIMA + ETS + TBATS + Linear Trend) projected to 2030.
          XGBoost is the most accurate model (MAE=43,467, R²=0.9975 on 2018-2019 test set).
        </p>
        <div className="overflow-x-auto max-h-[500px]">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border">
                <th className="text-left py-2 text-muted font-medium">State</th>
                <th className="text-right py-2 text-muted font-medium">XGBoost 2030</th>
                <th className="text-right py-2 text-muted font-medium">Final (Ensemble)</th>
                <th className="text-right py-2 text-muted font-medium">Lower</th>
                <th className="text-right py-2 text-muted font-medium">Upper</th>
              </tr>
            </thead>
            <tbody>
              {[...forecast2030]
                .sort((a: any, b: any) => b.forecast - a.forecast)
                .map((p: any) => (
                  <tr key={p.state} className="border-b border-border last:border-0">
                    <td className="py-2">{p.state}</td>
                    <td className="text-right py-2 font-mono text-xs">{formatNumber(p.forecast)}</td>
                    <td className="text-right py-2 font-mono text-xs">{formatNumber(p.final_forecast)}</td>
                    <td className="text-right py-2 font-mono text-xs text-muted">{formatNumber(p.lower_bound)}</td>
                    <td className="text-right py-2 font-mono text-xs text-muted">{formatNumber(p.upper_bound)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3 className="text-sm font-medium mb-4">National Aggregate (2030)</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-xs text-muted">XGBoost Total</p>
            <p className="text-lg font-semibold">{formatNumber(forecast2030.reduce((s: number, p: any) => s + p.forecast, 0))}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted">Ensemble Total</p>
            <p className="text-lg font-semibold">{formatNumber(forecast2030.reduce((s: number, p: any) => s + p.final_forecast, 0))}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted">States</p>
            <p className="text-lg font-semibold">{forecast2030.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ImportanceTab({ featureImportance }: any) {
  if (!featureImportance) return <ChartSkeleton />;

  const models = [...new Set(featureImportance.map((f: any) => f.model))];
  const [selectedModel, setSelectedModel] = useState(models[0]);

  const data = featureImportance
    .filter((f: any) => f.model === selectedModel)
    .sort((a: any, b: any) => b.importance - a.importance)
    .slice(0, 12);

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        {models.map((m: string) => (
          <button
            key={m}
            onClick={() => setSelectedModel(m)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              selectedModel === m ? "bg-accent text-white" : "bg-surface-2 text-muted hover:text-foreground"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="card">
        <h3 className="text-sm font-medium mb-4">Feature Importance — {selectedModel}</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
            <XAxis type="number" stroke="#6b6b7b" fontSize={11} />
            <YAxis dataKey="feature" type="category" stroke="#6b6b7b" fontSize={10} width={200} />
            <Tooltip
              contentStyle={{ background: "#1a1a24", border: "1px solid #2a2a3a", borderRadius: 8 }}
            />
            <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PredictTab() {
  const [state, setState] = useState("");
  const [year, setYear] = useState(2025);
  const [enrollmentLag1, setEnrollmentLag1] = useState(0);
  const [teachers, setTeachers] = useState(0);
  const [colleges, setColleges] = useState(0);
  const [ger, setGer] = useState(0);
  const [gpi, setGpi] = useState(1.0);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const { data: states } = useQuery({
    queryKey: ["states"],
    queryFn: api.states,
  });

  const handlePredict = async () => {
    if (!state || enrollmentLag1 <= 0) {
      setError("Please select a state and enter enrollment lag value.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await api.predict({
        state,
        year,
        enrollment_lag_1: enrollmentLag1,
        teachers_lag1: teachers,
        colleges_lag1: colleges,
        GER_lag1: ger,
        GPI_lag1: gpi,
      });
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
      }
    } catch (e: any) {
      setError(e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="card">
        <h3 className="text-sm font-medium mb-4">Custom Forecast (XGBoost)</h3>
        <p className="text-xs text-muted mb-4">
          Enter indicator values for a state and year to get a predicted enrollment forecast.
          Uses the trained XGBoost model.
        </p>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-muted block mb-1">State</label>
            <select
              value={state}
              onChange={(e) => setState(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
            >
              <option value="">Select state...</option>
              {states?.map((s: any) => (
                <option key={s.name} value={s.name}>{s.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Target Year</label>
            <input
              type="number"
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
            />
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Previous Year Enrollment (lag-1)</label>
            <input
              type="number"
              value={enrollmentLag1}
              onChange={(e) => setEnrollmentLag1(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
              placeholder="e.g., 4000000"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted block mb-1">Teachers (lag-1)</label>
              <input
                type="number"
                value={teachers}
                onChange={(e) => setTeachers(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
              />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Colleges (lag-1)</label>
              <input
                type="number"
                value={colleges}
                onChange={(e) => setColleges(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted block mb-1">GER (lag-1)</label>
              <input
                type="number"
                step="0.1"
                value={ger}
                onChange={(e) => setGer(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
              />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">GPI (lag-1)</label>
              <input
                type="number"
                step="0.01"
                value={gpi}
                onChange={(e) => setGpi(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm rounded-lg bg-surface-2 text-foreground border border-border"
              />
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary w-full"
          >
            {loading ? "Predicting..." : "Generate Forecast"}
          </button>

          {error && (
            <p className="text-xs text-danger">{error}</p>
          )}

          {result && (
            <div className="card bg-surface-2 mt-4">
              <h4 className="text-sm font-medium mb-2">Prediction Result</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-xs text-muted">State</span>
                  <span className="text-sm font-medium">{result.state}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-muted">Year</span>
                  <span className="text-sm font-medium">{result.year}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-muted">Predicted Enrollment</span>
                  <span className="text-lg font-semibold text-accent">{formatNumber(result.predicted_enrollment)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-muted">Model</span>
                  <span className="text-xs text-muted">{result.model}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function KpiCard({ icon: Icon, label, value }: { icon: any; label: string; value: string | number }) {
  return (
    <div className="card">
      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-accent/10 p-2">
          <Icon className="h-4 w-4 text-accent" />
        </div>
        <div>
          <p className="text-xs text-muted">{label}</p>
          <p className="text-lg font-semibold">{value}</p>
        </div>
      </div>
    </div>
  );
}
