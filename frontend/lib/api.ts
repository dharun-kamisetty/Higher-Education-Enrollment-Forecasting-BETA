const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  states: () => fetcher<State[]>("/api/states"),
  stateData: (state: string) => fetcher<StateData>(`/api/states/${encodeURIComponent(state)}/data`),
  latestEnrollment: () => fetcher<EnrollmentPoint[]>("/api/states/enrollment/latest"),
  stateForecast: (state: string) => fetcher<StateForecast>(`/api/states/${encodeURIComponent(state)}/forecast`),
  modelComparison: () => fetcher<ModelMetric[]>("/api/models/comparison"),
  bestModel: () => fetcher<BestModel>("/api/models/best"),
  featureImportance: () => fetcher<FeatureImportance[]>("/api/feature-importance"),
  bestParams: () => fetcher<Record<string, unknown>>("/api/best-params"),
  allForecasts: () => fetcher<ForecastPoint[]>("/api/forecast/all"),
  forecast2030: () => fetcher<Forecast2030Point[]>("/api/forecast/2030"),
  forecast2030State: (state: string) => fetcher<Forecast2030Series>(`/api/forecast/2030/${encodeURIComponent(state)}`),
  forecast2030All: () => fetcher<Forecast2030Row[]>("/api/forecast/2030-all"),
  forecastByYear: (year: number) => fetcher<ForecastByYearPoint[]>(`/api/forecast/by-year/${year}`),
  availableYears: () => fetcher<{ years: number[] }>("/api/forecast/available-years"),
  stateTimeline: (state: string) => fetcher<StateTimeline>(`/api/forecast/state-timeline/${encodeURIComponent(state)}`),
  assistantChat: (message: string, history: { role: string; content: string }[]) =>
    fetch(`${API_BASE}/api/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    }).then((r) => r.json()),
  indicatorTrends: () => fetcher<IndicatorTrend[]>("/api/forecast/indicator-trends"),
  methodology: () => fetcher<{ text: string }>("/api/forecast/methodology"),
  predict: (body: PredictRequest) =>
    fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
};

// Types
export interface State {
  name: string;
  code: number;
}

export interface StateData {
  state: string;
  years: number[];
  enrollment: number[];
  enrollment_lag_1: number[];
  teachers: number[];
  colleges: number[];
  GER: number[];
  GPI: number[];
}

export interface EnrollmentPoint {
  state: string;
  code: number;
  enrollment: number;
}

export interface StateForecast {
  state: string;
  scenarios: Record<string, { years: number[]; values: number[] }>;
}

export interface ModelMetric {
  Model: string;
  MAE: number;
  RMSE: number;
  MAPE: number;
  R2: number;
}

export interface BestModel {
  model: string;
  metrics: { MAE: number; RMSE: number; MAPE: number; R2: number };
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  model: string;
}

export interface ForecastPoint {
  state: string;
  forecast: number;
  year: number;
}

export interface Forecast2030Point {
  state: string;
  year: number;
  forecast: number;
  final_forecast: number;
  lower_bound: number;
  upper_bound: number;
}

export interface Forecast2030Series {
  state: string;
  years: number[];
  xgboost: number[];
  final: number[];
  lower: number[];
  upper: number[];
}

export interface Forecast2030Row {
  state: string;
  year: number;
  xgboost_forecast: number;
  final_forecast: number;
  lower_bound: number;
  upper_bound: number;
  [key: string]: unknown;
}

export interface ForecastByYearPoint {
  state: string;
  year: number;
  xgboost: number;
  ensemble: number;
  lower: number;
  upper: number;
}

export interface StateTimeline {
  state: string;
  years: number[];
  xgboost: number[];
  ensemble: number[];
  lower: number[];
  upper: number[];
}

export interface IndicatorTrend {
  State: string;
  Indicator: string;
  Method: string;
  R2: number;
  [key: string]: unknown;
}

export interface PredictRequest {
  state: string;
  year: number;
  enrollment_lag_1: number;
  teachers_lag1?: number;
  colleges_lag1?: number;
  GER_lag1?: number;
  GPI_lag1?: number;
}
