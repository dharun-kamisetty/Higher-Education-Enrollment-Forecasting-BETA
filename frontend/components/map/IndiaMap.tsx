"use client";

import { useEffect, useState, useRef } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const GEO_TO_DATA: Record<string, string> = {
  "Andaman and Nicobar": "Andaman And Nicobar Islands",
  "Andhra Pradesh": "Andhra Pradesh",
  "Arunachal Pradesh": "Arunachal Pradesh",
  "Assam": "Assam",
  "Bihar": "Bihar",
  "Chandigarh": "Chandigarh",
  "Chhattisgarh": "Chhattisgarh",
  "Dadra and Nagar Haveli": "The Dadra And Nagar Haveli And Daman And Diu",
  "Daman and Diu": "The Dadra And Nagar Haveli And Daman And Diu",
  "Delhi": "Delhi",
  "Goa": "Goa",
  "Gujarat": "Gujarat",
  "Haryana": "Haryana",
  "Himachal Pradesh": "Himachal Pradesh",
  "Jammu and Kashmir": "Jammu And Kashmir",
  "Jharkhand": "Jharkhand",
  "Karnataka": "Karnataka",
  "Kerala": "Kerala",
  "Lakshadweep": "Lakshadweep",
  "Madhya Pradesh": "Madhya Pradesh",
  "Maharashtra": "Maharashtra",
  "Manipur": "Manipur",
  "Meghalaya": "Meghalaya",
  "Mizoram": "Mizoram",
  "Nagaland": "Nagaland",
  "Orissa": "Odisha",
  "Puducherry": "Puducherry",
  "Punjab": "Punjab",
  "Rajasthan": "Rajasthan",
  "Sikkim": "Sikkim",
  "Tamil Nadu": "Tamil Nadu",
  "Telangana": "Telangana",
  "Tripura": "Tripura",
  "Uttar Pradesh": "Uttar Pradesh",
  "Uttaranchal": "Uttarakhand",
  "West Bengal": "West Bengal",
  "Odisha": "Odisha",
  "Uttarakhand": "Uttarakhand",
};

interface MapProps {
  data: Record<string, number>;
  onStateClick?: (state: string) => void;
  selectedState?: string | null;
  colorScale?: (value: number, min: number, max: number) => string;
  legendTitle?: string;
}

function getColor(value: number, min: number, max: number): string {
  if (value == null || isNaN(value)) return "#1a1a24";
  const ratio = (value - min) / (max - min || 1);
  const r = Math.round(99);
  const g = Math.round(102);
  const b = Math.round(241 - 100 * ratio);
  return `rgb(${r}, ${g}, ${b})`;
}

function ResetView() {
  const map = useMap();
  useEffect(() => {
    map.setView([22.5, 82.5], 5);
  }, [map]);
  return null;
}

export default function IndiaMap({ data, onStateClick, selectedState, colorScale, legendTitle }: MapProps) {
  const [geoData, setGeoData] = useState<any>(null);
  const [hoveredState, setHoveredState] = useState<string | null>(null);
  const dataRef = useRef(data);
  dataRef.current = data;

  useEffect(() => {
    fetch("/geo/india_states.geojson")
      .then((r) => r.json())
      .then(setGeoData)
      .catch(console.error);
  }, []);

  const values = Object.values(data).filter((v) => !isNaN(v));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;

  const styleFeature = (feature: any) => {
    const geoName = feature.properties.NAME_1;
    const dataName = GEO_TO_DATA[geoName] || geoName;
    const value = data[dataName];
    const isSelected = selectedState === dataName;
    const isHovered = hoveredState === dataName;
    return {
      fillColor: colorScale ? colorScale(value, min, max) : getColor(value, min, max),
      weight: isSelected ? 3 : isHovered ? 2 : 1,
      opacity: 1,
      color: isSelected ? "#6366f1" : isHovered ? "#a5a5b5" : "#2a2a3a",
      fillOpacity: 0.8,
    };
  };

  const onEachFeature = (feature: any, layer: L.Layer) => {
    const geoName = feature.properties.NAME_1;
    const dataName = GEO_TO_DATA[geoName] || geoName;

    layer.bindTooltip(
      `<div style="font-size:12px"><strong>${dataName}</strong><br/>Loading...</div>`,
      { sticky: true }
    );

    layer.on({
      mouseover: (e) => {
        setHoveredState(dataName);
        const l = e.target;
        l.setStyle({ weight: 2, color: "#a5a5b5", fillOpacity: 0.9 });
        l.bringToFront();
        const currentValue = dataRef.current[dataName];
        l.setTooltipContent(
          `<div style="font-size:12px">
            <strong>${dataName}</strong><br/>
            ${currentValue != null ? Number(currentValue).toLocaleString() : "No data"}
          </div>`
        );
      },
      mouseout: (e) => {
        setHoveredState(null);
        const l = e.target;
        l.setStyle(styleFeature(feature));
      },
      click: () => {
        if (onStateClick) onStateClick(dataName);
      },
    });
  };

  if (!geoData) {
    return (
      <div className="flex items-center justify-center h-[520px] bg-surface rounded-xl">
        <div className="skeleton h-8 w-32" />
      </div>
    );
  }

  return (
    <div className="relative">
      <MapContainer
        center={[22.5, 82.5]}
        zoom={5}
        style={{ height: "520px", width: "100%", borderRadius: "12px" }}
        scrollWheelZoom={true}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        <GeoJSON
          key={Object.keys(data).length + "-" + Object.values(data).reduce((a: number, b: number) => a + b, 0)}
          data={geoData}
          style={styleFeature}
          onEachFeature={onEachFeature}
        />
        <ResetView />
      </MapContainer>

      <div className="absolute bottom-4 right-4 bg-surface/90 backdrop-blur rounded-lg p-3 border border-border text-xs">
        <p className="font-medium mb-2">{legendTitle || "Value"}</p>
        <div className="flex items-center gap-2">
          <div className="w-24 h-2 rounded" style={{
            background: "linear-gradient(to right, rgb(99,102,241), rgb(99,102,141))"
          }} />
        </div>
        <div className="flex justify-between mt-1 text-muted">
          <span>{min.toLocaleString()}</span>
          <span>{max.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}
