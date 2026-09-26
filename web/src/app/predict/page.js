"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const EXAMPLE_MOLECULES = [
  { name: "Ethanol", smiles: "CCO" },
  { name: "Vanillin", smiles: "O=Cc1ccc(O)c(OC)c1" },
  { name: "Limonene", smiles: "CC(=C)C1CCC(=CC1)C" },
  { name: "Menthol", smiles: "CC(C)C1CCC(C)CC1O" },
  { name: "Benzaldehyde", smiles: "O=Cc1ccccc1" },
  { name: "Eugenol", smiles: "COc1cc(CC=C)ccc1O" },
  { name: "Acetic Acid", smiles: "CC(=O)O" },
  { name: "Camphor", smiles: "CC1(C)C2CCC1(C)C(=O)C2" },
];

const ODOR_COLORS = {
  Floral: "#FFB4D4",
  Fruity: "#FFD93D",
  Citrus: "#FFA62F",
  Green: "#7BB661",
  Woody: "#A0845C",
  Spicy: "#FF6B6B",
  Sweet: "#FF9ED2",
  Herbaceous: "#6BCB77",
  Earthy: "#8B6F47",
  Chemical: "#7EB8DA",
  Balsamic: "#D4A373",
  Smoky: "#808080",
  Minty: "#4ECDC4",
  Nutty: "#C4A35A",
  Alliaceous: "#9B8BB4",
};

function getOdorColor(odorName) {
  for (const [key, color] of Object.entries(ODOR_COLORS)) {
    if (odorName.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return "#FFA178";
}

export default function PredictPage() {
  const [smiles, setSmiles] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handlePredict() {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: smiles.trim(), top_k: 15, threshold: 0.05 }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Prediction failed");
      }

      const data = await res.json();
      setPredictions(data);
      if (data.source === "demo") {
        setError("Running in demo mode — showing pattern-based predictions. Deploy the Python API for full model predictions.");
      }
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Navbar />

      <div className="predict-page">
        <div className="container">
          <div className="section-header" style={{ paddingTop: "20px", marginBottom: "48px" }}>
            <div className="overline">Olfactory Prediction</div>
            <h2>Predict Odor from Molecules</h2>
            <p>
              Enter a SMILES string to predict how a molecule smells.
              Our AI analyzes the molecular structure and returns an odor profile.
            </p>
          </div>

          <div className="predict-container">
            {/* Input Panel */}
            <div className="input-panel">
              <div className="glass-card">
                <div className="smiles-input-group">
                  <label htmlFor="smiles-input">SMILES String</label>
                  <input
                    id="smiles-input"
                    type="text"
                    className="smiles-input"
                    placeholder="Enter SMILES (e.g., CCO for ethanol)"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handlePredict()}
                  />
                </div>

                <button
                  className="btn btn-primary"
                  onClick={handlePredict}
                  disabled={loading || !smiles.trim()}
                  style={{ width: "100%", justifyContent: "center" }}
                  id="predict-btn"
                >
                  {loading ? (
                    <>
                      <div className="loading-spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></div>
                      Analyzing...
                    </>
                  ) : (
                    "🧪 Predict Odor Profile"
                  )}
                </button>

                <div style={{ marginTop: "24px" }}>
                  <label style={{ display: "block", fontWeight: 600, marginBottom: 8, fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--warm-gray-dark)" }}>
                    Example Molecules
                  </label>
                  <div className="example-molecules">
                    {EXAMPLE_MOLECULES.map((mol) => (
                      <button
                        key={mol.smiles}
                        className="example-pill"
                        onClick={() => setSmiles(mol.smiles)}
                      >
                        {mol.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {smiles && (
                <div className="glass-card" style={{ marginTop: "20px", padding: "24px" }}>
                  <h4 style={{ marginBottom: 12 }}>Molecule Info</h4>
                  <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.9rem", color: "var(--warm-gray-dark)", wordBreak: "break-all" }}>
                    <strong>SMILES:</strong> {smiles}
                  </div>
                </div>
              )}
            </div>

            {/* Results Panel */}
            <div className="results-panel">
              {!predictions && !loading && (
                <div className="glass-card" style={{ textAlign: "center", padding: "80px 40px" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "16px" }}>🧬</div>
                  <h3 style={{ marginBottom: "12px", color: "var(--charcoal)" }}>
                    Enter a Molecule
                  </h3>
                  <p>
                    Type a SMILES string or click an example molecule to see its
                    predicted odor profile.
                  </p>
                </div>
              )}

              {error && (
                <div style={{ 
                  padding: "12px 20px", 
                  background: "rgba(255,107,107,0.1)", 
                  borderRadius: "12px", 
                  fontSize: "0.9rem",
                  color: "#c44",
                  marginBottom: "16px",
                  border: "1px solid rgba(255,107,107,0.2)"
                }}>
                  ⚠ {error} — Showing demo predictions below.
                </div>
              )}

              {predictions && (
                <div className="glass-card animate-in">
                  <h3 style={{ marginBottom: "8px" }}>
                    Odor Profile
                  </h3>
                  <p style={{ marginBottom: "24px", fontSize: "0.95rem" }}>
                    Top predicted odor categories for <code style={{ 
                      background: "var(--cream)", 
                      padding: "2px 8px", 
                      borderRadius: "6px",
                      fontSize: "0.85rem"
                    }}>{predictions.smiles}</code>
                  </p>

                  {predictions.predictions.map((pred, i) => (
                    <div className="prediction-bar" key={pred.odor}>
                      <div className="prediction-rank">{i + 1}</div>
                      <div className="prediction-name">
                        <span style={{ 
                          display: "inline-block",
                          width: 10, 
                          height: 10, 
                          borderRadius: "50%", 
                          background: getOdorColor(pred.odor),
                          marginRight: 8
                        }}></span>
                        {pred.odor}
                      </div>
                      <div className="prediction-confidence">
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{ width: `${pred.confidence * 100}%` }}
                          ></div>
                        </div>
                        <div className="confidence-value">
                          {(pred.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Visual Radar Summary */}
                  <div style={{ 
                    marginTop: "32px", 
                    padding: "24px",
                    background: "var(--cream)",
                    borderRadius: "var(--border-radius-sm)",
                    textAlign: "center"
                  }}>
                    <div style={{ 
                      display: "flex", 
                      flexWrap: "wrap", 
                      gap: "8px", 
                      justifyContent: "center" 
                    }}>
                      {predictions.predictions.slice(0, 6).map((pred) => (
                        <span
                          key={pred.odor}
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "6px",
                            padding: "8px 16px",
                            background: getOdorColor(pred.odor) + "30",
                            border: `2px solid ${getOdorColor(pred.odor)}`,
                            borderRadius: "100px",
                            fontSize: "0.85rem",
                            fontWeight: 600,
                          }}
                        >
                          {pred.odor}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </>
  );
}
