"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function ResearchPage() {
  return (
    <>
      <Navbar />

      <div className="research-page">
        <div className="container">
          <div className="section-header" style={{ paddingTop: "20px", marginBottom: "48px" }}>
            <div className="overline">Research & Architecture</div>
            <h2>OlfacNet v2 Deep Dive</h2>
            <p>
              Explore the model architecture, training methodology, and
              performance metrics behind our olfactory AI platform.
            </p>
          </div>

          {/* Performance Metrics */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">99.05%</div>
              <div className="metric-label">Per-Class Accuracy</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">0.037</div>
              <div className="metric-label">Test Loss</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">1.69M</div>
              <div className="metric-label">Parameters</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">105</div>
              <div className="metric-label">Output Classes</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">44.5K</div>
              <div className="metric-label">Training Samples</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">20</div>
              <div className="metric-label">Training Epochs</div>
            </div>
          </div>

          {/* Architecture */}
          <div className="glass-card" style={{ marginBottom: "32px" }}>
            <h3 style={{ marginBottom: "24px" }}>🏗 Model Architecture — OlfacNet v2</h3>
            <div className="arch-diagram">{`
                    SMILES Input
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
       Mol → Graph              Mol → Fingerprint
            │                         │
       ┌────┴────┐              Morgan FP (2048-bit)
       │  GCN    │                    │
       │ Layer 1 │              ┌─────┴─────┐
       │ (128)   │              │  FC Block  │
       │  + BN   │              │   (512)    │
       │  + ReLU │              │   + BN     │
       └────┬────┘              │   + ReLU   │
            │                   └─────┬─────┘
       ┌────┴────┐                    │
       │  GCN    │              ┌─────┴─────┐
       │ Layer 2 │              │  FC Block  │
       │ (128)   │              │   (256)    │
       │  + BN   │              │   + BN     │
       └────┬────┘              │   + ReLU   │
            │                   └─────┬─────┘
       ┌────┴────┐                    │
       │  GCN    │              ┌─────┴─────┐
       │ Layer 3 │              │  FC (128)  │
       │ (128)   │              │  output    │
       └────┬────┘              └─────┬─────┘
            │                         │
       Global Mean                    │
       Pooling                        │
            │                         │
            └────────┬────────────────┘
                     │
            Multi-Head Attention
               Fusion (256)
              4 Attention Heads
                     │
               ┌─────┴─────┐
               │ Classifier │
               │  FC (256)  │
               │  → BN+ReLU │
               │  → Dropout │
               │  FC (128)  │
               │  → BN+ReLU │
               │  FC (105)  │
               │  → Sigmoid │
               └────────────┘
            `}</div>
          </div>

          {/* Comparison Table */}
          <div className="glass-card" style={{ marginBottom: "32px" }}>
            <h3 style={{ marginBottom: "24px" }}>📊 v1 vs v2 Comparison</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid var(--warm-gray)" }}>
                    <th style={{ textAlign: "left", padding: "12px 16px", fontWeight: 700 }}>Feature</th>
                    <th style={{ textAlign: "left", padding: "12px 16px", fontWeight: 700, color: "var(--warm-gray-dark)" }}>OlfacNet v1</th>
                    <th style={{ textAlign: "left", padding: "12px 16px", fontWeight: 700, color: "var(--peach)" }}>OlfacNet v2 ✨</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Graph Convolution", "nn.Linear (fake GCN)", "True GCN with A·X·W"],
                    ["Molecular Features", "Random mock spectrum", "Morgan Fingerprints (2048-bit)"],
                    ["Fusion Method", "Simple concatenation", "Multi-Head Attention (4 heads)"],
                    ["Loss Function", "BCE Loss", "Focal Loss (α=0.25, γ=2.0)"],
                    ["Regularization", "None", "Dropout + BatchNorm + Weight Decay"],
                    ["LR Scheduling", "Fixed LR", "ReduceLROnPlateau"],
                    ["Early Stopping", "No", "Yes (patience=7)"],
                    ["Gradient Clipping", "No", "Max norm = 1.0"],
                    ["Metrics", "Accuracy only", "mAP, AUC-ROC, F1, Hamming Loss"],
                    ["Project Structure", "Single notebook", "Modular Python package"],
                    ["API", "None", "FastAPI REST endpoints"],
                    ["Web UI", "None", "Next.js premium interface"],
                  ].map(([feature, v1, v2]) => (
                    <tr key={feature} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                      <td style={{ padding: "12px 16px", fontWeight: 600 }}>{feature}</td>
                      <td style={{ padding: "12px 16px", color: "var(--warm-gray-dark)" }}>{v1}</td>
                      <td style={{ padding: "12px 16px" }}>{v2}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Tech Stack */}
          <div className="glass-card" style={{ marginBottom: "32px" }}>
            <h3 style={{ marginBottom: "24px" }}>🛠 Technology Stack</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px" }}>
              {[
                { name: "PyTorch", desc: "Deep learning framework", icon: "🔥" },
                { name: "RDKit", desc: "Chemical informatics", icon: "⚗️" },
                { name: "FastAPI", desc: "REST API server", icon: "⚡" },
                { name: "Next.js", desc: "React web framework", icon: "▲" },
                { name: "NumPy", desc: "Numerical computing", icon: "📐" },
                { name: "scikit-learn", desc: "ML evaluation metrics", icon: "📊" },
              ].map((tech) => (
                <div
                  key={tech.name}
                  style={{
                    padding: "20px",
                    background: "var(--cream)",
                    borderRadius: "var(--border-radius-sm)",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: "2rem", marginBottom: "8px" }}>{tech.icon}</div>
                  <div style={{ fontWeight: 700, marginBottom: "4px" }}>{tech.name}</div>
                  <div style={{ fontSize: "0.85rem", color: "var(--warm-gray-dark)" }}>{tech.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key Innovations */}
          <div className="glass-card" style={{ marginBottom: "48px" }}>
            <h3 style={{ marginBottom: "24px" }}>🔬 Key Innovations</h3>
            <div style={{ display: "grid", gap: "20px" }}>
              <div style={{ padding: "20px", background: "rgba(255,161,120,0.08)", borderRadius: "var(--border-radius-sm)", borderLeft: "4px solid var(--peach)" }}>
                <h4 style={{ marginBottom: "8px" }}>Multi-Modal Fusion via Attention</h4>
                <p style={{ fontSize: "0.95rem" }}>
                  Instead of simple concatenation, OlfacNet v2 uses multi-head self-attention to dynamically
                  weight the importance of graph structure vs. fingerprint features for each molecule.
                  This allows the model to rely more on 3D topology for some molecules and more on 
                  substructural patterns for others.
                </p>
              </div>
              <div style={{ padding: "20px", background: "rgba(213,247,195,0.15)", borderRadius: "var(--border-radius-sm)", borderLeft: "4px solid var(--sage-dark)" }}>
                <h4 style={{ marginBottom: "8px" }}>True Graph Convolution</h4>
                <p style={{ fontSize: "0.95rem" }}>
                  Our GCN implements D⁻¹/²·A·D⁻¹/² normalized adjacency multiplication following
                  Kipf & Welling (2017), unlike v1 which only used linear layers. This enables 
                  proper message passing between atoms through chemical bonds.
                </p>
              </div>
              <div style={{ padding: "20px", background: "rgba(232,201,122,0.1)", borderRadius: "var(--border-radius-sm)", borderLeft: "4px solid var(--gold)" }}>
                <h4 style={{ marginBottom: "8px" }}>Focal Loss for Class Imbalance</h4>
                <p style={{ fontSize: "0.95rem" }}>
                  With 105 odor categories of varying frequency, standard BCE loss leads to
                  models that predict only common odors. Focal loss down-weights easy negatives
                  and focuses training on rare, hard-to-classify odor categories.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </>
  );
}
