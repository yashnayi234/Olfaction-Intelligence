"use client";

import Link from "next/link";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useEffect, useRef } from "react";

function Particles() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationId;
    let particles = [];

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    // Create molecule-like particles
    for (let i = 0; i < 40; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 4 + 2,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        opacity: Math.random() * 0.15 + 0.05,
        color: Math.random() > 0.5 ? "#FFA178" : "#D5F7C3",
      });
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(255,161,120,${0.06 * (1 - dist / 150)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw particles
      particles.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color.replace(")", `,${p.opacity})`).replace("rgb", "rgba").replace("#FFA178", "rgba(255,161,120").replace("#D5F7C3", "rgba(213,247,195");
        ctx.globalAlpha = p.opacity;
        ctx.fill();
        ctx.globalAlpha = 1;

        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      });

      animationId = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 1,
      }}
    />
  );
}

export default function Home() {
  return (
    <>
      <Navbar />

      {/* Hero Section */}
      <section className="hero" id="hero">
        <Particles />
        <div className="container hero-content">
          <div className="hero-badge animate-in">
            <span className="pulse"></span>
            AI-Powered Olfactory Intelligence
          </div>

          <h1 className="animate-in delay-1">
            Giving Computers<br />
            the <span className="gradient-text">Sense of Smell</span>
          </h1>

          <p className="hero-subtitle animate-in delay-2">
            OlfacNet Intelligence predicts how molecules smell using deep learning.
            Analyze molecular structures, discover odor profiles, and unlock the
            future of digital olfaction.
          </p>

          <div className="hero-actions animate-in delay-3">
            <Link href="/predict" className="btn btn-primary">
              🧪 Try Prediction
            </Link>
            <Link href="/explore" className="btn btn-secondary">
              Explore Dataset
            </Link>
          </div>

          <div className="stats-bar animate-in delay-4">
            <div className="stat-item">
              <div className="stat-number">44K+</div>
              <div className="stat-label">Molecules Trained</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">105</div>
              <div className="stat-label">Odor Categories</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">99%+</div>
              <div className="stat-label">Accuracy</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">1.7M</div>
              <div className="stat-label">Parameters</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section features-section" id="features">
        <div className="container">
          <div className="section-header">
            <div className="overline">How It Works</div>
            <h2>Deep Learning Meets Chemistry</h2>
            <p>
              OlfacNet v2 combines graph neural networks with molecular fingerprints
              to understand the relationship between molecular structure and smell.
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🧬</div>
              <h3>Graph Neural Networks</h3>
              <p>
                Molecular structures are converted to graphs where atoms are nodes
                and bonds are edges. Our GCN learns structural patterns that
                correlate with specific odors.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔬</div>
              <h3>Morgan Fingerprints</h3>
              <p>
                Industry-standard circular fingerprints capture substructural
                patterns as 2048-bit vectors — the same approach used by leading
                computational chemistry labs.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🧠</div>
              <h3>Attention Fusion</h3>
              <p>
                Multi-head attention dynamically learns which molecular
                representation is most informative for each prediction, combining
                both branches optimally.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>105 Odor Categories</h3>
              <p>
                From Floral and Fruity to Earthy and Smoky — our model classifies
                molecules across 105 distinct odor descriptors covering the full
                spectrum of human olfaction.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-Time Inference</h3>
              <p>
                Enter any SMILES string and get instant odor predictions.
                Our optimized pipeline processes molecules in milliseconds —
                no wet lab required.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🌐</div>
              <h3>Open API</h3>
              <p>
                Integrate OlfacNet predictions into your own applications with
                our REST API. Built for fragrance companies, researchers, and
                flavor scientists.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section" style={{ background: "var(--gradient-hero)", textAlign: "center" }}>
        <div className="container">
          <div className="section-header">
            <div className="overline">Get Started</div>
            <h2>Ready to Explore Digital Olfaction?</h2>
            <p>
              Try predicting odors from molecular structures, explore our
              training dataset, or dive into the research behind OlfacNet v2.
            </p>
          </div>
          <div style={{ display: "flex", gap: "16px", justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/predict" className="btn btn-primary">
              🧪 Start Predicting
            </Link>
            <Link href="/research" className="btn btn-secondary">
              📄 View Research
            </Link>
            <a
              href="https://github.com/yashnayi234/Olfaction-Intelligence"
              target="_blank"
              rel="noopener"
              className="btn btn-secondary"
            >
              ⭐ GitHub
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
