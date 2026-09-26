import { NextResponse } from "next/server";

/**
 * Serverless prediction endpoint for Vercel deployment.
 * 
 * When deployed on Vercel (no Python backend), this returns realistic
 * demo predictions based on simple SMILES pattern matching.
 * For production, this would proxy to a hosted Python API.
 */

const ODOR_PROFILES = {
  // Alcohols
  "CCO": [
    { odor: "Sweet", confidence: 0.89 },
    { odor: "Chemical/ Hydrocarbon", confidence: 0.76 },
    { odor: "Fragrant", confidence: 0.65 },
    { odor: "Fruity", confidence: 0.42 },
    { odor: "Medicinal/ Alcohol", confidence: 0.38 },
  ],
  "CO": [
    { odor: "Chemical/ Hydrocarbon", confidence: 0.91 },
    { odor: "Pungent", confidence: 0.72 },
    { odor: "Sweet", confidence: 0.34 },
  ],
  // Vanillin
  "O=Cc1ccc(O)c(OC)c1": [
    { odor: "Sweet", confidence: 0.96 },
    { odor: "Fragrant", confidence: 0.91 },
    { odor: "Floral", confidence: 0.78 },
    { odor: "Bakery", confidence: 0.65 },
    { odor: "Caramel", confidence: 0.58 },
    { odor: "Chocolate", confidence: 0.42 },
  ],
  // Limonene
  "CC(=C)C1CCC(=CC1)C": [
    { odor: "Citrus", confidence: 0.95 },
    { odor: "Lemon", confidence: 0.88 },
    { odor: "Fruity", confidence: 0.82 },
    { odor: "Fresh", confidence: 0.71 },
    { odor: "Green", confidence: 0.45 },
    { odor: "Terpenes/ Pine/ Lemon", confidence: 0.41 },
  ],
  // Menthol
  "CC(C)C1CCC(C)CC1O": [
    { odor: "Minty", confidence: 0.97 },
    { odor: "Fresh", confidence: 0.85 },
    { odor: "Herbaceous", confidence: 0.62 },
    { odor: "Sweet", confidence: 0.41 },
    { odor: "Woody", confidence: 0.28 },
  ],
  // Benzaldehyde
  "O=Cc1ccccc1": [
    { odor: "Nutty", confidence: 0.88 },
    { odor: "Sweet", confidence: 0.82 },
    { odor: "Fruity", confidence: 0.76 },
    { odor: "Fragrant", confidence: 0.65 },
    { odor: "Floral", confidence: 0.38 },
  ],
  // Eugenol
  "COc1cc(CC=C)ccc1O": [
    { odor: "Spices", confidence: 0.94 },
    { odor: "Sweet", confidence: 0.78 },
    { odor: "Woody", confidence: 0.65 },
    { odor: "Herbaceous", confidence: 0.52 },
    { odor: "Fragrant", confidence: 0.45 },
    { odor: "Medicinal/ Phenolic", confidence: 0.38 },
  ],
  // Acetic acid
  "CC(=O)O": [
    { odor: "Pungent", confidence: 0.93 },
    { odor: "Sharp/ Pungent", confidence: 0.87 },
    { odor: "Sickening", confidence: 0.62 },
    { odor: "Fruity", confidence: 0.38 },
    { odor: "Sweet", confidence: 0.21 },
  ],
  // Camphor
  "CC1(C)C2CCC1(C)C(=O)C2": [
    { odor: "Minty", confidence: 0.86 },
    { odor: "Woody", confidence: 0.79 },
    { odor: "Herbaceous", confidence: 0.71 },
    { odor: "Medicinal/ Alcohol", confidence: 0.64 },
    { odor: "Fresh", confidence: 0.52 },
  ],
};

function generatePrediction(smiles) {
  // Check for exact match first
  if (ODOR_PROFILES[smiles]) {
    return ODOR_PROFILES[smiles];
  }

  // Pattern-based heuristic for unknown molecules
  const predictions = [];
  const s = smiles.toLowerCase();

  if (s.includes("o") && s.includes("=o")) {
    predictions.push({ odor: "Chemical/ Hydrocarbon", confidence: 0.72 });
  }
  if (s.includes("c1ccc") || s.includes("c1=cc")) {
    predictions.push({ odor: "Aromatic", confidence: 0.68 });
  }
  if (s.includes("oh") || s.includes("(o)")) {
    predictions.push({ odor: "Sweet", confidence: 0.61 });
  }
  if (s.includes("s")) {
    predictions.push({ odor: "Sulphur/ Cabbage/ Garlic", confidence: 0.74 });
  }
  if (s.includes("n")) {
    predictions.push({ odor: "Ammonia", confidence: 0.55 });
  }
  if (s.length > 20) {
    predictions.push({ odor: "Woody", confidence: 0.52 });
    predictions.push({ odor: "Fragrant", confidence: 0.48 });
  }

  // Always add some common predictions
  if (predictions.length === 0) {
    predictions.push(
      { odor: "Chemical/ Hydrocarbon", confidence: 0.65 },
      { odor: "Sweet", confidence: 0.52 },
      { odor: "Fragrant", confidence: 0.41 },
    );
  }

  predictions.push(
    { odor: "Fruity", confidence: 0.35 + Math.random() * 0.2 },
    { odor: "Green", confidence: 0.25 + Math.random() * 0.15 },
  );

  // Sort by confidence
  predictions.sort((a, b) => b.confidence - a.confidence);
  return predictions.slice(0, 8);
}

export async function POST(request) {
  try {
    const body = await request.json();
    const { smiles, top_k = 10, threshold = 0.05 } = body;

    if (!smiles || typeof smiles !== "string") {
      return NextResponse.json(
        { detail: "SMILES string is required" },
        { status: 400 }
      );
    }

    // Try the Python backend first (if running locally)
    try {
      const backendUrl = process.env.API_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, top_k, threshold }),
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const data = await res.json();
        return NextResponse.json(data);
      }
    } catch {
      // Backend not available — use built-in predictions
    }

    // Fallback: use demo predictions
    const predictions = generatePrediction(smiles);

    return NextResponse.json({
      smiles,
      predictions: predictions.filter((p) => p.confidence >= threshold).slice(0, top_k),
      full_profile: {},
      source: "demo",
    });
  } catch (err) {
    return NextResponse.json(
      { detail: err.message || "Internal server error" },
      { status: 500 }
    );
  }
}
