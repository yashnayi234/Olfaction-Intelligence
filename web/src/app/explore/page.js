"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useState } from "react";

const ODOR_CATEGORIES = [
  "Alliaceous","Ambrosial","Ammonia","Aromatic","Baked","Bakery","Balsamic",
  "Beverage","Bitter","Bland","Bloomy","Brown","Burnt","Caramel","Chemical",
  "Chlorinous","Chocolate","Citrus","Clean","Cooked","Dairy","Dry","Earthy",
  "Estery","Fecal","Fishy","Floral","Food","Foul","Fragrant","Fresh","Fried",
  "Fruity","Grassy","Green","Herbaceous","Honey","Ketonic","Lactonic","Meaty",
  "Medicinal","Metallic","Minty","Mossy","Musky","Musty","Nutty","Oily",
  "Onion","Orchid","Ozonic","Peppery","Phenolic","Pine","Plastic","Powdery",
  "Pungent","Ripe","Roasted","Rose","Rubber","Rum","Savory","Sharp","Smoky",
  "Soapy","Sour","Spicy","Sulfurous","Sweet","Tart","Tobacco","Tropical",
  "Vanilla","Vegetable","Vinegar","Waxy","Winey","Woody"
];

const ODOR_GROUPS = {
  "🌸 Floral & Sweet": ["Floral", "Rose", "Sweet", "Honey", "Vanilla", "Fragrant", "Powdery", "Orchid"],
  "🍊 Fruity & Citrus": ["Fruity", "Citrus", "Tropical", "Ripe", "Tart", "Winey", "Estery"],
  "🌿 Green & Herbal": ["Green", "Herbaceous", "Grassy", "Fresh", "Minty", "Pine", "Mossy"],
  "🪵 Woody & Earthy": ["Woody", "Earthy", "Musty", "Mossy", "Balsamic", "Smoky", "Tobacco"],
  "🍖 Savory & Food": ["Meaty", "Savory", "Baked", "Bakery", "Cooked", "Fried", "Roasted", "Food", "Dairy", "Nutty", "Caramel", "Chocolate"],
  "⚗️ Chemical & Sharp": ["Chemical", "Medicinal", "Metallic", "Plastic", "Rubber", "Pungent", "Sharp", "Chlorinous", "Ammonia", "Ozonic"],
  "🧅 Sulfurous & Pungent": ["Alliaceous", "Sulfurous", "Fecal", "Fishy", "Foul", "Onion"],
  "🧴 Clean & Aromatic": ["Clean", "Soapy", "Aromatic", "Musky", "Ambrosial"],
};

export default function ExplorePage() {
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredCategories = searchQuery
    ? ODOR_CATEGORIES.filter((c) =>
        c.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : ODOR_CATEGORIES;

  return (
    <>
      <Navbar />

      <div className="explore-page">
        <div className="container">
          <div className="section-header" style={{ paddingTop: "20px", marginBottom: "48px" }}>
            <div className="overline">Dataset Explorer</div>
            <h2>105 Odor Categories</h2>
            <p>
              Explore the full spectrum of olfactory descriptors used to train
              OlfacNet v2. Click a group to see its categories.
            </p>
          </div>

          {/* Stats Overview */}
          <div className="stats-bar" style={{ marginBottom: "48px", marginTop: 0 }}>
            <div className="stat-item">
              <div className="stat-number">44,543</div>
              <div className="stat-label">Molecules</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">105</div>
              <div className="stat-label">Odor Categories</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">52,947</div>
              <div className="stat-label">Odor Records</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">3,985</div>
              <div className="stat-label">Odorants</div>
            </div>
          </div>

          {/* Odor Groups */}
          <h3 style={{ marginBottom: "24px" }}>Odor Families</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "16px", marginBottom: "48px" }}>
            {Object.entries(ODOR_GROUPS).map(([group, odors]) => (
              <div
                key={group}
                className="glass-card"
                style={{
                  padding: "24px",
                  cursor: "pointer",
                  background: selectedGroup === group
                    ? "rgba(255,161,120,0.15)"
                    : "rgba(255,255,255,0.7)",
                  border: selectedGroup === group
                    ? "2px solid var(--peach)"
                    : "1px solid rgba(0,0,0,0.04)",
                }}
                onClick={() =>
                  setSelectedGroup(selectedGroup === group ? null : group)
                }
              >
                <h4 style={{ marginBottom: "12px" }}>{group}</h4>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {odors.map((odor) => (
                    <span
                      key={odor}
                      style={{
                        padding: "4px 12px",
                        background: "var(--cream)",
                        borderRadius: "100px",
                        fontSize: "0.8rem",
                        fontWeight: 500,
                      }}
                    >
                      {odor}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Full Category Grid */}
          <h3 style={{ marginBottom: "16px" }}>All Categories</h3>
          <div style={{ marginBottom: "24px" }}>
            <input
              type="text"
              className="smiles-input"
              placeholder="🔍 Search odor categories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ maxWidth: "400px" }}
              id="odor-search"
            />
          </div>

          <div className="odor-grid" style={{ marginBottom: "80px" }}>
            {filteredCategories.map((odor) => (
              <div className="odor-chip" key={odor}>
                {odor}
              </div>
            ))}
          </div>

          {/* Data Sources */}
          <div className="glass-card" style={{ marginBottom: "48px" }}>
            <h3 style={{ marginBottom: "16px" }}>📁 Data Sources</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "24px" }}>
              <div>
                <h4 style={{ color: "var(--peach)", marginBottom: "8px" }}>molecules.csv</h4>
                <p style={{ fontSize: "0.9rem" }}>5,105 molecules with CID, molecular weight, SMILES, and IUPAC names from PubChem</p>
              </div>
              <div>
                <h4 style={{ color: "var(--peach)", marginBottom: "8px" }}>odorants.csv</h4>
                <p style={{ fontSize: "0.9rem" }}>3,985 odorant compounds with SMILES, CAS numbers, and olfactory receptor counts</p>
              </div>
              <div>
                <h4 style={{ color: "var(--peach)", marginBottom: "8px" }}>odors.csv</h4>
                <p style={{ fontSize: "0.9rem" }}>52,947 odor records mapping chemicals to primary and sub-odor categories</p>
              </div>
              <div>
                <h4 style={{ color: "var(--peach)", marginBottom: "8px" }}>odorless.csv</h4>
                <p style={{ fontSize: "0.9rem" }}>1,124 confirmed odorless compounds for negative training examples</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </>
  );
}
